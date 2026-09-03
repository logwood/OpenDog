[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'restart')]
    [string] $Action = 'start',

    [switch] $NoBrowser,

    [ValidateSet('cuda', 'cpu')]
    [string] $Provider = 'cuda',

    [ValidateSet(
        'production', 'candidate', 'rollback', 'legacy-semantic',
        'research-bifor', 'research-agent'
    )]
    [string] $Model = 'production',

    [string] $PythonExe = $(
        if ($env:PET_REID_PYTHON) { $env:PET_REID_PYTHON }
        elseif (Test-Path -LiteralPath 'D:\CondaData\envs\torch312\python.exe') {
            'D:\CondaData\envs\torch312\python.exe'
        }
        else { 'python' }
    ),

    [string] $GalleryDir = '',

    [int] $PythonPort = 8000,
    [int] $JavaPort = 8080,
    [int] $FrontendPort = 3000,

    [switch] $Lan,

    [string] $LanAddress = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceRoot = Join-Path $workspaceRoot 'src\Pet-ReID-IMAG'
$deploymentManifest = Join-Path $workspaceRoot 'models\deployment.json'
if (-not (Test-Path -LiteralPath $deploymentManifest -PathType Leaf)) {
    throw "Deployment manifest was not found: $deploymentManifest"
}
$deployment = Get-Content -LiteralPath $deploymentManifest -Raw -Encoding UTF8 | ConvertFrom-Json
$profileProperty = $deployment.runtime_profiles.PSObject.Properties[$Model]
if ($null -eq $profileProperty) {
    throw "Runtime profile was not found in the deployment manifest: $Model"
}
$runtimeProfile = $profileProperty.Value
$runtimeDir = Join-Path $workspaceRoot 'artifacts\workspace_logs\quick_start'
$stateFile = Join-Path $runtimeDir 'runtime.json'
$adminKeyFile = Join-Path $runtimeDir 'admin-key.txt'

function Resolve-LanIPv4([string] $RequestedAddress) {
    if (-not [string]::IsNullOrWhiteSpace($RequestedAddress)) {
        $parsed = $null
        if (-not [System.Net.IPAddress]::TryParse($RequestedAddress, [ref] $parsed) -or
            $parsed.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork -or
            [System.Net.IPAddress]::IsLoopback($parsed)) {
            throw "LanAddress must be a non-loopback IPv4 address: $RequestedAddress"
        }
        return $parsed.IPAddressToString
    }

    $candidates = @()
    try {
        $configurations = @(Get-NetIPConfiguration -ErrorAction Stop |
            Where-Object { $_.NetAdapter.Status -eq 'Up' -and $_.IPv4DefaultGateway })
        $physicalConfigurations = @($configurations |
            Where-Object { $_.NetAdapter.HardwareInterface })
        $preferredConfigurations = if ($physicalConfigurations) {
            $physicalConfigurations
        }
        else { $configurations }
        $candidates = @($preferredConfigurations |
            ForEach-Object { $_.IPv4Address.IPAddress })
    }
    catch { $candidates = @() }

    if (-not $candidates) {
        $candidates = @([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
            Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } |
            ForEach-Object { $_.IPAddressToString })
    }
    $usable = @($candidates |
        Where-Object { $_ -and $_ -notlike '127.*' -and $_ -notlike '169.254.*' -and $_ -ne '0.0.0.0' } |
        Select-Object -Unique)
    $private = @($usable | Where-Object {
        $_ -match '^10\.' -or
        $_ -match '^192\.168\.' -or
        $_ -match '^172\.(1[6-9]|2[0-9]|3[01])\.'
    })
    $selected = if ($private) { @($private | Select-Object -First 1) }
        else { @($usable | Select-Object -First 1) }
    if (-not $selected) {
        throw 'No LAN IPv4 address was found. Pass -LanAddress with the address used by the phone.'
    }
    return [string] $selected
}

$pythonUrl = "http://127.0.0.1:$PythonPort"
$javaUrl = "http://127.0.0.1:$JavaPort"
$frontendCheckUrl = "http://127.0.0.1:$FrontendPort"
$frontendListenHost = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
$resolvedLanAddress = if ($Lan) { Resolve-LanIPv4 $LanAddress } else { '' }
$frontendUrl = if ($Lan) {
    "http://$($resolvedLanAddress):$FrontendPort"
}
else { "http://localhost:$FrontendPort" }

function Write-Step([string] $Message) {
    Write-Host "[Pet ReID] $Message" -ForegroundColor Cyan
}

function Write-Ok([string] $Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Resolve-RepoPath([string] $Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $Path))
}

function Resolve-Executable([string] $Candidate, [string] $Description) {
    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        $resolved = [System.IO.Path]::GetFullPath($Candidate)
        Assert-File $resolved $Description
        return $resolved
    }
    $command = Get-Command $Candidate -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "$Description was not found on PATH: $Candidate"
    }
    return $command.Source
}

function Test-TcpPort([int] $Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(400)) { return $false }
        $client.EndConnect($pending)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Get-Json([string] $Uri, [int] $TimeoutSeconds = 3) {
    try {
        return Invoke-RestMethod -Uri $Uri -Method Get -TimeoutSec $TimeoutSeconds
    }
    catch { return $null }
}

function Test-Frontend {
    try {
        $response = Invoke-WebRequest -Uri "$frontendCheckUrl/" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    }
    catch { return $false }
}

function Get-StackStatus {
    $python = Get-Json "$pythonUrl/health"
    $java = Get-Json "$javaUrl/v1/upstream-health"
    $frontend = Test-Frontend

    $provider = if ($python) { [string] $python.backend.provider } else { '-' }
    $backend = if ($python) { [string] $python.backend.backend } else { '-' }
    $profile = if ($python) { [string] $python.backend.deployment_profile } else { '-' }
    $fusion = if ($python) { [string] $python.backend.fusion_mode } else { '-' }
    $singleGraph = if ($python) { [bool] $python.backend.single_graph } else { $false }
    $modelHash = if ($python) { [string] $python.backend.model_sha256 } else { '-' }
    $agent = if ($python -and $python.backend.agent) {
        [string] $python.backend.agent.version
    }
    else { '-' }
    $pets = if ($java) { [string] $java.gallery.pets } else { '-' }
    $references = if ($java) { [string] $java.gallery.reference_images } else { '-' }

    [pscustomobject]@{
        Python = if ($python) { 'ready' } else { 'offline' }
        Java = if ($java) { 'ready' } else { 'offline' }
        Frontend = if ($frontend) { 'ready' } else { 'offline' }
        Provider = $provider
        Backend = $backend
        Profile = $profile
        Fusion = $fusion
        SingleGraph = $singleGraph
        ModelHash = $modelHash
        Agent = $agent
        Pets = $pets
        References = $references
        Ready = [bool] ($python -and $java -and $frontend)
    }
}

function Show-Status {
    $status = Get-StackStatus
    Write-Host ''
    Write-Host 'Service      Status' -ForegroundColor DarkGray
    Write-Host ('Python ONNX  {0}' -f $status.Python)
    Write-Host ('Java API     {0}' -f $status.Java)
    Write-Host ('Web UI       {0}' -f $status.Frontend)
    Write-Host ''
    Write-Host ('Provider     {0}' -f $status.Provider)
    Write-Host ('Backend      {0}' -f $status.Backend)
    Write-Host ('Profile      {0}' -f $status.Profile)
    Write-Host ('Fusion       {0}' -f $status.Fusion)
    Write-Host ('Agent        {0}' -f $status.Agent)
    Write-Host ('Gallery      {0} pets / {1} references' -f $status.Pets, $status.References)
    Write-Host ('URL          {0}' -f $frontendUrl)
    return $status
}

function Wait-JsonEndpoint(
    [string] $Name,
    [string] $Uri,
    [int] $TimeoutSeconds,
    [System.Diagnostics.Process] $Process
) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $result = Get-Json $Uri 2
        if ($result) { return $result }
        if ($Process -and $Process.HasExited) {
            throw "$Name exited before becoming ready (exit code $($Process.ExitCode))."
        }
        Start-Sleep -Milliseconds 700
    }
    throw "$Name did not become ready within $TimeoutSeconds seconds."
}

function Wait-Frontend([System.Diagnostics.Process] $Process, [int] $TimeoutSeconds = 90) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Frontend) { return }
        if ($Process -and $Process.HasExited) {
            throw "Frontend exited before becoming ready (exit code $($Process.ExitCode))."
        }
        Start-Sleep -Milliseconds 600
    }
    throw "Frontend did not become ready within $TimeoutSeconds seconds."
}

function New-ProcessRecord(
    [string] $Name,
    [System.Diagnostics.Process] $Process,
    [string] $Stdout,
    [string] $Stderr
) {
    return [pscustomobject]@{
        name = $Name
        pid = $Process.Id
        start_time_utc = $Process.StartTime.ToUniversalTime().ToString('o')
        owned = $true
        stdout = $Stdout
        stderr = $Stderr
    }
}

function Stop-RecordedProcess($Record) {
    if (-not $Record.owned -or -not $Record.pid) { return }
    $process = Get-Process -Id ([int] $Record.pid) -ErrorAction SilentlyContinue
    if (-not $process) { return }

    $actualStart = $process.StartTime.ToUniversalTime()
    # Windows PowerShell's ConvertFrom-Json materializes ISO timestamps as a
    # DateTime object. Casting that object back to string drops the trailing
    # `Z`, so an otherwise identical UTC time is misread as local time. Keep
    # the typed value when available and only parse strings as an offset.
    $recordedValue = $Record.start_time_utc
    if ($recordedValue -is [DateTimeOffset]) {
        $recordedStart = $recordedValue.UtcDateTime
    }
    elseif ($recordedValue -is [DateTime]) {
        $recordedStart = if ($recordedValue.Kind -eq [DateTimeKind]::Utc) {
            $recordedValue
        }
        elseif ($recordedValue.Kind -eq [DateTimeKind]::Local) {
            $recordedValue.ToUniversalTime()
        }
        else {
            [DateTime]::SpecifyKind($recordedValue, [DateTimeKind]::Utc)
        }
    }
    else {
        $recordedStart = [DateTimeOffset]::Parse(
            [string] $recordedValue,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind
        ).UtcDateTime
    }
    if ([Math]::Abs(($actualStart - $recordedStart).TotalSeconds) -gt 2) {
        Write-Warning "Skipped PID $($Record.pid): it now belongs to another process."
        return
    }

    & taskkill.exe /PID ([string] $Record.pid) /T /F *> $null
    Write-Host "Stopped $($Record.name) (PID $($Record.pid))."
}

function Stop-Stack {
    if (-not (Test-Path -LiteralPath $stateFile)) {
        Remove-Item -LiteralPath $adminKeyFile -Force -ErrorAction SilentlyContinue
        Write-Host 'No quick-start runtime state was found. Nothing managed by this script was stopped.'
        return
    }

    $state = Get-Content -LiteralPath $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $records = @($state.processes)
    [Array]::Reverse($records)
    foreach ($record in $records) { Stop-RecordedProcess $record }
    Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $adminKeyFile -Force -ErrorAction SilentlyContinue
    Write-Ok 'Quick-start services stopped.'
}

function Assert-File([string] $Path, [string] $Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Ensure-JavaJar([string] $JavaDir) {
    $jar = Join-Path $JavaDir 'target\pet-reid-java-api-1.0.0.jar'
    $sources = @(
        Get-ChildItem -LiteralPath (Join-Path $JavaDir 'src') -File -Recurse
        Get-Item -LiteralPath (Join-Path $JavaDir 'pom.xml')
    )
    $latestSource = $sources | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    $needsBuild = -not (Test-Path -LiteralPath $jar)
    if (-not $needsBuild) {
        $needsBuild = $latestSource.LastWriteTimeUtc -gt (Get-Item -LiteralPath $jar).LastWriteTimeUtc
    }
    if (-not $needsBuild) { return $jar }

    $maven = 'D:\Maven\apache-maven-3.9.16\bin\mvn.cmd'
    if (-not (Test-Path -LiteralPath $maven)) {
        $mavenCommand = Get-Command mvn.cmd -ErrorAction SilentlyContinue
        if (-not $mavenCommand) { throw 'Maven 3.9+ is required to build the Java gateway.' }
        $maven = $mavenCommand.Source
    }
    Write-Step 'Building the Java gateway (first run or source changed)...'
    Push-Location $JavaDir
    try {
        & $maven -q -DskipTests package
        if ($LASTEXITCODE -ne 0) { throw "Maven exited with code $LASTEXITCODE." }
    }
    finally { Pop-Location }
    Assert-File $jar 'Java gateway JAR'
    return $jar
}

function Ensure-FrontendDependencies([string] $FrontendDir) {
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) { throw 'Node.js 22.13+ and npm are required for the frontend.' }
    $modulesStamp = Join-Path $FrontendDir 'node_modules\.package-lock.json'
    $lockFile = Join-Path $FrontendDir 'package-lock.json'
    if ((Test-Path -LiteralPath $modulesStamp) -and
        (Get-Item -LiteralPath $modulesStamp).LastWriteTimeUtc -ge
        (Get-Item -LiteralPath $lockFile).LastWriteTimeUtc) {
        return $npmCommand.Source
    }

    Write-Step 'Installing exact frontend dependencies...'
    Push-Location $FrontendDir
    try {
        & $npmCommand.Source ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci exited with code $LASTEXITCODE." }
    }
    finally { Pop-Location }
    return $npmCommand.Source
}

function Start-Stack {
    $existing = Get-StackStatus
    $existingLan = $false
    if (Test-Path -LiteralPath $stateFile) {
        try {
            $existingState = Get-Content -LiteralPath $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json
            $existingLan = [bool] $existingState.lan_enabled
        }
        catch { $existingLan = $false }
    }
    $expectedExecutionProvider = if ($Provider -eq 'cpu') {
        'CPUExecutionProvider'
    }
    else {
        'CUDAExecutionProvider'
    }
    $inferenceDevice = if ($Provider -eq 'cpu') { 'cpu' } else { 'cuda' }
    $isUnified = [bool] $runtimeProfile.single_graph
    $expectedBackend = [string] $runtimeProfile.runtime_backend
    $expectedModelHash = [string] $runtimeProfile.model_sha256
    $expectedAgent = if ($runtimeProfile.agent_mode) {
        [string] $runtimeProfile.agent_mode
    }
    else { '-' }
    $expectedFusion = if ($runtimeProfile.fusion_mode) {
        [string] $runtimeProfile.fusion_mode
    }
    else { '-' }
    $modelLabel = [string] $runtimeProfile.display_name

    if ($existing.Ready) {
        $modelMatches = if ($isUnified) {
            $existing.Backend -eq $expectedBackend -and
            $existing.Profile -eq $Model -and
            $existing.SingleGraph -and
            $existing.ModelHash -eq $expectedModelHash
        }
        else {
            $existing.Backend -eq $expectedBackend -and
            $existing.Profile -eq $Model -and
            $existing.Fusion -eq $expectedFusion
        }
        if ($existing.Provider -eq $expectedExecutionProvider -and
            $modelMatches -and
            $existing.Agent -eq $expectedAgent -and
            $existingLan -eq [bool] $Lan) {
            Write-Ok "All services are already ready at $frontendUrl ($modelLabel / $expectedExecutionProvider)"
            if (-not $NoBrowser) { Start-Process $frontendUrl }
            return
        }
        if (Test-Path -LiteralPath $stateFile) {
            Write-Step "Switching inference from $($existing.Backend) / $($existing.Provider) to $expectedBackend / $expectedExecutionProvider..."
            Stop-Stack
        }
        else {
            throw "Services are already running with $($existing.Backend) / $($existing.Provider). Stop them before selecting $Model / $Provider."
        }
    }
    elseif (Test-Path -LiteralPath $stateFile) {
        Write-Step 'Cleaning up an incomplete previous quick-start run...'
        Stop-Stack
    }

    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    $resolvedPython = Resolve-Executable $PythonExe 'Python interpreter'

    $usingDefaultGallery = [string]::IsNullOrWhiteSpace($GalleryDir)
    $config = if ($runtimeProfile.config) {
        Resolve-RepoPath ([string] $runtimeProfile.config)
    }
    else { $null }
    $weights = if ($runtimeProfile.identity_weights) {
        Resolve-RepoPath ([string] $runtimeProfile.identity_weights)
    }
    else { $null }
    $onnxModel = Resolve-RepoPath ([string] $runtimeProfile.onnx)
    $bodyDetector = if ($runtimeProfile.body_detector) {
        Resolve-RepoPath ([string] $runtimeProfile.body_detector)
    }
    else { $null }
    $seedGallery = if ($runtimeProfile.seed_gallery) {
        Resolve-RepoPath ([string] $runtimeProfile.seed_gallery)
    }
    else { $null }
    $expertCheckpoint = if ($runtimeProfile.expert_checkpoint) {
        Resolve-RepoPath ([string] $runtimeProfile.expert_checkpoint)
    }
    else { $null }
    $warmupBatches = @($runtimeProfile.warmup_batches) -join ','
    $selectedGalleryDir = if ($usingDefaultGallery) {
        [string] $runtimeProfile.persistent_gallery
    }
    else { $GalleryDir }
    $resolvedGallery = Resolve-RepoPath $selectedGalleryDir
    $javaDir = Join-Path $sourceRoot 'java\pet-reid-spring-client'
    $frontendDir = Join-Path $sourceRoot 'frontend\pet-reid-web'

    if ($config) { Assert-File $config "$modelLabel config" }
    if ($weights) { Assert-File $weights "$modelLabel weights" }
    Assert-File $onnxModel "$modelLabel ONNX model"
    if ($bodyDetector) { Assert-File $bodyDetector 'BIFOR body detector' }
    if ($seedGallery) { Assert-File $seedGallery 'Seed gallery model' }
    if ([bool] $runtimeProfile.requires_existing_gallery -and $usingDefaultGallery -and
        -not (Test-Path -LiteralPath (Join-Path $resolvedGallery 'gallery.sqlite3'))) {
        throw "The migrated $modelLabel Gallery is missing: $resolvedGallery."
    }
    $javaJar = Ensure-JavaJar $javaDir
    $npm = Ensure-FrontendDependencies $frontendDir
    New-Item -ItemType Directory -Force -Path $resolvedGallery | Out-Null

    $owned = New-Object System.Collections.ArrayList
    $oldApiKey = $env:PET_REID_API_KEY
    $oldBaseUrl = $env:PET_REID_BASE_URL
    $oldCors = $env:FRONTEND_ALLOWED_ORIGINS
    $oldFrontendApi = $env:NEXT_PUBLIC_PET_REID_API_BASE_URL
    $oldGatewayProxy = $env:PET_REID_GATEWAY_PROXY_TARGET
    $oldFrontendHost = $env:PET_REID_FRONTEND_HOST
    $oldFrontendPort = $env:PET_REID_FRONTEND_PORT
    $oldAdminKey = $env:PET_REID_ADMIN_KEY
    $oldServerPort = $env:SERVER_PORT
    $oldSiteOrigin = $env:NEXT_PUBLIC_SITE_ORIGIN
    $apiKey = if ($env:PET_REID_API_KEY) {
        $env:PET_REID_API_KEY
    }
    else {
        [Convert]::ToBase64String([byte[]] (1..32 | ForEach-Object {
            Get-Random -Minimum 0 -Maximum 256
        }))
    }
    $adminKey = if ($env:PET_REID_ADMIN_KEY) {
        $env:PET_REID_ADMIN_KEY
    }
    else {
        $adminKeyBytes = New-Object byte[] 32
        $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try { $random.GetBytes($adminKeyBytes) }
        finally { $random.Dispose() }
        [Convert]::ToBase64String($adminKeyBytes)
    }

    try {
        $env:PET_REID_API_KEY = $apiKey
        Set-Content -LiteralPath $adminKeyFile -Value $adminKey -Encoding ASCII -NoNewline

        $pythonHealth = Get-Json "$pythonUrl/health"
        if (-not $pythonHealth) {
            if (Test-TcpPort $PythonPort) { throw "Port $PythonPort is already occupied by another process." }
            Write-Step "Starting $modelLabel on $($Provider.ToUpperInvariant()) (model loading can take about one minute)..."
            $pythonOut = Join-Path $runtimeDir 'python.stdout.log'
            $pythonErr = Join-Path $runtimeDir 'python.stderr.log'
            $pythonArgs = @(
                'tools\serve_pet_api.py',
                '--host', '127.0.0.1',
                '--port', [string] $PythonPort,
                '--profile', $Model,
                '--device', $inferenceDevice,
                '--onnx-provider', $Provider,
                '--onnx-warmup-batches', $warmupBatches,
                '--onnx-model', $onnxModel,
                '--storage-dir', $resolvedGallery
            )
            if ($config) {
                $pythonArgs += @('--config-file', $config)
            }
            if ($weights) {
                $pythonArgs += @('--identity-weights', $weights)
            }
            if ($bodyDetector) {
                $pythonArgs += @('--body-detector', $bodyDetector)
            }
            if ($runtimeProfile.agent_mode) {
                Assert-File $expertCheckpoint 'Expert checkpoint'
                $pythonArgs += @(
                    '--megadescriptor-checkpoint', $expertCheckpoint,
                    '--megadescriptor-device', $inferenceDevice
                )
            }
            if ($seedGallery -and -not (Test-Path -LiteralPath (Join-Path $resolvedGallery 'gallery.sqlite3'))) {
                $pythonArgs += @('--seed-gallery-model', $seedGallery)
            }
            $pythonProcess = Start-Process -FilePath $resolvedPython -ArgumentList $pythonArgs `
                -WorkingDirectory $sourceRoot -WindowStyle Hidden `
                -RedirectStandardOutput $pythonOut -RedirectStandardError $pythonErr -PassThru
            [void] $owned.Add((New-ProcessRecord "python-$Provider" $pythonProcess $pythonOut $pythonErr))
            $pythonHealth = Wait-JsonEndpoint "Python $($Provider.ToUpperInvariant()) service" "$pythonUrl/health" 180 $pythonProcess
        }
        if ([string] $pythonHealth.backend.provider -ne $expectedExecutionProvider) {
            throw "Python API is not using $expectedExecutionProvider (reported: $($pythonHealth.backend.provider))."
        }
        if ([int] $pythonHealth.backend.embedding_dim -ne 512) {
            throw "Python API is not serving the expected 512d $modelLabel model."
        }
        if ([string] $pythonHealth.backend.backend -ne $expectedBackend) {
            throw "Python API backend mismatch (expected: $expectedBackend; reported: $($pythonHealth.backend.backend))."
        }
        if ([string] $pythonHealth.backend.deployment_profile -ne $Model) {
            throw "Python API profile mismatch (expected: $Model; reported: $($pythonHealth.backend.deployment_profile))."
        }
        if ($isUnified) {
            if (-not [bool] $pythonHealth.backend.single_graph -or
                @($pythonHealth.backend.external_models).Count -ne 0 -or
                [string] $pythonHealth.backend.model_sha256 -ne $expectedModelHash) {
                throw "Python API is not serving the locked single-graph $modelLabel model."
            }
        }
        elseif ([string] $pythonHealth.backend.fusion_mode -ne $expectedFusion) {
            throw "Python API is not serving the expected $modelLabel fusion."
        }
        if ([string] $pythonHealth.backend.model_sha256 -ne $expectedModelHash) {
            throw "Python API model fingerprint differs from the deployment manifest."
        }
        $actualAgent = if ($pythonHealth.backend.agent) {
            [string] $pythonHealth.backend.agent.version
        }
        else { '-' }
        if ($actualAgent -ne $expectedAgent) {
            throw "Python API is not serving the expected Agent mode (reported: $actualAgent)."
        }
        Write-Ok "$($Provider.ToUpperInvariant()) $modelLabel is ready."

        $env:PET_REID_BASE_URL = $pythonUrl
        $env:SERVER_PORT = [string] $JavaPort
        $env:PET_REID_ADMIN_KEY = $adminKey
        $env:FRONTEND_ALLOWED_ORIGINS = "$frontendUrl,http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
        $javaHealth = Get-Json "$javaUrl/v1/upstream-health"
        if (-not $javaHealth) {
            if (Test-TcpPort $JavaPort) { throw "Port $JavaPort is already occupied by another process." }
            Write-Step 'Starting the Java gateway...'
            $javaOut = Join-Path $runtimeDir 'java.stdout.log'
            $javaErr = Join-Path $runtimeDir 'java.stderr.log'
            $javaExe = (Get-Command java.exe -ErrorAction Stop).Source
            $javaProcess = Start-Process -FilePath $javaExe -ArgumentList @('-jar', $javaJar) `
                -WorkingDirectory $javaDir -WindowStyle Hidden `
                -RedirectStandardOutput $javaOut -RedirectStandardError $javaErr -PassThru
            [void] $owned.Add((New-ProcessRecord 'java-gateway' $javaProcess $javaOut $javaErr))
            $javaHealth = Wait-JsonEndpoint 'Java gateway' "$javaUrl/v1/upstream-health" 60 $javaProcess
        }
        Write-Ok 'Java gateway is ready.'

        $env:NEXT_PUBLIC_PET_REID_API_BASE_URL = '/'
        $env:PET_REID_GATEWAY_PROXY_TARGET = $javaUrl
        $env:PET_REID_FRONTEND_HOST = $frontendListenHost
        $env:PET_REID_FRONTEND_PORT = [string] $FrontendPort
        $env:NEXT_PUBLIC_SITE_ORIGIN = $frontendUrl
        if (-not (Test-Frontend)) {
            if (Test-TcpPort $FrontendPort) { throw "Port $FrontendPort is already occupied by another process." }
            Write-Step 'Starting the browser workspace...'
            $frontendOut = Join-Path $runtimeDir 'frontend.stdout.log'
            $frontendErr = Join-Path $runtimeDir 'frontend.stderr.log'
            $frontendProcess = Start-Process -FilePath $npm -ArgumentList @('run', 'dev') `
                -WorkingDirectory $frontendDir -WindowStyle Hidden `
                -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru
            [void] $owned.Add((New-ProcessRecord 'frontend' $frontendProcess $frontendOut $frontendErr))
            Wait-Frontend $frontendProcess
        }
        Write-Ok 'Pawprint ID frontend is ready.'

        $final = Get-StackStatus
        if (-not $final.Ready) { throw 'One or more services failed the final health check.' }

        $state = [ordered]@{
            schema_version = 1
            started_at_utc = [DateTime]::UtcNow.ToString('o')
            workspace_root = $workspaceRoot
            source_root = $sourceRoot
            frontend_url = $frontendUrl
            frontend_listen_host = $frontendListenHost
            lan_enabled = [bool] $Lan
            gallery_dir = $resolvedGallery
            provider = $Provider
            profile = $Model
            capability = [string] $runtimeProfile.capability
            processes = @($owned)
        }
        $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $stateFile -Encoding UTF8

        Write-Host ''
        Write-Ok "Ready: $frontendUrl"
        if ($Lan) {
            Write-Host 'Phone: connect Android and this PC to the same trusted network, then open the URL above.'
        }
        Write-Host ('Gallery: {0} pets / {1} references' -f $final.Pets, $final.References)
        Write-Host "Logs: $runtimeDir"
        Write-Host "Admin key file: $adminKeyFile"
        if (-not $NoBrowser) { Start-Process $frontendUrl }
    }
    catch {
        Remove-Item -LiteralPath $adminKeyFile -Force -ErrorAction SilentlyContinue
        Write-Host ''
        Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
        $records = @($owned)
        [Array]::Reverse($records)
        foreach ($record in $records) { Stop-RecordedProcess $record }
        foreach ($log in Get-ChildItem -LiteralPath $runtimeDir -Filter '*.stderr.log' -ErrorAction SilentlyContinue) {
            if ($log.Length -gt 0) {
                Write-Host "`n--- $($log.Name) ---" -ForegroundColor DarkYellow
                Get-Content -LiteralPath $log.FullName -Tail 30
            }
        }
        throw
    }
    finally {
        if ($null -eq $oldApiKey) { Remove-Item Env:\PET_REID_API_KEY -ErrorAction SilentlyContinue } else { $env:PET_REID_API_KEY = $oldApiKey }
        if ($null -eq $oldBaseUrl) { Remove-Item Env:\PET_REID_BASE_URL -ErrorAction SilentlyContinue } else { $env:PET_REID_BASE_URL = $oldBaseUrl }
        if ($null -eq $oldServerPort) { Remove-Item Env:\SERVER_PORT -ErrorAction SilentlyContinue } else { $env:SERVER_PORT = $oldServerPort }
        if ($null -eq $oldCors) { Remove-Item Env:\FRONTEND_ALLOWED_ORIGINS -ErrorAction SilentlyContinue } else { $env:FRONTEND_ALLOWED_ORIGINS = $oldCors }
        if ($null -eq $oldFrontendApi) { Remove-Item Env:\NEXT_PUBLIC_PET_REID_API_BASE_URL -ErrorAction SilentlyContinue } else { $env:NEXT_PUBLIC_PET_REID_API_BASE_URL = $oldFrontendApi }
        if ($null -eq $oldGatewayProxy) { Remove-Item Env:\PET_REID_GATEWAY_PROXY_TARGET -ErrorAction SilentlyContinue } else { $env:PET_REID_GATEWAY_PROXY_TARGET = $oldGatewayProxy }
        if ($null -eq $oldFrontendHost) { Remove-Item Env:\PET_REID_FRONTEND_HOST -ErrorAction SilentlyContinue } else { $env:PET_REID_FRONTEND_HOST = $oldFrontendHost }
        if ($null -eq $oldFrontendPort) { Remove-Item Env:\PET_REID_FRONTEND_PORT -ErrorAction SilentlyContinue } else { $env:PET_REID_FRONTEND_PORT = $oldFrontendPort }
        if ($null -eq $oldSiteOrigin) { Remove-Item Env:\NEXT_PUBLIC_SITE_ORIGIN -ErrorAction SilentlyContinue } else { $env:NEXT_PUBLIC_SITE_ORIGIN = $oldSiteOrigin }
        if ($null -eq $oldAdminKey) { Remove-Item Env:\PET_REID_ADMIN_KEY -ErrorAction SilentlyContinue } else { $env:PET_REID_ADMIN_KEY = $oldAdminKey }
    }
}

Set-Location $workspaceRoot
switch ($Action) {
    'start' { Start-Stack }
    'stop' { Stop-Stack }
    'status' { [void] (Show-Status) }
    'restart' {
        Stop-Stack
        Start-Stack
    }
}
