[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'restart')]
    [string] $Action = 'start',

    [switch] $NoBrowser,

    [ValidateSet('cuda', 'cpu')]
    [string] $Provider = 'cuda',

    [ValidateSet('semantic-v3', 'semantic-v3-bifor', 'agent-v1')]
    [string] $Model = 'semantic-v3',

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
    [int] $FrontendPort = 3000
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceRoot = Join-Path $workspaceRoot 'src\Pet-ReID-IMAG'
$runtimeDir = Join-Path $workspaceRoot 'artifacts\workspace_logs\quick_start'
$stateFile = Join-Path $runtimeDir 'runtime.json'
$adminKeyFile = Join-Path $runtimeDir 'admin-key.txt'
$pythonUrl = "http://127.0.0.1:$PythonPort"
$javaUrl = "http://127.0.0.1:$JavaPort"
$frontendUrl = "http://localhost:$FrontendPort"

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
        $response = Invoke-WebRequest -Uri "$frontendUrl/" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    }
    catch { return $false }
}

function Get-StackStatus {
    $python = Get-Json "$pythonUrl/health"
    $java = Get-Json "$javaUrl/v1/upstream-health"
    $frontend = Test-Frontend

    $provider = if ($python) { [string] $python.backend.provider } else { '-' }
    $fusion = if ($python) { [string] $python.backend.fusion_mode } else { '-' }
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
        Fusion = $fusion
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

    $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
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
    $expectedExecutionProvider = if ($Provider -eq 'cpu') {
        'CPUExecutionProvider'
    }
    else {
        'CUDAExecutionProvider'
    }
    $inferenceDevice = if ($Provider -eq 'cpu') { 'cpu' } else { 'cuda' }
    $usesBifor = $Model -in @('semantic-v3-bifor', 'agent-v1')
    $expectedAgent = if ($Model -eq 'agent-v1') {
        'multi_expert_evidence_v1'
    }
    else { '-' }
    $expectedFusion = if ($usesBifor) {
        'semantic_residual_v3+bifor_lowrank_v1'
    }
    else {
        'semantic_residual_v3'
    }
    $modelLabel = if ($Model -eq 'agent-v1') {
        'BIFOR + MegaDescriptor Agent V1'
    }
    elseif ($Model -eq 'semantic-v3-bifor') {
        'Semantic V3 + BIFOR'
    }
    else {
        'Semantic V3'
    }

    if ($existing.Ready) {
        if ($existing.Provider -eq $expectedExecutionProvider -and
            $existing.Fusion -eq $expectedFusion -and
            $existing.Agent -eq $expectedAgent) {
            Write-Ok "All services are already ready at $frontendUrl ($modelLabel / $expectedExecutionProvider)"
            if (-not $NoBrowser) { Start-Process $frontendUrl }
            return
        }
        if (Test-Path -LiteralPath $stateFile) {
            Write-Step "Switching inference from $($existing.Fusion) / $($existing.Provider) to $expectedFusion / $expectedExecutionProvider..."
            Stop-Stack
        }
        else {
            throw "Services are already running with $($existing.Fusion) / $($existing.Provider). Stop them before selecting $Model / $Provider."
        }
    }
    elseif (Test-Path -LiteralPath $stateFile) {
        Write-Step 'Cleaning up an incomplete previous quick-start run...'
        Stop-Stack
    }

    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    $resolvedPython = Resolve-Executable $PythonExe 'Python interpreter'

    $weights = Join-Path $workspaceRoot 'models\selected\dogfacenet_semantic_v3_v1\model_final.pth'
    $usingDefaultGallery = [string]::IsNullOrWhiteSpace($GalleryDir)
    if ($usesBifor) {
        $config = Join-Path $workspaceRoot 'models\selected\dogfacenet_semantic_v3_bifor_lowrank_v1\config.yaml'
        $onnxModel = Join-Path $workspaceRoot 'models\selected\dogfacenet_semantic_v3_bifor_lowrank_v1\onnx\pet_embedding.onnx'
        $bodyDetector = Join-Path $workspaceRoot 'models\pretrained\body_detection\fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
        $identityBackend = 'onnx-bifor'
        $seedGallery = $null
        $selectedGalleryDir = if ($usingDefaultGallery) {
            if ($Model -eq 'agent-v1') {
                'data\gallery_store\pet_api_gallery_agent_v1'
            }
            else {
                'data\gallery_store\pet_api_gallery_semantic_v3_bifor_lowrank_v1'
            }
        }
        else { $GalleryDir }
    }
    else {
        $config = Join-Path $workspaceRoot 'models\selected\dogfacenet_semantic_v3_v1\config.yaml'
        $onnxModel = Join-Path $workspaceRoot 'models\selected\dogfacenet_semantic_v3_v1\onnx\pet_embedding.onnx'
        $bodyDetector = $null
        $identityBackend = 'onnx'
        $seedGallery = Join-Path $workspaceRoot 'models\selected\local_pet_gallery_semantic_v3_onnx_v1\gallery_model.json'
        $selectedGalleryDir = if ($usingDefaultGallery) {
            'data\gallery_store\pet_api_gallery_semantic_v3_v1'
        }
        else { $GalleryDir }
    }
    $resolvedGallery = Resolve-RepoPath $selectedGalleryDir
    $javaDir = Join-Path $sourceRoot 'java\pet-reid-spring-client'
    $frontendDir = Join-Path $sourceRoot 'frontend\pet-reid-web'

    Assert-File $config "$modelLabel config"
    Assert-File $weights 'Semantic V3 weights'
    Assert-File $onnxModel "$modelLabel ONNX model"
    if ($bodyDetector) { Assert-File $bodyDetector 'BIFOR body detector' }
    if ($seedGallery) { Assert-File $seedGallery 'Seed gallery model' }
    if ($usesBifor -and $usingDefaultGallery -and
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
    $oldAdminKey = $env:PET_REID_ADMIN_KEY
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
                '--backend', $identityBackend,
                '--device', $inferenceDevice,
                '--onnx-provider', $Provider,
                '--config-file', $config,
                '--identity-weights', $weights,
                '--onnx-model', $onnxModel,
                '--storage-dir', $resolvedGallery
            )
            if ($bodyDetector) {
                $pythonArgs += @('--body-detector', $bodyDetector)
            }
            if ($Model -eq 'agent-v1') {
                $megaDescriptor = Join-Path $workspaceRoot 'models\pretrained\megadescriptor\MegaDescriptor-B-224\pytorch_model.bin'
                Assert-File $megaDescriptor 'MegaDescriptor-B-224 checkpoint'
                $pythonArgs += @(
                    '--agent',
                    '--megadescriptor-checkpoint', $megaDescriptor,
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
        if ([int] $pythonHealth.backend.embedding_dim -ne 512 -or
            [string] $pythonHealth.backend.fusion_mode -ne $expectedFusion) {
            throw "Python API is not serving the expected 512d $modelLabel model."
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
        $env:PET_REID_ADMIN_KEY = $adminKey
        $env:FRONTEND_ALLOWED_ORIGINS = "$frontendUrl,http://127.0.0.1:$FrontendPort"
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

        $env:NEXT_PUBLIC_PET_REID_API_BASE_URL = $javaUrl
        if (-not (Test-Frontend)) {
            if (Test-TcpPort $FrontendPort) { throw "Port $FrontendPort is already occupied by another process." }
            Write-Step 'Starting the browser workspace...'
            $frontendOut = Join-Path $runtimeDir 'frontend.stdout.log'
            $frontendErr = Join-Path $runtimeDir 'frontend.stderr.log'
            $frontendProcess = Start-Process -FilePath $npm -ArgumentList @(
                'run', 'dev', '--', '--host', '127.0.0.1', '--port', [string] $FrontendPort
            ) -WorkingDirectory $frontendDir -WindowStyle Hidden `
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
            gallery_dir = $resolvedGallery
            provider = $Provider
            model = $Model
            processes = @($owned)
        }
        $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $stateFile -Encoding UTF8

        Write-Host ''
        Write-Ok "Ready: $frontendUrl"
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
        if ($null -eq $oldCors) { Remove-Item Env:\FRONTEND_ALLOWED_ORIGINS -ErrorAction SilentlyContinue } else { $env:FRONTEND_ALLOWED_ORIGINS = $oldCors }
        if ($null -eq $oldFrontendApi) { Remove-Item Env:\NEXT_PUBLIC_PET_REID_API_BASE_URL -ErrorAction SilentlyContinue } else { $env:NEXT_PUBLIC_PET_REID_API_BASE_URL = $oldFrontendApi }
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
