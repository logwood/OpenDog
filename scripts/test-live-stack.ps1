[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [string] $Provider = 'cpu',

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

    [string] $RunId = $(Get-Date -Format 'yyyyMMdd-HHmmss'),

    [int] $PythonPort = 8000,
    [int] $JavaPort = 8080,
    [int] $FrontendPort = 3000
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$deploymentManifest = Join-Path $workspaceRoot 'models\deployment.json'
$deployment = Get-Content -LiteralPath $deploymentManifest -Raw | ConvertFrom-Json
$profileProperty = $deployment.runtime_profiles.PSObject.Properties[$Model]
if ($null -eq $profileProperty) {
    throw "Runtime profile was not found in the deployment manifest: $Model"
}
$runtimeProfile = $profileProperty.Value
$stackScript = Join-Path $PSScriptRoot 'pet-reid-stack.ps1'
$smokeScript = Join-Path $PSScriptRoot 'smoke_test_live_stack.py'
$runtimeFile = Join-Path $workspaceRoot 'artifacts\workspace_logs\quick_start\runtime.json'
$liveRunsRoot = Join-Path $workspaceRoot 'artifacts\runs\live-stack-e2e'
$runDir = [System.IO.Path]::GetFullPath((Join-Path $liveRunsRoot $RunId))
$galleryDir = Join-Path $runDir 'gallery'
$started = $false
$primaryError = $null
$runtimeExisted = Test-Path -LiteralPath $runtimeFile

function Test-TcpPort([int] $Port) {
    foreach ($hostName in @('127.0.0.1', '::1')) {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $pending = $client.BeginConnect($hostName, $Port, $null, $null)
            if ($pending.AsyncWaitHandle.WaitOne(400)) {
                $client.EndConnect($pending)
                return $true
            }
        }
        catch { }
        finally { $client.Dispose() }
    }
    return $false
}

function Resolve-Executable([string] $Candidate) {
    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        $resolved = [System.IO.Path]::GetFullPath($Candidate)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "Python interpreter was not found: $resolved"
        }
        return $resolved
    }
    $command = Get-Command $Candidate -ErrorAction SilentlyContinue
    if (-not $command) { throw "Python interpreter was not found on PATH: $Candidate" }
    return $command.Source
}

function Assert-SafeRunDirectory {
    if ([string]::IsNullOrWhiteSpace($RunId) -or
        $RunId.IndexOfAny([System.IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
        $RunId -in '.', '..' -or
        $RunId.Contains([System.IO.Path]::DirectorySeparatorChar) -or
        $RunId.Contains([System.IO.Path]::AltDirectorySeparatorChar)) {
        throw '-RunId must be one safe directory name.'
    }
    $parent = [System.IO.Path]::GetDirectoryName($runDir)
    if (-not [string]::Equals(
        $parent,
        [System.IO.Path]::GetFullPath($liveRunsRoot),
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing a run directory outside $liveRunsRoot"
    }
    if (Test-Path -LiteralPath $runDir) {
        throw "Run directory already exists; choose another -RunId: $runDir"
    }
}

function Assert-PortsFree {
    $occupied = @(@($PythonPort, $JavaPort, $FrontendPort) |
        Where-Object { Test-TcpPort $_ })
    if ($occupied.Count -gt 0) {
        throw "Required local ports are already occupied: $($occupied -join ', ')"
    }
}

function Assert-ManagedGallery {
    if (-not (Test-Path -LiteralPath $runtimeFile -PathType Leaf)) {
        throw 'The quick-start script returned without creating runtime state.'
    }
    $state = Get-Content -LiteralPath $runtimeFile -Raw | ConvertFrom-Json
    $activeGallery = [System.IO.Path]::GetFullPath([string] $state.gallery_dir)
    $expectedGallery = [System.IO.Path]::GetFullPath($galleryDir)
    if ($activeGallery -ne $expectedGallery) {
        throw "Quick-start gallery mismatch: $activeGallery (expected $expectedGallery)"
    }
}

function Assert-PortsReleased {
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        $occupied = @(@($PythonPort, $JavaPort, $FrontendPort) |
            Where-Object { Test-TcpPort $_ })
        if ($occupied.Count -eq 0) { return }
        Start-Sleep -Milliseconds 300
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Quick-start stop left local ports occupied: $($occupied -join ', ')"
}

Set-Location $workspaceRoot
try {
    Assert-SafeRunDirectory
    if ($runtimeExisted) {
        throw 'A quick-start stack is already managed. Stop it before running isolated E2E.'
    }
    Assert-PortsFree
    $resolvedPython = Resolve-Executable $PythonExe
    New-Item -ItemType Directory -Path $runDir | Out-Null

    Write-Host "[E2E] Run directory: $runDir" -ForegroundColor Cyan
    & $stackScript start -Provider $Provider -Model $Model -NoBrowser -PythonExe $resolvedPython `
        -GalleryDir $galleryDir -PythonPort $PythonPort -JavaPort $JavaPort `
        -FrontendPort $FrontendPort
    Assert-ManagedGallery
    $started = $true

    $expectedProvider = if ($Provider -eq 'cpu') {
        'CPUExecutionProvider'
    }
    else {
        'CUDAExecutionProvider'
    }
    $expectedFusion = [string] $runtimeProfile.fusion_mode
    $expectedAgent = [string] $runtimeProfile.agent_mode
    $expectedBackend = [string] $runtimeProfile.runtime_backend
    $expectedModelHash = [string] $runtimeProfile.model_sha256
    $smokeArgs = @(
        $smokeScript,
        '--run-dir', $runDir,
        '--expected-provider', $expectedProvider,
        '--expected-backend', $expectedBackend,
        '--expected-profile', $Model,
        '--expected-capability', [string] $runtimeProfile.capability,
        '--expected-model-sha256', $expectedModelHash,
        '--base-url', "http://127.0.0.1:$JavaPort",
        '--python-url', "http://127.0.0.1:$PythonPort",
        '--frontend-url', "http://localhost:$FrontendPort"
    )
    if ($expectedFusion) {
        $smokeArgs += @('--expected-fusion', $expectedFusion)
    }
    if ($expectedAgent) {
        $smokeArgs += @('--expected-agent', $expectedAgent)
    }
    if ([bool] $runtimeProfile.single_graph) {
        $smokeArgs += '--expect-single-graph'
    }
    & $resolvedPython @smokeArgs
    if ($LASTEXITCODE -ne 0) { throw "Live-stack smoke exited with code $LASTEXITCODE" }
}
catch {
    $primaryError = $_
}
finally {
    if ($started -or (-not $runtimeExisted -and (Test-Path -LiteralPath $runtimeFile))) {
        try {
            & $stackScript stop -PythonPort $PythonPort -JavaPort $JavaPort `
                -FrontendPort $FrontendPort
            Assert-PortsReleased
            Write-Host '[E2E] Services stopped and ports released.' -ForegroundColor Green
        }
        catch {
            if ($null -eq $primaryError) { $primaryError = $_ }
            else { Write-Error "Teardown also failed: $($_.Exception.Message)" -ErrorAction Continue }
        }
    }
}

if ($null -ne $primaryError) { throw $primaryError }
Write-Host "[E2E] Passed. Report: $(Join-Path $runDir 'live-stack-smoke.json')" `
    -ForegroundColor Green
