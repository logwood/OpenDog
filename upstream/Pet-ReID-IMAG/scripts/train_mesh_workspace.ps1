param(
    [switch]$Resume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = "D:\CondaData\envs\torch312\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Training Python was not found at $pythonPath"
}

if ($Smoke) {
    $configPath = "configs/modern_mesh_workspace_smoke.yaml"
    $outputPath = Join-Path $repoRoot "logs/modern_mesh_workspace_smoke_d192_balanced"
} else {
    $configPath = "configs/modern_mesh_workspace_s101_224.yaml"
    $outputPath = Join-Path $repoRoot "logs/modern_mesh_workspace_s101_224_d192_balanced"
}

$checkpointPointer = Join-Path $outputPath "last_checkpoint"
$runMetrics = Join-Path $outputPath "metrics.json"
if ((-not $Resume) -and (
    (Test-Path -LiteralPath $checkpointPointer) -or
    (Test-Path -LiteralPath $runMetrics)
)) {
    throw "The output already contains a training run. Use -Resume or choose a fresh output directory: $outputPath"
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$consoleLog = Join-Path $outputPath "console.log"
$trainArguments = @(
    "pet_id/train_net.py",
    "--config-file", $configPath,
    "--num-gpus", "1"
)
if ($Resume) {
    $trainArguments += "--resume"
}

Push-Location $repoRoot
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$consoleWriter = [System.IO.StreamWriter]::new(
    $consoleLog,
    [bool]$Resume,
    $utf8NoBom
)
$consoleWriter.AutoFlush = $true
try {
    & $pythonPath @trainArguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        [Console]::Out.WriteLine($line)
        $consoleWriter.WriteLine($line)
    }
    $trainingExitCode = $LASTEXITCODE
} finally {
    $consoleWriter.Dispose()
    Pop-Location
}

if ($trainingExitCode -ne 0) {
    throw "Training exited with code $trainingExitCode. See $consoleLog"
}
