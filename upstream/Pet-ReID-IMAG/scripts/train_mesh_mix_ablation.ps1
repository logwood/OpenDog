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
    $configPath = "configs/ablation_mesh_mix_fixed005_smoke.yaml"
    $outputPath = Join-Path $repoRoot "logs/ablation_mesh_mix_fixed005_smoke_d192"
} else {
    $configPath = "configs/ablation_mesh_mix_fixed005_s101_224.yaml"
    $outputPath = Join-Path $repoRoot "logs/ablation_mesh_mix_fixed005_s101_224_d192"
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
$previousUnbuffered = $env:PYTHONUNBUFFERED
$previousEncoding = $env:PYTHONIOENCODING
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
try {
    if ($Resume) {
        & $pythonPath @trainArguments 2>&1 |
            Tee-Object -FilePath $consoleLog -Append
    } else {
        & $pythonPath @trainArguments 2>&1 |
            Tee-Object -FilePath $consoleLog
    }
    $trainingExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONUNBUFFERED = $previousUnbuffered
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}

if ($trainingExitCode -ne 0) {
    throw "Training exited with code $trainingExitCode. See $consoleLog"
}
