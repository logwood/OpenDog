param(
    [int]$WaitForProcessId = 0,
    [int]$BatchSize = 16
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = "D:\CondaData\envs\torch312\python.exe"
$checkpointDirectory = Join-Path $repoRoot "logs/modern_latent_v2_s101_224_d192"
$outputDirectory = Join-Path $repoRoot "logs/latent_v2_checkpoint_inference_ablation"
$reportPath = Join-Path $outputDirectory "summary.json"
$consoleLog = Join-Path $outputDirectory "console.log"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Evaluation Python was not found at $pythonPath"
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be positive"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
if ($WaitForProcessId -gt 0) {
    $process = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        "Waiting for training process $WaitForProcessId to exit." |
            Tee-Object -FilePath $consoleLog
        Wait-Process -Id $WaitForProcessId
    }
}

$checkpoints = @(
    Get-ChildItem -LiteralPath $checkpointDirectory -Filter "model_*.pth" |
        Where-Object { $_.Name -match '^model_[0-9]{4}\.pth$' } |
        Sort-Object Name |
        Select-Object -ExpandProperty FullName
)
$finalCheckpoint = Join-Path $checkpointDirectory "model_final.pth"
if (Test-Path -LiteralPath $finalCheckpoint) {
    $checkpoints += $finalCheckpoint
}
if ($checkpoints.Count -eq 0) {
    throw "No immutable V2 checkpoints were found in $checkpointDirectory"
}

$toolArguments = @(
    "tools/eval_latent_v2_checkpoint_ablation.py",
    "--config-file", "configs/modern_latent_v2_s101_224.yaml",
    "--checkpoints"
) + $checkpoints + @(
    "--output", $reportPath,
    "--batch-size", $BatchSize
)

Push-Location $repoRoot
try {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $pythonPath @toolArguments 2>&1 |
            Tee-Object -FilePath $consoleLog -Append
        $evaluationExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($evaluationExitCode -ne 0) {
        throw "Checkpoint ablation exited with code $evaluationExitCode. See $consoleLog"
    }
} finally {
    Pop-Location
}
