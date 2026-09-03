param(
    [int]$WaitForProcessId = 0,
    [int]$BatchSize = 16,
    [string]$PythonPath = '',
    [string]$CheckpointDirectory = '',
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'training_workspace.ps1')
if ([string]::IsNullOrWhiteSpace($PythonPath)) { $PythonPath = Get-PetReIDDefaultPython }
$python = Resolve-PetReIDPython $PythonPath
$repoRoot = $script:PetReIDSourceRoot
if ([string]::IsNullOrWhiteSpace($CheckpointDirectory)) {
    $CheckpointDirectory = Join-Path $script:PetReIDWorkspaceRoot "artifacts\runs\legacy\modern_latent_role_anchored_s101_224_d192"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $script:PetReIDWorkspaceRoot ("artifacts\evaluations\role-anchored-latent-checkpoint-ablation\{0}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
$checkpointDirectory = [System.IO.Path]::GetFullPath($CheckpointDirectory)
$outputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$reportPath = Join-Path $outputDirectory "summary.json"
$consoleLog = Join-Path $outputDirectory "console.log"

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
    throw "No immutable role-anchored checkpoints were found in $checkpointDirectory"
}

$toolArguments = @(
    "tools/evaluate_role_anchored_checkpoint_ablation.py",
    "--config-file", "configs/modern_latent_role_anchored_s101_224.yaml",
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
        & $python @toolArguments 2>&1 |
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
