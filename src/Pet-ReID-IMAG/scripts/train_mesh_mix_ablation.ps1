param(
    [switch]$Resume,
    [switch]$Smoke,
    [string]$RunId = '',
    [string]$PythonPath = '',
    [switch]$LegacyLayout
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'training_workspace.ps1')
if ([string]::IsNullOrWhiteSpace($PythonPath)) { $PythonPath = Get-PetReIDDefaultPython }
$python = Resolve-PetReIDPython $PythonPath
$repoRoot = $script:PetReIDSourceRoot

if ($Smoke) {
    $configPath = "configs/ablation_mesh_mix_fixed005_smoke.yaml"
    $legacyName = "ablation_mesh_mix_fixed005_smoke_d192"
    $workstream = "mesh-mix-ablation-smoke"
    $purpose = "smoke"
} else {
    $configPath = "configs/ablation_mesh_mix_fixed005_s101_224.yaml"
    $legacyName = "ablation_mesh_mix_fixed005_s101_224_d192"
    $workstream = "mesh-mix-ablation"
    $purpose = "train"
}
$run = Resolve-PetReIDRun -Workstream $workstream -Purpose $purpose -LegacyName $legacyName -RunId $RunId -Resume:$Resume -LegacyLayout:$LegacyLayout
$outputPath = $run.OutputPath

$checkpointPointer = if ($run.StandardLayout) {
    Join-Path $outputPath "checkpoints\last_checkpoint"
} else {
    Join-Path $outputPath "last_checkpoint"
}
$runMetrics = Join-Path $outputPath "metrics.json"
if ((-not $Resume) -and (
    (Test-Path -LiteralPath $checkpointPointer) -or
    (Test-Path -LiteralPath $runMetrics)
)) {
    throw "The output already contains a training run. Use -Resume or choose a fresh output directory: $outputPath"
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$consoleLog = Join-Path $outputPath $(if ($run.StandardLayout) { "stdout.log" } else { "console.log" })
$trainArguments = @(
    "pet_id/train_net.py",
    "--config-file", $configPath,
    "--num-gpus", "1"
) + @($run.CliArguments)
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
        & $python @trainArguments 2>&1 |
            Tee-Object -FilePath $consoleLog -Append
    } else {
        & $python @trainArguments 2>&1 |
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
