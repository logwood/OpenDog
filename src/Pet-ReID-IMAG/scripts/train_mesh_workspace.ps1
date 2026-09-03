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
    $configPath = "configs/modern_mesh_workspace_smoke.yaml"
    $legacyName = "modern_mesh_workspace_smoke_d192_balanced"
    $workstream = "mesh-workspace-smoke"
    $purpose = "smoke"
} else {
    $configPath = "configs/modern_mesh_workspace_s101_224.yaml"
    $legacyName = "modern_mesh_workspace_s101_224_d192_balanced"
    $workstream = "mesh-workspace"
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
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$consoleWriter = [System.IO.StreamWriter]::new(
    $consoleLog,
    [bool]$Resume,
    $utf8NoBom
)
$consoleWriter.AutoFlush = $true
try {
    & $python @trainArguments 2>&1 | ForEach-Object {
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
