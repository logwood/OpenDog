param(
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$Microfit,
    [switch]$PreflightOnly,
    [switch]$SkipPreflight,
    [string]$RunId = '',
    [string]$PythonPath = '',
    [switch]$LegacyLayout
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'training_workspace.ps1')
if ([string]::IsNullOrWhiteSpace($PythonPath)) { $PythonPath = Get-PetReIDDefaultPython }
$python = Resolve-PetReIDPython $PythonPath
$repoRoot = $script:PetReIDSourceRoot
if ($PreflightOnly -and $SkipPreflight) {
    throw "-PreflightOnly and -SkipPreflight cannot be used together"
}
if ($Smoke -and $Microfit) {
    throw "-Smoke and -Microfit cannot be used together"
}

if ($Microfit) {
    $configPath = "configs/modern_latent_v2_microfit.yaml"
    $legacyName = "modern_latent_v2_microfit_d192"
    $workstream = "latent-v2-microfit"
    $purpose = "microfit"
} elseif ($Smoke) {
    $configPath = "configs/modern_latent_v2_smoke.yaml"
    $legacyName = "modern_latent_v2_smoke_d192"
    $workstream = "latent-v2-smoke"
    $purpose = "smoke"
} else {
    $configPath = "configs/modern_latent_v2_s101_224.yaml"
    $legacyName = "modern_latent_v2_s101_224_d192"
    $workstream = "latent-v2"
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
if ((-not $PreflightOnly) -and (-not $Resume) -and (
    (Test-Path -LiteralPath $checkpointPointer) -or
    (Test-Path -LiteralPath $runMetrics)
)) {
    throw "The output already contains a training run. Use -Resume or choose a fresh output directory: $outputPath"
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$reportDirectory = if ($run.StandardLayout) {
    Join-Path $outputPath "reports"
} else {
    $outputPath
}
New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
$preflightReport = Join-Path $reportDirectory "preflight.json"

Push-Location $repoRoot
$previousUnbuffered = $env:PYTHONUNBUFFERED
$previousEncoding = $env:PYTHONIOENCODING
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
try {
    if (-not $SkipPreflight) {
        $preflightArguments = @(
            "tools/preflight_latent_v2.py",
            "--config-file", $configPath,
            "--device", "auto",
            "--json-output", $preflightReport
        )
        & $python @preflightArguments
        $preflightExitCode = $LASTEXITCODE
        if ($preflightExitCode -ne 0) {
            throw "Latent V2 preflight failed with code $preflightExitCode. See $preflightReport"
        }
    }

    if ($PreflightOnly) {
        Write-Host "Preflight passed. No training was started."
        return
    }

    $consoleLog = Join-Path $outputPath $(if ($run.StandardLayout) { "stdout.log" } else { "console.log" })
    $trainArguments = @(
        "pet_id/train_net.py",
        "--config-file", $configPath,
        "--num-gpus", "1"
    ) + @($run.CliArguments)
    if ($Resume) {
        $trainArguments += "--resume"
    }

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
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
        $ErrorActionPreference = $previousErrorAction
    }
    if ($trainingExitCode -ne 0) {
        throw "Training exited with code $trainingExitCode. See $consoleLog"
    }

    if ($Microfit) {
        $microfitReport = Join-Path $reportDirectory "microfit_analysis.json"
        $analysisArguments = @(
            "tools/analyze_latent_microfit.py",
            "--metrics", $runMetrics,
            "--json-output", $microfitReport
        )
        & $python @analysisArguments
        $analysisExitCode = $LASTEXITCODE
        if ($analysisExitCode -ne 0) {
            throw "Microfit quality gate failed. See $microfitReport"
        }
    }
} finally {
    $env:PYTHONUNBUFFERED = $previousUnbuffered
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
