param(
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$Microfit,
    [switch]$PreflightOnly,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = "D:\CondaData\envs\torch312\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Training Python was not found at $pythonPath"
}
if ($PreflightOnly -and $SkipPreflight) {
    throw "-PreflightOnly and -SkipPreflight cannot be used together"
}
if ($Smoke -and $Microfit) {
    throw "-Smoke and -Microfit cannot be used together"
}

if ($Microfit) {
    $configPath = "configs/modern_latent_v3_microfit.yaml"
    $outputPath = Join-Path $repoRoot "logs/modern_latent_v3_microfit_d192"
} elseif ($Smoke) {
    $configPath = "configs/modern_latent_v3_smoke.yaml"
    $outputPath = Join-Path $repoRoot "logs/modern_latent_v3_smoke_d192"
} else {
    $configPath = "configs/modern_latent_v3_s101_224.yaml"
    $outputPath = Join-Path $repoRoot "logs/modern_latent_v3_s101_224_d192"
}

$checkpointPointer = Join-Path $outputPath "last_checkpoint"
$runMetrics = Join-Path $outputPath "metrics.json"
if ((-not $PreflightOnly) -and (-not $Resume) -and (
    (Test-Path -LiteralPath $checkpointPointer) -or
    (Test-Path -LiteralPath $runMetrics)
)) {
    throw "The output already contains a run. Use -Resume or choose a fresh output directory: $outputPath"
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$preflightReport = Join-Path $outputPath "preflight.json"

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
        & $pythonPath @preflightArguments
        $preflightExitCode = $LASTEXITCODE
        if ($preflightExitCode -ne 0) {
            throw "Latent V3 preflight failed with code $preflightExitCode. See $preflightReport"
        }
    }

    if ($PreflightOnly) {
        Write-Host "V3 preflight passed. No training was started."
        return
    }

    $consoleLog = Join-Path $outputPath "console.log"
    $trainArguments = @(
        "pet_id/train_net.py",
        "--config-file", $configPath,
        "--num-gpus", "1"
    )
    if ($Resume) {
        $trainArguments += "--resume"
    }

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
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
        $ErrorActionPreference = $previousErrorAction
    }
    if ($trainingExitCode -ne 0) {
        throw "V3 training exited with code $trainingExitCode. See $consoleLog"
    }

    if ($Microfit) {
        $microfitReport = Join-Path $outputPath "microfit_analysis.json"
        $analysisArguments = @(
            "tools/analyze_latent_v3_microfit.py",
            "--metrics", $runMetrics,
            "--json-output", $microfitReport
        )
        & $pythonPath @analysisArguments
        $analysisExitCode = $LASTEXITCODE
        if ($analysisExitCode -ne 0) {
            throw "Latent V3 microfit quality gate failed. See $microfitReport"
        }
    }
} finally {
    $env:PYTHONUNBUFFERED = $previousUnbuffered
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
