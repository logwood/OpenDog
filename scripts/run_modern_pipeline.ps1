[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Train", "Resume", "Final", "ResumeFinal", "LatentSmoke", "LatentTrain", "LatentResume", "AuthorPhaseB", "RetrainedPhaseB")]
    [string]$Mode = "Smoke",

    [ValidateSet("s101_224", "s101_256", "s101_288", "s200_224")]
    [string]$Branch = "s101_224",

    [string]$Python = $(
        if (Test-Path -LiteralPath "D:\CondaData\envs\torch312\python.exe") {
            "D:\CondaData\envs\torch312\python.exe"
        } else {
            "python"
        }
    ),

    # Relative paths are resolved from the reproduction bundle root. Output
    # is appended so Resume/ResumeFinal can continue the same outer log.
    [string]$LogFile = "",

    # With -LogFile, suppress mirrored console output and write only to file.
    [switch]$LogOnly,

    # Standard runs use artifacts/runs/<workstream>/<run-id>. Resume modes pick
    # the latest manifest when RunId is omitted.
    [string]$RunId = "",

    # Explicit compatibility mode for continuing pre-cleanup runs.
    [switch]$LegacyLayout,

    [switch]$AllowCheckpointCleanup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Repo = (Resolve-Path (Join-Path $BundleRoot "src\Pet-ReID-IMAG")).Path
$LegacyRuns = (Resolve-Path (Join-Path $BundleRoot "artifacts\runs\legacy")).Path
$RunsRoot = (Resolve-Path (Join-Path $BundleRoot "artifacts\runs")).Path
$ResolvedLogFile = $null

if ($LogOnly -and [string]::IsNullOrWhiteSpace($LogFile)) {
    throw "-LogOnly requires -LogFile."
}

if (-not [string]::IsNullOrWhiteSpace($LogFile)) {
    if ([System.IO.Path]::IsPathRooted($LogFile)) {
        $ResolvedLogFile = [System.IO.Path]::GetFullPath($LogFile)
    } else {
        $ResolvedLogFile = [System.IO.Path]::GetFullPath((Join-Path $BundleRoot $LogFile))
    }
    $LogDirectory = Split-Path -Parent $ResolvedLogFile
    if (-not (Test-Path -LiteralPath $LogDirectory)) {
        New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    }
    Write-Host "Combined stdout/stderr log: $ResolvedLogFile"
}

function Invoke-Python {
    param([Parameter(Mandatory)][string[]]$Arguments)

    if ($null -eq $ResolvedLogFile) {
        & $Python @Arguments
        $ExitCode = $LASTEXITCODE
    } else {
        $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $CommandLine = "$Python $($Arguments -join ' ')"
        @("", "[$Timestamp] $CommandLine") |
            Out-File -LiteralPath $ResolvedLogFile -Append -Encoding utf8
        if ($LogOnly) {
            & $Python @Arguments 2>&1 |
                Out-File -LiteralPath $ResolvedLogFile -Append -Encoding utf8
        } else {
            & $Python @Arguments 2>&1 |
                ForEach-Object {
                    $_ | Out-File -LiteralPath $ResolvedLogFile -Append -Encoding utf8
                    $_
                }
        }
        $ExitCode = $LASTEXITCODE
    }

    if ($ExitCode -ne 0) {
        throw "Python command failed with exit code ${ExitCode}: $($Arguments -join ' ')"
    }
}

function Resolve-Run {
    param(
        [Parameter(Mandatory)][string]$Workstream,
        [Parameter(Mandatory)][string]$Purpose,
        [switch]$ResumeRun
    )

    if ($LegacyLayout) {
        return [pscustomobject]@{
            Standard = $false
            Arguments = @()
            Output = $null
            Id = $null
        }
    }
    foreach ($value in @($Workstream, $Purpose)) {
        if ($value -notmatch '^[A-Za-z0-9._-]+$') {
            throw "Invalid standard run name: $value"
        }
    }
    $selectedId = $RunId
    $workstreamRoot = Join-Path $RunsRoot $Workstream
    if ([string]::IsNullOrWhiteSpace($selectedId)) {
        if ($ResumeRun) {
            $latest = @(
                Get-ChildItem -LiteralPath $workstreamRoot -Directory -ErrorAction SilentlyContinue |
                    Where-Object {
                        Test-Path -LiteralPath (Join-Path $_.FullName 'run_manifest.json')
                    } |
                    Sort-Object LastWriteTimeUtc -Descending
            ) | Select-Object -First 1
            if (-not $latest) {
                throw "No standard run can be resumed under $workstreamRoot. Pass -RunId or use -LegacyLayout."
            }
            $selectedId = $latest.Name
        }
        else {
            $selectedId = '{0}_{1}_{2}' -f (
                Get-Date -Format 'yyyyMMdd-HHmmss'
            ), $Workstream, $Purpose
        }
    }
    if ($selectedId -notmatch '^[A-Za-z0-9._-]+$' -or $selectedId -in @('.', '..')) {
        throw "Invalid -RunId: $selectedId"
    }
    $output = Join-Path $workstreamRoot $selectedId
    return [pscustomobject]@{
        Standard = $true
        Arguments = @(
            '--run-workstream', $Workstream,
            '--run-id', $selectedId,
            '--run-purpose', $Purpose
        )
        Output = $output
        Id = $selectedId
    }
}

function Invoke-Training {
    param(
        [Parameter(Mandatory)][string]$Config,
        [Parameter(Mandatory)][string]$Workstream,
        [Parameter(Mandatory)][string]$Purpose,
        [switch]$ResumeRun
    )

    $run = Resolve-Run -Workstream $Workstream -Purpose $Purpose -ResumeRun:$ResumeRun
    $arguments = @('pet_id/train_net.py', '--config-file', $Config) + @($run.Arguments)
    if ($AllowCheckpointCleanup) { $arguments += '--allow-checkpoint-cleanup' }
    if ($ResumeRun) { $arguments += '--resume' }

    $previousLog = $script:ResolvedLogFile
    try {
        if ($run.Standard) {
            New-Item -ItemType Directory -Path $run.Output -Force | Out-Null
            $standardLog = Join-Path $run.Output 'stdout.log'
            if ($null -eq $script:ResolvedLogFile) {
                $script:ResolvedLogFile = $standardLog
            }
            elseif ([System.IO.Path]::GetFullPath($script:ResolvedLogFile) -ne
                [System.IO.Path]::GetFullPath($standardLog)) {
                "Console output was redirected to $($script:ResolvedLogFile)" |
                    Set-Content -LiteralPath $standardLog -Encoding UTF8
            }
            Write-Host "Standard run: $($run.Output)"
        }
        Invoke-Python -Arguments $arguments
    }
    finally {
        $script:ResolvedLogFile = $previousLog
    }
}

Invoke-Python -Arguments @(
    (Join-Path $BundleRoot "scripts\prepare_upstream_assets.py"),
    "--verify-only"
)

$ValidationConfigs = @{
    s101_224 = "configs/modern_s101_224.yaml"
    s101_256 = "configs/modern_s101_256.yaml"
    s101_288 = "configs/modern_s101_288.yaml"
    s200_224 = "configs/modern_s200_224.yaml"
}
$FinalConfigs = @{
    s101_224 = "configs/modern_final_s101_224.yaml"
    s101_256 = "configs/modern_final_s101_256.yaml"
    s101_288 = "configs/modern_final_s101_288.yaml"
    s200_224 = "configs/modern_final_s200_224.yaml"
}
$SubmitConfigs = @{
    s101_224 = "configs/s101_224_submit.yaml"
    s101_256 = "configs/s101_256_submit.yaml"
    s101_288 = "configs/s101_288_submit.yaml"
    s200_224 = "configs/s200_submit.yaml"
}

Push-Location $Repo
try {
    switch ($Mode) {
        "Smoke" {
            Invoke-Training -Config "configs/modern_smoke.yaml" -Workstream "reproduction-smoke" -Purpose "smoke"
        }
        "Train" {
            Invoke-Training -Config $ValidationConfigs[$Branch] -Workstream "reproduction-validation-$Branch" -Purpose "train"
        }
        "Resume" {
            Invoke-Training -Config $ValidationConfigs[$Branch] -Workstream "reproduction-validation-$Branch" -Purpose "train" -ResumeRun
        }
        "Final" {
            Invoke-Training -Config $FinalConfigs[$Branch] -Workstream "reproduction-final-$Branch" -Purpose "final"
        }
        "ResumeFinal" {
            Invoke-Training -Config $FinalConfigs[$Branch] -Workstream "reproduction-final-$Branch" -Purpose "final" -ResumeRun
        }
        "LatentSmoke" {
            Invoke-Training -Config "configs/modern_latent_workspace_smoke.yaml" -Workstream "latent-workspace-smoke" -Purpose "smoke"
        }
        "LatentTrain" {
            Invoke-Training -Config "configs/modern_latent_workspace_s101_224.yaml" -Workstream "latent-workspace" -Purpose "train"
        }
        "LatentResume" {
            Invoke-Training -Config "configs/modern_latent_workspace_s101_224.yaml" -Workstream "latent-workspace" -Purpose "train" -ResumeRun
        }
        "AuthorPhaseB" {
            foreach ($name in @("s101_224", "s101_256", "s101_288", "s200_224")) {
                Invoke-Python -Arguments @(
                    "pet_id/train_net.py", "--config-file", $SubmitConfigs[$name],
                    "--eval-only", "--save-features",
                    "DATALOADER.NUM_WORKERS", "0", "TEST.IMS_PER_BATCH", "32"
                )
            }
            Invoke-Python -Arguments @(
                (Join-Path $BundleRoot "scripts\fuse_and_score.py"),
                "--workspace-root", $BundleRoot,
                "--output", (Join-Path $LegacyRuns "fusion_submit\submit_modern.csv")
            )
        }
        "RetrainedPhaseB" {
            foreach ($name in @("s101_224", "s101_256", "s101_288", "s200_224")) {
                $featureDir = "retrained_features/$name"
                $weightFile = Join-Path $LegacyRuns "retrained_$name\model_final.pth"
                Invoke-Python -Arguments @(
                    "pet_id/train_net.py", "--config-file", $SubmitConfigs[$name],
                    "--eval-only", "--save-features",
                    "MODEL.WEIGHTS", $weightFile,
                    "OUTPUT_DIR", (Join-Path $LegacyRuns $featureDir),
                    "DATALOADER.NUM_WORKERS", "0", "TEST.IMS_PER_BATCH", "32"
                )
            }
            Invoke-Python -Arguments @(
                (Join-Path $BundleRoot "scripts\fuse_and_score.py"),
                "--workspace-root", $BundleRoot,
                "--model-dirs",
                "retrained_features/s101_224", "retrained_features/s101_256",
                "retrained_features/s101_288", "retrained_features/s200_224",
                "--query-names", (Join-Path $LegacyRuns "retrained_features\s101_224\query_filename.txt"),
                "--gallery-names", (Join-Path $LegacyRuns "retrained_features\s101_224\gallery_filename.txt"),
                "--output", (Join-Path $LegacyRuns "retrained_fusion\submit.csv")
            )
        }
    }
} finally {
    Pop-Location
}
