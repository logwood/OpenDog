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
    [switch]$LogOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Repo = (Resolve-Path (Join-Path $BundleRoot "upstream\Pet-ReID-IMAG")).Path
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
            Invoke-Python -Arguments @("pet_id/train_net.py", "--config-file", "configs/modern_smoke.yaml")
        }
        "Train" {
            Invoke-Python -Arguments @("pet_id/train_net.py", "--config-file", $ValidationConfigs[$Branch])
        }
        "Resume" {
            Invoke-Python -Arguments @("pet_id/train_net.py", "--config-file", $ValidationConfigs[$Branch], "--resume")
        }
        "Final" {
            Invoke-Python -Arguments @("pet_id/train_net.py", "--config-file", $FinalConfigs[$Branch])
        }
        "ResumeFinal" {
            Invoke-Python -Arguments @(
                "pet_id/train_net.py", "--config-file", $FinalConfigs[$Branch], "--resume"
            )
        }
        "LatentSmoke" {
            Invoke-Python -Arguments @(
                "pet_id/train_net.py", "--config-file", "configs/modern_latent_workspace_smoke.yaml"
            )
        }
        "LatentTrain" {
            Invoke-Python -Arguments @(
                "pet_id/train_net.py", "--config-file", "configs/modern_latent_workspace_s101_224.yaml"
            )
        }
        "LatentResume" {
            Invoke-Python -Arguments @(
                "pet_id/train_net.py", "--config-file", "configs/modern_latent_workspace_s101_224.yaml", "--resume"
            )
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
                "--root", ".",
                "--output", "logs/fusion_submit/submit_modern.csv"
            )
        }
        "RetrainedPhaseB" {
            foreach ($name in @("s101_224", "s101_256", "s101_288", "s200_224")) {
                $featureDir = "retrained_features/$name"
                $weightFile = "logs/retrained_$name/model_final.pth"
                Invoke-Python -Arguments @(
                    "pet_id/train_net.py", "--config-file", $SubmitConfigs[$name],
                    "--eval-only", "--save-features",
                    "MODEL.WEIGHTS", $weightFile,
                    "OUTPUT_DIR", "logs/$featureDir",
                    "DATALOADER.NUM_WORKERS", "0", "TEST.IMS_PER_BATCH", "32"
                )
            }
            Invoke-Python -Arguments @(
                (Join-Path $BundleRoot "scripts\fuse_and_score.py"),
                "--root", ".",
                "--model-dirs",
                "retrained_features/s101_224", "retrained_features/s101_256",
                "retrained_features/s101_288", "retrained_features/s200_224",
                "--query-names", "logs/retrained_features/s101_224/query_filename.txt",
                "--gallery-names", "logs/retrained_features/s101_224/gallery_filename.txt",
                "--output", "logs/retrained_fusion/submit.csv"
            )
        }
    }
} finally {
    Pop-Location
}
