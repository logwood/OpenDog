Set-StrictMode -Version Latest

$script:PetReIDSourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:PetReIDWorkspaceRoot = (
    Resolve-Path (Join-Path $script:PetReIDSourceRoot '..\..')
).Path

function Resolve-PetReIDPython {
    param([Parameter(Mandatory)][string] $Candidate)

    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        $resolved = [System.IO.Path]::GetFullPath($Candidate)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "Python interpreter was not found at $resolved"
        }
        return $resolved
    }
    $command = Get-Command $Candidate -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Python interpreter '$Candidate' was not found on PATH."
    }
    return $command.Source
}

function Resolve-PetReIDRun {
    param(
        [Parameter(Mandatory)][string] $Workstream,
        [Parameter(Mandatory)][string] $Purpose,
        [Parameter(Mandatory)][string] $LegacyName,
        [string] $RunId,
        [switch] $Resume,
        [switch] $LegacyLayout
    )

    if ($LegacyLayout) {
        return [pscustomobject]@{
            OutputPath = Join-Path $script:PetReIDWorkspaceRoot (
                "artifacts\runs\legacy\$LegacyName"
            )
            RunId = $LegacyName
            CliArguments = @()
            StandardLayout = $false
        }
    }
    foreach ($value in @($Workstream, $Purpose)) {
        if ($value -notmatch '^[A-Za-z0-9._-]+$') {
            throw "Run names may contain only letters, numbers, dot, underscore and dash: $value"
        }
    }
    $runRoot = Join-Path $script:PetReIDWorkspaceRoot "artifacts\runs\$Workstream"
    if ([string]::IsNullOrWhiteSpace($RunId)) {
        if ($Resume) {
            $latest = @(
                Get-ChildItem -LiteralPath $runRoot -Directory -ErrorAction SilentlyContinue |
                    Where-Object {
                        Test-Path -LiteralPath (Join-Path $_.FullName 'run_manifest.json')
                    } |
                    Sort-Object LastWriteTimeUtc -Descending
            ) | Select-Object -First 1
            if (-not $latest) {
                throw "No resumable standard run exists under $runRoot; pass -RunId or use -LegacyLayout."
            }
            $RunId = $latest.Name
        }
        else {
            $RunId = '{0}_{1}_{2}' -f (
                Get-Date -Format 'yyyyMMdd-HHmmss'
            ), $Workstream, $Purpose
        }
    }
    if ($RunId -notmatch '^[A-Za-z0-9._-]+$' -or $RunId -in @('.', '..')) {
        throw "Invalid run id: $RunId"
    }
    return [pscustomobject]@{
        OutputPath = Join-Path $runRoot $RunId
        RunId = $RunId
        CliArguments = @(
            '--run-workstream', $Workstream,
            '--run-id', $RunId,
            '--run-purpose', $Purpose
        )
        StandardLayout = $true
    }
}

function Get-PetReIDDefaultPython {
    if ($env:PET_REID_PYTHON) {
        return $env:PET_REID_PYTHON
    }
    $known = 'D:\CondaData\envs\torch312\python.exe'
    if (Test-Path -LiteralPath $known) {
        return $known
    }
    return 'python'
}
