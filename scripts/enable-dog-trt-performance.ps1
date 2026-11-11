# Only changes the dedicated executable's NVIDIA application profile.
[CmdletBinding()]
param(
    [switch]$Disable,
    [string]$EvidenceDirectory = ''
)
$ErrorActionPreference = 'Stop'
$trtRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$trtPython = Join-Path $trtRoot '.tmp/dog-trt-env/Scripts/python.exe'
$trtApplication = Join-Path $trtRoot '.tmp/dog-trt-env/Scripts/dog_capture_trt.exe'
$trtTool = Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/configure_dog_trt_windows.py'
$trtEvidence = $EvidenceDirectory
if (-not $trtEvidence) {
    $trtEvidence = Join-Path $trtRoot ('artifacts/runs/dog_capture/profile-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
}
$trtEvidence = [IO.Path]::GetFullPath($trtEvidence)
New-Item -ItemType Directory -Path $trtEvidence -Force | Out-Null
$trtLog = Join-Path $trtEvidence 'profile_apply.log'
$trtExitCode = 1
try {
    if (-not (Test-Path -LiteralPath $trtApplication)) { throw 'Run scripts/setup-dog-trt.ps1 first.' }
    $trtBeforeText = & $trtPython $trtTool --application $trtApplication
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA profile query failed.' }
    $trtBefore = ($trtBeforeText -join "`n") | ConvertFrom-Json
    $trtBeforeText | Set-Content -LiteralPath (Join-Path $trtEvidence 'profile_before.json') -Encoding UTF8
    $trtNeedsChange = if ($Disable) { $trtBefore.profile_exists } else { -not $trtBefore.profile_exists -or $trtBefore.preferred_pstate -ne 1 }
    if ($trtNeedsChange) {
        $trtIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $trtAdmin = (New-Object Security.Principal.WindowsPrincipal($trtIdentity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        if (-not $trtAdmin) {
            $trtArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $PSCommandPath + '"'), '-EvidenceDirectory', ('"' + $trtEvidence + '"'))
            if ($Disable) { $trtArguments += '-Disable' }
            $trtPowerShell = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
            $trtProcess = Start-Process -FilePath $trtPowerShell -ArgumentList $trtArguments -Verb RunAs -WindowStyle Hidden -Wait -PassThru
            if ($trtProcess.ExitCode -ne 0) { throw "NVIDIA profile change failed; see $trtLog" }
        } else {
            $trtAction = if ($Disable) { '--disable' } else { '--enable' }
            & $trtPython $trtTool --application $trtApplication $trtAction *> $trtLog
            if ($LASTEXITCODE -ne 0) { throw "NVIDIA profile write failed; see $trtLog" }
        }
    }
    $trtStateText = & $trtPython $trtTool --application $trtApplication
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA profile readback failed.' }
    $trtState = ($trtStateText -join "`n") | ConvertFrom-Json
    $trtVerified = if ($Disable) { -not $trtState.profile_exists } else { $trtState.profile_exists -and $trtState.preferred_pstate -eq 1 }
    if (-not $trtVerified -or $trtState.global_preferred_pstate -ne $trtBefore.global_preferred_pstate) {
        throw 'NVIDIA profile readback did not confirm the requested setting.'
    }
    $trtStateText | Set-Content -LiteralPath (Join-Path $trtEvidence 'profile_after.json') -Encoding UTF8
    $trtStateText | Write-Output
    $trtExitCode = 0
} catch {
    $_ | Out-String | Add-Content -LiteralPath $trtLog -Encoding UTF8
    Write-Warning $_
} finally {
    Set-Content -LiteralPath (Join-Path $trtEvidence 'profile_exit_code.txt') -Value $trtExitCode -Encoding ASCII
}
exit $trtExitCode
