[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)][string]$Video,
    [string]$Output = '',
    [switch]$Realtime,
    [switch]$Show,
    [switch]$NoPreview
)
$ErrorActionPreference = 'Stop'
$trtRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$trtPython = Join-Path $trtRoot '.tmp/dog-trt-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $trtPython)) {
    throw 'Run scripts/setup-dog-trt.ps1 once to install dependencies and build engines.'
}
if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) { throw "Video not found: $Video" }
$trtVideo = (Resolve-Path -LiteralPath $Video).Path
if (-not $Output) { $Output = Join-Path $trtRoot ('artifacts/runs/dog_capture/trt-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
if ($Realtime) {
    $trtArgs = @((Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/live_dog_views.py'), '--video', $trtVideo)
    if ($Show) { $trtArgs += '--show' }
} else {
    if ($Show) { throw '-Show requires -Realtime' }
    $trtArgs = @((Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/select_dog_views.py'), $trtVideo)
    if ($NoPreview) { $trtArgs += '--no-preview' }
}
$trtArgs += @('--device', 'trt', '--output', $Output)
$trtDedicated = Join-Path $trtRoot '.tmp/dog-trt-env/Scripts/dog_capture_trt.exe'
if (Test-Path -LiteralPath $trtDedicated) {
    $trtPython = $trtDedicated
    $trtArgs = @((Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/bootstrap_dog_trt.py')) + $trtArgs
}
& $trtPython @trtArgs
if ($LASTEXITCODE -ne 0) { throw 'TensorRT capture failed; see output above.' }
