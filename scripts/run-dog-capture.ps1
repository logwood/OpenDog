[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)][string]$Video,
    [string]$Output = "",
    [int]$Threads = 4,
    [double]$Fps = 0,
    [switch]$Mirror,
    [switch]$Realtime,
    [switch]$Show,
    [ValidateSet('challenge','views')][string]$Mode = 'challenge'
)
$ErrorActionPreference = "Stop"
$captureRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$capturePython = Join-Path $captureRoot '.tmp/dog-capture-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) { throw "Video not found: $Video" }
$captureInput = (Resolve-Path -LiteralPath $Video).Path
if (-not (Test-Path -LiteralPath $capturePython)) {
    $captureBasePython = if (Test-Path 'D:/CondaData/envs/torch312/python.exe') { 'D:/CondaData/envs/torch312/python.exe' } else { 'python' }
    & $captureBasePython -m venv (Join-Path $captureRoot '.tmp/dog-capture-env')
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    & $capturePython -m pip install -r (Join-Path $captureRoot 'requirements-dog-capture.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
$captureArgs = @((Join-Path $captureRoot 'src/Pet-ReID-IMAG/tools/run_dog_capture.py'), $captureInput, '--download-models', '--threads', "$Threads", '--fps', "$Fps")
if ($Realtime) {
    $captureArgs = @((Join-Path $captureRoot 'src/Pet-ReID-IMAG/tools/live_dog_capture.py'), '--video', $captureInput, '--download-models', '--threads', "$Threads")
    if (-not $Output) { $Output = Join-Path $captureRoot ('artifacts/runs/dog_capture/live-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
    if ($Show) { $captureArgs += '--show' }
} elseif ($Show) { throw '-Show requires -Realtime' }
if ($Output) { $captureArgs += @('--output', $Output) }
$captureArgs += @('--mode', $Mode)
if ($Mirror) { $captureArgs += '--mirror' }
& $capturePython @captureArgs
if ($LASTEXITCODE -ne 0) { throw 'Dog capture run failed; see output above' }
