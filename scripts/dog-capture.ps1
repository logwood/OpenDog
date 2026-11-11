# dog_capture v1: one entry for video / photos / camera / RTSP. Extra arguments go to `python -m dog_capture run`.
# First run creates .tmp/dog-capture-v1-env and downloads the YOLOX species gate (Apache-2.0, ~3.7 MB).
param([Parameter(Mandatory=$true, Position=0)][string]$Source,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$envDir = Join-Path $root '.tmp/dog-capture-v1-env'
$python = Join-Path $envDir 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $base = if (Test-Path 'D:/CondaData/envs/torch312/python.exe') { 'D:/CondaData/envs/torch312/python.exe' } else { 'python' }
    & (Join-Path $PSScriptRoot 'setup-dog-capture-v1.ps1') -Python $base
}
$env:PYTHONPATH = Join-Path $root 'src/Pet-ReID-IMAG'
& $python -c 'import numpy, cv2, onnxruntime, yaml' 2>$null
if ($LASTEXITCODE -ne 0) { & (Join-Path $PSScriptRoot 'setup-dog-capture-v1.ps1') -Repair }
& $python -m dog_capture models --download yolox_nano | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Species gate preparation failed.' }
if (Test-Path -LiteralPath $Source) { $Source = (Resolve-Path -LiteralPath $Source).Path }
& $python -m dog_capture run $Source @Rest
$captureExit = $LASTEXITCODE
if ($LASTEXITCODE -eq 2) { Write-Host 'No qualifying view was captured; see the guidance lines above.' }
elseif ($LASTEXITCODE -ne 0) { throw 'dog_capture failed; see the output above' }
exit $captureExit
