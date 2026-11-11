param([int]$Port = 8765, [string]$Config)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.tmp/dog-capture-v1-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup-dog-capture-v1.ps1 -Web first.' }
$captureArgs = @((Join-Path $root 'src/Pet-ReID-IMAG/tools/serve_dog_capture.py'), '--port', $Port)
if ($Config) { $captureArgs += @('--config', $Config) }
& $python @captureArgs
exit $LASTEXITCODE
