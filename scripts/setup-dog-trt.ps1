[CmdletBinding()]
param(
    [string]$Python = 'python',
    [switch]$Rebuild,
    [switch]$ExportOnnx,
    [switch]$EnableGpuPerformance
)
$ErrorActionPreference = 'Stop'
$trtRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$trtEnv = Join-Path $trtRoot '.tmp/dog-trt-env'
$trtPython = Join-Path $trtEnv 'Scripts/python.exe'
$trtModels = Join-Path $trtRoot 'models/pretrained/dog_capture_trt_v3'
if (-not (Test-Path -LiteralPath $trtPython)) {
    & $Python -m venv $trtEnv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
}
& $trtPython -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for this pinned environment'"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
& $trtPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip installation failed.' }
& $trtPython -m pip install -r (Join-Path $trtRoot 'requirements-dog-trt.txt') -r (Join-Path $trtRoot 'requirements-dog-trt-build.txt')
if ($LASTEXITCODE -ne 0) { throw 'TensorRT dependency installation failed.' }
$trtMissingOnnx = @('localizer','landmarks','detector') | Where-Object { -not (Test-Path (Join-Path $trtModels ($_ + '.onnx'))) }
if ($ExportOnnx -or $trtMissingOnnx.Count) {
    $trtExportEnv = Join-Path $trtRoot '.tmp/dog-model-export-env'
    $trtExportPython = Join-Path $trtExportEnv 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $trtExportPython)) {
        & $Python -m venv $trtExportEnv
        if ($LASTEXITCODE -ne 0) { throw 'Conversion environment creation failed.' }
    }
    & $trtExportPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Conversion pip installation failed.' }
    & $trtExportPython -m pip install -r (Join-Path $trtRoot 'requirements-dog-onnx-export.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Conversion dependency installation failed.' }
    & $trtPython (Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/prepare_dog_trt.py') --download-models --export-only --export-python $trtExportPython
    if ($LASTEXITCODE -ne 0) { throw 'ONNX export failed.' }
    $Rebuild = $true
}
$trtMissingEngine = @('localizer','landmarks','detector') | Where-Object { -not (Test-Path (Join-Path $trtModels ($_ + '.engine'))) }
if ($Rebuild -or $trtMissingEngine.Count) {
    & $trtPython (Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/prepare_dog_trt.py') --build-only
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT engine build failed.' }
}
& $trtPython (Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/prepare_dog_trt.py') --check
if ($LASTEXITCODE -ne 0) { throw 'Engine startup check failed. Re-run with -Rebuild for this GPU/runtime.' }
& $trtPython (Join-Path $trtRoot 'src/Pet-ReID-IMAG/tools/create_dog_trt_runtime.py')
if ($LASTEXITCODE -ne 0) { throw 'Dedicated runtime setup failed.' }
if ($EnableGpuPerformance) {
    & (Join-Path $trtRoot 'scripts/enable-dog-trt-performance.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA application profile write failed; see the profile evidence directory.' }
}
