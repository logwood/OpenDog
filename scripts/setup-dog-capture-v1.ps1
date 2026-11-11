# Install/repair native capture; optionally build H5 and install its protocol service.
param([string]$Model, [string]$Metadata, [string]$Python = 'python',
      [switch]$Web, [switch]$Repair, [switch]$NoDownload)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$envDir = Join-Path $root '.tmp/dog-capture-v1-env'
$runtime = Join-Path $envDir 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    & $Python -m venv $envDir
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment. Pass -Python PATH_TO_PYTHON_3_11_OR_NEWER.' }
}
& $runtime -c 'import numpy, cv2, onnxruntime, yaml' 2>$null
$install = $Repair -or $LASTEXITCODE -ne 0
if ($Web) {
    & $runtime -c 'import fastapi, uvicorn, multipart' 2>$null
    $install = $install -or $LASTEXITCODE -ne 0
}
if ($install) {
    $req = if ($Web) { 'requirements-dog-capture-service.txt' } else { 'requirements-dog-capture-v1.txt' }
    & $runtime -m pip install -r (Join-Path $root $req)
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; rerun setup to repair.' }
}
$env:PYTHONPATH = Join-Path $root 'src/Pet-ReID-IMAG'
if ($Model) {
    $importArgs = @('-m', 'dog_capture', 'models', '--import-face', $Model)
    if ($Metadata) { $importArgs += @('--metadata', $Metadata) }
    & $runtime @importArgs
    if ($LASTEXITCODE -ne 0) { throw 'Reference model import failed.' }
} elseif ($Metadata) { throw '-Metadata requires -Model.' }
if (-not $NoDownload) {
    & $runtime -m dog_capture models --download yolox_nano
    if ($LASTEXITCODE -ne 0) { throw 'Species gate download/integrity check failed.' }
}
& $runtime -c "from dog_capture.engine import CaptureConfig; from dog_capture.engine.zoo import build_engine; e=build_engine(CaptureConfig().merged({'runtime':{'providers':'cpu','threads':1},'body_gate':{'enabled':'on'}})); e.detector.warmup(1); print(e.describe())"
if ($LASTEXITCODE -ne 0) { throw 'Capture model check failed. Supply -Model dog_face_pose.onnx -Metadata dog_face_pose.json.' }
if ($Web) {
    Push-Location (Join-Path $root 'src/Pet-ReID-IMAG/frontend/dog-capture')
    try {
        & npm.cmd ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'H5 build failed.' }
    } finally { Pop-Location }
}
Write-Host 'dog_capture v1 ready. Run scripts/dog-capture.ps1 VIDEO --threads 4 --providers cpu'
