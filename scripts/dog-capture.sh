#!/usr/bin/env bash
# dog_capture v1 on Linux / macOS / Jetson / Raspberry Pi:
#   scripts/dog-capture.sh VIDEO|IMAGE|FOLDER|CAMERA_INDEX|rtsp://... [python -m dog_capture run options]
# First run creates .tmp/dog-capture-v1-env and downloads the YOLOX species gate (Apache-2.0, ~20 MB).
# On Jetson, install the NVIDIA onnxruntime-gpu wheel into the environment for CUDA/TensorRT.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$root/.tmp/dog-capture-v1-env"
if [ ! -x "$env_dir/bin/python" ]; then
  "${PYTHON:-python3}" -m venv "$env_dir"
  "$env_dir/bin/python" -m pip install -r "$root/requirements-dog-capture-v1.txt"
fi
export PYTHONPATH="$root/src/Pet-ReID-IMAG${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -f "$root/models/pretrained/dog_capture_v1/yolox_tiny.onnx" ]; then
  "$env_dir/bin/python" -m dog_capture models --download yolox_tiny >/dev/null
fi
exec "$env_dir/bin/python" -m dog_capture run "$@"
