#!/usr/bin/env bash
set -euo pipefail

STAGE1_WORKSPACE="${STAGE1_WORKSPACE:-/data/senwang/pet-reid-imag-unified-stage1-20260901}"
STAGE1_PYTHON="${STAGE1_PYTHON:-/data/senwang/envs/pet-reid-imag/bin/python}"
STAGE1_GPU="${STAGE1_GPU:-2}"
STAGE1_CONTROL="${STAGE1_WORKSPACE}/artifacts/runs/remote_full/control"
STAGE1_LOG="${STAGE1_CONTROL}/unified_nose_detail_batch80.log"
STAGE1_PID="${STAGE1_CONTROL}/unified_nose_detail_batch80.pid"

mkdir -p "${STAGE1_CONTROL}"
if [[ -f "${STAGE1_PID}" ]]; then
  existing_pid="$(<"${STAGE1_PID}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Training is already running with PID ${existing_pid}" >&2
    exit 2
  fi
fi

cd "${STAGE1_WORKSPACE}"
training_command=(
  "${STAGE1_PYTHON}"
  "src/Pet-ReID-IMAG/tools/train_unified_nose_detail.py"
  "--config"
  "src/Pet-ReID-IMAG/configs/unified_nose_detail_batch80.yaml"
  "--device"
  "cuda:0"
)
nohup env CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" PYTHONUNBUFFERED=1 "${training_command[@]}" >"${STAGE1_LOG}" 2>&1 </dev/null &
training_pid="$!"
printf '%s\n' "${training_pid}" >"${STAGE1_PID}"

sleep 2
if ! kill -0 "${training_pid}" 2>/dev/null; then
  tail -n 80 "${STAGE1_LOG}" >&2 || true
  exit 1
fi

printf '{"pid":%s,"physical_gpu":%s,"log":"%s","workspace":"%s"}\n' "${training_pid}" "${STAGE1_GPU}" "${STAGE1_LOG}" "${STAGE1_WORKSPACE}"
