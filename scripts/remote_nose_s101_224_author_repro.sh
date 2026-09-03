#!/usr/bin/env bash
set -euo pipefail

workspace_root="${PET_REID_REMOTE_ROOT:-/data/senwang/pet-reid-imag-20260901}"
python_bin="${PET_REID_PYTHON:-/data/senwang/envs/pet-reid-imag/bin/python}"
project_root="${workspace_root}/src/Pet-ReID-IMAG"
output_root="${workspace_root}/artifacts/runs/remote_full/nose_s101_224_author_repro_seed2022"
control_root="${workspace_root}/artifacts/runs/remote_full/control"
run_id="nose_s101_224_author_repro_seed2022"
pid_file="${control_root}/${run_id}.pid"
log_file="${control_root}/${run_id}.log"
cuda_device="${PET_REID_CUDA_DEVICE:-2}"

mkdir -p "${control_root}" "${output_root}"

start_run() {
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "already-running pid=$(cat "${pid_file}")"
    return
  fi
  cd "${project_root}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONUNBUFFERED=1 \
    PET_REID_WORKSPACE_ROOT="${workspace_root}" \
    "${python_bin}" -u -m pet_id.train_net \
      --config-file configs/remote_nose_s101_224_author_repro.yaml \
      --num-gpus 1 \
      >"${log_file}" 2>&1 &
  echo $! >"${pid_file}"
  echo "started pid=$! gpu=${cuda_device} world_size=1 batch=80 amp=on log=${log_file}"
}

show_status() {
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "running pid=$(cat "${pid_file}")"
  elif [[ -f "${pid_file}" ]]; then
    echo "finished-or-failed pid=$(cat "${pid_file}")"
  else
    echo "not-started"
  fi
  [[ -f "${log_file}" ]] && tail -n 30 "${log_file}"
}

case "${1:-status}" in
  start) start_run ;;
  status) show_status ;;
  *) echo "usage: $0 {start|status}" >&2; exit 2 ;;
esac
