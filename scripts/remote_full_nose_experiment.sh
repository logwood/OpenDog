#!/usr/bin/env bash
set -euo pipefail

workspace_root="${PET_REID_REMOTE_ROOT:-/data/senwang/pet-reid-imag-20260901}"
python_bin="${PET_REID_PYTHON:-/data/senwang/envs/pet-reid-imag/bin/python}"
project_root="${workspace_root}/src/Pet-ReID-IMAG"
run_root="${project_root}/artifacts/runs/remote_full"
control_root="${run_root}/control"

mkdir -p "${control_root}"

jobs=(
  "0|nose_s101_224_seed2022|configs/remote_full_s101_224.yaml"
  "1|nose_s101_256_seed2022|configs/remote_full_s101_256.yaml"
  "2|nose_s101_288_seed2022|configs/remote_full_s101_288.yaml"
  "3|nose_s200_224_seed2022|configs/remote_full_s200_224.yaml"
  "4|nose_latent_s101_224_seed2022|configs/remote_full_latent_s101_224.yaml"
)

start_jobs() {
  cd "${project_root}"
  for spec in "${jobs[@]}"; do
    IFS='|' read -r gpu run_id config_path <<< "${spec}"
    pid_file="${control_root}/${run_id}.pid"
    log_file="${control_root}/${run_id}.log"
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      echo "already-running ${run_id} pid=$(cat "${pid_file}")"
      continue
    fi
    nohup env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONUNBUFFERED=1 \
      PET_REID_WORKSPACE_ROOT="${workspace_root}" \
      "${python_bin}" -u -m pet_id.train_net \
        --config-file "${config_path}" \
        --num-gpus 1 \
        >"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
    echo "started ${run_id} gpu=${gpu} pid=$! log=${log_file}"
  done
}

show_status() {
  for spec in "${jobs[@]}"; do
    IFS='|' read -r gpu run_id config_path <<< "${spec}"
    pid_file="${control_root}/${run_id}.pid"
    log_file="${control_root}/${run_id}.log"
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      state="running"
    elif [[ -f "${pid_file}" ]]; then
      state="finished-or-failed"
    else
      state="not-started"
    fi
    echo "${run_id} gpu=${gpu} state=${state} pid=$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -f "${log_file}" ]]; then
      tail -n 3 "${log_file}"
    fi
  done
}

case "${1:-status}" in
  start) start_jobs ;;
  status) show_status ;;
  *) echo "usage: $0 {start|status}" >&2; exit 2 ;;
esac
