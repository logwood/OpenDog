#!/usr/bin/env bash
set -euo pipefail

root="${PET_REID_REMOTE_ROOT:-/data/senwang/pet-reid-imag-20260901}"
project="${root}/src/Pet-ReID-IMAG"
python_bin="${PET_REID_PYTHON:-/data/senwang/envs/pet-reid-imag/bin/python}"
python_dir="$(dirname "${python_bin}")"
config="${PET_REID_CONFIG:-configs/remote_structural_s101_224.yaml}"
run_id="${PET_REID_RUN_ID:-structural_s101_224_seed20260926}"
workstream="${PET_REID_WORKSTREAM:-structural-latent}"
gpu="${PET_REID_CUDA_DEVICE:-5}"
control="${root}/artifacts/runs/${workstream}/control"
run_dir="${root}/artifacts/runs/${workstream}/${run_id}"
pid_file="${control}/${run_id}.pid"
log_file="${control}/${run_id}.log"

# Runtime evaluation compiles FastReID's Cython rank helper through the
# command name ``python3``.  Keep that helper on the same environment as the
# trainer so it sees numpy/Cython and cannot silently fall back to /usr/bin.
export PATH="${python_dir}:${PATH}"

check() {
  cd "${project}"
  test -x "${python_bin}"
  test -f "${config}"
  "${python_bin}" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available()'
  nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  if git rev-parse HEAD >/dev/null 2>&1; then
    git rev-parse HEAD
  else
    echo "remote source is a snapshot; recording structural file hashes"
    sha256sum fastreid/config/defaults.py \
      fastreid/modeling/meta_arch/latent_workspace.py \
      pet_id/body_fusion.py configs/modern_structural_s101_224.yaml \
      configs/remote_structural_s101_224.yaml
  fi
}

case "${1:-status}" in
  check)
    check
    ;;
  start)
    mkdir -p "${control}"
    if [[ -e "${run_dir}" && "${PET_REID_RESUME:-0}" != "1" ]]; then
      echo "run directory already exists: ${run_dir}" >&2
      exit 2
    fi
    check
    cd "${project}"
    nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      PET_REID_WORKSPACE_ROOT="${root}" \
      "${python_bin}" -u -m pet_id.train_net \
      --config-file "${config}" --num-gpus 1 \
      --run-workstream "${workstream}" --run-id "${run_id}" \
      --run-purpose full-train >"${log_file}" 2>&1 < /dev/null &
    echo $! >"${pid_file}"
    sleep 2
    kill -0 "$(cat "${pid_file}")"
    echo "started ${run_id} pid=$(cat "${pid_file}")"
    ;;
  status)
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      echo "running pid=$(cat "${pid_file}")"
    else
      echo "finished-or-not-started"
    fi
    [[ ! -f "${log_file}" ]] || tail -n 60 "${log_file}"
    ;;
  *)
    echo "usage: $0 {check|start|status}" >&2
    exit 2
    ;;
esac
