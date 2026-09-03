#!/usr/bin/env python3
"""Run a bounded full-batch DDP optimizer step for the remote nose model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils import comm
from fastreid.utils.events import EventStorage
from pet_id import add_retri_config
from pet_id.train_net import Trainer
from pet_id.workspace_paths import normalize_runtime_config


def setup(args):
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    normalize_runtime_config(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def worker(args):
    cfg = setup(args)
    print(f"SMOKE_STAGE rank={comm.get_rank()} stage=config_ready", flush=True)
    trainer = Trainer(cfg)
    print(f"SMOKE_STAGE rank={comm.get_rank()} stage=trainer_ready", flush=True)
    trainer.resume_or_load(resume=False)
    print(f"SMOKE_STAGE rank={comm.get_rank()} stage=weights_ready", flush=True)
    torch.cuda.reset_peak_memory_stats()

    with EventStorage(0) as storage:
        for step in range(args.steps):
            print(
                f"SMOKE_STAGE rank={comm.get_rank()} stage=step_{step}_begin",
                flush=True,
            )
            trainer._trainer.iter = step
            trainer._trainer.run_step()
            print(
                f"SMOKE_STAGE rank={comm.get_rank()} stage=step_{step}_done",
                flush=True,
            )
        latest = {
            name: float(value[0])
            for name, value in storage.latest().items()
            if name == "total_loss" or name.startswith("loss_")
        }

    local = {
        "rank": comm.get_rank(),
        "cuda_device": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(),
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "metrics": latest,
    }
    gathered = comm.gather(local)
    if comm.is_main_process():
        report = {
            "status": "ok",
            "world_size": comm.get_world_size(),
            "global_batch": int(cfg.SOLVER.IMS_PER_BATCH),
            "amp": bool(cfg.SOLVER.AMP.ENABLED),
            "steps": args.steps,
            "workers": gathered,
        }
        report_path = Path(cfg.OUTPUT_DIR) / "ddp_smoke_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "REMOTE_NOSE_DDP_SMOKE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
    comm.synchronize()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    launch(
        worker,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
