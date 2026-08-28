#!/usr/bin/env python3
"""Measure steady training cost of baseline vs persistent latent workspace."""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BUNDLE_ROOT / "src" / "Pet-ReID-IMAG"
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from fastreid.config import get_cfg  # noqa: E402
from fastreid.modeling import build_model  # noqa: E402
from fastreid.solver import build_optimizer  # noqa: E402
from fastreid.utils.events import EventStorage  # noqa: E402
from pet_id import add_retri_config  # noqa: E402
from pet_id.workspace_paths import normalize_runtime_config  # noqa: E402


def build_config(config_path, num_classes, latent_dim=None):
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(config_path)
    normalize_runtime_config(cfg)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.LATENT_WORKSPACE.HEALTH_PERIOD = 0
    if latent_dim is not None:
        cfg.MODEL.LATENT_WORKSPACE.DIM = latent_dim
    cfg.SOLVER.FREEZE_ITERS = 0
    cfg.freeze()
    return cfg


def benchmark_one(
    name,
    config_path,
    images,
    targets,
    warmup,
    steps,
    latent_dim=None,
):
    torch.manual_seed(20260810)
    cfg = build_config(config_path, int(targets.max().item()) + 1, latent_dim)
    model = build_model(cfg).train()
    optimizer, _ = build_optimizer(cfg, model, contiguous=False)

    timings = []
    final_loss = None
    with EventStorage(0) as storage:
        for iteration in range(warmup + steps):
            storage.iter = iteration
            batch_images = images.clone()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                losses = model({"images": batch_images, "targets": targets})
                total_loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            final_loss = float(total_loss.detach())
            if iteration + 1 == warmup:
                torch.cuda.reset_peak_memory_stats()
            elif iteration >= warmup:
                timings.append(elapsed)

    result = {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "median_step_seconds": statistics.median(timings),
        "min_step_seconds": min(timings),
        "max_step_seconds": max(timings),
        "steady_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "final_loss": final_loss,
    }

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=28)
    parser.add_argument("--identities", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--latent-dim", type=int, default=192)
    args = parser.parse_args()
    if args.batch % args.identities:
        parser.error("--batch must be divisible by --identities")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("this benchmark requires a BF16-capable CUDA GPU")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(314159)
    images = torch.rand(args.batch, 3, 224, 224, device="cuda") * 255.0
    targets = torch.arange(args.identities, device="cuda").repeat_interleave(
        args.batch // args.identities
    )

    baseline = benchmark_one(
        "baseline", "configs/modern_smoke.yaml", images, targets, args.warmup, args.steps
    )
    latent = benchmark_one(
        "latent_workspace",
        "configs/modern_latent_workspace_smoke.yaml",
        images,
        targets,
        args.warmup,
        args.steps,
        args.latent_dim,
    )
    report = {
        "device": torch.cuda.get_device_name(),
        "batch": args.batch,
        "identities": args.identities,
        "warmup": args.warmup,
        "measured_steps": args.steps,
        "latent_dim": args.latent_dim,
        "baseline": baseline,
        "latent_workspace": latent,
        "time_overhead_percent": 100.0
        * (latent["median_step_seconds"] / baseline["median_step_seconds"] - 1.0),
        "peak_memory_overhead_mib": (
            latent["peak_allocated_mib"] - baseline["peak_allocated_mib"]
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
