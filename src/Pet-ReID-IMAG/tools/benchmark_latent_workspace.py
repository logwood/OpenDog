# encoding: utf-8
"""Compare one-GPU training-step time and memory for latent workspace configs."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.solver import build_optimizer
from fastreid.utils.events import EventStorage
from pet_id import add_retri_config
from pet_id.workspace_paths import normalize_runtime_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=5400)
    return parser.parse_args()


def benchmark(config_path, args):
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = args.num_classes
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.freeze()

    model = build_model(cfg).cuda().train()
    optimizer, _ = build_optimizer(cfg, model)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.SOLVER.AMP.ENABLED and amp_dtype == torch.float16
    )
    images = torch.randn(args.batch_size, 3, 224, 224, device="cuda") * 32 + 128
    identities = max(args.batch_size // 4, 1)
    targets = torch.arange(identities, device="cuda").repeat_interleave(4)
    targets = targets[: args.batch_size]
    if targets.numel() < args.batch_size:
        targets = torch.cat(
            [targets, torch.zeros(args.batch_size - targets.numel(), device="cuda", dtype=torch.long)]
        )

    timings = []
    torch.cuda.reset_peak_memory_stats()
    with EventStorage(0):
        for step_index in range(args.warmup_steps + args.measure_steps):
            optimizer.zero_grad(set_to_none=True)
            batch = {"images": images.clone(), "targets": targets}
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.SOLVER.AMP.ENABLED):
                losses = model(batch)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            end.record()
            torch.cuda.synchronize()
            if step_index >= args.warmup_steps:
                timings.append(start.elapsed_time(end) / 1000.0)

    result = {
        "config": config_path,
        "read_mode": cfg.MODEL.LATENT_WORKSPACE.READ_MODE,
        "median_step_seconds": statistics.median(timings),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    del model, optimizer, scaler, images, targets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    for config_path in args.configs:
        result = benchmark(config_path, args)
        print(
            "{config} mode={read_mode} median={median_step_seconds:.4f}s "
            "peak={peak_memory_gib:.3f}GiB".format(**result)
        )


if __name__ == "__main__":
    main()
