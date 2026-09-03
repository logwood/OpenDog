#!/usr/bin/env python3
"""Fail fast unless the remote run matches the upstream S101/224 recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fastreid.config import get_cfg
from pet_id import add_retri_config
from pet_id.workspace_paths import normalize_runtime_config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/remote_nose_s101_224_author_repro.yaml",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    normalize_runtime_config(cfg)

    checks = {
        "seed": int(cfg.SEED),
        "backbone": cfg.MODEL.BACKBONE.NAME,
        "depth": cfg.MODEL.BACKBONE.DEPTH,
        "train_size": list(cfg.INPUT.SIZE_TRAIN),
        "test_size": list(cfg.INPUT.SIZE_TEST),
        "classifier": cfg.MODEL.HEADS.CLS_LAYER,
        "losses": sorted(cfg.MODEL.LOSSES.NAME),
        "amp": bool(cfg.SOLVER.AMP.ENABLED),
        "optimizer": cfg.SOLVER.OPT,
        "max_epoch": int(cfg.SOLVER.MAX_EPOCH),
        "delay_epochs": int(cfg.SOLVER.DELAY_EPOCHS),
        "base_lr": float(cfg.SOLVER.BASE_LR),
        "global_batch": int(cfg.SOLVER.IMS_PER_BATCH),
        "warmup_iters": int(cfg.SOLVER.WARMUP_ITERS),
        "freeze_iters": int(cfg.SOLVER.FREEZE_ITERS),
        "sampler": cfg.DATALOADER.SAMPLER_TRAIN,
        "num_instance": int(cfg.DATALOADER.NUM_INSTANCE),
        "num_workers": int(cfg.DATALOADER.NUM_WORKERS),
        "train_dataset": list(cfg.DATASETS.NAMES),
        "validation_dataset": list(cfg.DATASETS.TESTS),
        "expected_world_size": 1,
        "per_rank_batch": int(cfg.SOLVER.IMS_PER_BATCH),
    }

    expected = {
        "seed": 2022,
        "backbone": "build_resnest_backbone",
        "depth": "101x",
        "train_size": [224, 224],
        "test_size": [224, 224],
        "classifier": "CosSoftmax",
        "losses": ["CircleLoss", "CrossEntropyLoss", "TripletLoss"],
        "amp": True,
        "optimizer": "Adam",
        "max_epoch": 35,
        "delay_epochs": 5,
        "base_lr": 0.00035,
        "global_batch": 80,
        "warmup_iters": 400,
        "freeze_iters": 1000,
        "sampler": "NaiveIdentitySampler",
        "num_instance": 4,
        "num_workers": 8,
        "train_dataset": ["PetID"],
        "validation_dataset": ["PetIDValidation"],
        "expected_world_size": 1,
        "per_rank_batch": 80,
    }
    require(checks == expected, f"author-recipe mismatch: expected {expected}, got {checks}")
    require(bool(cfg.MODEL.BACKBONE.PRETRAIN), "backbone pretraining is disabled")
    require(Path(cfg.MODEL.BACKBONE.PRETRAIN_PATH).is_file(), "pretraining checkpoint is missing")
    require(list(cfg.MODEL.FREEZE_LAYERS) == ["backbone"], "freeze layer mismatch")
    require(bool(cfg.INPUT.CROP.ENABLED), "random crop is disabled")
    require(bool(cfg.INPUT.BLUR.ENABLED), "blur augmentation is disabled")
    require(bool(cfg.INPUT.AUTOAUG.ENABLED), "auto augmentation is disabled")
    require(bool(cfg.INPUT.AUGMIX.ENABLED), "AugMix is disabled")
    require(bool(cfg.INPUT.AFFINE.ENABLED), "affine augmentation is disabled")

    report = {"status": "ok", "config": str(Path(args.config_file).resolve()), "checks": checks}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
