#!/usr/bin/env python3
"""Fail-fast audit for the full-capacity remote PetID nose experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fastreid.config import get_cfg
from pet_id import add_retri_config
from pet_id.workspace_paths import PROCESSED_DATA_ROOT, normalize_runtime_config


EXPECTED = {
    "seed": 2022,
    "train_identities": 5400,
    "validation_identities": 600,
    "positive_pairs": 1000,
    "negative_pairs": 1000,
    "max_epoch": 35,
    "global_batch": 80,
    "num_instance": 4,
    "warmup_iters": 400,
    "freeze_iters": 1000,
}


def nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/remote_nose_s101_224_original35.yaml",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(
        os.environ.get("PET_REID_WORKSPACE_ROOT", PROCESSED_DATA_ROOT.parents[2])
    ).resolve()
    split_root = PROCESSED_DATA_ROOT / "splits"
    image_root = PROCESSED_DATA_ROOT / "dir_train_fusai"

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    normalize_runtime_config(cfg)

    train_ids = nonempty_lines(split_root / "train_ids.txt")
    validation_ids = nonempty_lines(split_root / "validation_ids.txt")
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    require(len(train_ids) == len(train_set), "train_ids.txt contains duplicate identities")
    require(
        len(validation_ids) == len(validation_set),
        "validation_ids.txt contains duplicate identities",
    )
    overlap = sorted(train_set & validation_set, key=int)
    require(not overlap, f"training/validation identity leakage: {overlap[:10]}")

    available_ids = {path.name for path in image_root.iterdir() if path.is_dir()}
    require(train_set <= available_ids, "some training identities are missing on disk")
    require(validation_set <= available_ids, "some validation identities are missing on disk")

    pair_counts: Counter[int] = Counter()
    pair_identities: set[str] = set()
    pair_path = split_root / "validation_pairs.csv"
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            label = int(row["label"])
            require(label in (0, 1), f"invalid pair label at row {row_number}: {label}")
            pair_counts[label] += 1
            for field in ("imageA", "imageB"):
                relative = Path(row[field])
                identity = relative.parts[0]
                pair_identities.add(identity)
                require(
                    identity in validation_set,
                    f"pair row {row_number} uses non-validation identity {identity}",
                )
                require(
                    (image_root / relative).is_file(),
                    f"pair row {row_number} references missing image {relative}",
                )

    manifest_path = split_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pretrain_path = Path(cfg.MODEL.BACKBONE.PRETRAIN_PATH)

    checks = {
        "seed": int(cfg.SEED),
        "train_identities": len(train_ids),
        "validation_identities": len(validation_ids),
        "positive_pairs": pair_counts[1],
        "negative_pairs": pair_counts[0],
        "max_epoch": int(cfg.SOLVER.MAX_EPOCH),
        "global_batch": int(cfg.SOLVER.IMS_PER_BATCH),
        "num_instance": int(cfg.DATALOADER.NUM_INSTANCE),
        "warmup_iters": int(cfg.SOLVER.WARMUP_ITERS),
        "freeze_iters": int(cfg.SOLVER.FREEZE_ITERS),
    }
    require(
        checks == EXPECTED,
        f"full-capacity protocol mismatch: expected {EXPECTED}, got {checks}",
    )
    require(manifest.get("seed") == EXPECTED["seed"], "split manifest seed mismatch")
    require(list(cfg.INPUT.SIZE_TRAIN) == [224, 224], "training input is not 224x224")
    require(list(cfg.INPUT.SIZE_TEST) == [224, 224], "test input is not 224x224")
    require(cfg.MODEL.BACKBONE.NAME == "build_resnest_backbone", "backbone is not ResNeSt")
    require(cfg.MODEL.BACKBONE.DEPTH == "101x", "backbone is not ResNeSt-101")
    require(bool(cfg.MODEL.BACKBONE.PRETRAIN), "backbone pretraining is disabled")
    require(pretrain_path.is_file(), f"pretraining checkpoint is missing: {pretrain_path}")
    require(not bool(cfg.SOLVER.AMP.ENABLED), "AMP compromise is unexpectedly enabled")
    require(cfg.SOLVER.OPT == "Adam", "optimizer is not Adam")
    require(abs(float(cfg.SOLVER.BASE_LR) - 3.5e-4) < 1e-12, "base LR mismatch")
    require(
        cfg.DATALOADER.SAMPLER_TRAIN == "NaiveIdentitySampler",
        "identity sampler mismatch",
    )
    require(list(cfg.DATASETS.NAMES) == ["PetID"], "training dataset mismatch")
    require(
        list(cfg.DATASETS.TESTS) == ["PetIDValidation"],
        "validation dataset mismatch",
    )
    require(
        set(cfg.MODEL.LOSSES.NAME)
        == {"CrossEntropyLoss", "TripletLoss", "CircleLoss"},
        "loss recipe mismatch",
    )
    require(cfg.MODEL.HEADS.CLS_LAYER == "CosSoftmax", "classification head mismatch")
    require(
        list(cfg.MODEL.FREEZE_LAYERS) == ["backbone"],
        "backbone freeze layer mismatch",
    )
    require(int(cfg.TEST.EVAL_PERIOD) == 1, "validation is not scheduled every epoch")
    require(
        int(cfg.SOLVER.CHECKPOINT_PERIOD) == 1,
        "checkpoint is not scheduled every epoch",
    )

    train_images = sum(
        len(list((image_root / identity).glob("*.jpg"))) for identity in train_ids
    )
    validation_images = sum(
        len(list((image_root / identity).glob("*.jpg")))
        for identity in validation_ids
    )
    report = {
        "status": "ok",
        "workspace": str(workspace),
        "config": str(Path(args.config_file).resolve()),
        "protocol": checks,
        "identity_overlap": 0,
        "train_images": train_images,
        "validation_images": validation_images,
        "pair_identities": len(pair_identities),
        "pretrain_checkpoint": str(pretrain_path),
        "pretrain_sha256": sha256(pretrain_path),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256(manifest_path),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
