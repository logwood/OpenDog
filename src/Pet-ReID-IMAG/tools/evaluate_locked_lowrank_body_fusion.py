#!/usr/bin/env python3
"""Evaluate a locked 512-D low-rank body-fusion model without reselection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOOL = ROOT / "tools" / "train_evaluate_body_primary_fusion.py"
SPEC = importlib.util.spec_from_file_location("_body_primary_train_eval", TRAIN_TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--multimodal-features", type=Path, required=True)
    parser.add_argument("--body-features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    model_path = args.model.resolve()
    manifest_path = args.manifest.resolve()
    if sha256_file(model_path) != lock["model"]["sha256"]:
        raise ValueError("Locked model hash mismatch")
    locked_test = lock["protocol_files"]["blind_test"]
    if manifest_path != Path(locked_test["path"]).resolve():
        raise ValueError("Requested manifest is not the locked blind_test split")
    if sha256_file(manifest_path) != locked_test["sha256"]:
        raise ValueError("Locked blind_test manifest hash mismatch")

    dataset = MODULE.FrozenFeatureSet(
        args.multimodal_features,
        args.body_features,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    body_weight = float(checkpoint["body_weight"])
    semantic = F.normalize(dataset.baseline.float(), dim=1) * math.sqrt(
        1.0 - body_weight
    )
    body = F.normalize(dataset.body.float(), dim=1) * math.sqrt(body_weight)
    joint = torch.cat((semantic, body), dim=1)
    mean = checkpoint["mean"].float()
    projection = checkpoint["projection"].float()
    features = (joint - mean) @ projection.T
    output_dim = int(checkpoint["architecture"]["output_dim"])
    if features.shape[1] < output_dim:
        features = F.pad(features, (0, output_dim - features.shape[1]))
    features = F.normalize(features, dim=1)
    if features.shape[1] != 512:
        raise RuntimeError(f"Locked model output is not 512-D: {features.shape}")
    metrics = MODULE.evaluate_features(
        features,
        dataset.identities,
        dataset.source_paths,
    )
    baseline = MODULE.evaluate_features(
        dataset.baseline,
        dataset.identities,
        dataset.source_paths,
    )
    report = {
        "schema_version": 1,
        "evaluation_purpose": "spent_test_diagnostic",
        "fresh_blind_claim": False,
        "historical_caveat": lock["historical_caveat"],
        "lock": str(args.lock.resolve()),
        "locked_model": {
            "path": str(model_path),
            "sha256": lock["model"]["sha256"],
            "body_weight": body_weight,
            "output_dim": features.shape[1],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": locked_test["sha256"],
        },
        "records": len(dataset),
        "identities": dataset.num_classes,
        "baseline_semantic_embedding": baseline,
        "body_fusion": metrics,
        "delta_vs_nose_face": {
            "gallery_rank1": metrics["gallery_rank1"] - baseline["gallery_rank1"],
            "leave_one_out_rank1": metrics["leave_one_out_rank1"]
            - baseline["leave_one_out_rank1"],
            "auc": metrics["auc"] - baseline["auc"],
            "balanced_accuracy": metrics["best_balanced_threshold"][
                "balanced_accuracy"
            ]
            - baseline["best_balanced_threshold"]["balanced_accuracy"],
            "mean_gap": metrics["mean_gap"] - baseline["mean_gap"],
        },
        "output_contract": {
            "input": "one dog image through internal branches",
            "embedding_shape": [len(dataset), features.shape[1]],
            "unit_norm_max_error": float((features.norm(dim=1) - 1.0).abs().max()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features.npz",
        features=features.numpy(),
        identities=np.asarray(dataset.identities),
        source_paths=np.asarray(dataset.source_paths),
    )
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(report_path.resolve()),
                "evaluation_purpose": report["evaluation_purpose"],
                "baseline": {
                    key: baseline[key]
                    for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                },
                "body_fusion": {
                    key: metrics[key]
                    for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
