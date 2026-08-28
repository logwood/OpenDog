#!/usr/bin/env python3
"""Evaluate a validation-selected body-primary fusion checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOOL = ROOT / "tools" / "train_evaluate_body_primary_fusion.py"
SPEC = importlib.util.spec_from_file_location("_body_primary_train_eval", TRAIN_TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--multimodal-features", type=Path, required=True)
    parser.add_argument("--body-features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation-purpose",
        choices=("development", "spent_test_diagnostic", "locked_final"),
        default="spent_test_diagnostic",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset = MODULE.FrozenFeatureSet(
        args.multimodal_features,
        args.body_features,
    )
    if len(dataset) != len(manifest["records"]):
        raise ValueError("Feature archive and manifest record counts differ")

    device = torch.device(args.device)
    dataset.to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = MODULE.BodyPrimaryFusionNeck(
        body_dim=dataset.body.shape[1],
        embedding_dim=dataset.face.shape[1],
        body_quality_dim=dataset.body_quality.shape[1],
        nose_quality_dim=dataset.nose_quality.shape[1],
    ).to(device)
    model.load_state_dict(checkpoint["neck"], strict=True)
    output = MODULE.feature_outputs(model, dataset)

    fused = MODULE.evaluate_features(
        output["features"], dataset.identities, dataset.source_paths
    )
    baseline = MODULE.evaluate_features(
        dataset.baseline.detach().cpu(), dataset.identities, dataset.source_paths
    )
    raw_body = MODULE.evaluate_features(
        dataset.body.detach().cpu(), dataset.identities, dataset.source_paths
    )
    both = dataset.available[:, 1] & dataset.available[:, 2]
    report = {
        "schema_version": 1,
        "evaluation_purpose": args.evaluation_purpose,
        "fresh_blind_claim": args.evaluation_purpose == "locked_final",
        "historical_caveat": (
            None
            if args.evaluation_purpose == "locked_final"
            else "These identities were used by earlier project experiments; this is a spent-test diagnostic."
        ),
        "model": str(args.model.resolve()),
        "selected_step": int(checkpoint["step"]),
        "manifest": str(args.manifest.resolve()),
        "records": len(dataset),
        "identities": dataset.num_classes,
        "baseline_semantic_v3_nose_face": baseline,
        "raw_bifor_body": raw_body,
        "body_primary_fusion": fused,
        "delta_vs_nose_face": {
            "gallery_rank1": fused["gallery_rank1"] - baseline["gallery_rank1"],
            "leave_one_out_rank1": fused["leave_one_out_rank1"]
            - baseline["leave_one_out_rank1"],
            "auc": fused["auc"] - baseline["auc"],
            "balanced_accuracy": fused["best_balanced_threshold"][
                "balanced_accuracy"
            ]
            - baseline["best_balanced_threshold"]["balanced_accuracy"],
            "mean_gap": fused["mean_gap"] - baseline["mean_gap"],
        },
        "fusion_diagnostics": {
            "mean_body_weight_when_face_and_body_available": float(
                output["body_weights"][both.cpu(), 0].mean()
            ),
            "mean_nose_weight": float(output["nose_weights"][:, 0].mean()),
        },
        "output_contract": {
            "input": "one dog image through frozen nose, face and BIFOR body encoders",
            "embedding_shape": list(output["features"].shape),
            "unit_norm_max_error": float(
                (output["features"].norm(dim=1) - 1.0).abs().max()
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features.npz",
        features=output["features"].numpy(),
        primary_features=output["primary_features"].numpy(),
        adapted_body_features=output["adapted_body_features"].numpy(),
        body_weights=output["body_weights"].numpy(),
        nose_weights=output["nose_weights"].numpy(),
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
                "baseline": {
                    key: baseline[key]
                    for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                },
                "raw_bifor": {
                    key: raw_body[key]
                    for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                },
                "fusion": {
                    key: fused[key]
                    for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
