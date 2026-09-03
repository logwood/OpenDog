#!/usr/bin/env python3
"""Evaluate a 512-D low-rank projection of semantic and body features.

This is the compatibility-first alternative to an aligned additive body neck.
The two modalities keep separate subspaces before a training-only SVD
projection, so the body descriptor does not need to imitate face coordinates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOOL = ROOT / "tools" / "train_evaluate_body_primary_fusion.py"
SPEC = importlib.util.spec_from_file_location("_body_primary_train_eval", TRAIN_TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def joint_features(dataset, body_weight: float) -> torch.Tensor:
    return torch.cat(
        (
            F.normalize(dataset.baseline.float(), dim=1)
            * math.sqrt(1.0 - body_weight),
            F.normalize(dataset.body.float(), dim=1) * math.sqrt(body_weight),
        ),
        dim=1,
    )


def selection_key(candidate: dict) -> tuple:
    metrics = candidate["metrics"]
    return (
        metrics["gallery_rank1"],
        metrics["leave_one_out_rank1"],
        metrics["auc"],
        metrics["best_balanced_threshold"]["balanced_accuracy"],
        metrics["mean_gap"],
        metrics["worst_gap"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-multimodal", type=Path, required=True)
    parser.add_argument("--train-body", type=Path, required=True)
    parser.add_argument("--validation-multimodal", type=Path, required=True)
    parser.add_argument("--validation-body", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--body-weights",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.05, 0.08),
    )
    parser.add_argument("--output-dim", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    train = MODULE.FrozenFeatureSet(args.train_multimodal, args.train_body).to(device)
    validation = MODULE.FrozenFeatureSet(
        args.validation_multimodal, args.validation_body
    ).to(device)
    if train.num_classes != 100 or validation.num_classes != 20:
        raise ValueError(
            f"Expected 100/20 identities, got {train.num_classes}/{validation.num_classes}"
        )
    if set(train.identities) & set(validation.identities):
        raise ValueError("Train and validation identities overlap")
    if not 0 < args.output_dim <= 512:
        raise ValueError("output_dim must be between 1 and 512")

    baseline = MODULE.evaluate_features(
        validation.baseline,
        validation.identities,
        validation.source_paths,
    )
    candidates = []
    projectors: dict[tuple[float, bool], dict[str, torch.Tensor]] = {}
    for body_weight in args.body_weights:
        body_weight = float(body_weight)
        if not 0.0 < body_weight < 1.0:
            raise ValueError("Every body weight must be strictly between 0 and 1")
        train_joint = joint_features(train, body_weight)
        validation_joint = joint_features(validation, body_weight)
        for centered in (False, True):
            mean = (
                train_joint.mean(dim=0, keepdim=True)
                if centered
                else torch.zeros(
                    (1, train_joint.shape[1]),
                    device=device,
                    dtype=train_joint.dtype,
                )
            )
            _, singular_values, right_vectors = torch.linalg.svd(
                train_joint - mean,
                full_matrices=False,
            )
            rank = min(args.output_dim, right_vectors.shape[0])
            projection = right_vectors[:rank]
            projected = (validation_joint - mean) @ projection.T
            if rank < args.output_dim:
                projected = F.pad(projected, (0, args.output_dim - rank))
            projected = F.normalize(projected, dim=1)
            metrics = MODULE.evaluate_features(
                projected,
                validation.identities,
                validation.source_paths,
            )
            candidate = {
                "body_weight": body_weight,
                "semantic_weight": 1.0 - body_weight,
                "centered": centered,
                "projection_rank": rank,
                "output_dim": args.output_dim,
                "retained_train_energy": float(
                    singular_values[:rank].square().sum()
                    / singular_values.square().sum().clamp_min(1e-12)
                ),
                "metrics": metrics,
            }
            candidates.append(candidate)
            projectors[(body_weight, centered)] = {
                "mean": mean.detach().cpu(),
                "projection": projection.detach().cpu(),
            }
            print(
                json.dumps(
                    {
                        "body_weight": body_weight,
                        "centered": centered,
                        "rank1": metrics["gallery_rank1"],
                        "loo_rank1": metrics["leave_one_out_rank1"],
                        "auc": metrics["auc"],
                    }
                ),
                flush=True,
            )

    ranked = sorted(candidates, key=selection_key, reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    selected = ranked[0]
    projector = projectors[(selected["body_weight"], selected["centered"])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model_final.pth"
    torch.save(
        {
            "schema_version": 1,
            "architecture": {
                "name": "lowrank_semantic_body_fusion",
                "input_dim": int(projector["projection"].shape[1]),
                "output_dim": args.output_dim,
                "body_classifier_head": None,
                "formula": (
                    "normalize(pad((concat(sqrt(1-w)*semantic_embedding, "
                    "sqrt(w)*body)-mean) @ projection.T, 512))"
                ),
            },
            "body_weight": selected["body_weight"],
            "semantic_weight": selected["semantic_weight"],
            "centered": selected["centered"],
            "mean": projector["mean"],
            "projection": projector["projection"],
            "source": {
                "train_multimodal": str(args.train_multimodal.resolve()),
                "train_body": str(args.train_body.resolve()),
            },
        },
        model_path,
    )
    report = {
        "schema_version": 1,
        "purpose": "compatibility-first 512-D body fusion ablation",
        "protocol": {
            "train_identities": train.num_classes,
            "train_records": len(train),
            "validation_identities": validation.num_classes,
            "validation_records": len(validation),
            "identity_overlap": 0,
            "model_selection_data": "validation identities only",
        },
        "interface": {
            "input": "one dog image; body crop and branches are internal",
            "output": "512-D L2-normalized identity embedding",
            "retrieval": "unchanged cosine similarity/prototype gallery",
        },
        "architecture": {
            "semantic_branch": "frozen semantic nose+face 512-D embedding",
            "body_branch": "frozen headless Swin-V2-B 1024-D embedding",
            "fusion": "separate weighted subspaces followed by training-only SVD projection",
            "body_classifier_head": "removed",
            "classification_head_in_deployment": None,
        },
        "selection_rule": (
            "lexicographic descending: gallery_rank1, leave_one_out_rank1, "
            "auc, balanced_accuracy, mean_gap, worst_gap"
        ),
        "baseline_semantic_embedding": baseline,
        "candidates_ranked": ranked,
        "selected": selected,
        "selected_model": str(model_path.resolve()),
        "delta_vs_nose_face": {
            "gallery_rank1": selected["metrics"]["gallery_rank1"]
            - baseline["gallery_rank1"],
            "leave_one_out_rank1": selected["metrics"]["leave_one_out_rank1"]
            - baseline["leave_one_out_rank1"],
            "auc": selected["metrics"]["auc"] - baseline["auc"],
            "balanced_accuracy": selected["metrics"]["best_balanced_threshold"][
                "balanced_accuracy"
            ]
            - baseline["best_balanced_threshold"]["balanced_accuracy"],
            "mean_gap": selected["metrics"]["mean_gap"] - baseline["mean_gap"],
        },
    }
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(report_path.resolve()),
                "model": str(model_path.resolve()),
                "baseline": {
                    key: baseline[key]
                    for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                },
                "selected": {
                    "body_weight": selected["body_weight"],
                    "centered": selected["centered"],
                    **{
                        key: selected["metrics"][key]
                        for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
