#!/usr/bin/env python3
"""Search face-anchored fusion weights on a locked development feature cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_training import retrieval_metrics, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-nose-weight", type=float, default=0.35)
    parser.add_argument("--scalar-steps", type=int, default=141)
    parser.add_argument("--maximum-gate-scale", type=float, default=1.5)
    parser.add_argument("--gate-scale-steps", type=int, default=151)
    parser.add_argument("--minimum-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-top5-correct", type=int, default=197)
    return parser.parse_args()


def fuse(
    face: torch.Tensor,
    nose: torch.Tensor,
    nose_weight: torch.Tensor,
) -> torch.Tensor:
    if nose_weight.ndim == 1:
        nose_weight = nose_weight[:, None]
    return F.normalize(
        face * (1.0 - nose_weight) + nose * nose_weight,
        dim=1,
    )


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "top1_correct",
            "top1_accuracy",
            "top5_correct",
            "top5_accuracy",
            "mean_reciprocal_rank",
            "auc",
            "same_score_mean",
            "different_score_mean",
        )
    }


def candidate_row(
    *,
    mode: str,
    value: float,
    weights: torch.Tensor,
    face: torch.Tensor,
    nose: torch.Tensor,
    identities: list[str],
    source_paths: list[str],
) -> dict[str, Any]:
    features = fuse(face, nose, weights)
    metrics = retrieval_metrics(
        features,
        identities,
        source_paths,
        gallery_images_per_identity=2,
    )
    return {
        "mode": mode,
        "value": float(value),
        "mean_nose_weight": float(weights.mean()),
        "minimum_nose_weight": float(weights.min()),
        "maximum_nose_weight": float(weights.max()),
        "metrics": compact(metrics),
    }


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["top1_correct"]),
        float(metrics["top5_correct"]),
        float(metrics["mean_reciprocal_rank"]),
        float(metrics["auc"]),
        -float(row["mean_nose_weight"]),
    )


def query_transitions(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_path = {
        row["query_source_path"]: row for row in baseline["queries"]
    }
    candidate_by_path = {
        row["query_source_path"]: row for row in candidate["queries"]
    }
    recovered = []
    regressed = []
    rank_improved = []
    rank_worsened = []
    for path, base in baseline_by_path.items():
        current = candidate_by_path[path]
        if not base["correct"] and current["correct"]:
            recovered.append(path)
        if base["correct"] and not current["correct"]:
            regressed.append(path)
        if current["true_identity_rank"] < base["true_identity_rank"]:
            rank_improved.append(path)
        if current["true_identity_rank"] > base["true_identity_rank"]:
            rank_worsened.append(path)
    return {
        "top1_recovered": len(recovered),
        "top1_regressed": len(regressed),
        "rank_improved": len(rank_improved),
        "rank_worsened": len(rank_worsened),
        "recovered_source_paths": recovered,
        "regressed_source_paths": regressed,
    }


def main() -> None:
    args = parse_args()
    if args.scalar_steps < 2 or args.gate_scale_steps < 2:
        raise ValueError("Search grids require at least two steps")
    if args.maximum_nose_weight <= 0 or args.maximum_gate_scale <= 0:
        raise ValueError("Search maxima must be positive")

    cache_path = args.cache.expanduser().resolve()
    payload = np.load(cache_path, allow_pickle=False)
    required = {
        "face_embedding",
        "adapted_nose_embedding",
        "fusion_weights",
        "identities",
        "source_paths",
    }
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ValueError(f"Feature cache is missing arrays: {missing}")

    face = F.normalize(
        torch.from_numpy(payload["face_embedding"]).float(),
        dim=1,
    )
    nose = F.normalize(
        torch.from_numpy(payload["adapted_nose_embedding"]).float(),
        dim=1,
    )
    original_gate = torch.from_numpy(payload["fusion_weights"]).float()[:, 0]
    identities = payload["identities"].astype(str).tolist()
    source_paths = payload["source_paths"].astype(str).tolist()
    if face.shape != nose.shape or face.shape[0] != len(identities):
        raise ValueError("Feature cache arrays have inconsistent shapes")

    rows = []
    for value in np.linspace(
        0.0,
        args.maximum_nose_weight,
        args.scalar_steps,
    ):
        weights = torch.full(
            (face.shape[0],),
            float(value),
            dtype=face.dtype,
        )
        rows.append(
            candidate_row(
                mode="scalar",
                value=float(value),
                weights=weights,
                face=face,
                nose=nose,
                identities=identities,
                source_paths=source_paths,
            )
        )
    for value in np.linspace(
        0.0,
        args.maximum_gate_scale,
        args.gate_scale_steps,
    ):
        weights = (original_gate * float(value)).clamp(
            0.0,
            args.maximum_nose_weight,
        )
        rows.append(
            candidate_row(
                mode="gate_scale",
                value=float(value),
                weights=weights,
                face=face,
                nose=nose,
                identities=identities,
                source_paths=source_paths,
            )
        )

    best = max(rows, key=ranking_key)
    eligible = [
        row
        for row in rows
        if row["metrics"]["top1_correct"] >= args.minimum_top1_correct
        and row["metrics"]["top5_correct"] >= args.minimum_top5_correct
    ]
    best_eligible = max(eligible, key=ranking_key) if eligible else None
    selected = best_eligible or best
    if selected["mode"] == "scalar":
        selected_weights = torch.full(
            (face.shape[0],),
            selected["value"],
            dtype=face.dtype,
        )
    else:
        selected_weights = (original_gate * selected["value"]).clamp(
            0.0,
            args.maximum_nose_weight,
        )

    baseline_queries = retrieval_metrics(
        face,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    selected_queries = retrieval_metrics(
        fuse(face, nose, selected_weights),
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    report = {
        "schema_version": 1,
        "purpose": "development_only_safe_fusion_search",
        "fusion": "face_nose_convex_combination_no_interaction",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "thresholds": {
            "minimum_top1_correct": args.minimum_top1_correct,
            "minimum_top5_correct": args.minimum_top5_correct,
        },
        "baseline": compact(baseline_queries),
        "best": best,
        "best_eligible": best_eligible,
        "eligible_candidates": len(eligible),
        "selected": selected,
        "transitions": query_transitions(
            baseline_queries,
            selected_queries,
        ),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline": report["baseline"],
                "best": best,
                "best_eligible": best_eligible,
                "eligible_candidates": len(eligible),
                "transitions": report["transitions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
