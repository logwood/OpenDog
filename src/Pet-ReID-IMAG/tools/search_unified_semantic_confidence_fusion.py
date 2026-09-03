#!/usr/bin/env python3
"""Search a simple geometry-confidence prior for face-anchored fusion."""

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
    parser.add_argument("--weight-steps", type=int, default=71)
    parser.add_argument("--minimum-threshold", type=float, default=0.30)
    parser.add_argument("--maximum-threshold", type=float, default=0.80)
    parser.add_argument("--threshold-steps", type=int, default=51)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=(0.02, 0.04, 0.06, 0.08, 0.12, 0.18),
    )
    parser.add_argument("--minimum-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-top5-correct", type=int, default=198)
    return parser.parse_args()


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


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["top1_correct"]),
        float(metrics["top5_correct"]),
        float(metrics["mean_reciprocal_rank"]),
        float(metrics["auc"]),
        -float(row["mean_nose_weight"]),
    )


def fuse(
    face: torch.Tensor,
    nose: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(
        face * (1.0 - weights[:, None]) + nose * weights[:, None], dim=1
    )


def transitions(
    baseline: dict[str, Any], candidate: dict[str, Any]
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
    if min(args.weight_steps, args.threshold_steps) < 2:
        raise ValueError("Search grids require at least two steps")
    if not 0.0 < args.maximum_nose_weight < 0.5:
        raise ValueError("maximum nose weight must be in (0, 0.5)")
    if any(value <= 0 for value in args.temperatures):
        raise ValueError("All temperatures must be positive")

    cache_path = args.cache.expanduser().resolve()
    payload = np.load(cache_path, allow_pickle=False)
    required = {
        "face_embedding",
        "adapted_nose_embedding",
        "geometry_confidence",
        "identities",
        "source_paths",
    }
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ValueError(f"Feature cache is missing arrays: {missing}")
    face = F.normalize(torch.from_numpy(payload["face_embedding"]).float(), dim=1)
    nose = F.normalize(
        torch.from_numpy(payload["adapted_nose_embedding"]).float(), dim=1
    )
    confidence = torch.from_numpy(payload["geometry_confidence"]).float()
    face_confidence = confidence[:, 0].clamp(0.0, 1.0)
    identities = payload["identities"].astype(str).tolist()
    source_paths = payload["source_paths"].astype(str).tolist()

    rows = []
    for temperature in args.temperatures:
        for threshold in np.linspace(
            args.minimum_threshold,
            args.maximum_threshold,
            args.threshold_steps,
        ):
            reliability = torch.sigmoid(
                (float(threshold) - face_confidence) / float(temperature)
            )
            for maximum_weight in np.linspace(
                0.0,
                args.maximum_nose_weight,
                args.weight_steps,
            ):
                weights = reliability * float(maximum_weight)
                metrics = retrieval_metrics(
                    fuse(face, nose, weights),
                    identities,
                    source_paths,
                    gallery_images_per_identity=2,
                )
                rows.append(
                    {
                        "mode": "low_face_confidence_sigmoid",
                        "maximum_weight": float(maximum_weight),
                        "face_confidence_threshold": float(threshold),
                        "temperature": float(temperature),
                        "mean_nose_weight": float(weights.mean()),
                        "minimum_nose_weight": float(weights.min()),
                        "maximum_nose_weight": float(weights.max()),
                        "metrics": compact(metrics),
                    }
                )
    eligible = [
        row
        for row in rows
        if row["metrics"]["top1_correct"] >= args.minimum_top1_correct
        and row["metrics"]["top5_correct"] >= args.minimum_top5_correct
    ]
    best = max(rows, key=ranking_key)
    selected = max(eligible, key=ranking_key) if eligible else None
    selected_or_best = selected or best
    selected_reliability = torch.sigmoid(
        (
            selected_or_best["face_confidence_threshold"]
            - face_confidence
        )
        / selected_or_best["temperature"]
    )
    selected_weights = (
        selected_reliability * selected_or_best["maximum_weight"]
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
        "purpose": "locked_development_geometry_confidence_fusion_search",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "feature": "predicted_face_geometry_confidence",
        "formula": "w=max_weight*sigmoid((threshold-face_confidence)/temperature)",
        "thresholds": {
            "minimum_top1_correct": args.minimum_top1_correct,
            "minimum_top5_correct": args.minimum_top5_correct,
        },
        "grid": {
            "maximum_nose_weight": args.maximum_nose_weight,
            "weight_steps": args.weight_steps,
            "minimum_threshold": args.minimum_threshold,
            "maximum_threshold": args.maximum_threshold,
            "threshold_steps": args.threshold_steps,
            "temperatures": args.temperatures,
        },
        "baseline": compact(baseline_queries),
        "candidate_count": len(rows),
        "eligible_candidates": len(eligible),
        "best": best,
        "selected": selected,
        "transitions": transitions(baseline_queries, selected_queries),
        "top_candidates": sorted(rows, key=ranking_key, reverse=True)[:100],
        "eligible": sorted(eligible, key=ranking_key, reverse=True)[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline": report["baseline"],
                "candidate_count": len(rows),
                "eligible_candidates": len(eligible),
                "best": best,
                "selected": selected,
                "transitions": report["transitions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

