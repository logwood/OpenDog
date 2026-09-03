#!/usr/bin/env python3
"""Search a smooth semantic threshold guard for confidence-prior fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pet_id.model_profiles import get_runtime_profile
from pet_id.unified_training import retrieval_metrics, sha256_file
from search_unified_semantic_guarded_fusion import (
    build_wrapper,
    compact,
    donor_indices,
    fuse,
    query_mask,
    ranking_key,
    semantic_guard,
)


def parse_args() -> argparse.Namespace:
    identity_profile = get_runtime_profile("legacy-semantic")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=identity_profile.config,
    )
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=identity_profile.identity_weights,
    )
    parser.add_argument("--minimum-weight", type=float, default=0.15)
    parser.add_argument("--maximum-weight", type=float, default=0.35)
    parser.add_argument("--weight-steps", type=int, default=21)
    parser.add_argument("--confidence-thresholds", type=float, nargs="+", default=(0.78, 0.80, 0.82))
    parser.add_argument("--temperatures", type=float, nargs="+", default=(0.02, 0.03, 0.04))
    parser.add_argument("--guard-thresholds", type=float, nargs="+", default=(0.45, 0.50, 0.55, 0.60, 0.65))
    parser.add_argument("--guard-temperatures", type=float, nargs="+", default=(0.02, 0.04, 0.06, 0.08))
    parser.add_argument("--minimum-clean-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-clean-top5-correct", type=int, default=198)
    parser.add_argument("--minimum-corrupted-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-corrupted-top5-correct", type=int, default=198)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.weight_steps < 2:
        raise ValueError("--weight-steps must be at least two")
    if not 0.0 <= args.minimum_weight <= args.maximum_weight < 0.5:
        raise ValueError("Weight range must be inside [0, 0.5)")
    if any(
        value <= 0
        for value in (*args.temperatures, *args.guard_temperatures)
    ):
        raise ValueError("All temperatures must be positive")

    cache_path = args.cache.expanduser().resolve()
    payload = np.load(cache_path, allow_pickle=False)
    face = F.normalize(torch.from_numpy(payload["face_embedding"]).float(), dim=1)
    nose = F.normalize(
        torch.from_numpy(payload["adapted_nose_embedding"]).float(), dim=1
    )
    quality = torch.from_numpy(payload["quality_signals"]).float()
    viewpoint = torch.from_numpy(payload["viewpoint_signals"]).float()
    confidence = torch.from_numpy(payload["geometry_confidence"]).float()
    identities = payload["identities"].astype(str).tolist()
    source_paths = payload["source_paths"].astype(str).tolist()
    donors = donor_indices(identities)
    held_out = query_mask(identities)
    corrupted_nose = nose.index_select(0, donors)

    device = torch.device(args.device)
    wrapper = build_wrapper(
        args.semantic_config.resolve(),
        args.semantic_checkpoint.resolve(),
        device,
    )
    clean_guard = semantic_guard(wrapper, quality, viewpoint, nose, face)
    corrupted_guard = semantic_guard(
        wrapper, quality, viewpoint, corrupted_nose, face
    )
    del wrapper

    rows = []
    weights_grid = np.linspace(
        args.minimum_weight, args.maximum_weight, args.weight_steps
    )
    for confidence_threshold in args.confidence_thresholds:
        for temperature in args.temperatures:
            confidence_prior = torch.sigmoid(
                (float(confidence_threshold) - confidence[:, 0])
                / float(temperature)
            )
            for guard_threshold in args.guard_thresholds:
                for guard_temperature in args.guard_temperatures:
                    clean_factor = torch.sigmoid(
                        (clean_guard - float(guard_threshold))
                        / float(guard_temperature)
                    )
                    corrupted_factor = torch.sigmoid(
                        (corrupted_guard - float(guard_threshold))
                        / float(guard_temperature)
                    )
                    for maximum_weight in weights_grid:
                        clean_weights = (
                            float(maximum_weight)
                            * confidence_prior
                            * clean_factor
                        )
                        corrupted_weights = (
                            float(maximum_weight)
                            * confidence_prior
                            * corrupted_factor
                        )
                        clean_features = fuse(face, nose, clean_weights)
                        corrupted_features = fuse(
                            face, corrupted_nose, corrupted_weights
                        )
                        mixed = clean_features.clone()
                        mixed[held_out] = corrupted_features[held_out]
                        clean_metrics = retrieval_metrics(
                            clean_features,
                            identities,
                            source_paths,
                            gallery_images_per_identity=2,
                        )
                        corrupted_metrics = retrieval_metrics(
                            mixed,
                            identities,
                            source_paths,
                            gallery_images_per_identity=2,
                        )
                        rows.append(
                            {
                                "maximum_weight": float(maximum_weight),
                                "face_confidence_threshold": float(
                                    confidence_threshold
                                ),
                                "temperature": float(temperature),
                                "guard_threshold": float(guard_threshold),
                                "guard_temperature": float(guard_temperature),
                                "mean_clean_nose_weight": float(
                                    clean_weights.mean()
                                ),
                                "mean_corrupted_query_nose_weight": float(
                                    corrupted_weights[held_out].mean()
                                ),
                                "clean": compact(clean_metrics),
                                "corrupted": compact(corrupted_metrics),
                            }
                        )
    eligible = [
        row
        for row in rows
        if row["clean"]["top1_correct"] >= args.minimum_clean_top1_correct
        and row["clean"]["top5_correct"] >= args.minimum_clean_top5_correct
        and row["corrupted"]["top1_correct"]
        >= args.minimum_corrupted_top1_correct
        and row["corrupted"]["top5_correct"]
        >= args.minimum_corrupted_top5_correct
    ]
    best = max(rows, key=ranking_key)
    selected = max(eligible, key=ranking_key) if eligible else None
    report = {
        "schema_version": 1,
        "purpose": "locked_development_semantic_threshold_guard_search",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "semantic_config": str(args.semantic_config.resolve()),
        "semantic_config_sha256": sha256_file(args.semantic_config.resolve()),
        "semantic_checkpoint": str(args.semantic_checkpoint.resolve()),
        "semantic_checkpoint_sha256": sha256_file(
            args.semantic_checkpoint.resolve()
        ),
        "formula": "confidence_prior*sigmoid((semantic_guard-threshold)/temperature)",
        "thresholds": {
            "minimum_clean_top1_correct": args.minimum_clean_top1_correct,
            "minimum_clean_top5_correct": args.minimum_clean_top5_correct,
            "minimum_corrupted_top1_correct": args.minimum_corrupted_top1_correct,
            "minimum_corrupted_top5_correct": args.minimum_corrupted_top5_correct,
        },
        "candidate_count": len(rows),
        "eligible_candidates": len(eligible),
        "best": best,
        "selected": selected,
        "top_candidates": sorted(rows, key=ranking_key, reverse=True)[:100],
        "eligible": sorted(eligible, key=ranking_key, reverse=True)[:200],
        "guard_statistics": {
            "clean_mean": float(clean_guard.mean()),
            "corrupted_query_mean": float(corrupted_guard[held_out].mean()),
            "decreased_query_count": int(
                (corrupted_guard[held_out] < clean_guard[held_out]).sum()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_count": len(rows),
                "eligible_candidates": len(eligible),
                "best": best,
                "selected": selected,
                "guard_statistics": report["guard_statistics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
