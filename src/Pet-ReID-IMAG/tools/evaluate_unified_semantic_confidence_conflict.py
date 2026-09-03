#!/usr/bin/env python3
"""Evaluate a geometry-confidence fusion policy under nose identity conflict."""

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
    parser.add_argument("--maximum-nose-weight", type=float, default=0.165)
    parser.add_argument("--face-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--minimum-clean-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-clean-top5-correct", type=int, default=198)
    parser.add_argument("--minimum-corrupted-top1-correct", type=int, default=193)
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


def donor_indices(identities: list[str], gallery_count: int = 2) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    names = sorted(grouped)
    donor = torch.arange(len(identities))
    for identity_index, identity in enumerate(names):
        source_queries = grouped[identity][gallery_count:]
        donor_identity = names[(identity_index + 1) % len(names)]
        donor_queries = grouped[donor_identity][gallery_count:]
        if len(source_queries) != len(donor_queries):
            raise ValueError("Conflict identities need equal query counts")
        for source, replacement in zip(source_queries, donor_queries):
            donor[source] = replacement
    return donor


def fuse(
    face: torch.Tensor,
    nose: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(
        face * (1.0 - weights[:, None]) + nose * weights[:, None], dim=1
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.maximum_nose_weight < 0.5:
        raise ValueError("maximum nose weight must be in [0, 0.5)")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    cache_path = args.cache.expanduser().resolve()
    payload = np.load(cache_path, allow_pickle=False)
    face = F.normalize(torch.from_numpy(payload["face_embedding"]).float(), dim=1)
    nose = F.normalize(
        torch.from_numpy(payload["adapted_nose_embedding"]).float(), dim=1
    )
    confidence = torch.from_numpy(payload["geometry_confidence"]).float()
    identities = payload["identities"].astype(str).tolist()
    source_paths = payload["source_paths"].astype(str).tolist()
    weights = float(args.maximum_nose_weight) * torch.sigmoid(
        (
            float(args.face_confidence_threshold)
            - confidence[:, 0].clamp(0.0, 1.0)
        )
        / float(args.temperature)
    )
    donors = donor_indices(identities)
    clean_features = fuse(face, nose, weights)
    corrupted_nose = nose.index_select(0, donors)
    corrupted_features = fuse(face, corrupted_nose, weights)
    # Gallery records retain clean fusion; only held-out queries are corrupted.
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    mixed_features = clean_features.clone()
    query_mask = torch.zeros(len(identities), dtype=torch.bool)
    for indices in grouped.values():
        query_mask[indices[2:]] = True
    mixed_features[query_mask] = corrupted_features[query_mask]

    clean = retrieval_metrics(
        clean_features,
        identities,
        source_paths,
        gallery_images_per_identity=2,
    )
    corrupted = retrieval_metrics(
        mixed_features,
        identities,
        source_paths,
        gallery_images_per_identity=2,
    )
    checks = {
        "clean_top1": clean["top1_correct"]
        >= args.minimum_clean_top1_correct,
        "clean_top5": clean["top5_correct"]
        >= args.minimum_clean_top5_correct,
        "corrupted_top1": corrupted["top1_correct"]
        >= args.minimum_corrupted_top1_correct,
    }
    report = {
        "schema_version": 1,
        "purpose": "development_cross_identity_nose_injection",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "policy": {
            "maximum_nose_weight": args.maximum_nose_weight,
            "face_confidence_threshold": args.face_confidence_threshold,
            "temperature": args.temperature,
        },
        "corruption": {
            "gallery": "clean_identity_prototypes",
            "query_face": "unchanged",
            "query_nose": "next_sorted_different_identity_same_query_offset",
            "query_geometry_confidence": "unchanged",
        },
        "mean_clean_nose_weight": float(weights.mean()),
        "clean": compact(clean),
        "corrupted": compact(corrupted),
        "thresholds": {
            "minimum_clean_top1_correct": args.minimum_clean_top1_correct,
            "minimum_clean_top5_correct": args.minimum_clean_top5_correct,
            "minimum_corrupted_top1_correct": args.minimum_corrupted_top1_correct,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

