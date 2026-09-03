#!/usr/bin/env python3
"""Diagnose one guarded confidence-fusion policy on development conflicts."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from pet_id.model_profiles import get_runtime_profile
from pet_id.unified_training import retrieval_metrics, sha256_file
from search_unified_semantic_guarded_fusion import (
    build_wrapper,
    donor_indices,
    fuse,
    query_mask,
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
    parser.add_argument("--maximum-weight", type=float, default=0.19)
    parser.add_argument("--face-confidence-threshold", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--guard-floor", type=float, default=0.50)
    parser.add_argument("--guard-power", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def transition_rows(baseline: dict, candidate: dict) -> list[dict]:
    candidate_by_path = {
        row["query_source_path"]: row for row in candidate["queries"]
    }
    rows = []
    for base in baseline["queries"]:
        current = candidate_by_path[base["query_source_path"]]
        if base["true_identity_rank"] != current["true_identity_rank"]:
            rows.append(
                {
                    "query_index": base["query_index"],
                    "query_identity": base["query_identity"],
                    "query_source_path": base["query_source_path"],
                    "baseline_rank": base["true_identity_rank"],
                    "candidate_rank": current["true_identity_rank"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
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
    wrapper = build_wrapper(
        args.semantic_config.resolve(),
        args.semantic_checkpoint.resolve(),
        torch.device(args.device),
    )
    clean_guard = semantic_guard(wrapper, quality, viewpoint, nose, face)
    corrupted_guard = semantic_guard(
        wrapper, quality, viewpoint, corrupted_nose, face
    )
    prior = torch.sigmoid(
        (float(args.face_confidence_threshold) - confidence[:, 0])
        / float(args.temperature)
    )
    clean_factor = float(args.guard_floor) + (
        1.0 - float(args.guard_floor)
    ) * clean_guard.pow(float(args.guard_power))
    corrupted_factor = float(args.guard_floor) + (
        1.0 - float(args.guard_floor)
    ) * corrupted_guard.pow(float(args.guard_power))
    clean_weight = float(args.maximum_weight) * prior * clean_factor
    corrupted_weight = float(args.maximum_weight) * prior * corrupted_factor
    baseline_features = face
    clean_features = fuse(face, nose, clean_weight)
    corrupted_features = fuse(face, corrupted_nose, corrupted_weight)
    mixed = clean_features.clone()
    mixed[held_out] = corrupted_features[held_out]
    baseline = retrieval_metrics(
        baseline_features,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    clean = retrieval_metrics(
        clean_features,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    corrupted = retrieval_metrics(
        mixed,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    record_rows = []
    changed_indices = {
        row["query_index"] for row in transition_rows(baseline, clean)
    } | {row["query_index"] for row in transition_rows(baseline, corrupted)}
    for index in sorted(changed_indices):
        donor = int(donors[index])
        record_rows.append(
            {
                "index": index,
                "identity": identities[index],
                "source_path": source_paths[index],
                "donor_index": donor,
                "donor_identity": identities[donor],
                "face_confidence": float(confidence[index, 0]),
                "confidence_prior": float(prior[index]),
                "clean_guard": float(clean_guard[index]),
                "corrupted_guard": float(corrupted_guard[index]),
                "clean_weight": float(clean_weight[index]),
                "corrupted_weight": float(corrupted_weight[index]),
            }
        )
    report = {
        "schema_version": 1,
        "purpose": "guarded_policy_query_diagnostic",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "policy": {
            "maximum_weight": args.maximum_weight,
            "face_confidence_threshold": args.face_confidence_threshold,
            "temperature": args.temperature,
            "guard_floor": args.guard_floor,
            "guard_power": args.guard_power,
        },
        "baseline": baseline,
        "clean": clean,
        "corrupted": corrupted,
        "clean_transitions": transition_rows(baseline, clean),
        "corrupted_transitions": transition_rows(baseline, corrupted),
        "changed_records": record_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline": {
                    "top1_correct": baseline["top1_correct"],
                    "top5_correct": baseline["top5_correct"],
                },
                "clean": {
                    "top1_correct": clean["top1_correct"],
                    "top5_correct": clean["top5_correct"],
                },
                "corrupted": {
                    "top1_correct": corrupted["top1_correct"],
                    "top5_correct": corrupted["top5_correct"],
                },
                "clean_transitions": report["clean_transitions"],
                "corrupted_transitions": report["corrupted_transitions"],
                "changed_records": record_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
