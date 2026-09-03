#!/usr/bin/env python3
"""Search confidence-prior fusion guarded by semantic nose/face agreement."""

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
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.config import add_retri_config
from pet_id.model_profiles import get_runtime_profile
from pet_id.multimodal import build_local_identity_model
from pet_id.onnx_export import PreCroppedPetEmbeddingModel
from pet_id.unified_training import retrieval_metrics, sha256_file
from pet_id.workspace_paths import normalize_runtime_config


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
    parser.add_argument("--guard-floors", type=float, nargs="+", default=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50))
    parser.add_argument("--guard-powers", type=float, nargs="+", default=(0.5, 1.0, 1.5, 2.0))
    parser.add_argument("--minimum-clean-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-clean-top5-correct", type=int, default=198)
    parser.add_argument("--minimum-corrupted-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-corrupted-top5-correct", type=int, default=198)
    parser.add_argument("--device", default="cuda")
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


def build_wrapper(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> PreCroppedPetEmbeddingModel:
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_path.resolve()))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    identity_model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=checkpoint_path.resolve(),
    )
    return PreCroppedPetEmbeddingModel(identity_model).to(device).eval()


def donor_indices(identities: list[str], gallery_count: int = 2) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    names = sorted(grouped)
    donor = torch.arange(len(identities))
    for identity_index, identity in enumerate(names):
        source_queries = grouped[identity][gallery_count:]
        donor_identity = names[(identity_index + 1) % len(names)]
        replacements = grouped[donor_identity][gallery_count:]
        if len(source_queries) != len(replacements):
            raise ValueError("Conflict identities need equal query counts")
        for source, replacement in zip(source_queries, replacements):
            donor[source] = replacement
    return donor


def query_mask(identities: list[str], gallery_count: int = 2) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    mask = torch.zeros(len(identities), dtype=torch.bool)
    for indices in grouped.values():
        mask[indices[gallery_count:]] = True
    return mask


@torch.inference_mode()
def semantic_guard(
    wrapper: PreCroppedPetEmbeddingModel,
    quality: torch.Tensor,
    viewpoint: torch.Tensor,
    nose: torch.Tensor,
    face: torch.Tensor,
) -> torch.Tensor:
    device = next(wrapper.parameters()).device
    quality = quality.to(device)
    viewpoint = viewpoint.to(device)
    nose = nose.to(device)
    face = face.to(device)
    pose_magnitude = viewpoint[:, :3].float().norm(dim=1)
    frontality = (
        wrapper.viewpoint_nose_floor
        + (1.0 - wrapper.viewpoint_nose_floor)
        * torch.exp(-wrapper.viewpoint_nose_penalty * pose_magnitude)
    ).to(dtype=quality.dtype)
    joint_quality = torch.cat(
        (quality[:, 0:1] * frontality[:, None], quality[:, 1:]), dim=1
    )
    joint_inputs = torch.cat((joint_quality, viewpoint), dim=1)
    available = torch.ones(
        (quality.shape[0], 2), dtype=torch.bool, device=device
    )
    weights = wrapper._apply_semantic_gate(
        wrapper.gate,
        joint_inputs,
        nose,
        face,
        available,
    )
    return (
        weights[:, 0] / float(wrapper.gate.max_nose_weight)
    ).clamp(0.0, 1.0).cpu()


def fuse(
    face: torch.Tensor,
    nose: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(
        face * (1.0 - weights[:, None]) + nose * weights[:, None], dim=1
    )


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    clean = row["clean"]
    corrupted = row["corrupted"]
    return (
        float(clean["top1_correct"]),
        float(corrupted["top1_correct"]),
        float(clean["top5_correct"]),
        float(corrupted["top5_correct"]),
        float(clean["mean_reciprocal_rank"]),
        float(corrupted["mean_reciprocal_rank"]),
        float(clean["auc"]),
        -float(row["mean_clean_nose_weight"]),
    )


def main() -> None:
    args = parse_args()
    if args.weight_steps < 2:
        raise ValueError("--weight-steps must be at least two")
    if not 0.0 <= args.minimum_weight <= args.maximum_weight < 0.5:
        raise ValueError("Weight range must be inside [0, 0.5)")
    if any(not 0.0 <= value <= 1.0 for value in args.guard_floors):
        raise ValueError("Guard floors must be in [0, 1]")
    if any(value <= 0 for value in (*args.temperatures, *args.guard_powers)):
        raise ValueError("Temperatures and guard powers must be positive")

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
        args.semantic_config.expanduser().resolve(),
        args.semantic_checkpoint.expanduser().resolve(),
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
    for threshold in args.confidence_thresholds:
        for temperature in args.temperatures:
            confidence_prior = torch.sigmoid(
                (float(threshold) - confidence[:, 0]) / float(temperature)
            )
            for floor in args.guard_floors:
                for power in args.guard_powers:
                    clean_factor = float(floor) + (1.0 - float(floor)) * clean_guard.pow(
                        float(power)
                    )
                    corrupted_factor = float(floor) + (
                        1.0 - float(floor)
                    ) * corrupted_guard.pow(float(power))
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
                                "face_confidence_threshold": float(threshold),
                                "temperature": float(temperature),
                                "guard_floor": float(floor),
                                "guard_power": float(power),
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
        "purpose": "locked_development_guarded_confidence_fusion_search",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "semantic_config": str(args.semantic_config.resolve()),
        "semantic_config_sha256": sha256_file(args.semantic_config.resolve()),
        "semantic_checkpoint": str(args.semantic_checkpoint.resolve()),
        "semantic_checkpoint_sha256": sha256_file(
            args.semantic_checkpoint.resolve()
        ),
        "formula": "confidence_prior*(floor+(1-floor)*semantic_guard**power)",
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
