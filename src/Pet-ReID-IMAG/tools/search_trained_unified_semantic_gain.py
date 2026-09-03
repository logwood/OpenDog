#!/usr/bin/env python3
"""Search the bounded global gain of a trained semantic fusion checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_semantic import FaceAnchoredSemanticFusion
from pet_id.unified_training import atomic_torch_save, retrieval_metrics, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("validation_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=401)
    parser.add_argument("--maximum-gain", type=float, default=1.0)
    parser.add_argument("--minimum-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-top5-correct", type=int, default=198)
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


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["top1_correct"]),
        float(metrics["top5_correct"]),
        float(metrics["mean_reciprocal_rank"]),
        float(metrics["auc"]),
        -float(row["gain"]),
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


@torch.inference_mode()
def precompute(
    model: FaceAnchoredSemanticFusion,
    payload: Any,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    arrays = {
        "face": torch.from_numpy(payload["face_embedding"]).float(),
        "nose": torch.from_numpy(payload["nose_embedding"]).float(),
        "quality": torch.from_numpy(payload["quality_signals"]).float(),
        "viewpoint": torch.from_numpy(payload["viewpoint_signals"]).float(),
        "confidence": torch.from_numpy(payload["geometry_confidence"]).float(),
    }
    rows: dict[str, list[torch.Tensor]] = {
        "face": [],
        "nose": [],
        "proposed": [],
        "interaction": [],
    }
    model.eval()
    for index in range(arrays["face"].shape[0]):
        output = model(
            arrays["face"][index : index + 1].to(device),
            arrays["nose"][index : index + 1].to(device),
            arrays["quality"][index : index + 1].to(device),
            arrays["viewpoint"][index : index + 1].to(device),
            arrays["confidence"][index : index + 1].to(device),
            return_aux=True,
        )
        face = output["face_descriptor"]
        nose = output["adapted_nose_descriptor"]
        interaction = torch.tanh(model.cross_modal_residual(nose, face))
        interaction = interaction / math.sqrt(model.descriptor_dim)
        rows["face"].append(face.cpu())
        rows["nose"].append(nose.cpu())
        rows["proposed"].append(output["proposed_nose_weight"].cpu())
        rows["interaction"].append(interaction.cpu())
    return {key: torch.cat(values) for key, values in rows.items()}


def fused_features(
    components: dict[str, torch.Tensor],
    gain: float,
    residual_scale: float,
) -> torch.Tensor:
    face = components["face"]
    nose = components["nose"]
    effective = components["proposed"] * float(gain)
    return F.normalize(
        face
        + effective * (nose - face)
        + float(residual_scale) * effective * components["interaction"],
        dim=1,
    )


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if not 0.0 < args.maximum_gain <= 1.0:
        raise ValueError("--maximum-gain must be in (0, 1]")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    cache_path = args.validation_cache.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "face_anchored_semantic_fusion":
        raise ValueError("Unexpected fusion checkpoint type")
    model = FaceAnchoredSemanticFusion(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)

    payload = np.load(cache_path, allow_pickle=False)
    required = {
        "face_embedding",
        "nose_embedding",
        "quality_signals",
        "viewpoint_signals",
        "geometry_confidence",
        "identities",
        "source_paths",
    }
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ValueError(f"Validation cache is missing arrays: {missing}")
    identities = payload["identities"].astype(str).tolist()
    source_paths = payload["source_paths"].astype(str).tolist()
    components = precompute(model, payload, device=torch.device(args.device))

    rows = []
    for gain in np.linspace(0.0, args.maximum_gain, args.steps):
        features = fused_features(
            components,
            float(gain),
            model.semantic_residual_scale,
        )
        metrics = retrieval_metrics(
            features,
            identities,
            source_paths,
            gallery_images_per_identity=2,
        )
        effective = components["proposed"] * float(gain)
        rows.append(
            {
                "gain": float(gain),
                "mean_effective_nose_weight": float(effective.mean()),
                "maximum_effective_nose_weight": float(effective.max()),
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
    baseline_queries = retrieval_metrics(
        fused_features(components, 0.0, model.semantic_residual_scale),
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    selected_queries = retrieval_metrics(
        fused_features(
            components,
            selected_or_best["gain"],
            model.semantic_residual_scale,
        ),
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=True,
    )
    report = {
        "schema_version": 1,
        "purpose": "locked_development_trained_fusion_gain_search",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_cache": str(cache_path),
        "validation_cache_sha256": sha256_file(cache_path),
        "thresholds": {
            "minimum_top1_correct": args.minimum_top1_correct,
            "minimum_top5_correct": args.minimum_top5_correct,
        },
        "baseline": compact(baseline_queries),
        "best": best,
        "selected": selected,
        "eligible_candidates": len(eligible),
        "transitions": transitions(baseline_queries, selected_queries),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if selected is not None and args.selected_checkpoint is not None:
        selected_payload = dict(checkpoint)
        selected_state = dict(checkpoint["model"])
        gain = min(float(selected["gain"]), 1.0 - 1e-7)
        selected_state["direction_gain_logit"] = torch.tensor(
            math.atanh(gain), dtype=torch.float32
        )
        selected_payload.update(
            {
                "model": selected_state,
                "stage": "development_gain_selection",
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_sha256": sha256_file(checkpoint_path),
                "gain_search_report": str(args.output.resolve()),
                "gain_search_report_sha256": sha256_file(args.output.resolve()),
                "selected_gain": float(selected["gain"]),
                "validation": selected,
                "promotion_eligible": True,
            }
        )
        atomic_torch_save(selected_payload, args.selected_checkpoint.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline": report["baseline"],
                "best": best,
                "selected": selected,
                "eligible_candidates": len(eligible),
                "transitions": report["transitions"],
                "selected_checkpoint": str(args.selected_checkpoint.resolve())
                if selected is not None and args.selected_checkpoint is not None
                else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

