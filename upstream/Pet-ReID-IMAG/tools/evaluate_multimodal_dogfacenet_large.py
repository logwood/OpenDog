#!/usr/bin/env python3
"""Scalable identity-prototype evaluation for large DogFaceNet galleries."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model


def exact_auc(positive: list[float], negative: list[float]) -> float:
    """Return the exact Mann-Whitney AUC without a quadratic allocation."""
    negative_sorted = sorted(float(value) for value in negative)
    wins = 0.0
    for value in positive:
        left = bisect_left(negative_sorted, float(value))
        right = bisect_right(negative_sorted, float(value))
        wins += left + 0.5 * (right - left)
    return float(wins / (len(positive) * len(negative_sorted)))


def best_balanced_threshold(positive: list[float], negative: list[float]) -> dict:
    """Find the exact balanced-accuracy threshold in O(n log n)."""
    positives = [float(value) for value in positive]
    negatives = [float(value) for value in negative]
    rows = sorted(
        [(value, 1) for value in positives]
        + [(value, 0) for value in negatives],
        reverse=True,
    )
    positive_count = len(positives)
    negative_count = len(negatives)
    best = (0.5, 0.0, 1.0, rows[0][0] + 1e-6)
    true_positives = false_positives = 0
    cursor = 0
    while cursor < len(rows):
        score = rows[cursor][0]
        next_cursor = cursor
        while next_cursor < len(rows) and rows[next_cursor][0] == score:
            if rows[next_cursor][1]:
                true_positives += 1
            else:
                false_positives += 1
            next_cursor += 1
        next_score = rows[next_cursor][0] if next_cursor < len(rows) else score - 2e-6
        threshold = 0.5 * (score + next_score)
        true_positive_rate = true_positives / positive_count
        true_negative_rate = (negative_count - false_positives) / negative_count
        candidate = (
            0.5 * (true_positive_rate + true_negative_rate),
            true_positive_rate,
            true_negative_rate,
            threshold,
        )
        if candidate > best:
            best = candidate
        cursor = next_cursor
    return {
        "threshold": best[3],
        "balanced_accuracy": best[0],
        "same_recall": best[1],
        "different_recall": best[2],
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def evaluate_branch(
    features: torch.Tensor,
    identities: list[str],
    source_paths: list[str],
    gallery_per_identity: int,
) -> dict:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity, []).append(index)
    incomplete = {
        identity: len(indices)
        for identity, indices in grouped.items()
        if len(indices) <= gallery_per_identity
    }
    if incomplete:
        raise ValueError(f"Identities without held-out queries: {incomplete}")

    prototype_identities = sorted(grouped)
    prototypes = []
    query_indices = []
    for identity in prototype_identities:
        indices = grouped[identity]
        gallery_indices = indices[:gallery_per_identity]
        query_indices.extend(indices[gallery_per_identity:])
        prototype = features.index_select(0, torch.tensor(gallery_indices)).mean(dim=0)
        prototypes.append(F.normalize(prototype, dim=0))
    prototype_matrix = torch.stack(prototypes)
    query_matrix = features.index_select(0, torch.tensor(query_indices))
    similarities = query_matrix @ prototype_matrix.T
    rankings = similarities.argsort(dim=1, descending=True)
    identity_to_column = {
        identity: column for column, identity in enumerate(prototype_identities)
    }

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    query_results = []
    for row, query_index in enumerate(query_indices):
        query_identity = identities[query_index]
        true_column = identity_to_column[query_identity]
        ranking = rankings[row].tolist()
        true_rank = ranking.index(true_column) + 1
        positive_scores.append(float(similarities[row, true_column]))
        negative_scores.extend(
            float(value)
            for column, value in enumerate(similarities[row].tolist())
            if column != true_column
        )
        top_columns = ranking[:5]
        query_results.append(
            {
                "query_index": query_index,
                "query_identity": query_identity,
                "query_source_path": source_paths[query_index],
                "true_identity_rank": true_rank,
                "correct": true_rank == 1,
                "top5": [
                    {
                        "identity": prototype_identities[column],
                        "score": float(similarities[row, column]),
                    }
                    for column in top_columns
                ],
            }
        )

    ranks = [row["true_identity_rank"] for row in query_results]
    query_count = len(query_results)
    return {
        "retrieval_unit": "l2_normalized_mean_identity_prototype",
        "gallery_identities": len(prototype_identities),
        "gallery_images_per_identity": gallery_per_identity,
        "gallery_records": len(prototype_identities) * gallery_per_identity,
        "query_records": query_count,
        "top1_correct": sum(rank <= 1 for rank in ranks),
        "top1_accuracy": sum(rank <= 1 for rank in ranks) / query_count,
        "top5_correct": sum(rank <= 5 for rank in ranks),
        "top5_accuracy": sum(rank <= 5 for rank in ranks) / query_count,
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "true_rank": summarize([float(rank) for rank in ranks]),
        "same_score": summarize(positive_scores),
        "different_score": summarize(negative_scores),
        "auc": exact_auc(positive_scores, negative_scores),
        "best_balanced_threshold": best_balanced_threshold(
            positive_scores, negative_scores
        ),
        "queries": query_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", nargs="?", default="")
    parser.add_argument("--config-file", default="configs/multimodal_joint100_frozen.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PreparedDogFaceNetDataset(args.manifest, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prepared_dogfacenet,
    )
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=args.checkpoint,
    )
    model.eval()

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    feature_rows: dict[str, list[torch.Tensor]] = {
        "fused": [],
        "nose": [],
        "face": [],
    }
    identities: list[str] = []
    source_paths: list[str] = []
    for batch in loader:
        inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"targets", "identities", "source_paths"}
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output = model(**inputs)
        feature_rows["fused"].append(output["features"].float().cpu())
        feature_rows["nose"].append(output["nose_features"].float().cpu())
        feature_rows["face"].append(output["face_features"].float().cpu())
        identities.extend(identity.casefold() for identity in batch["identities"])
        source_paths.extend(batch["source_paths"])

    evaluations = {
        name: evaluate_branch(
            torch.cat(rows),
            identities,
            source_paths,
            args.gallery_images_per_identity,
        )
        for name, rows in feature_rows.items()
    }
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "model_source": "trained_joint_checkpoint" if args.checkpoint else "frozen_pretrained",
        "records": len(dataset),
        "identities": len(set(identities)),
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "branches": evaluations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "evaluation.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {
        "output": str(output_path.resolve()),
        "model_source": summary["model_source"],
        "identities": summary["identities"],
        "records": summary["records"],
        "branches": {
            name: {
                key: value
                for key, value in metrics.items()
                if key in {"top1_correct", "top1_accuracy", "top5_correct", "top5_accuracy", "auc", "mean_reciprocal_rank"}
            }
            for name, metrics in evaluations.items()
        },
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
