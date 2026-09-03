#!/usr/bin/env python3
"""Evaluate the complete one-input semantic model and conflict robustness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)
from pet_id.unified_training import retrieval_metrics, sha256_file  # noqa: E402
from pet_id.release_compatibility import historical_run_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "shared-fusion-baseline")
        / "dev_validation_manifest.json",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=historical_run_path(WORKSPACE, "semantic-prototype-validation"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--include-queries", action="store_true")
    parser.add_argument("--minimum-clean-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-clean-top5-correct", type=int, default=198)
    parser.add_argument("--minimum-conflict-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-conflict-top5-correct", type=int, default=198)
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
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


def conflict_features(
    clean: torch.Tensor,
    face: torch.Tensor,
    nose: torch.Tensor,
    weights: torch.Tensor,
    identities: list[str],
) -> torch.Tensor:
    donors = donor_indices(identities)
    corrupted = F.normalize(
        face * (1.0 - weights[:, None])
        + nose.index_select(0, donors) * weights[:, None],
        dim=1,
    )
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    query_mask = torch.zeros(len(identities), dtype=torch.bool)
    for indices in grouped.values():
        query_mask[indices[2:]] = True
    mixed = clean.clone()
    mixed[query_mask] = corrupted[query_mask]
    return mixed


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "top1_correct",
        "top1_accuracy",
        "top5_correct",
        "top5_accuracy",
        "mean_reciprocal_rank",
        "auc",
        "same_score_mean",
        "different_score_mean",
    )
    result = {key: metrics[key] for key in keys}
    if "queries" in metrics:
        result["queries"] = metrics["queries"]
    return result


def parity(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    expected = F.normalize(expected.float(), dim=1)
    actual = F.normalize(actual.float(), dim=1)
    difference = (expected - actual).abs()
    cosine = (expected * actual).sum(dim=1)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    reference_path = args.reference_cache.expanduser().resolve()
    for path in (checkpoint_path, manifest_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    model, checkpoint = build_unified_semantic_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=True,
    )
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    embeddings = []
    faces = []
    noses = []
    raw_noses = []
    confidences = []
    weights = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    processed = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["rgb"], return_aux=True)
            embeddings.append(output["embedding"].float().cpu())
            faces.append(output["face_descriptor"].float().cpu())
            noses.append(output["adapted_nose_descriptor"].float().cpu())
            raw_noses.append(output["raw_nose_descriptor"].float().cpu())
            confidences.append(output["geometry_confidence"].float().cpu())
            weights.append(output["nose_weight"].float().cpu())
            identities.extend(raw_batch["identity"])
            source_paths.extend(raw_batch["source_path"])
            source_sha256.extend(raw_batch["source_sha256"])
            processed += int(batch["rgb"].shape[0])
            if processed == args.batch_size or processed % 25 == 0:
                print(
                    f"unified semantic full graph: {processed}/{len(dataset)}",
                    flush=True,
                )

    embedding = torch.cat(embeddings)
    face = torch.cat(faces)
    nose = torch.cat(noses)
    raw_nose = torch.cat(raw_noses)
    confidence = torch.cat(confidences)
    nose_weight = torch.cat(weights)
    conflict = conflict_features(embedding, face, nose, nose_weight, identities)
    clean_metrics = retrieval_metrics(
        embedding,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=args.include_queries,
    )
    conflict_metrics = retrieval_metrics(
        conflict,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=args.include_queries,
    )

    reference = np.load(reference_path, allow_pickle=False)
    reference_sha = reference["source_sha256"].astype(str).tolist()
    if reference_sha != source_sha256:
        raise RuntimeError("Reference cache and manifest record order differ")
    reference_face = torch.from_numpy(reference["face_embedding"]).float()
    reference_nose = torch.from_numpy(reference["adapted_nose_embedding"]).float()
    reference_raw_nose = torch.from_numpy(reference["nose_embedding"]).float()
    reference_confidence = torch.from_numpy(reference["geometry_confidence"]).float()
    policy = checkpoint["model_config"]["fusion"]
    reference_weight = float(policy["maximum_nose_weight"]) * torch.sigmoid(
        (
            float(policy["face_confidence_threshold"])
            - reference_confidence[:, 0].clamp(0.0, 1.0)
        )
        / float(policy["temperature"])
    )
    reference_embedding = F.normalize(
        reference_face * (1.0 - reference_weight[:, None])
        + reference_nose * reference_weight[:, None],
        dim=1,
    )
    parity_report = {
        "face_descriptor": parity(reference_face, face),
        "raw_nose_descriptor": parity(reference_raw_nose, raw_nose),
        "adapted_nose_descriptor": parity(reference_nose, nose),
        "embedding": parity(reference_embedding, embedding),
        "geometry_confidence_max_abs_error": float(
            (reference_confidence - confidence).abs().max()
        ),
        "nose_weight_max_abs_error": float(
            (reference_weight - nose_weight).abs().max()
        ),
    }
    discretization = checkpoint["model_config"].get(
        "geometry_discretization", {"enabled": False}
    )
    prototype_gate_applicable = not bool(discretization.get("enabled", False))
    prototype_parity_passed = (
        parity_report["embedding"]["minimum_cosine"] >= 0.99999
    )
    checks = {
        "clean_top1": clean_metrics["top1_correct"] >= args.minimum_clean_top1_correct,
        "clean_top5": clean_metrics["top5_correct"] >= args.minimum_clean_top5_correct,
        "conflict_top1": conflict_metrics["top1_correct"]
        >= args.minimum_conflict_top1_correct,
        "conflict_top5": conflict_metrics["top5_correct"]
        >= args.minimum_conflict_top5_correct,
    }
    if prototype_gate_applicable:
        checks["complete_graph_formula_parity"] = prototype_parity_passed
    prototype_parity_gate = {
        "applicable": prototype_gate_applicable,
        "minimum_cosine_threshold": 0.99999,
        "passed": prototype_parity_passed if prototype_gate_applicable else None,
        "reason": (
            None
            if prototype_gate_applicable
            else "locked graph-internal geometry discretization intentionally changes crops"
        ),
    }
    cache_record = None
    if args.feature_cache is not None:
        feature_cache = args.feature_cache.expanduser().resolve()
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_cache,
            embedding=embedding.numpy(),
            face_embedding=face.numpy(),
            raw_nose_embedding=raw_nose.numpy(),
            adapted_nose_embedding=nose.numpy(),
            geometry_confidence=confidence.numpy(),
            nose_weight=nose_weight.numpy(),
            identities=np.asarray(identities),
            source_paths=np.asarray(source_paths),
            source_sha256=np.asarray(source_sha256),
        )
        cache_record = {
            "path": str(feature_cache),
            "sha256": sha256_file(feature_cache),
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_semantic_pet_reid",
        "runtime_contract": checkpoint["runtime_contract"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(dataset),
        "policy": policy,
        "clean": compact(clean_metrics),
        "conflict": compact(conflict_metrics),
        "parity_with_prototype_cache": {
            "cache": str(reference_path),
            "cache_sha256": sha256_file(reference_path),
            **parity_report,
        },
        "prototype_parity_gate": prototype_parity_gate,
        "mean_nose_weight": float(nose_weight.mean()),
        "nose_weight_range": [
            float(nose_weight.min()),
            float(nose_weight.max()),
        ],
        "thresholds": {
            "minimum_clean_top1_correct": args.minimum_clean_top1_correct,
            "minimum_clean_top5_correct": args.minimum_clean_top5_correct,
            "minimum_conflict_top1_correct": args.minimum_conflict_top1_correct,
            "minimum_conflict_top5_correct": args.minimum_conflict_top5_correct,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "feature_cache": cache_record,
        "default_backend_changed": False,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
