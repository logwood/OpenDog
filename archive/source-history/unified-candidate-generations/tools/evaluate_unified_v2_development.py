#!/usr/bin/env python3
"""Evaluate a single-graph UnifiedSemanticPetReID on the locked v2 development set."""

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
from pet_id.unified_training import (  # noqa: E402
    geometry_losses,
    load_acceptance,
    retrieval_metrics,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional candidate embedding cache for full ONNX parity checks.",
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v2.json",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/unified/v2/teacher_development_semantic_v3.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
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
        source = grouped[identity][gallery_count:]
        replacement = grouped[names[(identity_index + 1) % len(names)]][gallery_count:]
        if len(source) != len(replacement):
            raise ValueError("Development conflict identities need equal query counts")
        for source_index, replacement_index in zip(source, replacement):
            donor[source_index] = replacement_index
    return donor


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "gallery_identities",
            "gallery_records",
            "query_records",
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


def parity(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    expected = F.normalize(expected.float(), dim=1)
    actual = F.normalize(actual.float(), dim=1)
    cosine = (expected * actual).sum(dim=1)
    difference = (expected - actual).abs()
    return {
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    reference_path = args.reference_cache.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, acceptance_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    acceptance = load_acceptance(
        acceptance_path,
        expected_protocol="unified_pet_reid_v2_strict_noninferiority",
    )
    manifest_path = Path(acceptance["development"]["path"]).expanduser().resolve()
    if sha256_file(manifest_path) != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest differs from v2 acceptance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("protocol_split", "")).casefold()
    if "blind" in split or "test" in split:
        raise RuntimeError("This development evaluator refuses protected blind data")

    device = torch.device(args.device)
    model, checkpoint = build_unified_semantic_from_checkpoint(
        checkpoint_path, device=device, verify_sources=True
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
    weights = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    geometry_sums: dict[str, float] = {
        "geometry_center": 0.0,
        "geometry_size": 0.0,
        "geometry_angle": 0.0,
        "geometry_containment": 0.0,
        "geometry_total": 0.0,
    }
    records = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["rgb"], return_aux=True)
            losses = geometry_losses(
                output["boxes_cxcywh"].float(),
                output["angle_radians"].float(),
                batch["boxes_cxcywh"].float(),
                batch["angle_radians"].float(),
            )
            count = int(batch["rgb"].shape[0])
            for name in geometry_sums:
                geometry_sums[name] += float(losses[name]) * count
            records += count
            embeddings.append(output["embedding"].float().cpu())
            faces.append(output["face_descriptor"].float().cpu())
            noses.append(output["adapted_nose_descriptor"].float().cpu())
            weights.append(output["nose_weight"].float().cpu())
            identities.extend(raw_batch["identity"])
            source_paths.extend(raw_batch["source_path"])
            source_sha256.extend(raw_batch["source_sha256"])
            if records == count or records % 25 == 0:
                print(f"v2 development: {records}/{len(dataset)}", flush=True)

    embedding = torch.cat(embeddings)
    face = torch.cat(faces)
    nose = torch.cat(noses)
    nose_weight = torch.cat(weights)
    donors = donor_indices(identities)
    corrupted = F.normalize(
        face * (1.0 - nose_weight[:, None])
        + nose.index_select(0, donors) * nose_weight[:, None],
        dim=1,
    )
    query_mask = torch.zeros(len(identities), dtype=torch.bool)
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    for indices in grouped.values():
        query_mask[indices[2:]] = True
    conflict = embedding.clone()
    conflict[query_mask] = corrupted[query_mask]
    clean_metrics = retrieval_metrics(embedding, identities, source_paths)
    conflict_metrics = retrieval_metrics(conflict, identities, source_paths)

    with np.load(reference_path, allow_pickle=False) as reference:
        reference_sha = reference["source_sha256"].astype(str).tolist()
        if reference_sha != source_sha256:
            raise RuntimeError("Reference cache and development record order differ")
        reference_embedding = torch.from_numpy(
            np.asarray(reference["embedding"], dtype=np.float32)
        )
        reference_face = torch.from_numpy(
            np.asarray(reference["face_embedding"], dtype=np.float32)
        )
    baseline_lock_path = Path(acceptance["baseline_lock"]["path"]).resolve()
    if sha256_file(baseline_lock_path) != acceptance["baseline_lock"]["sha256"]:
        raise RuntimeError("Baseline lock differs from v2 acceptance")
    baseline_lock = json.loads(baseline_lock_path.read_text(encoding="utf-8"))
    baseline = baseline_lock["reports"]["development"]["metrics"]
    checks = {
        "top1": clean_metrics["top1_correct"] >= baseline["top1_correct"],
        "top5": clean_metrics["top5_correct"] >= baseline["top5_correct"],
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "unified_v2_development_only_model_selection",
        "blind_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_type": checkpoint["model_type"],
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
            "protocol_name": acceptance["protocol_name"],
        },
        "baseline_lock": {
            "path": str(baseline_lock_path),
            "sha256": sha256_file(baseline_lock_path),
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reference_cache": str(reference_path),
        "reference_cache_sha256": sha256_file(reference_path),
        "records": records,
        "clean": compact(clean_metrics),
        "conflict": compact(conflict_metrics),
        "geometry": {name: value / records for name, value in geometry_sums.items()},
        "parity_to_semantic_v3": {
            "embedding": parity(reference_embedding, embedding),
            "face_descriptor": parity(reference_face, face),
        },
        "development_noninferiority": {
            "minimum_top1_correct": baseline["top1_correct"],
            "minimum_top5_correct": baseline["top5_correct"],
            "passed": all(checks.values()),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "default_backend_changed": False,
    }
    if args.feature_cache is not None:
        feature_cache_path = args.feature_cache.expanduser().resolve()
        if feature_cache_path.exists():
            raise FileExistsError(feature_cache_path)
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_cache_path,
            embedding=embedding.numpy(),
            face=face.numpy(),
            nose=nose.numpy(),
            nose_weight=nose_weight.numpy(),
            identities=np.asarray(identities),
            source_paths=np.asarray(source_paths),
            source_sha256=np.asarray(source_sha256),
        )
        report["feature_cache"] = {
            "path": str(feature_cache_path),
            "sha256": sha256_file(feature_cache_path),
        }
    else:
        report["feature_cache"] = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
