#!/usr/bin/env python3
"""Evaluate a PyTorch external-joint UnifiedPetReID on the development split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_data import UnifiedRawManifestDataset  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    build_external_joint_from_checkpoint,
    configure_strict_cuda_precision,
    sha256_file,
)
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def compact(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "queries"}


def main() -> None:
    args = parse_args()
    precision = configure_strict_cuda_precision()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, acceptance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    manifest_path = Path(acceptance["development"]["path"]).resolve()
    if sha256_file(manifest_path) != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest differs from acceptance")
    device = torch.device(args.device)
    model, payload = build_external_joint_from_checkpoint(
        checkpoint_path, device=device, verify_sources=True
    )
    if payload.get("training", {}).get("blind_data_used") is not False:
        raise RuntimeError("Candidate training provenance is not blind-safe")
    dataset = UnifiedRawManifestDataset(
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
    base_embeddings = []
    face_descriptors = []
    nose_descriptors = []
    geometry_confidences = []
    residual_weights = []
    reliabilities = []
    interaction_deltas = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    records = 0
    with torch.inference_mode():
        for raw in loader:
            output = model(raw["rgb"].to(device, non_blocking=True), return_aux=True)
            embeddings.append(output["embedding"].float().cpu())
            base_embeddings.append(output["base_embedding"].float().cpu())
            face_descriptors.append(output["face_descriptor"].float().cpu())
            nose_descriptors.append(
                output["adapted_nose_descriptor"].float().cpu()
            )
            geometry_confidences.append(
                output["geometry_confidence"][:, 0].float().cpu()
            )
            residual_weights.append(
                output["refiner_residual_weight"].float().cpu()
            )
            reliabilities.append(output["refiner_reliability"].float().cpu())
            interaction_deltas.append(
                (
                    output["refiner_interaction_scale"][:, None]
                    * output["refiner_interaction"]
                )
                .float()
                .cpu()
            )
            identities.extend(raw["identity"])
            source_paths.extend(raw["source_path"])
            source_sha256.extend(raw["source_sha256"])
            records += int(raw["rgb"].shape[0])
            if records == len(raw["rgb"]) or records % 64 == 0:
                print(f"joint development: {records}/{len(dataset)}", flush=True)
    embedding = torch.cat(embeddings)
    base = torch.cat(base_embeddings)
    face_descriptor = torch.cat(face_descriptors)
    nose_descriptor = torch.cat(nose_descriptors)
    geometry_confidence = torch.cat(geometry_confidences)
    weight = torch.cat(residual_weights)
    reliability = torch.cat(reliabilities)
    interaction_delta = torch.cat(interaction_deltas)
    metrics = retrieval_metrics(embedding, identities, source_paths)
    base_metrics = retrieval_metrics(base, identities, source_paths)
    baseline_checks = {
        "top1": metrics["top1_correct"]
        >= acceptance["development"]["minimum_top1_correct"],
        "top5": metrics["top5_correct"]
        >= acceptance["development"]["minimum_top5_correct"],
    }
    parent_checks = {
        "top1": metrics["top1_correct"] >= base_metrics["top1_correct"],
        "top5": metrics["top5_correct"] >= base_metrics["top5_correct"],
    }
    feature_cache_record = None
    if args.feature_cache is not None:
        feature_cache_path = args.feature_cache.expanduser().resolve()
        if feature_cache_path.exists():
            raise FileExistsError(feature_cache_path)
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_cache_path,
            embedding=embedding.numpy(),
            base_embedding=base.numpy(),
            face_descriptor=face_descriptor.numpy(),
            nose_descriptor=nose_descriptor.numpy(),
            geometry_confidence=geometry_confidence.numpy(),
            interaction_delta=interaction_delta.numpy(),
            source_sha256=np.asarray(source_sha256),
        )
        feature_cache_record = {
            "path": str(feature_cache_path),
            "sha256": sha256_file(feature_cache_path),
            "records": len(source_sha256),
        }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "external_joint_development",
        "blind_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_type": payload["model_type"],
        "cuda_precision": precision,
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "records": len(dataset),
            "identities": dataset.num_classes,
        },
        "candidate": compact(metrics),
        "parent_base": compact(base_metrics),
        "refiner": {
            "global_gain": float(
                model.refiner.maximum_residual_weight
                * model.refiner.direction_gain_logit.detach().tanh()
            ),
            "residual_weight_minimum": float(weight.min()),
            "residual_weight_mean": float(weight.mean()),
            "residual_weight_maximum": float(weight.max()),
            "reliability_minimum": float(reliability.min()),
            "reliability_mean": float(reliability.mean()),
            "reliability_maximum": float(reliability.max()),
            "interaction_norm_minimum": float(
                interaction_delta.norm(dim=1).min()
            ),
            "interaction_norm_mean": float(
                interaction_delta.norm(dim=1).mean()
            ),
            "interaction_norm_maximum": float(
                interaction_delta.norm(dim=1).max()
            ),
        },
        "embedding_norm_range": [
            float(embedding.norm(dim=1).min()),
            float(embedding.norm(dim=1).max()),
        ],
        "semantic_noninferiority": {
            "checks": baseline_checks,
            "passed": all(baseline_checks.values()),
        },
        "parent_noninferiority": {
            "checks": parent_checks,
            "passed": all(parent_checks.values()),
        },
        "feature_cache": feature_cache_record,
        "passed": all(baseline_checks.values()) and all(parent_checks.values()),
        "default_backend_changed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
