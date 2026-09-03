#!/usr/bin/env python3
"""Evaluate a one-input geometry frontend with the compatibility identity model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.config import add_retri_config
from pet_id.model_profiles import get_runtime_profile
from pet_id.multimodal import build_local_identity_model
from pet_id.onnx_export import PreCroppedPetEmbeddingModel
from pet_id.unified import NormalizedRotatedCropper
from pet_id.unified_data import UnifiedManifestDataset
from pet_id.unified_training import (
    build_model_from_checkpoint,
    geometry_losses,
    retrieval_metrics,
    sha256_file,
)
from pet_id.workspace_paths import normalize_runtime_config


QUALITY_MEAN = (
    0.8534528664733094,
    0.7257118686828046,
    0.9248313263697283,
    0.84109377179827,
    0.7066592261904762,
    0.8330098854452375,
)
VIEWPOINT_MEAN = (
    -0.021129616576627087,
    0.00010473374667394556,
    -0.03786277081765418,
    0.8105865410015186,
)


def parse_args() -> argparse.Namespace:
    identity_profile = get_runtime_profile("legacy-semantic")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--identity-checkpoint",
        type=Path,
        default=identity_profile.identity_weights,
    )
    parser.add_argument(
        "--identity-config",
        type=Path,
        default=identity_profile.config,
    )
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--include-queries", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--signal-mode",
        choices=("constant", "manifest", "internal"),
        default="constant",
    )
    parser.add_argument(
        "--geometry-source",
        choices=("predicted", "teacher"),
        default="predicted",
    )
    parser.add_argument("--letterbox-upscale", action="store_true")
    return parser.parse_args()


def build_semantic_wrapper(
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


def internal_signals(
    geometry,
    *,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = geometry.boxes_cxcywh
    face_resolution = (
        boxes[:, 0, 2:].amin(dim=1) * float(input_size) / 160.0
    ).clamp(0.0, 1.0)
    nose_resolution = (
        boxes[:, 1, 2:].amin(dim=1) * float(input_size) / 96.0
    ).clamp(0.0, 1.0)
    face_confidence = geometry.confidence[:, 0].clamp(0.0, 1.0)
    nose_confidence = geometry.confidence[:, 1].clamp(0.0, 1.0)
    quality = torch.stack(
        (
            nose_confidence * nose_resolution,
            face_confidence * face_resolution,
            face_confidence,
            nose_confidence,
            nose_resolution,
            face_resolution,
        ),
        dim=1,
    )
    viewpoint = torch.as_tensor(
        VIEWPOINT_MEAN,
        device=boxes.device,
        dtype=boxes.dtype,
    ).unsqueeze(0).expand(boxes.shape[0], -1)
    return quality, viewpoint


def semantic_signals(
    mode: str,
    batch: dict[str, Any],
    geometry,
    records_by_sha256: dict[str, dict[str, Any]],
    *,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = geometry.boxes_cxcywh.device
    dtype = geometry.boxes_cxcywh.dtype
    if mode == "internal":
        return internal_signals(geometry, input_size=input_size)
    if mode == "manifest":
        records = [records_by_sha256[value] for value in batch["source_sha256"]]
        quality = torch.tensor(
            [record["quality_signals"] for record in records],
            device=device,
            dtype=dtype,
        )
        viewpoint = torch.tensor(
            [record["viewpoint_signals"] for record in records],
            device=device,
            dtype=dtype,
        )
        return quality, viewpoint
    quality = torch.as_tensor(
        QUALITY_MEAN,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(geometry.boxes_cxcywh.shape[0], -1)
    viewpoint = torch.as_tensor(
        VIEWPOINT_MEAN,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(geometry.boxes_cxcywh.shape[0], -1)
    return quality, viewpoint


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    geometry_model, geometry_payload = build_model_from_checkpoint(
        args.geometry_checkpoint.resolve(),
        args.arcface_checkpoint.resolve(),
        device=device,
    )
    geometry_model.eval()
    semantic_model = build_semantic_wrapper(
        args.identity_config,
        args.identity_checkpoint,
        device,
    )
    dataset = UnifiedManifestDataset(
        args.manifest,
        input_size=geometry_model.input_size,
        training=False,
        allow_letterbox_upscale=args.letterbox_upscale,
    )
    records_by_sha256 = {
        str(record["source_sha256"]): record for record in dataset.records
    }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    nose_cropper = NormalizedRotatedCropper((244, 244)).to(device)

    embeddings = []
    face_embeddings = []
    adapted_nose_embeddings = []
    nose_embeddings = []
    fusion_weights = []
    viewpoint_rows = []
    confidence_rows = []
    target_rows = []
    quality_rows = []
    geometry_sums = {
        "geometry_center": 0.0,
        "geometry_size": 0.0,
        "geometry_angle": 0.0,
        "geometry_containment": 0.0,
        "geometry_total": 0.0,
    }
    identities: list[str] = []
    source_paths: list[str] = []
    records = 0
    with torch.inference_mode():
        for step, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            geometry, _ = geometry_model._localize(batch["rgb"])
            if args.geometry_source == "teacher":
                geometry = geometry_model._override_geometry(
                    geometry,
                    {
                        "boxes_cxcywh": batch["boxes_cxcywh"],
                        "angle_radians": batch["angle_radians"],
                    },
                )
            face_crop = geometry_model.cropper(
                batch["rgb"],
                geometry.boxes_cxcywh[:, 0],
                geometry.angle_radians,
            )
            nose_crop = nose_cropper(
                batch["rgb"],
                geometry.boxes_cxcywh[:, 1],
                geometry.angle_radians,
            )
            quality, viewpoint = semantic_signals(
                args.signal_mode,
                raw_batch,
                geometry,
                records_by_sha256,
                input_size=geometry_model.input_size,
            )
            available = torch.ones(
                (batch["rgb"].shape[0], 2),
                device=device,
                dtype=torch.bool,
            )
            nose_mask = torch.ones(
                (batch["rgb"].shape[0], 1, 244, 244),
                device=device,
                dtype=batch["rgb"].dtype,
            )
            output = semantic_model(
                nose_crop,
                face_crop,
                nose_mask,
                quality,
                viewpoint,
                available,
            )
            losses = geometry_losses(
                geometry.boxes_cxcywh.float(),
                geometry.angle_radians.float(),
                batch["boxes_cxcywh"].float(),
                batch["angle_radians"].float(),
            )
            count = int(batch["rgb"].shape[0])
            for name in geometry_sums:
                geometry_sums[name] += float(losses[name]) * count
            records += count
            embeddings.append(output[0].float().cpu())
            nose_embeddings.append(output[1].float().cpu())
            face_embeddings.append(output[2].float().cpu())
            adapted_nose_embeddings.append(
                semantic_model.nose_adapter(output[1]).float().cpu()
            )
            fusion_weights.append(output[3].float().cpu())
            quality_rows.append(quality.float().cpu())
            identities.extend(raw_batch["identity"])
            viewpoint_rows.append(viewpoint.float().cpu())
            confidence_rows.append(geometry.confidence.float().cpu())
            target_rows.append(batch["target"].cpu())
            source_paths.extend(raw_batch["source_path"])
            if step == 1 or records % 25 == 0:
                print(f"unified semantic prototype: {records}/{len(dataset)}", flush=True)

    embeddings_tensor = torch.cat(embeddings)
    nose_tensor = torch.cat(nose_embeddings)
    face_tensor = torch.cat(face_embeddings)
    weights_tensor = torch.cat(fusion_weights)
    quality_tensor = torch.cat(quality_rows)
    adapted_nose_tensor = torch.cat(adapted_nose_embeddings)
    viewpoint_tensor = torch.cat(viewpoint_rows)
    confidence_tensor = torch.cat(confidence_rows)
    targets_tensor = torch.cat(target_rows)
    cache_record = None
    if args.feature_cache is not None:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_cache,
            embedding=embeddings_tensor.numpy(),
            face_embedding=face_tensor.numpy(),
            nose_embedding=nose_tensor.numpy(),
            adapted_nose_embedding=adapted_nose_tensor.numpy(),
            fusion_weights=weights_tensor.numpy(),
            quality_signals=quality_tensor.numpy(),
            viewpoint_signals=viewpoint_tensor.numpy(),
            geometry_confidence=confidence_tensor.numpy(),
            targets=targets_tensor.numpy(),
            identities=np.asarray(identities),
            source_paths=np.asarray(source_paths),
            source_sha256=np.asarray(
                [record["source_sha256"] for record in dataset.records]
            ),
        )
        cache_record = {
            "path": str(args.feature_cache.resolve()),
            "sha256": sha256_file(args.feature_cache.resolve()),
        }
    report = {
        "schema_version": 1,
        "model_type": "unified_semantic_prototype",
        "deployment_eligible": False,
        "reason": "development-only architecture diagnostic",
        "signal_mode": args.signal_mode,
        "geometry_source": args.geometry_source,
        "nose_mask_mode": "full",
        "manifest": str(dataset.manifest_path),
        "records": records,
        "geometry_checkpoint": str(args.geometry_checkpoint.resolve()),
        "geometry_checkpoint_sha256": sha256_file(args.geometry_checkpoint.resolve()),
        "geometry_stage": geometry_payload.get("stage"),
        "identity_checkpoint": str(args.identity_checkpoint.resolve()),
        "identity_checkpoint_sha256": sha256_file(args.identity_checkpoint.resolve()),
        "identity_config": str(args.identity_config.resolve()),
        "identity_config_sha256": sha256_file(args.identity_config.resolve()),
        "feature_cache": cache_record,
        "geometry": {
            name: value / records for name, value in geometry_sums.items()
        },
        "embedding": retrieval_metrics(
            embeddings_tensor,
            identities,
            source_paths,
            gallery_images_per_identity=2,
            include_queries=args.include_queries,
        ),
        "face_descriptor": retrieval_metrics(
            face_tensor,
            identities,
            source_paths,
            gallery_images_per_identity=2,
            include_queries=args.include_queries,
        ),
        "nose_descriptor": retrieval_metrics(
            nose_tensor,
            identities,
            source_paths,
            gallery_images_per_identity=2,
            include_queries=args.include_queries,
        ),
        "mean_fusion_weights": weights_tensor.mean(dim=0).tolist(),
        "mean_quality_signals": quality_tensor.mean(dim=0).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "embedding": {
                    key: report["embedding"][key]
                    for key in (
                        "top1_correct",
                        "top1_accuracy",
                        "top5_correct",
                        "top5_accuracy",
                    )
                },
                "face": {
                    key: report["face_descriptor"][key]
                    for key in ("top1_correct", "top5_correct")
                },
                "nose": {
                    key: report["nose_descriptor"][key]
                    for key in ("top1_correct", "top5_correct")
                },
                "mean_fusion_weights": report["mean_fusion_weights"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
