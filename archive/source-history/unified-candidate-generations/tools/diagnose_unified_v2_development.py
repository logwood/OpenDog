#!/usr/bin/env python3
"""Dump development-only geometry and branch diagnostics for v2 candidates.

This tool intentionally accepts only the locked v2 development manifest.  It
does not read the protected blind split and writes embeddings/geometry only so
that model selection can be audited without altering the deployment backend.
"""

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
    retrieval_metrics,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_unified_fresh_v2_protocol_20260831"
        / "prepared/development/manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Diagnostic-only: enlarge images smaller than the fixed canvas.",
    )
    parser.add_argument(
        "--minimum-size",
        type=float,
        help="Optional diagnostic override for the shared geometry size floor.",
    )
    parser.add_argument(
        "--nose-minimum-size",
        type=float,
        help="Optional diagnostic override for only the nose size floor.",
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Diagnostic-only: reset the learned box calibration to identity.",
    )
    return parser.parse_args()


def _iou_xyxy(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_xyxy = torch.stack(
        (
            first[..., 0] - first[..., 2] * 0.5,
            first[..., 1] - first[..., 3] * 0.5,
            first[..., 0] + first[..., 2] * 0.5,
            first[..., 1] + first[..., 3] * 0.5,
        ),
        dim=-1,
    )
    second_xyxy = torch.stack(
        (
            second[..., 0] - second[..., 2] * 0.5,
            second[..., 1] - second[..., 3] * 0.5,
            second[..., 0] + second[..., 2] * 0.5,
            second[..., 1] + second[..., 3] * 0.5,
        ),
        dim=-1,
    )
    left_top = torch.maximum(first_xyxy[..., :2], second_xyxy[..., :2])
    right_bottom = torch.minimum(first_xyxy[..., 2:], second_xyxy[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    first_area = (first_xyxy[..., 2:] - first_xyxy[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second_xyxy[..., 2:] - second_xyxy[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-8)


def _query_rows(
    features: torch.Tensor,
    identities: list[str],
    gallery_count: int = 2,
) -> list[dict[str, Any]]:
    features = F.normalize(features.float(), dim=1).cpu()
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    names = sorted(grouped)
    gallery_indices = [index for name in names for index in grouped[name][:gallery_count]]
    gallery = features[gallery_indices]
    gallery_names = [identities[index].casefold() for index in gallery_indices]
    rows: list[dict[str, Any]] = []
    for name in names:
        for query_index in grouped[name][gallery_count:]:
            scores = gallery @ features[query_index]
            prototype_scores = {
                gallery_name: float(scores[[i for i, n in enumerate(gallery_names) if n == gallery_name]].mean())
                for gallery_name in names
            }
            ranked = sorted(prototype_scores.items(), key=lambda item: (-item[1], item[0]))
            rank = next(i + 1 for i, item in enumerate(ranked) if item[0] == name)
            rows.append(
                {
                    "query_index": query_index,
                    "identity": identities[query_index],
                    "rank": rank,
                    "correct": rank == 1,
                    "top_score": ranked[0][1],
                    "true_score": prototype_scores[name],
                    "runner_up": ranked[0][0],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("checkpoint or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "blind" in str(manifest.get("protocol_split", "")).casefold():
        raise RuntimeError("This diagnostic refuses protected blind data")

    device = torch.device(args.device)
    model, payload = build_unified_semantic_from_checkpoint(
        checkpoint_path, device=device, verify_sources=True
    )
    geometry = model.geometry_frontend.geometry
    if args.minimum_size is not None:
        if any(
            not 0.0 < args.minimum_size < maximum
            for maximum in geometry.maximum_sizes
        ):
            raise ValueError("minimum-size must be positive and below every part maximum")
        geometry.minimum_sizes = (float(args.minimum_size),) * len(geometry.minimum_sizes)
        geometry.minimum_size = float(args.minimum_size)
        geometry.minimum_sizes_tensor.fill_(float(args.minimum_size))
    if args.nose_minimum_size is not None:
        if not 0.0 < args.nose_minimum_size < geometry.maximum_sizes[1]:
            raise ValueError("nose-minimum-size must be positive and below its maximum")
        minimums = list(geometry.minimum_sizes)
        minimums[1] = float(args.nose_minimum_size)
        geometry.minimum_sizes = tuple(minimums)
        geometry.minimum_size = float(min(minimums))
        geometry.minimum_sizes_tensor[1] = float(args.nose_minimum_size)
    if args.disable_calibration:
        calibration = model.geometry_frontend.geometry_calibration
        with torch.no_grad():
            calibration.center_offset_logits.zero_()
            calibration.log_size_scales.zero_()
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=args.allow_upscale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows: list[str] = []
    source_sha: list[str] = []
    targets: list[torch.Tensor] = []
    raw_boxes: list[torch.Tensor] = []
    stable_boxes: list[torch.Tensor] = []
    target_angles: list[torch.Tensor] = []
    raw_angles: list[torch.Tensor] = []
    stable_angles: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    noses: list[torch.Tensor] = []
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            output = model(rgb, return_aux=True)
            rows.extend(str(value) for value in batch["source_path"])
            source_sha.extend(str(value) for value in batch["source_sha256"])
            targets.append(batch["boxes_cxcywh"].float().cpu())
            raw_boxes.append(output["raw_boxes_cxcywh"].float().cpu())
            stable_boxes.append(output["boxes_cxcywh"].float().cpu())
            target_angles.append(batch["angle_radians"].float().cpu())
            raw_angles.append(output["raw_angle_radians"].float().cpu())
            stable_angles.append(output["angle_radians"].float().cpu())
            confidences.append(output["geometry_confidence"].float().cpu())
            embeddings.append(output["embedding"].float().cpu())
            faces.append(output["face_descriptor"].float().cpu())
            noses.append(output["adapted_nose_descriptor"].float().cpu())
            processed += int(rgb.shape[0])
            if processed == args.batch_size or processed % 25 == 0:
                print(f"v2 diagnostic: {processed}/{len(dataset)}", flush=True)

    target_box = torch.cat(targets)
    raw_box = torch.cat(raw_boxes)
    stable_box = torch.cat(stable_boxes)
    target_angle = torch.cat(target_angles)
    raw_angle = torch.cat(raw_angles)
    stable_angle = torch.cat(stable_angles)
    confidence = torch.cat(confidences)
    embedding = torch.cat(embeddings)
    face = torch.cat(faces)
    nose = torch.cat(noses)
    losses = geometry_losses(raw_box, raw_angle, target_box, target_angle)
    iou = _iou_xyxy(raw_box, target_box)
    stable_iou = _iou_xyxy(stable_box, target_box)
    identities = [str(record["identity"]) for record in dataset.records]
    retrieval = retrieval_metrics(embedding, identities, rows, gallery_images_per_identity=2)
    face_retrieval = retrieval_metrics(face, identities, rows, gallery_images_per_identity=2)
    nose_retrieval = retrieval_metrics(nose, identities, rows, gallery_images_per_identity=2)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development_only_unified_v2_geometry_diagnostic",
        "blind_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(dataset),
        "identities": len(set(identities)),
        "model_config": payload.get("model_config"),
        "retrieval": {
            "fused": retrieval,
            "face": face_retrieval,
            "nose": nose_retrieval,
        },
        "geometry": {
            name: float(value)
            for name, value in losses.items()
        },
        "geometry_iou": {
            "raw_mean": float(iou.mean()),
            "raw_median": float(iou.median()),
            "raw_min": float(iou.min()),
            "stable_mean": float(stable_iou.mean()),
            "stable_median": float(stable_iou.median()),
            "stable_min": float(stable_iou.min()),
        },
        "confidence": {
            "face_mean": float(confidence[:, 0].mean()),
            "face_min": float(confidence[:, 0].min()),
            "face_max": float(confidence[:, 0].max()),
            "nose_mean": float(confidence[:, 1].mean()),
        },
        "queries": _query_rows(embedding, identities),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.arrays is not None:
        arrays_path = args.arrays.expanduser().resolve()
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            target_boxes=target_box.numpy(),
            raw_boxes=raw_box.numpy(),
            stable_boxes=stable_box.numpy(),
            target_angles=target_angle.numpy(),
            raw_angles=raw_angle.numpy(),
            stable_angles=stable_angle.numpy(),
            confidence=confidence.numpy(),
            embedding=embedding.numpy(),
            face=face.numpy(),
            nose=nose.numpy(),
            identities=np.asarray(identities),
            source_paths=np.asarray(rows),
            source_sha256=np.asarray(source_sha),
        )
        report["arrays"] = str(arrays_path)
        report["arrays_sha256"] = sha256_file(arrays_path)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "fused_top1": retrieval["top1_correct"],
        "fused_top5": retrieval["top5_correct"],
        "face_top1": face_retrieval["top1_correct"],
        "raw_iou_mean": float(iou.mean()),
        "raw_iou_median": float(iou.median()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
