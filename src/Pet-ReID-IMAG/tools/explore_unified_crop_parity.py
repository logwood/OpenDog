#!/usr/bin/env python3
"""Read-only crop/preprocessing parity study on the locked development set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import _read_bgr  # noqa: E402
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
        default=historical_run_path(WORKSPACE, "fresh-baseline")
        / "prepared/development/manifest.json",
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=historical_run_path(WORKSPACE, "semantic-teacher-development"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--use-predicted-geometry",
        action="store_true",
        help=(
            "Use the model's localized boxes/angles instead of manifest boxes. "
            "The default target-box mode measures crop preprocessing parity only."
        ),
    )
    return parser.parse_args()


def transform_box(box: np.ndarray, scale: float, pad: tuple[int, int]) -> np.ndarray:
    result = np.asarray(box, dtype=np.float32).copy()
    result[[0, 2]] = result[[0, 2]] * scale + pad[0]
    result[[1, 3]] = result[[1, 3]] * scale + pad[1]
    result /= 1280.0
    x1, y1, x2, y2 = result
    return np.asarray(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dtype=np.float32
    )


def make_variant(
    image: np.ndarray, *, mode: str
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    scale = min(1.0, 1280.0 / max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if mode == "center_black":
        out = np.zeros((1280, 1280, 3), dtype=np.uint8)
        px, py = (1280 - new_w) // 2, (1280 - new_h) // 2
        out[py : py + new_h, px : px + new_w] = resized
        return out, scale, (px, py)
    out = np.zeros((1280, 1280, 3), dtype=np.uint8)
    out[:new_h, :new_w] = resized
    if mode == "top_left_black":
        return out, scale, (0, 0)
    if mode == "top_left_edge":
        if new_w < 1280:
            out[:new_h, new_w:] = resized[:, -1:, :]
        if new_h < 1280:
            out[new_h:, :] = out[new_h - 1 : new_h, :]
        return out, scale, (0, 0)
    if mode == "top_left_reflect":
        reflected = np.pad(
            resized,
            ((0, max(0, 1280 - new_h)), (0, max(0, 1280 - new_w)), (0, 0)),
            mode="reflect",
        )
        return reflected[:1280, :1280], scale, (0, 0)
    raise ValueError(mode)


def move(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32)


def run_variant(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    teacher: dict[str, np.ndarray],
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    use_predicted_geometry: bool = False,
) -> dict[str, Any]:
    features, faces, teacher_features, teacher_faces = [], [], [], []
    identities, paths = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        images, boxes, angles = [], [], []
        for row in chunk:
            image_bgr = _read_bgr(Path(row["source_path"]))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            variant, scale, padding = make_variant(image_rgb, mode=mode)
            images.append(variant.transpose(2, 0, 1).copy())
            boxes.append(
                np.stack(
                    (
                        transform_box(
                            np.asarray(row["face_roi_xyxy"]), scale, padding
                        ),
                        transform_box(
                            np.asarray(row["nose_roi_xyxy"]), scale, padding
                        ),
                    )
                )
            )
            angles.append(float(row["roll_angle_radians"]))
            identities.append(str(row["identity"]))
            paths.append(str(row["source_path"]))
        rgb = torch.from_numpy(np.stack(images)).float().to(device)
        target_boxes = torch.from_numpy(np.stack(boxes)).float().to(device)
        target_angles = torch.tensor(angles, dtype=torch.float32, device=device)
        with torch.inference_mode():
            geometry, geometry_features = model.geometry_frontend._localize(rgb)
            if use_predicted_geometry:
                boxes_for_crop = geometry.boxes_cxcywh
                angles_for_crop = geometry.angle_radians
            else:
                boxes_for_crop = target_boxes
                angles_for_crop = target_angles
            stable_boxes, stable_angles = model.geometry_discretizer(
                boxes_for_crop,
                angles_for_crop,
            )
            face_crop = model.geometry_frontend.cropper(
                rgb, stable_boxes[:, 0], stable_angles
            )
            nose_crop = model.nose_cropper(rgb, stable_boxes[:, 1], stable_angles)
            face_descriptor = model.geometry_frontend._backbone_descriptor(
                model.geometry_frontend._normalize(face_crop)
            )
            feather = model.nose_feather_mask.to(dtype=nose_crop.dtype)
            background = model.nose_encoder.model.pixel_mean.to(dtype=nose_crop.dtype)
            feathered = nose_crop * feather + background * (1.0 - feather)
            raw_nose = F.normalize(
                model.nose_encoder(nose_crop) + model.nose_encoder(feathered), dim=1
            )
            adapted_nose = model.nose_adapter(raw_nose)
            fusion_output = model.fusion(
                face_descriptor,
                adapted_nose,
                geometry.confidence[:, 0],
                return_aux=True,
            )
            # ConfidenceGatedNoseFusion returns a mapping when ``return_aux``
            # is enabled; the old parity script assumed a tuple contract.
            fused = fusion_output["embedding"]
        features.append(fused.float().cpu())
        faces.append(face_descriptor.float().cpu())
        teacher_features.append(
            torch.from_numpy(teacher["embedding"][start : start + len(chunk)])
        )
        teacher_faces.append(
            torch.from_numpy(teacher["face_embedding"][start : start + len(chunk)])
        )
    feature = torch.cat(features)
    face = torch.cat(faces)
    tfeature = torch.cat(teacher_features)
    tface = torch.cat(teacher_faces)
    cosine = (F.normalize(feature, dim=1) * F.normalize(tfeature, dim=1)).sum(dim=1)
    face_cosine = (F.normalize(face, dim=1) * F.normalize(tface, dim=1)).sum(dim=1)
    metrics = retrieval_metrics(feature, identities, paths)
    return {
        "retrieval": {
            key: metrics[key]
            for key in ("top1_correct", "top5_correct", "top1_accuracy", "top5_accuracy", "mean_reciprocal_rank")
        },
        "parity": {
            "minimum_cosine": float(cosine.min()),
            "mean_cosine": float(cosine.mean()),
            "minimum_face_cosine": float(face_cosine.min()),
            "mean_face_cosine": float(face_cosine.mean()),
        },
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    teacher_path = args.teacher_cache.expanduser().resolve()
    for path in (checkpoint, manifest_path, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest["records"])
    with np.load(teacher_path, allow_pickle=False) as payload:
        source_sha = payload["source_sha256"].astype(str).tolist()
        row_sha = [str(row["source_sha256"]) for row in rows]
        if source_sha != row_sha:
            raise RuntimeError("Teacher cache order differs from manifest")
        teacher = {name: np.asarray(payload[name]) for name in ("embedding", "face_embedding")}
    device = torch.device(args.device)
    model, _ = build_unified_semantic_from_checkpoint(checkpoint, device=device, verify_sources=True)
    model.eval()
    modes = ("center_black", "top_left_black", "top_left_edge", "top_left_reflect")
    report = {
        "schema_version": 1,
        "blind_data_used": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "variants": {},
        "geometry_source": "predicted" if args.use_predicted_geometry else "manifest_targets",
    }
    for mode in modes:
        print(f"crop parity: {mode}", flush=True)
        report["variants"][mode] = run_variant(
            model,
            rows,
            teacher,
            mode=mode,
            device=device,
            batch_size=args.batch_size,
            use_predicted_geometry=args.use_predicted_geometry,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
