#!/usr/bin/env python3
"""Search deployment-safe face-crop calibration on locked development data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset
from pet_id.unified_training import (
    build_model_from_checkpoint,
    load_acceptance,
    retrieval_metrics,
    sha256_file,
)
from pet_id.release_compatibility import (  # noqa: E402
    acceptance_path,
    historical_run_path,
)


@dataclass(frozen=True)
class FaceCalibration:
    """Small deterministic adjustment applied after predicted geometry."""

    name: str
    center_x_in_widths: float = 0.0
    center_y_in_heights: float = 0.0
    width_scale: float = 1.0
    height_scale: float = 1.0
    angle_scale: float = 1.0
    angle_offset_radians: float = 0.0

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "FaceCalibration":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(f"Unknown calibration fields: {unknown}")
        return cls(**row)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Calibration names must be non-empty")
        if self.width_scale <= 0 or self.height_scale <= 0:
            raise ValueError("Calibration width/height scales must be positive")


def default_calibrations() -> list[FaceCalibration]:
    rows = [FaceCalibration("baseline")]
    rows.extend(
        FaceCalibration(f"isotropic_{scale:.2f}", width_scale=scale, height_scale=scale)
        for scale in (0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20)
    )
    for scale in (0.85, 0.90, 0.95, 1.05, 1.10, 1.15):
        rows.append(FaceCalibration(f"width_{scale:.2f}", width_scale=scale))
        rows.append(FaceCalibration(f"height_{scale:.2f}", height_scale=scale))
    for offset in (-0.15, -0.10, -0.05, 0.05, 0.10, 0.15):
        rows.append(
            FaceCalibration(
                f"center_x_{offset:+.2f}", center_x_in_widths=offset
            )
        )
        rows.append(
            FaceCalibration(
                f"center_y_{offset:+.2f}", center_y_in_heights=offset
            )
        )
    rows.extend(
        FaceCalibration(f"angle_scale_{scale:.2f}", angle_scale=scale)
        for scale in (0.0, 0.50, 0.75, 1.25, 1.50)
    )
    return rows


def refined_calibrations() -> list[FaceCalibration]:
    rows = []
    width_scales = (1.08, 1.12, 1.16, 1.20, 1.24, 1.28, 1.32)
    height_scales = (1.08, 1.12, 1.16, 1.20, 1.24, 1.28, 1.32)
    for width_scale in width_scales:
        for height_scale in height_scales:
            rows.append(
                FaceCalibration(
                    f"width_{width_scale:.2f}_height_{height_scale:.2f}",
                    width_scale=width_scale,
                    height_scale=height_scale,
                )
            )
    for scale in (1.10, 1.15, 1.20, 1.25, 1.30):
        for offset in (-0.15, -0.10, -0.05, 0.05):
            rows.append(
                FaceCalibration(
                    f"isotropic_{scale:.2f}_center_y_{offset:+.2f}",
                    center_y_in_heights=offset,
                    width_scale=scale,
                    height_scale=scale,
                )
            )
    for scale in (1.10, 1.15, 1.20, 1.25):
        for angle_scale in (0.75, 1.00, 1.25):
            rows.append(
                FaceCalibration(
                    f"isotropic_{scale:.2f}_angle_{angle_scale:.2f}",
                    width_scale=scale,
                    height_scale=scale,
                    angle_scale=angle_scale,
                )
            )
    return rows


def deployment_calibrations() -> list[FaceCalibration]:
    rows = []
    for width_scale in (1.12, 1.16, 1.20, 1.24, 1.28):
        for height_scale in (1.00, 1.04, 1.08, 1.12, 1.16):
            rows.append(
                FaceCalibration(
                    f"width_{width_scale:.2f}_height_{height_scale:.2f}",
                    width_scale=width_scale,
                    height_scale=height_scale,
                )
            )
    return rows


def deployment_refined_calibrations() -> list[FaceCalibration]:
    rows = []
    for width_scale in (1.08, 1.10, 1.12, 1.14, 1.16):
        for height_scale in (1.04, 1.06, 1.08, 1.10, 1.12):
            rows.append(
                FaceCalibration(
                    f"width_{width_scale:.2f}_height_{height_scale:.2f}",
                    width_scale=width_scale,
                    height_scale=height_scale,
                )
            )
    for offset in (-0.15, -0.10, -0.05, 0.05, 0.10):
        rows.append(
            FaceCalibration(
                f"best_center_y_{offset:+.2f}",
                center_y_in_heights=offset,
                width_scale=1.12,
                height_scale=1.08,
            )
        )
    for offset in (-0.10, -0.05, 0.05, 0.10):
        rows.append(
            FaceCalibration(
                f"best_center_x_{offset:+.2f}",
                center_x_in_widths=offset,
                width_scale=1.12,
                height_scale=1.08,
            )
        )
    for angle_scale in (0.75, 0.90, 1.10, 1.25):
        rows.append(
            FaceCalibration(
                f"best_angle_{angle_scale:.2f}",
                width_scale=1.12,
                height_scale=1.08,
                angle_scale=angle_scale,
            )
        )
    return rows


def load_calibrations(
    path: Path | None, *, preset: str = "coarse"
) -> list[FaceCalibration]:
    if path is not None:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("--grid-json must contain a JSON list")
        rows = [FaceCalibration.from_mapping(dict(row)) for row in payload]
    elif preset == "refine":
        rows = refined_calibrations()
    elif preset == "deployment":
        rows = deployment_calibrations()
    elif preset == "deployment-refine":
        rows = deployment_refined_calibrations()
    else:
        rows = default_calibrations()
    if not rows:
        raise ValueError("At least one calibration is required")
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Calibration names must be unique")
    for row in rows:
        row.validate()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "shared-fusion-baseline")
        / "dev_validation_manifest.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=acceptance_path(WORKSPACE, "legacy-training"),
    )
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--grid-json", type=Path)
    grid.add_argument(
        "--preset",
        choices=(
            "coarse",
            "refine",
            "deployment",
            "deployment-refine",
        ),
        default="coarse",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--exact-branch-batching", action="store_true")
    parser.add_argument(
        "--descriptor-source",
        choices=("face", "fused"),
        default="face",
        help="Rank calibrations by the face branch or final fused embedding.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def adjust_face_geometry(
    boxes: torch.Tensor,
    angles: torch.Tensor,
    calibration: FaceCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjusted = boxes.clone()
    adjusted[:, 0] = adjusted[:, 0] + (
        calibration.center_x_in_widths * adjusted[:, 2]
    )
    adjusted[:, 1] = adjusted[:, 1] + (
        calibration.center_y_in_heights * adjusted[:, 3]
    )
    adjusted[:, 2] = adjusted[:, 2] * calibration.width_scale
    adjusted[:, 3] = adjusted[:, 3] * calibration.height_scale
    adjusted[:, :2] = adjusted[:, :2].clamp(0.0, 1.0)
    adjusted[:, 2:] = adjusted[:, 2:].clamp(1e-4, 1.0)
    adjusted_angles = (
        angles * calibration.angle_scale + calibration.angle_offset_radians
    )
    return adjusted, adjusted_angles


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.encoder_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.descriptor_source == "fused" and not args.exact_branch_batching:
        raise ValueError(
            "--descriptor-source fused requires --exact-branch-batching"
        )
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    arcface_path = args.arcface_checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    acceptance = load_acceptance(acceptance_path)
    manifest_hash = sha256_file(manifest_path)
    expected_manifest_hash = acceptance["development"]["validation_manifest"][
        "sha256"
    ]
    if manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            "Geometry calibration search only accepts the locked development "
            "validation manifest"
        )
    expected_arcface_hash = acceptance["source_weight_locks"][
        "dog_arcface_checkpoint"
    ]["sha256"]
    if sha256_file(arcface_path) != expected_arcface_hash:
        raise RuntimeError("ArcFace checkpoint differs from the acceptance lock")

    calibrations = load_calibrations(args.grid_json, preset=args.preset)
    device = torch.device(args.device)
    model, checkpoint = build_model_from_checkpoint(
        checkpoint_path, arcface_path, device=device
    )
    model.eval()
    allow_upscale = bool(
        checkpoint.get("preprocessing", {}).get("letterbox_allow_upscale", True)
    )
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=allow_upscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    feature_rows: list[list[torch.Tensor]] = [
        [] for _ in calibrations
    ]
    identities: list[str] = []
    source_paths: list[str] = []
    records = 0
    with torch.inference_mode():
        for raw_batch in loader:
            rgb = raw_batch["rgb"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                geometry, geometry_features = model._localize(
                    model._validate_rgb(rgb)
                )
                reduced_geometry = model.geometry.reduction(geometry_features)
                global_query = F.adaptive_avg_pool2d(
                    reduced_geometry, output_size=1
                ).flatten(1)
                semantic_queries = torch.cat(
                    (geometry.pooled_queries, global_query.unsqueeze(1)), dim=1
                )
                if args.exact_branch_batching:
                    nose_crops = model.cropper(
                        rgb,
                        geometry.boxes_cxcywh[:, 1],
                        geometry.angle_radians,
                    )
                    descriptor_rows = []
                    for calibration in calibrations:
                        boxes, angles = adjust_face_geometry(
                            geometry.boxes_cxcywh[:, 0],
                            geometry.angle_radians,
                            calibration,
                        )
                        face_crops = model.cropper(rgb, boxes, angles)
                        branches = model._backbone_descriptor(
                            model._normalize(
                                torch.cat((face_crops, nose_crops), dim=0)
                            )
                        )
                        face_descriptor = branches[: rgb.shape[0]]
                        if args.descriptor_source == "fused":
                            nose_descriptor = branches[rgb.shape[0] :]
                            fused, _, _ = model.semantic_fusion(
                                face_descriptor,
                                nose_descriptor,
                                semantic_queries,
                                geometry.confidence,
                            )
                            descriptor_rows.append(fused)
                        else:
                            descriptor_rows.append(face_descriptor)
                    descriptor_matrix = torch.stack(descriptor_rows)
                else:
                    crops = []
                    for calibration in calibrations:
                        boxes, angles = adjust_face_geometry(
                            geometry.boxes_cxcywh[:, 0],
                            geometry.angle_radians,
                            calibration,
                        )
                        crops.append(model.cropper(rgb, boxes, angles))
                    all_crops = torch.cat(crops, dim=0)
                    descriptors = []
                    for start in range(
                        0, all_crops.shape[0], args.encoder_batch_size
                    ):
                        part = all_crops[start : start + args.encoder_batch_size]
                        descriptors.append(
                            model._backbone_descriptor(model._normalize(part))
                        )
                    descriptor_matrix = torch.cat(descriptors).reshape(
                        len(calibrations), rgb.shape[0], -1
                    )
            for index in range(len(calibrations)):
                feature_rows[index].append(descriptor_matrix[index].float().cpu())
            records += int(rgb.shape[0])
            identities.extend(identity.casefold() for identity in raw_batch["identity"])
            source_paths.extend(raw_batch["source_path"])
            print(f"geometry calibration: {records}/{len(dataset)}", flush=True)

    results = []
    for calibration, rows in zip(calibrations, feature_rows):
        metrics = retrieval_metrics(
            torch.cat(rows),
            identities,
            source_paths,
            gallery_images_per_identity=2,
        )
        results.append({"calibration": asdict(calibration), "metrics": metrics})
    results.sort(
        key=lambda row: (
            row["metrics"]["top1_correct"],
            row["metrics"]["top5_correct"],
            row["metrics"]["mean_reciprocal_rank"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_guard": "locked_development_validation_only",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "letterbox_allow_upscale": allow_upscale,
        "exact_branch_batching": bool(args.exact_branch_batching),
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "descriptor_source": args.descriptor_source,
        "records": records,
        "identities": len(set(identities)),
        "candidate_count": len(calibrations),
        "best": results[0],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "candidate_count": len(calibrations),
                "top10": [
                    {
                        "calibration": row["calibration"],
                        "top1_correct": row["metrics"]["top1_correct"],
                        "top5_correct": row["metrics"]["top5_correct"],
                        "mean_reciprocal_rank": row["metrics"][
                            "mean_reciprocal_rank"
                        ],
                    }
                    for row in results[:10]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
