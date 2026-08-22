#!/usr/bin/env python3
"""Extract and compare AnyFace-guided dog face + nose descriptors."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.localization import crop_aligned_face, crop_nose_views
from pet_id.multimodal import (
    DescriptorCache,
    build_multimodal_pipeline,
    compare_descriptors,
    pipeline_namespace,
)


def _collect_images(values) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            images.extend(
                item for item in sorted(path.iterdir()) if item.suffix.lower() in suffixes
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Input image path does not exist: {path}")
    if not images:
        raise RuntimeError("No input images found")
    return images


def _artifact_root(image_path: Path, output_root: Path) -> Path:
    return output_root / f"{image_path.stem}_{image_path.stat().st_size}"


def _artifacts_complete(image_path: Path, output_root: Path, pet_count: int) -> bool:
    root = _artifact_root(image_path, output_root)
    if not (root / "metadata.json").is_file() or not (root / "detections.jpg").is_file():
        return False
    return all(
        (root / f"pet_{index:02d}" / "descriptor.npz").is_file()
        for index in range(pet_count)
    )


def _save_artifacts(image_path: Path, descriptors, output_root: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if descriptors and descriptors[0].inference_size:
        target_size = tuple(descriptors[0].inference_size)
        if image.shape[1::-1] != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    stem_root = _artifact_root(image_path, output_root)
    stem_root.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    metadata = {"source": str(image_path.resolve()), "pets": []}
    for index, descriptor in enumerate(descriptors):
        pet_root = stem_root / f"pet_{index:02d}"
        pet_root.mkdir(parents=True, exist_ok=True)
        detection = descriptor.detection
        segmentation = descriptor.segmentation
        if detection is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for landmark_index, point in enumerate(detection.landmarks_xy):
                color = (0, 0, 255) if landmark_index == 2 else (255, 255, 0)
                cv2.circle(
                    annotated,
                    tuple(int(round(value)) for value in point),
                    3,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(pet_root / "face_aligned.jpg"), crop_aligned_face(image, detection))
        if segmentation is not None:
            raw, soft_masked, roi_mask = crop_nose_views(image, segmentation)
            cv2.imwrite(str(pet_root / "nose_raw.jpg"), raw)
            cv2.imwrite(str(pet_root / "nose_soft_masked.jpg"), soft_masked)
            cv2.imwrite(str(pet_root / "nose_mask.png"), roi_mask.astype(np.uint8) * 255)
            mask_overlay = image.copy()
            mask_overlay[segmentation.mask] = cv2.addWeighted(
                image, 0.40, np.full_like(image, (0, 255, 0)), 0.60, 0
            )[segmentation.mask]
            cv2.imwrite(str(pet_root / "mask_overlay.jpg"), mask_overlay)
        np.savez_compressed(
            pet_root / "descriptor.npz",
            fused=descriptor.fused_feature.numpy(),
            nose=descriptor.nose_feature.numpy(),
            face=descriptor.face_feature.numpy(),
        )
        item = descriptor.metadata_dict()
        item["index"] = index
        metadata["pets"].append(item)
    cv2.imwrite(str(stem_root / "detections.jpg"), annotated)
    (stem_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="image files and/or directories")
    parser.add_argument(
        "--config-file",
        default="configs/multimodal_inference.yaml",
        help="multimodal pipeline config",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="logs/multimodal_runs")
    parser.add_argument(
        "--identity-weights",
        default="",
        help="trained multimodal checkpoint; enables closed-set identity scores",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE = args.device
    if args.identity_weights:
        cfg.MULTIMODAL.IDENTITY_WEIGHTS = args.identity_weights
    cfg.freeze()
    options = cfg.MULTIMODAL
    image_paths = _collect_images(args.images)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_files = [
        options.NOSE_CONFIG,
        options.NOSE_WEIGHTS,
        options.ARCFACE_WEIGHTS,
        options.ANYFACE_WEIGHTS,
        options.SAM2_CHECKPOINT,
    ]
    if options.IDENTITY_WEIGHTS:
        model_files.append(options.IDENTITY_WEIGHTS)
    namespace = pipeline_namespace(
        model_files,
        {
            "nose_prior": options.NOSE_PRIOR,
            "face_prior": options.FACE_PRIOR,
            "nose_size": list(options.NOSE_SIZE),
            "face_size": list(options.FACE_SIZE),
            "max_long_side": options.MAX_LONG_SIDE,
        },
    )
    cache = DescriptorCache(options.CACHE_DIR, namespace)
    results = {}
    missing = []
    for path in image_paths:
        descriptors = None if args.no_cache else cache.load(path)
        if descriptors is None or not _artifacts_complete(
            path, output_root, len(descriptors)
        ):
            missing.append(path)
        else:
            results[path] = descriptors

    pipeline = build_multimodal_pipeline(cfg, device=args.device) if missing else None
    for path in missing:
        descriptors = pipeline.encode_image(path)
        results[path] = descriptors
        if not args.no_cache:
            cache.save(path, descriptors)
        _save_artifacts(path, descriptors, output_root)

    comparisons = []
    for left_path, right_path in itertools.combinations(image_paths, 2):
        matrix = []
        for left_index, left in enumerate(results[left_path]):
            for right_index, right in enumerate(results[right_path]):
                matrix.append(
                    {
                        "left_pet": left_index,
                        "right_pet": right_index,
                        **compare_descriptors(left, right).to_dict(),
                    }
                )
        comparisons.append(
            {
                "left": str(left_path.resolve()),
                "right": str(right_path.resolve()),
                "scores": matrix,
            }
        )

    summary = {
        "namespace": namespace,
        "images": [
            {
                "path": str(path.resolve()),
                "pet_count": len(results[path]),
                "descriptors": [item.metadata_dict() for item in results[path]],
            }
            for path in image_paths
        ],
        "comparisons": comparisons,
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console_summary = {
        "namespace": namespace,
        "images": len(image_paths),
        "pets": sum(len(items) for items in results.values()),
        "pair_comparisons": sum(len(item["scores"]) for item in comparisons),
        "cache_hits": len(image_paths) - len(missing),
        "fresh_inferences": len(missing),
        "summary": str(summary_path.resolve()),
    }
    print(json.dumps(console_summary, indent=2))


if __name__ == "__main__":
    main()
