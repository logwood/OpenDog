#!/usr/bin/env python3
"""Generate preliminary dog nose masks from AnyFace-guided nose crops.

The AnyFace crop convention places its nose landmark at approximately
``(0.50 * width, 0.39 * height)``.  That point is used as a positive SAM 2
prompt, while the four crop corners are negative prompts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def component_at_point(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """Keep only the connected component containing the positive prompt."""

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    x, y = point
    label = labels[min(max(y, 0), mask.shape[0] - 1), min(max(x, 0), mask.shape[1] - 1)]
    if label:
        return labels == label

    # Very small images can move the decoded boundary by one pixel. Fall back
    # to the closest foreground component rather than silently returning empty.
    best_label, best_distance = 0, float("inf")
    for component in range(1, count):
        ys, xs = np.nonzero(labels == component)
        distance = np.min((xs - x) ** 2 + (ys - y) ** 2)
        if distance < best_distance:
            best_label, best_distance = component, distance
    return labels == best_label


def refine_mask(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """Remove one-pixel tendrils and fill holes inside the nose region."""

    mask_u8 = mask.astype(np.uint8) * 255
    if int(mask_u8.sum() // 255) >= 25:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    cleaned = component_at_point(mask_u8 > 0, point)
    contours, _ = cv2.findContours(
        cleaned.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(mask_u8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def candidate_score(
    mask: np.ndarray,
    predicted_iou: float,
    point: tuple[int, int],
    box: tuple[int, int, int, int] | None = None,
) -> float:
    """Prefer confident, compact masks that contain the AnyFace nose point."""

    if not mask.any():
        return -1e9
    x, y = point
    contains_point = bool(mask[y, x])
    if box is None:
        region = mask
    else:
        x1, y1, x2, y2 = box
        region = mask[y1:y2, x1:x2]
    area_fraction = float(region.mean())
    border_fraction = float(
        np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).mean()
    )
    size_penalty = 0.10 * abs(np.log(max(area_fraction, 1e-6) / 0.35))
    return float(predicted_iou) + (0.20 if contains_point else -1.0) - size_penalty - 0.20 * border_fraction


def save_outputs(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    stem: str,
    output: Path,
    preview_box: tuple[int, int, int, int] | None = None,
) -> None:
    mask_u8 = mask.astype(np.uint8) * 255
    cv2.imwrite(str(output / "masks" / f"{stem}.png"), mask_u8)

    overlay = image_bgr.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[mask] = cv2.addWeighted(overlay, 0.40, tint, 0.60, 0)[mask]
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "overlays" / f"{stem}.jpg"), overlay)
    if preview_box is not None:
        x1, y1, x2, y2 = preview_box
        cv2.imwrite(
            str(output / "preview_crops" / f"{stem}.jpg"),
            overlay[y1:y2, x1:x2],
        )

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_u8
    cv2.imwrite(str(output / "cutouts" / f"{stem}.png"), rgba)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--detections-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    args = parser.parse_args()

    if bool(args.source) == bool(args.detections_json):
        raise SystemExit("Provide exactly one of --source or --detections-json")

    jobs = []
    if args.detections_json:
        detections = json.loads(args.detections_json.read_text(encoding="utf-8"))
        for record in detections:
            if not record["detections"]:
                continue
            detection = max(record["detections"], key=lambda item: item["confidence"])
            landmarks = np.asarray(detection["landmarks_xy"], dtype=np.float32).reshape(5, 2)
            face_box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
            face_w = max(float(face_box[2] - face_box[0]), 1.0)
            face_h = max(float(face_box[3] - face_box[1]), 1.0)
            nose_x, nose_y = landmarks[2]
            nose_box = np.asarray(
                [
                    nose_x - 0.24 * face_w,
                    nose_y - 0.14 * face_h,
                    nose_x + 0.24 * face_w,
                    nose_y + 0.22 * face_h,
                ],
                dtype=np.float32,
            )
            jobs.append(
                {
                    "path": Path(record["image"]),
                    "point_coords": landmarks[[2, 0, 1, 3, 4]],
                    "point_labels": np.asarray([1, 0, 0, 0, 0], dtype=np.int32),
                    "box": nose_box,
                    "source_confidence": float(detection["confidence"]),
                }
            )
    else:
        paths = sorted(
            path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in paths:
            jobs.append({"path": path})
    if not jobs:
        raise SystemExit("No input images with detections found")

    for folder in ("masks", "overlays", "cutouts", "preview_crops"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam2(
        args.config,
        str(args.checkpoint),
        device=device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    records = []
    with torch.inference_mode(), autocast:
        for job in jobs:
            path = job["path"]
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read {path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            if "box" in job:
                box_values = job["box"].copy()
                box_values[[0, 2]] = np.clip(box_values[[0, 2]], 0, width - 1)
                box_values[[1, 3]] = np.clip(box_values[[1, 3]], 0, height - 1)
                box = tuple(int(round(value)) for value in box_values)
                nose_xy = job["point_coords"][0]
                nose_point = tuple(int(round(value)) for value in nose_xy)
                point_coords = job["point_coords"]
                point_labels = job["point_labels"]
            else:
                box = None
                nose_point = (
                    int(round(0.50 * (width - 1))),
                    int(round(0.39 * (height - 1))),
                )
                point_coords = np.asarray(
                    [
                        nose_point,
                        (0.04 * width, 0.04 * height),
                        (0.96 * width, 0.04 * height),
                        (0.04 * width, 0.96 * height),
                        (0.96 * width, 0.96 * height),
                    ],
                    dtype=np.float32,
                )
                point_labels = np.asarray([1, 0, 0, 0, 0], dtype=np.int32)

            predictor.set_image(image_rgb)
            masks, predicted_ious, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.asarray(box, dtype=np.float32) if box else None,
                multimask_output=True,
            )
            scores = [
                candidate_score(mask.astype(bool), iou, nose_point, box)
                for mask, iou in zip(masks, predicted_ious)
            ]
            selected = int(np.argmax(scores))
            selected_mask = masks[selected].astype(bool)
            if box is not None:
                x1, y1, x2, y2 = box
                box_mask = np.zeros_like(selected_mask)
                box_mask[y1:y2, x1:x2] = True
                selected_mask &= box_mask
            mask = refine_mask(selected_mask, nose_point)
            save_outputs(image_bgr, mask, path.stem, args.output, box)

            record = {
                "image": path.name,
                "width": width,
                "height": height,
                "nose_point_xy": list(nose_point),
                "nose_box_xyxy": list(box) if box else None,
                "anyface_confidence": job.get("source_confidence"),
                "candidate_predicted_ious": [float(value) for value in predicted_ious],
                "selected_candidate": selected,
                "selected_predicted_iou": float(predicted_ious[selected]),
                "mask_area_fraction": float(mask.mean()),
            }
            records.append(record)
            print(
                f"{path.name}: candidate={selected}, "
                f"pred_iou={predicted_ious[selected]:.3f}, area={mask.mean():.3f}"
            )

    metadata = {
        "model": "SAM 2.1 Hiera Tiny",
        "device": str(device),
        "prompt": (
            "AnyFace nose point + nose box + negative eye/mouth landmarks"
            if args.detections_json
            else "AnyFace nose point + four negative crop corners"
        ),
        "images": records,
    }
    (args.output / "results.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
