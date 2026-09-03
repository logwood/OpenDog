"""Frozen dog-body localization used by the BIFOR deployment path.

The identity ONNX graph starts from fixed-size crops, just like the existing
AnyFace/SAM2 deployment boundary.  This module reproduces the detector and
target-selection policy used by the locked 100/20 BIFOR experiment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


def box_area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(
        float(box[3]) - float(box[1]), 0.0
    )


def intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    return max(
        min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])),
        0.0,
    ) * max(
        min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])),
        0.0,
    )


def select_target_dog_box(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    face_box: Sequence[float],
    *,
    dog_label: int,
    score_threshold: float,
) -> tuple[list[float] | None, float]:
    """Choose the detected dog most strongly associated with one dog face."""

    face_area = max(box_area(face_box), 1e-6)
    face_center = (
        0.5 * (float(face_box[0]) + float(face_box[2])),
        0.5 * (float(face_box[1]) + float(face_box[3])),
    )
    candidates: list[tuple[tuple[float, float, float], list[float], float]] = []
    for box_tensor, label, score_tensor in zip(boxes, labels, scores):
        score = float(score_tensor)
        if int(label) != dog_label or score < score_threshold:
            continue
        box = [float(value) for value in box_tensor.tolist()]
        contains_center = float(
            box[0] <= face_center[0] <= box[2] and box[1] <= face_center[1] <= box[3]
        )
        face_coverage = intersection_area(box, face_box) / face_area
        candidates.append(((contains_center, face_coverage, score), box, score))
    if not candidates:
        return None, 0.0
    _, selected_box, selected_score = max(candidates, key=lambda row: row[0])
    return selected_box, selected_score


def expand_and_clip_box(
    box: Sequence[float],
    width: int,
    height: int,
    expansion: float,
) -> list[int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    pad_x = (x2 - x1) * expansion
    pad_y = (y2 - y1) * expansion
    return [
        max(int(math.floor(x1 - pad_x)), 0),
        max(int(math.floor(y1 - pad_y)), 0),
        min(int(math.ceil(x2 + pad_x)), width),
        min(int(math.ceil(y2 + pad_y)), height),
    ]


class FrozenDogBodyDetector:
    """Torchvision COCO dog detector with explicit local checkpoint loading."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device,
        score_threshold: float = 0.5,
        crop_expansion: float = 0.04,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if crop_expansion < 0.0:
            raise ValueError("crop_expansion must be non-negative")
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Dog body detector checkpoint not found: {self.checkpoint_path}"
            )
        self.device = torch.device(device)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transform = weights.transforms()
        self.dog_label = weights.meta["categories"].index("dog")
        self.score_threshold = float(score_threshold)
        self.crop_expansion = float(crop_expansion)
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
        )
        state_dict = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def locate(
        self,
        images_0_255: torch.Tensor,
        face_rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return body ROIs, availability flags, and detector scores per face ROI."""

        if images_0_255.ndim != 4 or images_0_255.shape[1] != 3:
            raise ValueError("images_0_255 must have shape [N, 3, H, W]")
        if face_rois.ndim != 2 or face_rois.shape[1] != 5:
            raise ValueError("face_rois must have shape [R, 5]")
        detector_inputs = [
            self.transform(image.detach().clamp(0, 255).to(torch.uint8).cpu()).to(
                self.device
            )
            for image in images_0_255
        ]
        with torch.inference_mode():
            predictions = self.model(detector_inputs)

        rois: list[list[float]] = []
        detected_rows: list[bool] = []
        score_rows: list[float] = []
        for roi in face_rois.detach().cpu():
            image_index = int(roi[0])
            if image_index < 0 or image_index >= len(predictions):
                raise ValueError(f"Invalid face ROI image index: {image_index}")
            height, width = images_0_255.shape[-2:]
            prediction = predictions[image_index]
            face_box = [float(value) for value in roi[1:].tolist()]
            selected, score = select_target_dog_box(
                prediction["boxes"].detach().cpu(),
                prediction["labels"].detach().cpu(),
                prediction["scores"].detach().cpu(),
                face_box,
                dog_label=self.dog_label,
                score_threshold=self.score_threshold,
            )
            detected = selected is not None
            if selected is None:
                selected = [0.0, 0.0, float(width), float(height)]
            crop_box = expand_and_clip_box(
                selected,
                width,
                height,
                self.crop_expansion if detected else 0.0,
            )
            rois.append([float(image_index), *(float(value) for value in crop_box)])
            detected_rows.append(detected)
            score_rows.append(score)

        output_device = face_rois.device
        return (
            torch.tensor(rois, dtype=torch.float32, device=output_device),
            torch.tensor(detected_rows, dtype=torch.bool, device=output_device),
            torch.tensor(score_rows, dtype=torch.float32, device=output_device),
        )
