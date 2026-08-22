# encoding: utf-8
"""Animal-face localization and AnyFace-guided nose segmentation."""

from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


VIEWPOINT_DIM = 4


@dataclass(frozen=True)
class FaceDetection:
    """One AnyFace detection in original-image pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    landmarks_xy: tuple[tuple[float, float], ...]
    class_id: int = 0

    def __post_init__(self):
        if len(self.landmarks_xy) != 5:
            raise ValueError("AnyFace detections require five facial landmarks")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid face box: {self.bbox_xyxy}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid detection confidence: {self.confidence}")

    @property
    def left_eye(self) -> tuple[float, float]:
        return self.landmarks_xy[0]

    @property
    def right_eye(self) -> tuple[float, float]:
        return self.landmarks_xy[1]

    @property
    def nose_top(self) -> tuple[float, float]:
        return self.landmarks_xy[2]

    @property
    def left_mouth(self) -> tuple[float, float]:
        return self.landmarks_xy[3]

    @property
    def right_mouth(self) -> tuple[float, float]:
        return self.landmarks_xy[4]

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    def to_dict(self) -> dict:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "landmarks_xy": [list(point) for point in self.landmarks_xy],
            "class_id": self.class_id,
        }


def viewpoint_signals(detection: FaceDetection) -> np.ndarray:
    """Return continuous, scale/roll-normalized pose cues from five landmarks.

    The values are deliberately soft geometry signals rather than discrete pose
    labels.  Horizontal mirroring negates the first three entries and preserves
    the vertical entry.
    """

    points = np.asarray(detection.landmarks_xy, dtype=np.float32)
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_axis = right_eye - left_eye
    eye_distance = max(float(np.linalg.norm(eye_axis)), 1e-6)
    horizontal = eye_axis / eye_distance
    vertical = np.asarray((-horizontal[1], horizontal[0]), dtype=np.float32)
    eye_midpoint = 0.5 * (left_eye + right_eye)
    mouth_midpoint = 0.5 * (left_mouth + right_mouth)

    nose_horizontal = float(np.dot(nose - eye_midpoint, horizontal) / eye_distance)
    mouth_horizontal = float(
        np.dot(mouth_midpoint - eye_midpoint, horizontal) / eye_distance
    )
    left_distance = max(float(np.linalg.norm(nose - left_mouth)), 1e-6)
    right_distance = max(float(np.linalg.norm(nose - right_mouth)), 1e-6)
    distance_asymmetry = float(math.log(left_distance / right_distance))
    nose_vertical = float(np.dot(nose - eye_midpoint, vertical) / eye_distance)
    values = np.asarray(
        (nose_horizontal, mouth_horizontal, distance_asymmetry, nose_vertical),
        dtype=np.float32,
    )
    return np.clip(values, -2.0, 2.0)


@dataclass(frozen=True)
class NoseSegmentation:
    """A full-image nose mask and its prompt/quality metadata."""

    mask: np.ndarray
    roi_box_xyxy: tuple[int, int, int, int]
    predicted_iou: float
    selected_candidate: int

    def __post_init__(self):
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise TypeError("Nose mask must be a 2-D boolean NumPy array")

    @property
    def roi_mask_fraction(self) -> float:
        x1, y1, x2, y2 = self.roi_box_xyxy
        region = self.mask[y1:y2, x1:x2]
        return float(region.mean()) if region.size else 0.0


def _clip_box(
    box: Sequence[float], image_shape: Sequence[int]
) -> tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(int(math.floor(x1)), 0), max(width - 1, 0))
    y1 = min(max(int(math.floor(y1)), 0), max(height - 1, 0))
    x2 = min(max(int(math.ceil(x2)), x1 + 1), width)
    y2 = min(max(int(math.ceil(y2)), y1 + 1), height)
    return x1, y1, x2, y2


def nose_roi_box(
    detection: FaceDetection,
    image_shape: Sequence[int],
    *,
    width_scale: float = 0.48,
    height_scale: float = 0.36,
    vertical_offset: float = 0.04,
) -> tuple[int, int, int, int]:
    """Derive a high-resolution nose ROI from AnyFace geometry."""

    nose_x, nose_y = detection.nose_top
    crop_w = width_scale * detection.width
    crop_h = height_scale * detection.height
    center_y = nose_y + vertical_offset * detection.height
    return _clip_box(
        (
            nose_x - crop_w / 2,
            center_y - crop_h / 2,
            nose_x + crop_w / 2,
            center_y + crop_h / 2,
        ),
        image_shape,
    )


def crop_aligned_face(
    image_bgr: np.ndarray,
    detection: FaceDetection,
    *,
    padding: float = 0.12,
) -> np.ndarray:
    """Roll-align the eye line, then crop a padded AnyFace face box."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape [H, W, 3]")
    left_eye = np.asarray(detection.left_eye, dtype=np.float32)
    right_eye = np.asarray(detection.right_eye, dtype=np.float32)
    eye_delta = right_eye - left_eye
    angle = math.degrees(math.atan2(float(eye_delta[1]), float(eye_delta[0])))

    x1, y1, x2, y2 = detection.bbox_xyxy
    pad_x, pad_y = padding * detection.width, padding * detection.height
    padded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        matrix,
        (image_bgr.shape[1], image_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    corners = np.asarray(
        [
            [padded[0], padded[1], 1.0],
            [padded[2], padded[1], 1.0],
            [padded[2], padded[3], 1.0],
            [padded[0], padded[3], 1.0],
        ],
        dtype=np.float32,
    )
    rotated_corners = corners @ matrix.T
    crop_box = _clip_box(
        (
            rotated_corners[:, 0].min(),
            rotated_corners[:, 1].min(),
            rotated_corners[:, 0].max(),
            rotated_corners[:, 1].max(),
        ),
        rotated.shape,
    )
    cx1, cy1, cx2, cy2 = crop_box
    crop = rotated[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        raise RuntimeError("AnyFace produced an empty aligned face crop")
    return crop


def crop_nose_views(
    image_bgr: np.ndarray,
    segmentation: NoseSegmentation,
    *,
    feather_fraction: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw ROI, softly masked ROI, and the binary ROI mask."""

    x1, y1, x2, y2 = segmentation.roi_box_xyxy
    raw = image_bgr[y1:y2, x1:x2].copy()
    mask = segmentation.mask[y1:y2, x1:x2]
    if raw.size == 0 or mask.size == 0:
        raise RuntimeError("Nose segmentation produced an empty ROI")

    min_side = min(mask.shape)
    sigma = max(float(min_side) * feather_fraction, 0.8)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    soft = np.clip(soft, 0.0, 1.0)[..., None]
    # ImageNet mean in BGR byte space. Filling instead of blacking out avoids
    # a strong artificial edge after the FastReID normalization step.
    background = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
    masked = raw.astype(np.float32) * soft + background * (1.0 - soft)
    return raw, np.clip(masked, 0, 255).astype(np.uint8), mask


def laplacian_sharpness_quality(image_bgr: np.ndarray) -> float:
    """Map Laplacian variance onto a conservative [0, 1] quality scale."""

    if image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low, high = math.log1p(8.0), math.log1p(450.0)
    return float(np.clip((math.log1p(variance) - low) / (high - low), 0.0, 1.0))


def face_quality(detection: FaceDetection, aligned_face_bgr: np.ndarray) -> float:
    """Estimate whether the aligned face is reliable for identity encoding."""

    resolution = float(np.clip(min(detection.width, detection.height) / 160.0, 0.05, 1.0))
    left = np.asarray(detection.left_eye)
    right = np.asarray(detection.right_eye)
    nose = np.asarray(detection.nose_top)
    left_distance = float(np.linalg.norm(nose - left))
    right_distance = float(np.linalg.norm(nose - right))
    pose = min(left_distance, right_distance) / max(left_distance, right_distance, 1e-6)
    sharpness = max(laplacian_sharpness_quality(aligned_face_bgr), 0.05)
    auxiliary = max(resolution * pose * sharpness, 1e-6) ** (1.0 / 3.0)
    return float(np.clip(detection.confidence * auxiliary, 0.0, 1.0))


def nose_quality(
    segmentation: NoseSegmentation,
    raw_nose_bgr: np.ndarray,
) -> float:
    """Estimate whether the nose branch should influence pair scoring."""

    height, width = raw_nose_bgr.shape[:2]
    resolution = float(np.clip(min(height, width) / 96.0, 0.03, 1.0))
    sharpness = max(laplacian_sharpness_quality(raw_nose_bgr), 0.03)
    area = segmentation.roi_mask_fraction
    if 0.10 <= area <= 0.75:
        area_quality = 1.0
    elif area < 0.10:
        area_quality = max(area / 0.10, 0.05)
    else:
        area_quality = max((1.0 - area) / 0.25, 0.05)
    x1, y1, x2, y2 = segmentation.roi_box_xyxy
    region = segmentation.mask[y1:y2, x1:x2]
    border = np.concatenate((region[0], region[-1], region[:, 0], region[:, -1]))
    border_quality = float(np.clip(1.0 - 0.5 * border.mean(), 0.1, 1.0))
    predicted = float(np.clip(segmentation.predicted_iou, 0.0, 1.0))
    product = max(predicted * resolution * sharpness * area_quality * border_quality, 0.0)
    return float(np.clip(product ** (1.0 / 5.0), 0.0, 1.0))


class AnyFaceDetector:
    """Thin programmatic wrapper around the official AnyFace YOLOv5l6 model."""

    def __init__(
        self,
        weights,
        *,
        repository_root=None,
        device=None,
        image_size=800,
        confidence_threshold=0.20,
        iou_threshold=0.50,
    ):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"AnyFace weights not found: {self.weights}")
        if repository_root is None:
            repository_root = Path(__file__).resolve().parents[1] / "third_party" / "AnyFace"
        self.yolo_root = Path(repository_root) / "yolov5-face"
        if not self.yolo_root.is_dir():
            raise FileNotFoundError(f"AnyFace source not found: {self.yolo_root}")

        yolo_root_string = str(self.yolo_root.resolve())
        if yolo_root_string not in sys.path:
            sys.path.insert(0, yolo_root_string)
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression_face, scale_coords

        self._letterbox = letterbox
        self._nms = non_max_suppression_face
        self._scale_coords = scale_coords
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = attempt_load(str(self.weights), map_location=self.device).float().eval()
        stride = int(self.model.stride.max())
        self.image_size = int(check_img_size(int(image_size), s=stride))
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape [H, W, 3]")
        prepared, ratio, pad = self._letterbox(image_bgr, new_shape=self.image_size)
        tensor_array = prepared[:, :, ::-1].transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(tensor_array).to(self.device).float().div_(255.0).unsqueeze(0)
        with torch.inference_mode():
            prediction = self.model(tensor)[0]
            detections = self._nms(
                prediction,
                self.confidence_threshold,
                self.iou_threshold,
            )[0]
        if not len(detections):
            return []

        detections = detections.detach().clone()
        detections[:, :4] = self._scale_coords(
            tensor.shape[2:],
            detections[:, :4],
            image_bgr.shape,
            ratio_pad=(ratio, pad),
        )
        landmarks = detections[:, 5:15].reshape(-1, 5, 2)
        landmarks[:, :, 0].sub_(pad[0]).div_(ratio[0]).clamp_(0, image_bgr.shape[1])
        landmarks[:, :, 1].sub_(pad[1]).div_(ratio[1]).clamp_(0, image_bgr.shape[0])

        results = []
        for row, points in zip(detections, landmarks):
            results.append(
                FaceDetection(
                    bbox_xyxy=tuple(float(value) for value in row[:4].tolist()),
                    confidence=float(row[4]),
                    landmarks_xy=tuple(
                        (float(point[0]), float(point[1])) for point in points.tolist()
                    ),
                    class_id=int(row[15]),
                )
            )
        return sorted(results, key=lambda item: item.confidence, reverse=True)


def _component_at_point(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    x = min(max(int(point[0]), 0), mask.shape[1] - 1)
    y = min(max(int(point[1]), 0), mask.shape[0] - 1)
    label = int(labels[y, x])
    if label:
        return labels == label
    best_label, best_distance = 0, float("inf")
    for component in range(1, count):
        ys, xs = np.nonzero(labels == component)
        distance = float(np.min((xs - x) ** 2 + (ys - y) ** 2))
        if distance < best_distance:
            best_label, best_distance = component, distance
    return labels == best_label


def _refine_mask(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    if int(mask_u8.sum() // 255) >= 25:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    component = _component_at_point(mask_u8 > 0, point)
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(mask_u8)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled > 0


class SAM2NoseSegmenter:
    """Use AnyFace geometry as prompts for an official SAM 2 image predictor."""

    def __init__(
        self,
        checkpoint,
        *,
        config="configs/sam2.1/sam2.1_hiera_t.yaml",
        device=None,
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = build_sam2(
            config,
            str(checkpoint),
            device=self.device,
            apply_postprocessing=False,
        )
        self.predictor = SAM2ImagePredictor(model)

    @staticmethod
    def _candidate_score(
        mask: np.ndarray,
        predicted_iou: float,
        point: tuple[int, int],
        box: tuple[int, int, int, int],
    ) -> float:
        x1, y1, x2, y2 = box
        region = mask[y1:y2, x1:x2]
        if not region.any():
            return -1e9
        area_fraction = float(region.mean())
        contains_point = bool(mask[point[1], point[0]])
        size_penalty = 0.10 * abs(math.log(max(area_fraction, 1e-6) / 0.35))
        return float(predicted_iou) + (0.20 if contains_point else -1.0) - size_penalty

    def segment(self, image_bgr: np.ndarray, detection: FaceDetection) -> NoseSegmentation:
        height, width = image_bgr.shape[:2]
        roi_box = nose_roi_box(detection, image_bgr.shape)
        nose_point = (
            min(max(int(round(detection.nose_top[0])), 0), width - 1),
            min(max(int(round(detection.nose_top[1])), 0), height - 1),
        )
        landmarks = np.asarray(detection.landmarks_xy, dtype=np.float32)
        point_coords = landmarks[[2, 0, 1, 3, 4]]
        point_labels = np.asarray([1, 0, 0, 0, 0], dtype=np.int32)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            self.predictor.set_image(image_rgb)
            masks, predicted_ious, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.asarray(roi_box, dtype=np.float32),
                multimask_output=True,
            )
        scores = [
            self._candidate_score(mask.astype(bool), iou, nose_point, roi_box)
            for mask, iou in zip(masks, predicted_ious)
        ]
        selected = int(np.argmax(scores))
        selected_mask = masks[selected].astype(bool)
        x1, y1, x2, y2 = roi_box
        roi_limit = np.zeros_like(selected_mask)
        roi_limit[y1:y2, x1:x2] = True
        selected_mask &= roi_limit
        selected_mask = _refine_mask(selected_mask, nose_point)
        return NoseSegmentation(
            mask=selected_mask,
            roi_box_xyxy=roi_box,
            predicted_iou=float(predicted_ious[selected]),
            selected_candidate=selected,
        )
