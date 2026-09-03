# encoding: utf-8
"""Manifest-backed fixed-RGB data path for pet_id.unified."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dogfacenet_alignment import _read_bgr
from .workspace_paths import resolve_legacy_path


def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.asarray(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1),
        dtype=np.float32,
    )


def letterbox_rgb(
    image_rgb: np.ndarray,
    *,
    size: int = 1280,
    fill_value: int = 0,
    allow_upscale: bool = True,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Center-letterbox an RGB image and return scale plus integer padding."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape [height, width, 3]")
    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive")
    height, width = image_rgb.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    if not allow_upscale:
        scale = min(1.0, scale)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image_rgb, (resized_width, resized_height), interpolation=interpolation
    )
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    output = np.full((size, size, 3), int(fill_value), dtype=np.uint8)
    output[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized
    return output, float(scale), (pad_x, pad_y)


def transform_box_to_letterbox(
    box_xyxy: Any,
    *,
    scale: float,
    padding: tuple[int, int],
    size: int,
) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    box[[0, 2]] = box[[0, 2]] * float(scale) + int(padding[0])
    box[[1, 3]] = box[[1, 3]] * float(scale) + int(padding[1])
    box /= float(size)
    box = np.clip(box, 0.0, 1.0)
    center = _xyxy_to_cxcywh(box)
    center[2:] = np.maximum(center[2:], 1.0 / float(size))
    return center


class UnifiedTeacherCache:
    """Read an immutable per-image teacher descriptor NPZ."""

    REQUIRED = ("source_sha256", "embedding", "face_embedding")

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        payload = np.load(self.path, allow_pickle=False)
        missing = [name for name in self.REQUIRED if name not in payload]
        if missing:
            raise ValueError(f"Teacher cache is missing arrays: {missing}")
        keys = payload["source_sha256"].astype(str).tolist()
        if len(keys) != len(set(keys)):
            raise ValueError("Teacher cache has duplicate source_sha256 values")
        self._rows = {key: index for index, key in enumerate(keys)}
        self.embedding = np.asarray(payload["embedding"], dtype=np.float32)
        self.face_embedding = np.asarray(payload["face_embedding"], dtype=np.float32)
        if self.embedding.ndim != 2 or self.face_embedding.ndim != 2:
            raise ValueError("Teacher embedding arrays must both be two-dimensional")
        if self.embedding.shape[0] != len(keys):
            raise ValueError("Teacher embedding row count differs from source keys")
        if self.face_embedding.shape != (len(keys), 512):
            raise ValueError("Teacher face_embedding must have shape [records,512]")
        self.embedding_dim = int(self.embedding.shape[1])

    def get(self, source_sha256: str) -> tuple[np.ndarray, np.ndarray]:
        index = self._rows[str(source_sha256)]
        return self.embedding[index], self.face_embedding[index]

    def __contains__(self, source_sha256: str) -> bool:
        return str(source_sha256) in self._rows


class UnifiedManifestDataset(Dataset):
    """Load fixed-size RGB plus model-internal geometry supervision."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        input_size: int = 1280,
        training: bool = False,
        horizontal_flip_probability: float = 0.0,
        color_jitter: float = 0.0,
        min_images_per_identity: int = 1,
        teacher_cache: UnifiedTeacherCache | None = None,
        allow_letterbox_upscale: bool = True,
    ):
        self.manifest_path = resolve_legacy_path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = list(manifest["records"])
        counts = Counter(record["identity"].casefold() for record in records)
        allowed = {
            identity
            for identity, count in counts.items()
            if count >= int(min_images_per_identity)
        }
        self.records = [
            record for record in records if record["identity"].casefold() in allowed
        ]
        identities = sorted({record["identity"].casefold() for record in self.records})
        self.identity_to_label = {
            identity: index for index, identity in enumerate(identities)
        }
        self.targets = [
            self.identity_to_label[record["identity"].casefold()]
            for record in self.records
        ]
        self.input_size = int(input_size)
        self.training = bool(training)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.color_jitter = float(color_jitter)
        self.teacher_cache = teacher_cache
        self.allow_letterbox_upscale = bool(allow_letterbox_upscale)
        if not self.records:
            raise RuntimeError("Unified manifest has no eligible records")
        if teacher_cache is not None:
            missing = [
                record.get("source_sha256", "")
                for record in self.records
                if record.get("source_sha256", "") not in teacher_cache
            ]
            if missing:
                raise ValueError(
                    f"Teacher cache is missing {len(missing)} manifest records"
                )

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _read_bgr(resolve_legacy_path(record["source_path"]))
        width, height = (int(value) for value in record["resized_size"])
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_box = np.asarray(record["face_roi_xyxy"], dtype=np.float32)
        nose_box = np.asarray(record["nose_roi_xyxy"], dtype=np.float32)
        angle = float(record["roll_angle_radians"])
        viewpoint = record.get("viewpoint_signals")
        if viewpoint is not None:
            viewpoint = np.asarray(viewpoint, dtype=np.float32).reshape(-1)
        quality = record.get("quality_signals")

        if self.training and random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            face_box = np.asarray(
                (width - face_box[2], face_box[1], width - face_box[0], face_box[3]),
                dtype=np.float32,
            )
            nose_box = np.asarray(
                (width - nose_box[2], nose_box[1], width - nose_box[0], nose_box[3]),
                dtype=np.float32,
            )
            angle = -angle
            if viewpoint is not None and viewpoint.size >= 3:
                viewpoint = viewpoint.copy()
                viewpoint[:3] *= -1.0
        if self.training and self.color_jitter > 0:
            alpha = random.uniform(1.0 - self.color_jitter, 1.0 + self.color_jitter)
            beta = random.uniform(-32.0, 32.0) * self.color_jitter
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )

        image, scale, padding = letterbox_rgb(
            image,
            size=self.input_size,
            fill_value=0,
            allow_upscale=self.allow_letterbox_upscale,
        )
        boxes = np.stack(
            (
                transform_box_to_letterbox(
                    face_box,
                    scale=scale,
                    padding=padding,
                    size=self.input_size,
                ),
                transform_box_to_letterbox(
                    nose_box,
                    scale=scale,
                    padding=padding,
                    size=self.input_size,
                ),
            )
        )
        sample: dict[str, Any] = {
            "rgb": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "boxes_cxcywh": torch.from_numpy(boxes),
            "angle_radians": torch.tensor(angle, dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "identity": record["identity"],
            "source_path": record["source_path"],
            "source_sha256": record.get("source_sha256", ""),
        }
        # Preserve continuous geometry cues for optional reference-set
        # supervision. Older manifests may not contain either field.
        if viewpoint is not None:
            sample["viewpoint_signals"] = viewpoint.tolist()
        if quality is not None:
            sample["quality_signals"] = quality
        if self.teacher_cache is not None:
            embedding, face_embedding = self.teacher_cache.get(record["source_sha256"])
            sample["teacher_embedding"] = torch.from_numpy(embedding.copy())
            sample["teacher_face_embedding"] = torch.from_numpy(face_embedding.copy())
        return sample
