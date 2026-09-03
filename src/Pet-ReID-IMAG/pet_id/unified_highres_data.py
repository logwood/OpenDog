"""Data paths for spatial-detail high-resolution training and evaluation."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dogfacenet_alignment import _read_bgr
from .release_compatibility import is_high_resolution_protocol_name
from .unified_data import letterbox_rgb


HIGHRES_MIN_INPUT_SIDE = 64


def load_highres_manifest(
    manifest_path: str | Path,
    *,
    expected_split: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not is_high_resolution_protocol_name(payload.get("protocol_name")):
        raise RuntimeError("Unexpected high-resolution protocol")
    if expected_split is not None and payload.get("protocol_split") != expected_split:
        raise RuntimeError(
            f"Expected {expected_split!r}, got {payload.get('protocol_split')!r}"
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("High-resolution manifest records must be non-empty")
    return path, payload


def _resize_long_side(image: np.ndarray, maximum_side: int) -> np.ndarray:
    maximum_side = int(maximum_side)
    if maximum_side <= 0:
        raise ValueError("maximum_side must be positive")
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / float(max(height, width)))
    if scale >= 1.0:
        return image
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
def validate_highres_dimensions(
    height: int,
    width: int,
    *,
    minimum_side: int = HIGHRES_MIN_INPUT_SIDE,
    maximum_side: int = 4096,
) -> tuple[int, int]:
    """Validate a tensor/image shape against the spatial-detail contract."""

    height = int(height)
    width = int(width)
    minimum_side = int(minimum_side)
    maximum_side = int(maximum_side)
    if minimum_side < 2:
        raise ValueError("minimum_side must be at least two")
    if maximum_side < minimum_side:
        raise ValueError("maximum_side must be at least minimum_side")
    if height < minimum_side or width < minimum_side:
        raise ValueError(
            f"Input dimensions must both be at least {minimum_side}; "
            f"got {height}x{width}"
        )
    if max(height, width) > maximum_side:
        raise ValueError(
            f"Input maximum side is {maximum_side}; got {height}x{width}. "
            "The runtime refuses to resize it outside the ONNX graph."
        )
    return height, width




def load_raw_rgb(
    source_path: str | Path,
    *,
    maximum_side: int = 4096,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = Path(source_path).expanduser().resolve()
    image = cv2.cvtColor(_read_bgr(source), cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    image = _resize_long_side(image, maximum_side)
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
    return tensor, {
        "source_path": str(source),
        "original_height": int(original_height),
        "original_width": int(original_width),
        "fed_height": int(image.shape[0]),
        "fed_width": int(image.shape[1]),
        "maximum_side": int(maximum_side),
    }


def degraded_raw_rgb(
    source_path: str | Path,
    *,
    detail_cap: int = 1280,
    maximum_side: int = 4096,
) -> torch.Tensor:
    source = Path(source_path).expanduser().resolve()
    image = cv2.cvtColor(_read_bgr(source), cv2.COLOR_BGR2RGB)
    image = _resize_long_side(image, detail_cap)
    image = _resize_long_side(image, maximum_side)
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float()


class UnifiedHighResolutionTrainingDataset(Dataset):
    """Return synchronized high-detail and 1280-detail square views."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        training_size: int = 2048,
        degraded_detail_size: int = 1280,
        horizontal_flip_probability: float = 0.5,
        color_jitter: float = 0.08,
    ) -> None:
        self.manifest_path, payload = load_highres_manifest(
            manifest_path,
            expected_split="training_extension",
        )
        self.records = list(payload["records"])
        self.training_size = int(training_size)
        self.degraded_detail_size = int(degraded_detail_size)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.color_jitter = float(color_jitter)
        if self.training_size < 1280:
            raise ValueError("training_size must be at least 1280")
        if not 64 <= self.degraded_detail_size <= self.training_size:
            raise ValueError("degraded_detail_size is outside the training canvas")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0,1]")
        if not 0.0 <= self.color_jitter <= 1.0:
            raise ValueError("color_jitter must be in [0,1]")

        identities = sorted(
            {str(record["identity"]).casefold() for record in self.records}
        )
        self.identity_to_label = {
            identity: index for index, identity in enumerate(identities)
        }
        self.targets = [
            self.identity_to_label[str(record["identity"]).casefold()]
            for record in self.records
        ]
        self.indices_by_target: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(self.targets):
            self.indices_by_target[target].append(index)
        declared = int(payload.get("images_per_identity", 0))
        counts = Counter(self.targets)
        if declared < 3 or any(count != declared for count in counts.values()):
            raise ValueError("High-resolution identity counts are inconsistent")
        self.images_per_identity = declared

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = Path(record["source_path"]).expanduser().resolve()
        image = cv2.cvtColor(_read_bgr(source), cv2.COLOR_BGR2RGB)
        if random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
        if self.color_jitter > 0.0:
            alpha = random.uniform(
                1.0 - self.color_jitter,
                1.0 + self.color_jitter,
            )
            beta = random.uniform(-24.0, 24.0) * self.color_jitter
            image = np.clip(
                image.astype(np.float32) * alpha + beta,
                0.0,
                255.0,
            ).astype(np.uint8)

        high, _, _ = letterbox_rgb(
            image,
            size=self.training_size,
            fill_value=0,
            allow_upscale=True,
        )
        degraded, _, _ = letterbox_rgb(
            image,
            size=self.degraded_detail_size,
            fill_value=0,
            allow_upscale=True,
        )
        if self.degraded_detail_size != self.training_size:
            degraded = cv2.resize(
                degraded,
                (self.training_size, self.training_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return {
            "high_rgb": torch.from_numpy(
                high.transpose(2, 0, 1).copy()
            ).float(),
            "degraded_rgb": torch.from_numpy(
                degraded.transpose(2, 0, 1).copy()
            ).float(),
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "identity": str(record["identity"]).casefold(),
            "source_path": str(source),
            "source_sha256": str(record["source_sha256"]),
        }


class UnifiedHighResolutionReferenceDataset(Dataset):
    """Materialize high-resolution records for image-reference episodes.

    The reference-aware trainer works with a small, fixed tensor per episode,
    while the spatial-detail model itself accepts dynamic raw shapes. This
    adapter preserves source detail by letterboxing each raw image onto a
    configurable high-resolution square (normally 2048) and exposes the
    resulting tensor under the common ``rgb`` key expected by the episode
    sampler. It does not alter the production runtime contract.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int = 2048,
        expected_split: str | None = None,
        training: bool = False,
        horizontal_flip_probability: float = 0.0,
        color_jitter: float = 0.0,
        maximum_side: int = 4096,
        verify_source_hash: bool = True,
    ) -> None:
        self.manifest_path, payload = load_highres_manifest(
            manifest_path,
            expected_split=expected_split,
        )
        self.records = list(payload["records"])
        self.image_size = int(image_size)
        self.maximum_side = int(maximum_side)
        self.training = bool(training)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.color_jitter = float(color_jitter)
        self.verify_source_hash = bool(verify_source_hash)
        if self.image_size < 1280:
            raise ValueError("image_size must be at least 1280")
        if self.maximum_side < self.image_size:
            raise ValueError("maximum_side must be at least image_size")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0,1]")
        if not 0.0 <= self.color_jitter <= 1.0:
            raise ValueError("color_jitter must be in [0,1]")
        if not self.records:
            raise ValueError("high-resolution reference manifest has no records")

        identities = sorted(
            {str(record["identity"]).casefold() for record in self.records}
        )
        self.identity_to_label = {
            identity: index for index, identity in enumerate(identities)
        }
        self.targets = [
            self.identity_to_label[str(record["identity"]).casefold()]
            for record in self.records
        ]
        self.indices_by_target: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(self.targets):
            self.indices_by_target[target].append(index)
        self.images_per_identity = int(payload.get("images_per_identity", 0))
        if self.images_per_identity < 1:
            raise ValueError("high-resolution manifest has no images_per_identity")
        counts = Counter(self.targets)
        if any(count != self.images_per_identity for count in counts.values()):
            raise ValueError("High-resolution identity counts are inconsistent")
        self._verified_sources: set[str] = set()

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def __len__(self) -> int:
        return len(self.records)

    def _verify_source(self, source: Path, expected: str) -> str:
        # Import lazily to keep this data helper usable in lightweight tools.
        from .unified_fresh_protocol import sha256_file

        key = str(source)
        if self.verify_source_hash and key not in self._verified_sources:
            actual = sha256_file(source).casefold()
            if expected and actual != expected.casefold():
                raise RuntimeError(
                    f"Source hash differs from high-resolution manifest: {source}"
                )
            self._verified_sources.add(key)
            return actual
        return expected.casefold() if expected else ""

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        record = self.records[index]
        source = Path(record["source_path"]).expanduser().resolve()
        image = cv2.cvtColor(_read_bgr(source), cv2.COLOR_BGR2RGB)
        viewpoint = record.get("viewpoint_signals")
        if viewpoint is not None:
            viewpoint = np.asarray(viewpoint, dtype=np.float32).reshape(-1)
        quality = record.get("quality_signals")
        source_sha256 = self._verify_source(
            source, str(record.get("source_sha256", ""))
        )
        if self.training and random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            if viewpoint is not None and viewpoint.size >= 3:
                viewpoint = viewpoint.copy()
                viewpoint[:3] *= -1.0
        if self.training and self.color_jitter > 0.0:
            alpha = random.uniform(1.0 - self.color_jitter, 1.0 + self.color_jitter)
            beta = random.uniform(-24.0, 24.0) * self.color_jitter
            image = np.clip(
                image.astype(np.float32) * alpha + beta,
                0.0,
                255.0,
            ).astype(np.uint8)
        square, _scale, _padding = letterbox_rgb(
            image,
            size=self.image_size,
            fill_value=0,
            allow_upscale=True,
        )
        sample = {
            "rgb": torch.from_numpy(square.transpose(2, 0, 1).copy()).float(),
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "identity": str(record["identity"]).casefold(),
            "source_path": str(source),
            "source_sha256": source_sha256,
            "record": record,
        }
        if viewpoint is not None:
            sample["viewpoint_signals"] = viewpoint.tolist()
        if quality is not None:
            sample["quality_signals"] = quality
        return sample

