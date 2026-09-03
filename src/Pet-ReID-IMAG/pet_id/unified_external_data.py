"""Raw external-manifest data path for joint UnifiedPetReID fusion fitting."""

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
from .unified_data import letterbox_rgb


class UnifiedRawManifestDataset(Dataset):
    """Load one identity-labelled image without external geometry annotations."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        input_size: int = 1280,
        training: bool = False,
        horizontal_flip_probability: float = 0.0,
        color_jitter: float = 0.0,
        allow_letterbox_upscale: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Raw manifest records must be a non-empty list")
        self.records = list(records)
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
        if declared < 1 or any(count != declared for count in counts.values()):
            raise ValueError("Raw manifest identity counts are inconsistent")
        self.images_per_identity = declared
        self.input_size = int(input_size)
        self.training = bool(training)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.color_jitter = float(color_jitter)
        self.allow_letterbox_upscale = bool(allow_letterbox_upscale)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0,1]")
        if not 0.0 <= self.color_jitter <= 1.0:
            raise ValueError("color_jitter must be in [0,1]")

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = Path(record["source_path"]).expanduser().resolve()
        image = _read_bgr(source)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.training and random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
        if self.training and self.color_jitter > 0.0:
            alpha = random.uniform(
                1.0 - self.color_jitter, 1.0 + self.color_jitter
            )
            beta = random.uniform(-24.0, 24.0) * self.color_jitter
            image = np.clip(
                image.astype(np.float32) * alpha + beta, 0.0, 255.0
            ).astype(np.uint8)
        image, _, _ = letterbox_rgb(
            image,
            size=self.input_size,
            fill_value=0,
            allow_upscale=self.allow_letterbox_upscale,
        )
        return {
            "rgb": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "identity": str(record["identity"]).casefold(),
            "source_path": str(source),
            "source_sha256": str(record["source_sha256"]),
        }


def identity_batches(
    dataset: UnifiedRawManifestDataset,
    *,
    identities_per_batch: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Return deterministic batches containing all records for each identity."""

    identities_per_batch = int(identities_per_batch)
    if identities_per_batch < 2:
        raise ValueError("At least two identities are required per metric batch")
    generator = random.Random(int(seed) + 1_000_003 * int(epoch))
    targets = list(dataset.indices_by_target)
    generator.shuffle(targets)
    batches = []
    for start in range(0, len(targets), identities_per_batch):
        selected = targets[start : start + identities_per_batch]
        if len(selected) != identities_per_batch:
            continue
        rows: list[int] = []
        for target in selected:
            indices = list(dataset.indices_by_target[target])
            generator.shuffle(indices)
            rows.extend(indices)
        batches.append(rows)
    return batches
