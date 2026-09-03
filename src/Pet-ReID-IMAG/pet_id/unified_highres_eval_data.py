"""Variable-shape data helpers for spatial-detail evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .unified_fresh_protocol import sha256_file
from .unified_highres_data import (
    HIGHRES_MIN_INPUT_SIDE,
    load_highres_manifest,
    load_raw_rgb,
)


class UnifiedHighResolutionRawDataset(Dataset):
    """Load high-resolution records without collapsing them to 1280x1280.

    Every item contains a raw ``[3,H,W]`` tensor.  Shapes are intentionally
    allowed to vary; evaluation code can process items individually or group
    equal dimensions before stacking.  Source hashes are checked by default
    so a changed image cannot silently invalidate a locked protocol.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_split: str | None = None,
        maximum_side: int = 4096,
        minimum_side: int = HIGHRES_MIN_INPUT_SIDE,
        verify_source_hash: bool = True,
    ) -> None:
        self.manifest_path, payload = load_highres_manifest(
            manifest_path,
            expected_split=expected_split,
        )
        self.records = list(payload["records"])
        self.maximum_side = int(maximum_side)
        self.minimum_side = int(minimum_side)
        self.verify_source_hash = bool(verify_source_hash)
        if self.minimum_side < 2:
            raise ValueError("minimum_side must be at least two")
        if self.maximum_side < self.minimum_side:
            raise ValueError("maximum_side must be at least minimum_side")

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
        rgb, dimensions = load_raw_rgb(source, maximum_side=self.maximum_side)
        if (
            dimensions["fed_height"] < self.minimum_side
            or dimensions["fed_width"] < self.minimum_side
        ):
            raise ValueError(
                f"Source image is below the minimum tensor side: {source}"
            )
        expected_digest = str(record.get("source_sha256", "")).casefold()
        actual_digest = sha256_file(source).casefold()
        if self.verify_source_hash and expected_digest and actual_digest != expected_digest:
            raise RuntimeError(
                f"Source hash differs from high-resolution manifest: {source}"
            )
        return {
            "rgb": rgb,
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "identity": str(record["identity"]).casefold(),
            "source_path": str(source),
            "source_sha256": actual_digest,
            "record": record,
            "original_height": dimensions["original_height"],
            "original_width": dimensions["original_width"],
            "fed_height": dimensions["fed_height"],
            "fed_width": dimensions["fed_width"],
        }
