from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from pet_id.unified_highres_data import (
    HIGHRES_MIN_INPUT_SIDE,
    UnifiedHighResolutionReferenceDataset,
    validate_highres_dimensions,
)
from pet_id.unified_highres_eval_data import UnifiedHighResolutionRawDataset
from pet_id.unified_highres_protocol import PROTOCOL_NAME


def test_highres_dimension_contract() -> None:
    assert validate_highres_dimensions(64, 80, maximum_side=4096) == (64, 80)
    with pytest.raises(ValueError, match="at least"):
        validate_highres_dimensions(HIGHRES_MIN_INPUT_SIDE - 1, 80)
    with pytest.raises(ValueError, match="maximum side"):
        validate_highres_dimensions(4097, 80, maximum_side=4096)


def test_raw_dataset_preserves_dynamic_shape_and_verifies_hash(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.full((96, 128, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "development.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_name": PROTOCOL_NAME,
                "protocol_split": "development",
                "images_per_identity": 3,
                "records": [
                    {
                        "source_path": str(image_path),
                        "source_sha256": digest,
                        "identity": "pet-a",
                    },
                    {
                        "source_path": str(image_path),
                        "source_sha256": digest,
                        "identity": "pet-a",
                    },
                    {
                        "source_path": str(image_path),
                        "source_sha256": digest,
                        "identity": "pet-a",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = UnifiedHighResolutionRawDataset(
        manifest_path,
        expected_split="development",
        maximum_side=256,
    )
    row = dataset[0]
    assert tuple(row["rgb"].shape) == (3, 96, 128)
    assert row["source_sha256"] == digest
    assert row["fed_height"] == 96
    assert row["fed_width"] == 128


def test_reference_dataset_materializes_stackable_highres_canvas(tmp_path) -> None:
    image_path = tmp_path / "reference.png"
    image = np.full((96, 128, 3), 91, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "development.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_name": PROTOCOL_NAME,
                "protocol_split": "development",
                "images_per_identity": 3,
                "records": [
                    {
                        "source_path": str(image_path),
                        "source_sha256": digest,
                        "identity": "pet-a",
                    }
                    for _ in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = UnifiedHighResolutionReferenceDataset(
        manifest_path,
        expected_split="development",
        image_size=1280,
    )
    row = dataset[0]
    assert tuple(row["rgb"].shape) == (3, 1280, 1280)
    assert row["source_sha256"] == digest
    assert row["identity"] == "pet-a"
