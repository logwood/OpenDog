"""Checkpoint-envelope dispatch tests for the reference-aware entry point."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from pet_id.reference_aware_model import (
    build_reference_aware_encoder_from_checkpoint,
)
from pet_id.reference_aware_training import validate_reference_image_manifest


class TinyEncoder(nn.Module):
    descriptor_dim = 8
    input_size = 4


class ReferenceAwareCheckpointDispatchTest(unittest.TestCase):
    def _checkpoint(self, model_type: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "base.pth"
        torch.save({"model_type": model_type}, path)
        return path

    def test_external_joint_envelope_uses_verified_loader(self):
        path = self._checkpoint("unified_external_joint_pet_reid")
        expected = TinyEncoder()
        payload = {"model_type": "unified_external_joint_pet_reid"}
        with patch(
            "pet_id.unified_external_model.build_external_joint_from_checkpoint",
            return_value=(expected, payload),
        ) as loader:
            actual, restored = build_reference_aware_encoder_from_checkpoint(path)
        self.assertIs(actual, expected)
        self.assertEqual(restored, payload)
        loader.assert_called_once()

    def test_high_resolution_envelope_uses_verified_loader(self):
        path = self._checkpoint("unified_high_resolution_pet_reid")
        expected = TinyEncoder()
        payload = {"model_type": "unified_high_resolution_pet_reid"}
        with patch(
            "pet_id.unified_highres.build_highres_from_checkpoint",
            return_value=(expected, payload),
        ) as loader:
            actual, restored = build_reference_aware_encoder_from_checkpoint(path)
        self.assertIs(actual, expected)
        self.assertEqual(restored, payload)
        loader.assert_called_once()

    def test_untyped_envelope_requires_legacy_arcface_source(self):
        path = self._checkpoint("unknown")
        with self.assertRaisesRegex(ValueError, "arcface checkpoint"):
            build_reference_aware_encoder_from_checkpoint(path)

    def test_manifest_validation_rejects_raw_and_blind_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "development.manifest.json"
            raw.write_text(
                json.dumps({"protocol_split": "development", "records": [{"identity": "a"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing fields"):
                validate_reference_image_manifest(raw)

            blind = root / "blind.manifest.json"
            blind.write_text(
                json.dumps(
                    {
                        "protocol_split": "blind_test",
                        "records": [
                            {
                                "identity": "a",
                                "source_path": "a.jpg",
                                "resized_size": [4, 4],
                                "face_roi_xyxy": [0, 0, 4, 4],
                                "nose_roi_xyxy": [1, 1, 3, 3],
                                "roll_angle_radians": 0.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "blind manifests"):
                validate_reference_image_manifest(blind)


if __name__ == "__main__":
    unittest.main()
