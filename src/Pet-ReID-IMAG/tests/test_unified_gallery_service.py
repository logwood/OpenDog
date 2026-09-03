"""Regression tests for unified descriptors in the legacy gallery container."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pet_id.api import create_app
from pet_id.gallery_service import (
    EncodedPetImage,
    PetGalleryStore,
    PetIdentificationService,
    is_unified_single_graph_descriptor,
    reference_quality,
)


def image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


class UnifiedFakeEncoder:
    def __init__(self, output_norm: float = 0.999):
        self.output_norm = float(output_norm)

    def backend_info(self) -> dict:
        return {
            "backend": "onnxruntime-unified",
            "model_sha256": "unified-fake-fingerprint",
            "provider": "CPUExecutionProvider",
            "embedding_dim": 3,
            "single_graph": True,
            "external_models": [],
        }

    def encode_file(self, path: Path) -> EncodedPetImage:
        with Image.open(path) as image:
            feature = np.asarray(image.convert("RGB"), dtype=np.float32).mean(
                axis=(0, 1)
            )
        feature = feature / np.linalg.norm(feature) * self.output_norm
        descriptor = {
            # These values are an API-container compatibility detail, not real
            # branches. The explicit unified diagnostic must take precedence.
            "branch_available": [False, True],
            "branch_quality": [0.0, 0.01],
            "fusion_weights": [0.0, 1.0],
            "detection": None,
            "runtime_diagnostics": {
                "unified": {
                    "single_graph": True,
                    "external_models": [],
                    "provider": "CPUExecutionProvider",
                }
            },
        }
        return EncodedPetImage(
            fused=feature,
            nose=feature,
            face=feature,
            metadata={
                "detections": 1,
                "selected_detection": 0,
                "descriptor": descriptor,
            },
        )


class UnifiedGalleryServiceTest(unittest.TestCase):
    def test_unified_diagnostic_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "red.png"
            path.write_bytes(image_bytes((250, 10, 10)))
            descriptor = UnifiedFakeEncoder().encode_file(path).metadata["descriptor"]
        self.assertTrue(is_unified_single_graph_descriptor(descriptor))
        quality = reference_quality({"descriptor": descriptor})
        self.assertEqual(quality["status"], "good")
        self.assertEqual(quality["reasons"], [])
        self.assertIsNone(quality["branch_available"])
        self.assertIsNone(quality["branch_quality"])
        self.assertEqual(quality["architecture"], "unified_single_graph")

    def test_unified_enrollment_and_identification_have_no_fake_branch_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PetGalleryStore(Path(directory) / "gallery")
            service = PetIdentificationService(
                store,
                UnifiedFakeEncoder(),
            )
            with TestClient(create_app(service)) as client:
                red = image_bytes((250, 10, 10))
                green = image_bytes((10, 250, 10))
                enrolled = client.post(
                    "/v1/pets/red/images",
                    files={"files": ("red.png", red, "image/png")},
                )
                self.assertEqual(enrolled.status_code, 201, enrolled.text)
                quality = enrolled.json()["pet"]["images"][0]["quality"]
                self.assertEqual(quality["status"], "good")
                self.assertEqual(quality["reasons"], [])

                second = client.post(
                    "/v1/pets/green/images",
                    files={"files": ("green.png", green, "image/png")},
                )
                self.assertEqual(second.status_code, 201, second.text)

                response = client.post(
                    "/v1/identify?top_k=2",
                    files={"file": ("query.png", red, "image/png")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["predicted_pet_id"], "red")
                self.assertTrue(result["diagnostics"]["single_graph"])
                self.assertEqual(
                    result["diagnostics"]["mode"], "unified_single_graph"
                )
                self.assertIsNone(result["diagnostics"]["branch_available"])
                self.assertNotIn("branch_top1", result["diagnostics"])
                self.assertNotIn("single_branch", result["hard_case_reasons"])
                self.assertNotIn("low_quality", result["hard_case_reasons"])
                self.assertAlmostEqual(result["top1_score"], 0.999, places=5)

                with closing(sqlite3.connect(store.database_path)) as connection:
                    row = connection.execute(
                        "SELECT fused, fused_dim FROM reference_images "
                        "WHERE pet_id = 'red'"
                    ).fetchone()
                stored = np.frombuffer(row[0], dtype="<f4", count=int(row[1]))
                self.assertAlmostEqual(float(np.linalg.norm(stored)), 0.999, places=5)

    def test_unified_service_rejects_a_non_normalized_graph_output(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PetIdentificationService(
                PetGalleryStore(Path(directory) / "gallery"),
                UnifiedFakeEncoder(output_norm=0.5),
            )
            with TestClient(create_app(service)) as client:
                response = client.post(
                    "/v1/pets/red/images",
                    files={
                        "files": (
                            "red.png",
                            image_bytes((250, 10, 10)),
                            "image/png",
                        )
                    },
                )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("L2-normalized", response.text)

if __name__ == "__main__":
    unittest.main()
