"""Regression tests for wiring the learned matcher into the gallery service."""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pet_id.api import create_app
from pet_id.gallery_service import (
    EncodedPetImage,
    InvalidGalleryRequest,
    PetGalleryStore,
    PetIdentificationService,
    UploadPayload,
)
from pet_id.reference_scoring import LEARNED_REFERENCE_SET_SCORING


def image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


class ColorEncoder:
    def backend_info(self) -> dict[str, object]:
        return {
            "backend": "test",
            "model_sha256": "reference-set-service-test",
            "embedding_dim": 3,
        }

    def encode_file(self, path: Path) -> EncodedPetImage:
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        return EncodedPetImage(
            fused=value + 1.0,
            nose=value[:2] + 1.0,
            face=value[1:] + 1.0,
            metadata={"detections": 1},
        )


class RecordingMatcher:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def backend_info(self) -> dict[str, object]:
        return {
            "type": "test-reference-matcher",
            "model_sha256": "matcher-test",
            "descriptor_dim": 3,
        }

    def score_gallery(self, query, prototypes):
        self.calls.append(
            [int(np.asarray(item["reference_features"]).shape[0]) for item in prototypes]
        )
        scores = np.asarray(
            [
                float(np.max(np.asarray(item["reference_features"]) @ query))
                for item in prototypes
            ],
            dtype=np.float32,
        )
        details = {
            str(item["pet_id"]): {
                "mode": LEARNED_REFERENCE_SET_SCORING,
                "score": float(scores[index]),
                "reference_count": self.calls[-1][index],
            }
            for index, item in enumerate(prototypes)
        }
        return scores, details


class ReferenceSetServiceIntegrationTest(unittest.TestCase):
    def test_learned_mode_reaches_matcher_and_keeps_references(self):
        matcher = RecordingMatcher()
        with tempfile.TemporaryDirectory() as directory:
            service = PetIdentificationService(
                PetGalleryStore(Path(directory) / "gallery"),
                ColorEncoder(),
                default_scoring_mode=LEARNED_REFERENCE_SET_SCORING,
                reference_matcher=matcher,
            )
            client = TestClient(create_app(service))
            red = image_bytes((250, 10, 10))
            red_two = image_bytes((240, 20, 10))
            blue = image_bytes((10, 10, 250))
            self.assertEqual(
                client.post(
                    "/v1/pets/red/images",
                    files=[
                        ("files", ("red.png", red, "image/png")),
                        ("files", ("red-two.png", red_two, "image/png")),
                    ],
                ).status_code,
                201,
            )
            self.assertEqual(
                client.post(
                    "/v1/pets/blue/images",
                    files=[("files", ("blue.png", blue, "image/png"))],
                ).status_code,
                201,
            )
            response = client.post(
                "/v1/identify?top_k=2",
                files=[("file", ("query.png", red, "image/png"))],
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["scoring"]["mode"], LEARNED_REFERENCE_SET_SCORING)
            self.assertEqual(result["predicted_pet_id"], "red")
            # Prototypes are sorted by pet_id, so blue precedes red.
            self.assertEqual(matcher.calls, [[1, 2]])
            self.assertEqual(
                service.health()["backend"]["reference_matcher"]["model_sha256"],
                "matcher-test",
            )

    def test_learned_request_without_matcher_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PetIdentificationService(
                PetGalleryStore(Path(directory) / "gallery"), ColorEncoder()
            )
            with self.assertRaises(InvalidGalleryRequest):
                service.identify(
                    # The request is rejected before image decoding, so a minimal
                    # payload is sufficient for this constructor-level contract.
                    UploadPayload(filename="x", content_type="", data=b""),
                    scoring_mode=LEARNED_REFERENCE_SET_SCORING,
                    record_history=False,
                )


if __name__ == "__main__":
    unittest.main()
