"""Tests for independent expert storage and score-level Agent decisions."""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from pet_id.gallery_service import (
    EncodedPetImage,
    PetGalleryStore,
    PetIdentificationService,
    UploadPayload,
)
from pet_id.recognition_agent import (
    MEGADESCRIPTOR_EXPERT_ID,
    MegaDescriptorEncoder,
    build_agent_decision,
)


def image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, format="PNG")
    return buffer.getvalue()


def metadata(*, detected: bool = True) -> dict:
    return {
        "detections": 1,
        "descriptor": {
            "branch_available": [True, True],
            "branch_quality": [0.9, 0.9],
            "inference_size": [80, 60],
            "runtime_diagnostics": {
                "body": {
                    "detected": detected,
                    "score": 0.95 if detected else 0.0,
                    "bbox_xyxy": [10.0, 5.0, 70.0, 55.0],
                }
            },
        },
    }


class FakeAgentEncoder:
    def backend_info(self):
        return {
            "backend": "fake-agent",
            "model_sha256": "fake-bifor-fingerprint",
            "experts": {
                MEGADESCRIPTOR_EXPERT_ID: {
                    "backend": "fake",
                    "model_sha256": "fake-mega-fingerprint",
                    "feature_dim": 3,
                }
            },
        }

    def encode_file(self, path: Path) -> EncodedPetImage:
        with Image.open(path) as image:
            mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        feature = mean + 1.0
        return EncodedPetImage(
            fused=feature,
            nose=feature[:2],
            face=feature[1:],
            metadata=metadata(),
            expert_features={MEGADESCRIPTOR_EXPERT_ID: feature.copy()},
            expert_metadata={
                MEGADESCRIPTOR_EXPERT_ID: {
                    "body_detected": True,
                    "body_detection_score": 0.95,
                    "quality": {"sharpness": 0.9, "exposure": 0.9},
                }
            },
        )


class RecognitionAgentTest(unittest.TestCase):
    def test_expert_features_are_transactional_and_survive_rebind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gallery"
            service = PetIdentificationService(
                PetGalleryStore(root),
                FakeAgentEncoder(),
            )
            red = UploadPayload("red.png", "image/png", image_bytes((240, 10, 10)))
            green = UploadPayload(
                "green.png", "image/png", image_bytes((10, 240, 10))
            )
            service.enroll("red", [red])
            service.enroll("green", [green])
            result = service.identify(red, top_k=2, record_history=False)
            self.assertEqual(result["decision"], "agent_evidence")
            self.assertEqual(result["agent"]["decision"], "matched")
            self.assertTrue(result["agent"]["expert_agreement"])
            self.assertEqual(result["predicted_pet_id"], "red")
            self.assertIn("expert_scores", result["candidates"][0])

            reopened = PetIdentificationService(
                PetGalleryStore(root),
                FakeAgentEncoder(),
            )
            self.assertEqual(
                reopened.store.summary()["experts"],
                [MEGADESCRIPTOR_EXPERT_ID],
            )
            self.assertEqual(
                len(
                    reopened.store.prototypes()[0]["expert_prototypes"][
                        MEGADESCRIPTOR_EXPERT_ID
                    ]
                ),
                3,
            )
            image_id = service.store.get_pet("red")["images"][0]["image_id"]
            service.store.delete_image("red", image_id)
            with service.store._connect() as connection:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM expert_features WHERE image_id = ?",
                    (image_id,),
                ).fetchone()[0]
            self.assertEqual(remaining, 0)

    def test_reliable_expert_conflict_requests_more_evidence(self):
        prototypes = [
            {
                "pet_id": "a",
                "display_name": "A",
                "reference_count": 2,
            },
            {
                "pet_id": "b",
                "display_name": "B",
                "reference_count": 2,
            },
        ]
        encoded = EncodedPetImage(
            fused=np.asarray([1.0, 0.0], dtype=np.float32),
            nose=np.asarray([1.0, 0.0], dtype=np.float32),
            face=np.asarray([1.0, 0.0], dtype=np.float32),
            metadata=metadata(),
            expert_features={
                MEGADESCRIPTOR_EXPERT_ID: np.asarray([1.0, 0.0], dtype=np.float32)
            },
            expert_metadata={
                MEGADESCRIPTOR_EXPERT_ID: {
                    "body_detected": True,
                    "body_detection_score": 0.99,
                    "quality": {"sharpness": 1.0, "exposure": 1.0},
                }
            },
        )
        result = build_agent_decision(
            prototypes=prototypes,
            encoded=encoded,
            bifor_scores=np.asarray([0.9, 0.0], dtype=np.float32),
            expert_scores={
                MEGADESCRIPTOR_EXPERT_ID: np.asarray([0.0, 0.9], dtype=np.float32)
            },
            top_k=2,
            requested_threshold=None,
            requested_margin=0.0,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["agent"]["decision"], "needs_more_evidence")
        self.assertIn("expert_conflict", result["agent"]["reasons"])
        self.assertTrue(result["agent"]["capture_recommendations"])

    def test_low_agreed_scores_can_be_possible_unknown(self):
        prototypes = [
            {"pet_id": "a", "display_name": "A", "reference_count": 1},
            {"pet_id": "b", "display_name": "B", "reference_count": 1},
        ]
        encoded = EncodedPetImage(
            fused=np.asarray([1.0, 0.0], dtype=np.float32),
            nose=np.asarray([1.0, 0.0], dtype=np.float32),
            face=np.asarray([1.0, 0.0], dtype=np.float32),
            metadata=metadata(),
            expert_features={
                MEGADESCRIPTOR_EXPERT_ID: np.asarray([1.0, 0.0], dtype=np.float32)
            },
            expert_metadata={
                MEGADESCRIPTOR_EXPERT_ID: {
                    "body_detected": True,
                    "body_detection_score": 0.99,
                    "quality": {"sharpness": 1.0, "exposure": 1.0},
                }
            },
        )
        result = build_agent_decision(
            prototypes=prototypes,
            encoded=encoded,
            bifor_scores=np.asarray([-0.7, -0.8], dtype=np.float32),
            expert_scores={
                MEGADESCRIPTOR_EXPERT_ID: np.asarray(
                    [-0.75, -0.85], dtype=np.float32
                )
            },
            top_k=2,
            requested_threshold=None,
            requested_margin=0.0,
        )
        self.assertEqual(result["agent"]["decision"], "possible_unknown")
        self.assertFalse(result["accepted"])

    def test_multiple_quality_failures_request_more_evidence(self):
        prototypes = [
            {"pet_id": "a", "display_name": "A", "reference_count": 1},
            {"pet_id": "b", "display_name": "B", "reference_count": 1},
        ]
        low_quality_metadata = metadata(detected=False)
        low_quality_metadata["descriptor"]["branch_available"] = [False, True]
        encoded = EncodedPetImage(
            fused=np.asarray([1.0, 0.0], dtype=np.float32),
            nose=np.asarray([1.0, 0.0], dtype=np.float32),
            face=np.asarray([1.0, 0.0], dtype=np.float32),
            metadata=low_quality_metadata,
            expert_features={
                MEGADESCRIPTOR_EXPERT_ID: np.asarray([1.0, 0.0], dtype=np.float32)
            },
            expert_metadata={
                MEGADESCRIPTOR_EXPERT_ID: {
                    "body_detected": False,
                    "body_detection_score": 0.0,
                    "quality": {"sharpness": 0.1, "exposure": 0.2},
                }
            },
        )
        result = build_agent_decision(
            prototypes=prototypes,
            encoded=encoded,
            bifor_scores=np.asarray([0.95, 0.0], dtype=np.float32),
            expert_scores={
                MEGADESCRIPTOR_EXPERT_ID: np.asarray([0.9, 0.0], dtype=np.float32)
            },
            top_k=2,
            requested_threshold=None,
            requested_margin=0.0,
        )
        self.assertEqual(result["agent"]["decision"], "needs_more_evidence")
        self.assertFalse(result["accepted"])
        self.assertIn("body_not_detected", result["agent"]["reasons"])
        self.assertIn("motion_or_focus_blur", result["agent"]["reasons"])
        self.assertTrue(result["agent"]["capture_recommendations"])

    def test_body_box_is_scaled_back_to_original_image(self):
        image = Image.new("RGB", (160, 120), "white")
        crop, crop_metadata = MegaDescriptorEncoder._body_crop(image, metadata())
        self.assertTrue(crop_metadata["body_detected"])
        self.assertGreater(crop.width, 115)
        self.assertGreater(crop.height, 95)
        self.assertLess(crop.width, image.width)
        self.assertLess(crop.height, image.height)


if __name__ == "__main__":
    unittest.main()
