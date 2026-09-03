"""Tests for incremental gallery storage and the FastAPI surface."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pet_id.api import create_app
from pet_id.gallery import sha256_file
from pet_id.gallery_service import (
    EncodedPetImage,
    GalleryModelMismatch,
    PetGalleryStore,
    PetIdentificationService,
)


def image_bytes(color: tuple[int, int, int], image_format: str = "JPEG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format=image_format, quality=100)
    return buffer.getvalue()


class FakeEncoder:
    def __init__(self, fingerprint: str = "fake-model-fingerprint"):
        self.fingerprint = fingerprint

    def backend_info(self):
        return {
            "backend": "fake",
            "model_sha256": self.fingerprint,
            "provider": "TestExecutionProvider",
        }

    def encode_file(self, path: Path) -> EncodedPetImage:
        with Image.open(path) as image:
            mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        fused = mean + 1.0
        nose = np.asarray([mean[0] + 1.0, mean[1] + 1.0], dtype=np.float32)
        face = np.asarray([mean[1] + 1.0, mean[2] + 1.0], dtype=np.float32)
        return EncodedPetImage(
            fused=fused,
            nose=nose,
            face=face,
            metadata={"detections": 1, "fake_mean_rgb": mean.tolist()},
        )


class GalleryAPITest(unittest.TestCase):
    def make_service(
        self, root: Path, fingerprint: str = "fake-model-fingerprint"
    ):
        return PetIdentificationService(
            PetGalleryStore(root),
            FakeEncoder(fingerprint),
            maximum_upload_bytes=1024 * 1024,
            maximum_image_pixels=1_000_000,
        )

    def test_enroll_identify_list_download_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory) / "gallery")
            client = TestClient(create_app(service))
            red = image_bytes((250, 10, 10))
            red_two = image_bytes((240, 20, 10), "PNG")
            green = image_bytes((10, 250, 10))

            response = client.post(
                "/v1/pets/dog-red/images",
                data={"display_name": "Red dog"},
                files=[
                    ("files", ("red.jpg", red, "image/jpeg")),
                    ("files", ("red-two.png", red_two, "image/png")),
                ],
            )
            self.assertEqual(response.status_code, 201, response.text)
            enrolled = response.json()
            self.assertEqual(enrolled["pet"]["reference_count"], 2)
            self.assertEqual(len(enrolled["added_image_ids"]), 2)

            duplicate = client.post(
                "/v1/pets/dog-red/images",
                files={"files": ("again.jpg", red, "image/jpeg")},
            )
            self.assertEqual(duplicate.status_code, 201, duplicate.text)
            self.assertEqual(duplicate.json()["added_image_ids"], [])
            self.assertEqual(len(duplicate.json()["duplicate_image_ids"]), 1)

            conflict = client.post(
                "/v1/pets/dog-other/images",
                files={"files": ("same.jpg", red, "image/jpeg")},
            )
            self.assertEqual(conflict.status_code, 409, conflict.text)
            self.assertEqual(conflict.json()["error"]["code"], "gallery_conflict")

            response = client.post(
                "/v1/pets/dog-green/images",
                files={"files": ("green.jpg", green, "image/jpeg")},
            )
            self.assertEqual(response.status_code, 201, response.text)

            identified = client.post(
                "/v1/identify?top_k=2",
                files={"file": ("query.jpg", red, "image/jpeg")},
            )
            self.assertEqual(identified.status_code, 200, identified.text)
            result = identified.json()
            self.assertTrue(result["accepted"])
            self.assertEqual(result["predicted_pet_id"], "dog-red")
            self.assertEqual(len(result["candidates"]), 2)
            self.assertGreater(result["margin"], 0)

            pets = client.get("/v1/pets")
            self.assertEqual(pets.status_code, 200)
            self.assertEqual(pets.json()["count"], 2)

            image_id = enrolled["added_image_ids"][0]
            downloaded = client.get(f"/v1/pets/dog-red/images/{image_id}")
            self.assertEqual(downloaded.status_code, 200)
            self.assertEqual(downloaded.content, red)

            deleted_image = client.delete(f"/v1/pets/dog-red/images/{image_id}")
            self.assertEqual(deleted_image.status_code, 200)
            self.assertEqual(deleted_image.json()["remaining_references"], 1)
            deleted_pet = client.delete("/v1/pets/dog-red")
            self.assertEqual(deleted_pet.status_code, 200)
            self.assertEqual(deleted_pet.json()["deleted_images"], 1)

    def test_api_key_and_invalid_upload_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory) / "gallery")
            client = TestClient(create_app(service, api_key="secret"))
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/v1/pets").status_code, 401)
            self.assertEqual(
                client.get("/v1/pets", headers={"X-API-Key": "secret"}).status_code,
                200,
            )
            invalid = client.post(
                "/v1/pets/dog-one/images",
                headers={"X-API-Key": "secret"},
                files={"files": ("bad.jpg", b"not an image", "image/jpeg")},
            )
            self.assertEqual(invalid.status_code, 422, invalid.text)
            self.assertEqual(invalid.json()["error"]["code"], "invalid_pet_image")

    def test_threshold_can_reject_a_query(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory) / "gallery")
            client = TestClient(create_app(service))
            red = image_bytes((250, 10, 10))
            client.post(
                "/v1/pets/dog-red/images",
                files={"files": ("red.jpg", red, "image/jpeg")},
            )
            response = client.post(
                "/v1/identify?match_threshold=1.0",
                files={"file": ("green.jpg", image_bytes((10, 250, 10)), "image/jpeg")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(response.json()["accepted"])
            self.assertIsNone(response.json()["predicted_pet_id"])

    def test_reference_set_scoring_is_selectable_for_single_and_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory) / "gallery")
            client = TestClient(create_app(service))
            red = image_bytes((250, 10, 10))
            red_two = image_bytes((240, 20, 10), "PNG")
            green = image_bytes((10, 250, 10))
            client.post(
                "/v1/pets/dog-red/images",
                files=[
                    ("files", ("red.jpg", red, "image/jpeg")),
                    ("files", ("red-two.png", red_two, "image/png")),
                ],
            )
            client.post(
                "/v1/pets/dog-green/images",
                files={"files": ("green.jpg", green, "image/jpeg")},
            )

            identified = client.post(
                "/v1/identify?scoring_mode=reference_set&reference_top_k=1"
                "&reference_score_weight=1.0",
                files={"file": ("query.jpg", red, "image/jpeg")},
            )
            self.assertEqual(identified.status_code, 200, identified.text)
            result = identified.json()
            self.assertEqual(result["predicted_pet_id"], "dog-red")
            self.assertEqual(result["scoring"]["mode"], "reference_set")
            self.assertEqual(result["scoring"]["reference_top_k"], 1)
            self.assertEqual(result["scoring"]["reference_score_weight"], 1.0)
            self.assertEqual(result["diagnostics"]["scoring"]["mode"], "reference_set")

            invalid_mode = client.post(
                "/v1/identify?scoring_mode=not-a-mode",
                files={"file": ("query.jpg", red, "image/jpeg")},
            )
            self.assertEqual(invalid_mode.status_code, 400, invalid_mode.text)
            self.assertEqual(invalid_mode.json()["error"]["code"], "invalid_request")

            created = client.post(
                "/v1/batches?scoring_mode=reference_set&reference_top_k=2"
                "&reference_score_weight=0.25",
                data={"expected_pet_ids": ["dog-red"]},
                files=[("files", ("query.jpg", red, "image/jpeg"))],
            )
            self.assertEqual(created.status_code, 202, created.text)
            batch_id = created.json()["batch_id"]
            batch = None
            for _ in range(100):
                batch = client.get(f"/v1/batches/{batch_id}").json()
                if batch["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(batch)
            self.assertEqual(batch["status"], "completed", batch)
            self.assertEqual(batch["parameters"]["scoring_mode"], "reference_set")
            self.assertEqual(batch["parameters"]["reference_top_k"], 2)
            self.assertEqual(batch["parameters"]["reference_score_weight"], 0.25)
            history_id = batch["results"][0]["history_id"]
            history = client.get(f"/v1/history/{history_id}")
            self.assertEqual(history.status_code, 200, history.text)
            self.assertEqual(history.json()["result"]["scoring"]["mode"], "reference_set")

    def test_history_review_pet_edit_and_hard_case_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory) / "gallery")
            client = TestClient(create_app(service))
            red = image_bytes((250, 10, 10))
            client.post(
                "/v1/pets/dog-red/images",
                data={"display_name": "Red dog"},
                files={"files": ("red.jpg", red, "image/jpeg")},
            )

            identified = client.post(
                "/v1/identify",
                files={"file": ("query.jpg", red, "image/jpeg")},
            )
            self.assertEqual(identified.status_code, 200, identified.text)
            history_id = identified.json()["history_id"]
            self.assertGreaterEqual(identified.json()["latency_ms"], 0)

            history = client.get("/v1/history")
            self.assertEqual(history.status_code, 200, history.text)
            self.assertEqual(history.json()["total"], 1)
            self.assertEqual(history.json()["items"][0]["history_id"], history_id)
            image = client.get(f"/v1/history/{history_id}/image")
            self.assertEqual(image.status_code, 200, image.text)
            self.assertEqual(image.content, red)

            reviewed = client.patch(
                f"/v1/history/{history_id}/review",
                json={"status": "incorrect", "note": "manual check"},
            )
            self.assertEqual(reviewed.status_code, 200, reviewed.text)
            self.assertEqual(reviewed.json()["review_status"], "incorrect")
            hard_cases = client.get("/v1/hard-cases")
            self.assertEqual(hard_cases.status_code, 200, hard_cases.text)
            self.assertEqual(hard_cases.json()["total"], 1)

            updated = client.patch(
                "/v1/pets/dog-red", json={"display_name": "Ruby"}
            )
            self.assertEqual(updated.status_code, 200, updated.text)
            self.assertEqual(updated.json()["display_name"], "Ruby")
            self.assertIn("quality", updated.json()["images"][0])

    def test_batch_progress_metrics_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory) / "gallery")
            client = TestClient(create_app(service))
            red = image_bytes((250, 10, 10))
            green = image_bytes((10, 250, 10))
            client.post(
                "/v1/pets/dog-red/images",
                files={"files": ("red.jpg", red, "image/jpeg")},
            )
            client.post(
                "/v1/pets/dog-green/images",
                files={"files": ("green.jpg", green, "image/jpeg")},
            )
            created = client.post(
                "/v1/batches?top_k=2",
                data={
                    "name": "regression",
                    "expected_pet_ids": ["dog-red", "dog-green"],
                },
                files=[
                    ("files", ("red-query.jpg", red, "image/jpeg")),
                    ("files", ("green-query.jpg", green, "image/jpeg")),
                ],
            )
            self.assertEqual(created.status_code, 202, created.text)
            batch_id = created.json()["batch_id"]
            batch = None
            for _ in range(100):
                batch = client.get(f"/v1/batches/{batch_id}").json()
                if batch["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(batch)
            self.assertEqual(batch["status"], "completed", batch)
            self.assertEqual(batch["completed"], 2)
            self.assertEqual(batch["metrics"]["labelled"], 2)
            self.assertEqual(batch["metrics"]["top1_accuracy"], 1.0)
            self.assertEqual(len(batch["results"]), 2)
            csv_response = client.get(f"/v1/batches/{batch_id}/results.csv")
            self.assertEqual(csv_response.status_code, 200, csv_response.text)
            self.assertIn("expected_pet_id", csv_response.text)

    def test_gallery_backup_can_be_merged_into_an_empty_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_service(root / "source")
            source_client = TestClient(create_app(source))
            red = image_bytes((250, 10, 10))
            source_client.post(
                "/v1/pets/dog-red/images",
                data={"display_name": "Ruby"},
                files={"files": ("red.jpg", red, "image/jpeg")},
            )
            backup = source_client.get("/v1/gallery/backup")
            self.assertEqual(backup.status_code, 200, backup.text)

            target = self.make_service(root / "target")
            target_client = TestClient(create_app(target))
            restored = target_client.post(
                "/v1/gallery/restore",
                files={"file": ("gallery.zip", backup.content, "application/zip")},
            )
            self.assertEqual(restored.status_code, 200, restored.text)
            self.assertEqual(restored.json()["added_images"], 1)
            pet = target_client.get("/v1/pets/dog-red")
            self.assertEqual(pet.status_code, 200, pet.text)
            self.assertEqual(pet.json()["display_name"], "Ruby")

    def test_store_refuses_features_from_a_different_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gallery"
            self.make_service(root, "model-a")
            with self.assertRaises(GalleryModelMismatch):
                self.make_service(root, "model-b")

    def test_existing_npz_gallery_can_seed_the_incremental_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_image = root / "seed.jpg"
            source_image.write_bytes(image_bytes((250, 10, 10)))
            feature_path = root / "gallery_features.npz"
            fused = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
            nose = np.asarray([[1.0, 0.0]], dtype=np.float32)
            face = np.asarray([[1.0, 0.0]], dtype=np.float32)
            np.savez_compressed(
                feature_path,
                selected_fused_references=fused,
                selected_nose_references=nose,
                selected_face_references=face,
                reference_identity_indices=np.asarray([0], dtype=np.int64),
            )
            model_path = root / "gallery_model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "identities": ["seed-dog"],
                        "references": [
                            {
                                "path": str(source_image),
                                "sha256": sha256_file(source_image),
                                "selected_inference": {"detections": 1},
                            }
                        ],
                        "selected_backend": {
                            "model_sha256": "fake-model-fingerprint"
                        },
                        "features_file": feature_path.name,
                        "features_sha256": sha256_file(feature_path),
                    }
                ),
                encoding="utf-8",
            )
            service = self.make_service(root / "incremental")
            imported = service.import_gallery_model(model_path)
            self.assertEqual(imported["added"], 1)
            self.assertEqual(service.store.get_pet("seed-dog")["reference_count"], 1)
            imported_again = service.import_gallery_model(model_path)
            self.assertEqual(imported_again["added"], 0)
            self.assertEqual(imported_again["duplicates"], 1)


if __name__ == "__main__":
    unittest.main()
