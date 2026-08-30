"""Tests for safe model-space Gallery migration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pet_id.gallery_migration import GalleryMigrationError, migrate_gallery
from pet_id.gallery_service import (
    EncodedPetImage,
    PetGalleryStore,
    PetIdentificationService,
    UploadPayload,
)


class FakeEncoder:
    def __init__(self, fingerprint: str, *, fail: bool = False):
        self.fingerprint = fingerprint
        self.fail = fail

    def backend_info(self):
        return {
            "backend": "fake",
            "embedding_dim": 512,
            "model_sha256": self.fingerprint,
        }

    def encode_file(self, path: Path) -> EncodedPetImage:
        if self.fail:
            raise RuntimeError("intentional encoder failure")
        with Image.open(path) as image:
            mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        fused = np.zeros(512, dtype=np.float32)
        fused[:3] = mean + 1.0
        nose = np.zeros(8, dtype=np.float32)
        nose[:3] = mean + 1.0
        face = np.zeros(6, dtype=np.float32)
        face[:3] = mean + 1.0
        return EncodedPetImage(
            fused=fused,
            nose=nose,
            face=face,
            metadata={"detections": 1},
        )


def image_payload(root: Path, name: str, color: tuple[int, int, int]) -> UploadPayload:
    path = root / name
    Image.new("RGB", (32, 24), color).save(path)
    return UploadPayload(filename=name, content_type="image/png", data=path.read_bytes())


class GalleryMigrationTest(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "source"
        service = PetIdentificationService(
            PetGalleryStore(source), FakeEncoder("old-model")
        )
        service.enroll(
            "pet-a",
            [
                image_payload(root, "a1.png", (200, 20, 10)),
                image_payload(root, "a2.png", (180, 30, 20)),
            ],
            display_name="Alpha",
        )
        service.enroll(
            "pet-b",
            [image_payload(root, "b1.png", (10, 20, 200))],
            display_name="Beta",
        )
        return source

    def test_reencodes_and_atomically_publishes_separate_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            target = root / "target"
            report = migrate_gallery(source, target, FakeEncoder("new-model"))

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["source"]["model_fingerprint"], "old-model")
            self.assertEqual(report["target"]["model_fingerprint"], "new-model")
            self.assertEqual(report["target"]["pets"], 2)
            self.assertEqual(report["target"]["reference_images"], 3)
            self.assertTrue((target / "migration_report.json").is_file())
            self.assertEqual(
                PetGalleryStore(source).metadata()["model_fingerprint"], "old-model"
            )
            migrated = PetGalleryStore(target)
            self.assertEqual(migrated.metadata()["model_fingerprint"], "new-model")
            self.assertEqual(migrated.get_pet("pet-a")["display_name"], "Alpha")
            self.assertEqual(migrated.summary()["reference_images"], 3)

    def test_existing_target_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            target = root / "target"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(GalleryMigrationError):
                migrate_gallery(source, target, FakeEncoder("new-model"))
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_failure_does_not_publish_partial_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            target = root / "target"
            with self.assertRaises(Exception):
                migrate_gallery(source, target, FakeEncoder("new-model", fail=True))
            self.assertFalse(target.exists())
            self.assertEqual(
                list(root.glob(".target.migration-*.tmp")), [], "staging must be removed"
            )
            self.assertEqual(PetGalleryStore(source).summary()["reference_images"], 3)


if __name__ == "__main__":
    unittest.main()
