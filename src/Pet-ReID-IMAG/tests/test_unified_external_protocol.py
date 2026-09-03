from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from pet_id.unified_external_data import (
    UnifiedRawManifestDataset,
    identity_batches,
)
from pet_id.unified_external_protocol import (
    build_external_protocol,
    collect_identity_images,
    sha256_file,
    validate_raw_manifest,
)


class UnifiedExternalProtocolTest(unittest.TestCase):
    def test_collection_deduplicates_and_excludes_cross_identity_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for identity in ("0", "1", "2"):
                (root / identity).mkdir()
            (root / "0/a.jpg").write_bytes(b"same-inside")
            (root / "0/b.jpg").write_bytes(b"same-inside")
            (root / "0/c.jpg").write_bytes(b"unique-zero")
            (root / "1/a.jpg").write_bytes(b"cross")
            (root / "2/a.jpg").write_bytes(b"cross")

            groups, audit = collect_identity_images(
                root,
                dataset_namespace="fixture",
                source_split="train",
            )

            self.assertEqual(set(groups), {"fixture:0"})
            self.assertEqual(len(groups["fixture:0"]), 2)
            self.assertEqual(
                audit["cross_identity_conflicting_identities_excluded"],
                ["fixture:1", "fixture:2"],
            )
            self.assertEqual(
                len(audit["within_identity_exact_duplicates_excluded"]), 1
            )

    def test_build_is_disjoint_and_excludes_historical_identity(self):
        def rows(prefix: str, identity_count: int, images: int):
            result = {}
            for identity_index in range(identity_count):
                identity = f"fixture:{prefix}:{identity_index}"
                result[identity] = [
                    {
                        "identity": identity,
                        "source_path": f"/{identity}/{image_index}.jpg",
                        "source_sha256": f"{prefix}-{identity_index}-{image_index}",
                        "source_split": prefix,
                        "source_filename": f"{image_index}.jpg",
                    }
                    for image_index in range(images)
                ]
            return result

        training = rows("train", 8, 6)
        evaluation = rows("test", 8, 6)
        historical = {"train-0-0", "test-0-0"}
        manifests, audit = build_external_protocol(
            training_groups=training,
            evaluation_groups=evaluation,
            historical_sha256=historical,
            training_identities=4,
            development_identities=2,
            blind_identities=2,
            training_images_per_identity=4,
            evaluation_images_per_identity=4,
            seed=7,
        )

        self.assertEqual(len(manifests["training_extension"]["records"]), 16)
        self.assertEqual(len(manifests["development"]["records"]), 8)
        self.assertEqual(len(manifests["blind_test"]["records"]), 8)
        self.assertIn(
            "fixture:train:0",
            audit["historical_overlap"]["training_identities_excluded"],
        )
        self.assertIn(
            "fixture:test:0",
            audit["historical_overlap"]["evaluation_identities_excluded"],
        )
        for overlap in audit["pairwise_disjointness"].values():
            self.assertEqual(overlap["identity_overlap"], [])
            self.assertEqual(overlap["source_sha256_overlap"], [])

    def test_validate_raw_manifest_checks_source_hash_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for identity in ("a", "b"):
                for index in range(4):
                    path = root / f"{identity}-{index}.jpg"
                    path.write_bytes(f"{identity}-{index}".encode())
                    records.append(
                        {
                            "identity": identity,
                            "source_path": str(path),
                            "source_sha256": sha256_file(path),
                        }
                    )
            payload = {
                "protocol_split": "development",
                "images_per_identity": 4,
                "records": records,
            }
            self.assertEqual(
                validate_raw_manifest(payload, expected_split="development"),
                {"records": 8, "identities": 2},
            )
            records[0]["source_sha256"] = "incorrect"
            with self.assertRaises(RuntimeError):
                validate_raw_manifest(payload, expected_split="development")

    def test_raw_dataset_and_identity_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for identity in ("a", "b", "c", "d"):
                for index in range(2):
                    path = root / f"{identity}-{index}.jpg"
                    image = np.full((12, 20, 3), index * 40 + 20, np.uint8)
                    self.assertTrue(cv2.imwrite(str(path), image))
                    records.append(
                        {
                            "identity": identity,
                            "source_path": str(path),
                            "source_sha256": sha256_file(path),
                        }
                    )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "protocol_split": "training_extension",
                        "images_per_identity": 2,
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            dataset = UnifiedRawManifestDataset(manifest, input_size=32)
            self.assertEqual(dataset.num_classes, 4)
            self.assertEqual(tuple(dataset[0]["rgb"].shape), (3, 32, 32))
            batches = identity_batches(
                dataset, identities_per_batch=2, seed=11, epoch=0
            )
            self.assertEqual(len(batches), 2)
            self.assertTrue(all(len(batch) == 4 for batch in batches))
            for batch in batches:
                self.assertEqual(len({dataset.targets[index] for index in batch}), 2)


if __name__ == "__main__":
    unittest.main()
