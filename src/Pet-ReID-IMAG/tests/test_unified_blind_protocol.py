"""Tests for locked one-shot UnifiedPetReID protocol guards."""

import json
import tempfile
import unittest
from pathlib import Path

from pet_id.unified_blind_protocol import (
    reserve_single_attempt,
    sha256_file,
    validate_disjoint_splits,
    validate_manifest,
)


class UnifiedBlindProtocolTest(unittest.TestCase):
    def _manifest(self, root: Path, name: str, split: str, identities: list[str]):
        records = []
        for identity in identities:
            for index in range(4):
                records.append(
                    {
                        "identity": identity,
                        "source_sha256": f"{name}-{identity}-{index}",
                    }
                )
        path = root / f"{name}.json"
        path.write_text(
            json.dumps({"protocol_split": split, "records": records}),
            encoding="utf-8",
        )
        return path

    def _validate(self, path: Path, split: str, identities: int):
        return validate_manifest(
            path,
            expected_sha256=sha256_file(path),
            expected_split=split,
            expected_records=identities * 4,
            expected_identities=identities,
            expected_images_per_identity=4,
        )

    def test_manifest_shape_and_disjointness_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = self._manifest(root, "train", "train", ["a", "b"])
            blind_path = self._manifest(root, "blind", "blind_test", ["c"])
            train = self._validate(train_path, "train", 2)
            blind = self._validate(blind_path, "blind_test", 1)
            validate_disjoint_splits(train, blind)

            overlap_path = self._manifest(root, "overlap", "blind_test", ["a"])
            overlap = self._validate(overlap_path, "blind_test", 1)
            with self.assertRaisesRegex(RuntimeError, "identity overlap"):
                validate_disjoint_splits(train, overlap)

    def test_wrong_manifest_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._manifest(Path(directory), "train", "train", ["a"])
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_manifest(
                    path,
                    expected_sha256="0" * 64,
                    expected_split="train",
                    expected_records=4,
                    expected_identities=1,
                    expected_images_per_identity=4,
                )

    def test_single_attempt_marker_is_atomic_and_permanent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            marker = reserve_single_attempt(
                output,
                candidate_lock_sha256="a" * 64,
            )
            self.assertTrue(marker.is_file())
            with self.assertRaises(FileExistsError):
                reserve_single_attempt(
                    output,
                    candidate_lock_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
