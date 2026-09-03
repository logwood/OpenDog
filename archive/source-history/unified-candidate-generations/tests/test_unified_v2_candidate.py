"""Tests for unified v2 acceptance and one-shot controls."""

import json
import tempfile
import unittest
from pathlib import Path

from pet_id.unified_training import load_acceptance
from pet_id.unified_v2_candidate import (
    complete_blind_attempt,
    reserve_blind_attempt,
)


class UnifiedV2CandidateTest(unittest.TestCase):
    def test_acceptance_loader_supports_v1_and_v2_with_explicit_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for protocol, schema in (
                ("unified_pet_reid_v1_noninferiority", 1),
                ("unified_pet_reid_v2_strict_noninferiority", 2),
                ("unified_pet_reid_v3_external_strict_noninferiority", 3),
            ):
                path = root / f"{schema}.json"
                path.write_text(
                    json.dumps({
                        "schema_version": schema,
                        "protocol_name": protocol,
                    }),
                    encoding="utf-8",
                )
                self.assertEqual(load_acceptance(path)["protocol_name"], protocol)
                self.assertEqual(
                    load_acceptance(path, expected_protocol=protocol)["schema_version"],
                    schema,
                )

    def test_acceptance_loader_rejects_wrong_expected_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acceptance.json"
            path.write_text(
                json.dumps({
                    "schema_version": 2,
                    "protocol_name": "unified_pet_reid_v2_strict_noninferiority",
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_acceptance(
                    path,
                    expected_protocol="unified_pet_reid_v1_noninferiority",
                )

    def test_blind_reservation_is_permanent_and_completable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "attempt.json"
            output = root / "result.json"
            reserved = reserve_blind_attempt(
                marker,
                output_path=output,
                candidate_lock_sha256="abc",
            )
            self.assertEqual(reserved, marker.resolve())
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "RUNNING")
            with self.assertRaises(FileExistsError):
                reserve_blind_attempt(
                    marker,
                    output_path=output,
                    candidate_lock_sha256="abc",
                )
            complete_blind_attempt(marker, "report-hash")
            completed = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "COMPLETED")
            self.assertEqual(completed["report_sha256"], "report-hash")


if __name__ == "__main__":
    unittest.main()
