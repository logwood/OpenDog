"""Tests for external unified v3 acceptance and one-shot controls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pet_id.unified_training import load_acceptance
from pet_id.unified_v3_candidate import (
    complete_blind_attempt,
    reserve_blind_attempt,
)


class UnifiedV3CandidateTest(unittest.TestCase):
    def test_v3_acceptance_loader_requires_the_exact_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acceptance.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "protocol_name": (
                            "unified_pet_reid_v3_external_strict_noninferiority"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            payload = load_acceptance(
                path,
                expected_protocol=(
                    "unified_pet_reid_v3_external_strict_noninferiority"
                ),
            )
            self.assertEqual(payload["schema_version"], 3)

    def test_blind_reservation_is_permanent_and_completable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "candidate.attempt.json"
            output = root / "blind.json"
            reserved = reserve_blind_attempt(
                marker,
                output_path=output,
                candidate_lock_sha256="locked",
            )
            self.assertEqual(reserved, marker.resolve())
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["status"],
                "RUNNING",
            )
            with self.assertRaises(FileExistsError):
                reserve_blind_attempt(
                    marker,
                    output_path=output,
                    candidate_lock_sha256="locked",
                )
            complete_blind_attempt(marker, "result")
            completed = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "COMPLETED")
            self.assertEqual(completed["report_sha256"], "result")


if __name__ == "__main__":
    unittest.main()
