from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pet_id.dogfacenet_alignment import AlignmentIndexRecord
from pet_id.unified_fresh_protocol import build_fresh_protocol


class UnifiedFreshProtocolTests(unittest.TestCase):
    def _record(
        self,
        root: Path,
        identity: str,
        index: int,
        *,
        content: bytes | None = None,
    ) -> AlignmentIndexRecord:
        path = root / f"{identity}.{index}.jpg"
        path.write_bytes(content if content is not None else f"{identity}:{index}".encode())
        return AlignmentIndexRecord(
            source_path=path,
            canonical_filename=path.name,
            identity=identity,
            left_eye=(10.0, 10.0),
            right_eye=(20.0 + index, 10.0),
            nose=(15.0, 20.0),
        )

    def test_deterministic_disjoint_split_and_duplicate_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for identity in (
                "history",
                "conflict-a",
                "conflict-b",
                "dog-1",
                "dog-2",
                "dog-3",
                "dog-4",
                "dog-5",
                "dog-6",
            ):
                records.extend(self._record(root, identity, index) for index in range(4))
            duplicate = b"same-image-under-two-identities"
            records.append(self._record(root, "conflict-a", 10, content=duplicate))
            records.append(self._record(root, "conflict-b", 10, content=duplicate))

            first, audit = build_fresh_protocol(
                records,
                historical_identities={"history"},
                training_identities=2,
                development_identities=2,
                blind_identities=2,
                seed=17,
            )
            second, _ = build_fresh_protocol(
                records,
                historical_identities={"history"},
                training_identities=2,
                development_identities=2,
                blind_identities=2,
                seed=17,
            )
            self.assertEqual(first, second)
            self.assertEqual(audit["eligible_identities"], 6)
            self.assertEqual(
                set(audit["cross_identity_conflicting_identities_excluded"]),
                {"conflict-a", "conflict-b"},
            )
            split_ids = [
                {row["identity"] for row in manifest["records"]}
                for manifest in first.values()
            ]
            self.assertFalse(split_ids[0] & split_ids[1])
            self.assertFalse(split_ids[0] & split_ids[2])
            self.assertFalse(split_ids[1] & split_ids[2])
            self.assertTrue(
                all(not row["identity"].startswith("conflict") for manifest in first.values() for row in manifest["records"])
            )


if __name__ == "__main__":
    unittest.main()
