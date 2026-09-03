import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "split_prepared_dogfacenet_protocol.py"
SPEC = importlib.util.spec_from_file_location("split_prepared_dogfacenet_protocol", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
split_protocol = MODULE.split_protocol


class DogFaceNetProtocolSplitTest(unittest.TestCase):
    def test_identity_and_hash_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for identity_index in range(7):
                for image_index in range(2):
                    path = root / f"{identity_index}_{image_index}.jpg"
                    path.write_bytes(f"{identity_index}:{image_index}".encode())
                    records.append(
                        {
                            "identity": f"dog-{identity_index}",
                            "source_path": str(path),
                            "canonical_filename": path.name,
                            "eye_distance": 100 + image_index,
                        }
                    )
            splits, audit = split_protocol(
                {"records": records},
                train_identities=3,
                validation_identities=2,
                blind_identities=1,
                min_images_per_identity=2,
                seed=7,
            )
            self.assertEqual(len(splits["train"]["records"]), 6)
            self.assertEqual(len(splits["validation"]["records"]), 4)
            self.assertEqual(len(splits["blind_test"]["records"]), 2)
            for result in audit["pairwise_disjointness"].values():
                self.assertEqual(result["identity_overlap"], [])
                self.assertEqual(result["sha256_overlap"], [])
            self.assertEqual(len(audit["reserve_identities"]), 1)

    def test_cross_identity_exact_duplicate_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for identity in ("a", "b", "c"):
                for image_index in range(2):
                    path = root / f"{identity}_{image_index}.jpg"
                    path.write_bytes(
                        b"shared" if image_index == 0 and identity in {"a", "b"}
                        else f"{identity}:{image_index}".encode()
                    )
                    records.append(
                        {
                            "identity": identity,
                            "source_path": str(path),
                            "canonical_filename": path.name,
                        }
                    )
            splits, audit = split_protocol(
                {"records": records},
                train_identities=1,
                validation_identities=0,
                blind_identities=0,
                min_images_per_identity=2,
                seed=1,
            )
            self.assertEqual({item["identity"] for item in splits["train"]["records"]}, {"c"})
            self.assertEqual(audit["conflicting_identities_excluded"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
