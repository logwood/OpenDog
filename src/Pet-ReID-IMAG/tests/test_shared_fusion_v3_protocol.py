import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SPLIT_MODULE = _load_module(
    "split_prepared_dogfacenet_protocol",
    "tools/split_prepared_dogfacenet_protocol.py",
)
V3_MODULE = _load_module(
    "build_shared_fusion_v3_protocol",
    "tools/build_shared_fusion_v3_protocol.py",
)


class SharedFusionV3ProtocolTest(unittest.TestCase):
    def _fixture(self, root: Path):
        records = []
        for identity_index in range(10):
            for image_index in range(2):
                path = root / f"{identity_index}_{image_index}.jpg"
                path.write_bytes(f"{identity_index}:{image_index}".encode())
                records.append(
                    {
                        "identity": f"dog-{identity_index}",
                        "source_path": str(path),
                        "canonical_filename": path.name,
                    }
                )
        source = {"schema_version": 1, "records": records}
        old_splits, old_audit = SPLIT_MODULE.split_protocol(
            source,
            train_identities=5,
            validation_identities=0,
            blind_identities=3,
            min_images_per_identity=2,
            seed=17,
        )
        return source, old_splits, old_audit

    def test_preserves_spent_test_and_uses_only_reserve_for_fresh_blind(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, old_splits, old_audit = self._fixture(Path(temporary))
            manifests, audit = V3_MODULE.build_v3_protocol(
                source,
                old_splits["train"],
                old_splits["blind_test"],
                old_audit,
                dev_train_identities=4,
                dev_validation_identities=1,
                dev_seed=23,
            )

            self.assertEqual(audit["splits"]["dev_train"]["identities"], 4)
            self.assertEqual(audit["splits"]["dev_validation"]["identities"], 1)
            self.assertEqual(audit["splits"]["spent_test"]["identities"], 3)
            self.assertEqual(audit["splits"]["fresh_blind"]["identities"], 2)
            self.assertEqual(audit["fresh_blind_status"], "LOCKED_UNSCORED")
            self.assertEqual(
                audit["splits"]["fresh_blind"]["identity_names"],
                old_audit["reserve_identities"],
            )
            self.assertEqual(
                manifests["spent_test"]["usage_policy"],
                "postmortem_only_prohibited_for_v3_model_selection",
            )
            for overlap in audit["pairwise_disjointness"].values():
                self.assertEqual(overlap["identity_overlap"], [])
                self.assertEqual(overlap["sha256_overlap"], [])

    def test_rejects_a_tampered_original_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, old_splits, old_audit = self._fixture(Path(temporary))
            old_audit["reserve_identities"] = old_audit["reserve_identities"][:-1]
            with self.assertRaisesRegex(ValueError, "reserve identities"):
                V3_MODULE.build_v3_protocol(
                    source,
                    old_splits["train"],
                    old_splits["blind_test"],
                    old_audit,
                    dev_train_identities=4,
                    dev_validation_identities=1,
                    dev_seed=23,
                )


if __name__ == "__main__":
    unittest.main()
