# encoding: utf-8
"""Tests for post-blind semantic-v3 deployment packaging."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "package_semantic_v3_deployment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_semantic_v3_packaging_tool", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
PACKAGING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGING
SPEC.loader.exec_module(PACKAGING)


class SemanticV3PackagingTest(unittest.TestCase):
    def _fixtures(self, root: Path) -> dict[str, Path]:
        checkpoint = root / "source.pth"
        config = root / "config.yaml"
        model_lock = root / "model_lock.json"
        completion = root / "completion.json"
        checkpoint.write_bytes(b"locked-model")
        config.write_text("MULTIMODAL: {}\n", encoding="utf-8")
        checkpoint_hash = PACKAGING.sha256_file(checkpoint)
        config_hash = PACKAGING.sha256_file(config)
        model_lock.write_text(
            json.dumps(
                {
                    "status": "LOCKED_BEFORE_FRESH_BLIND_EVALUATION",
                    "artifacts": {
                        "model_final.pth": {"sha256": checkpoint_hash},
                        "config": {"sha256": config_hash},
                    },
                    "decision": {"candidate_recipe": "v3b-500"},
                }
            ),
            encoding="utf-8",
        )
        completion.write_text(
            json.dumps(
                {
                    "identities": 64,
                    "queries": 128,
                    "model_lock": {"sha256": PACKAGING.sha256_file(model_lock)},
                    "selected_checkpoint": {
                        "sha256_before_and_after_evaluation": checkpoint_hash,
                        "unchanged": True,
                    },
                    "acceptance": {"selected_for_deployment": "semantic_residual_v3"},
                    "no_post_blind_training_or_model_selection": True,
                    "evaluations": {
                        "legacy_production": {
                            "fused": {
                                "top1_accuracy": 0.96,
                                "top5_accuracy": 1.0,
                            }
                        },
                        "semantic_residual_v3": {
                            "fused": {
                                "top1_accuracy": 0.96,
                                "top5_accuracy": 1.0,
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "checkpoint": checkpoint,
            "config": config,
            "model_lock": model_lock,
            "blind_completion": completion,
        }

    def test_package_preserves_hashes_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = self._fixtures(root)
            output = root / "package"
            record = PACKAGING.package_deployment(**fixtures, output_dir=output)
            self.assertEqual(record["embedding_dim"], 512)
            self.assertEqual(record["onnx"]["status"], "pending_export")
            self.assertEqual(
                PACKAGING.sha256_file(output / "model_final.pth"),
                PACKAGING.sha256_file(fixtures["checkpoint"]),
            )
            with self.assertRaises(FileExistsError):
                PACKAGING.package_deployment(**fixtures, output_dir=output)

    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = self._fixtures(root)
            fixtures["checkpoint"].write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "pre-blind model lock"):
                PACKAGING.package_deployment(**fixtures, output_dir=root / "package")


if __name__ == "__main__":
    unittest.main()
