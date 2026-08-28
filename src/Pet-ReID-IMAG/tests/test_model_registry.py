"""Structural consistency tests for the portable model registry."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from pet_id.workspace_paths import WORKSPACE_ROOT


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry_path = WORKSPACE_ROOT / "models/registry.json"
        cls.registry = load_json(cls.registry_path)

    def test_registry_paths_are_portable_and_files_match_sizes(self):
        self.assertEqual(self.registry["schema_version"], 1)
        names = [package["name"] for package in self.registry["packages"]]
        self.assertEqual(len(names), len(set(names)))
        for section in ("packages", "pretrained"):
            records = (
                [
                    artifact
                    for package in self.registry[section]
                    for artifact in package["artifacts"]
                ]
                if section == "packages"
                else self.registry[section]
            )
            for record in records:
                with self.subTest(path=record["path"]):
                    relative = Path(record["path"])
                    self.assertFalse(relative.is_absolute())
                    self.assertNotIn("..", relative.parts)
                    path = WORKSPACE_ROOT / relative
                    self.assertTrue(path.is_file(), path)
                    self.assertEqual(path.stat().st_size, record["bytes"])
                    self.assertRegex(record["sha256"], SHA256)

    def test_default_deployment_is_in_production_package(self):
        default = self.registry["default_deployment"]
        package = next(
            item
            for item in self.registry["packages"]
            if item["name"] == default["model_package"]
        )
        self.assertEqual(package["role"], "production")
        artifact_by_path = {
            artifact["path"]: artifact for artifact in package["artifacts"]
        }
        for key in ("config", "checkpoint", "onnx"):
            with self.subTest(key=key):
                self.assertIn(default[key], artifact_by_path)
                self.assertTrue((WORKSPACE_ROOT / default[key]).is_file())
        self.assertTrue((WORKSPACE_ROOT / default["seed_gallery"]).is_file())
        self.assertTrue((WORKSPACE_ROOT / default["persistent_gallery"]).is_dir())

    def test_production_metadata_hashes_agree(self):
        root = WORKSPACE_ROOT / "models/selected/dogfacenet_semantic_v3_v1"
        metadata = load_json(root / "onnx/metadata.json")
        deployment = load_json(root / "deployment_record.json")
        package = next(
            item
            for item in self.registry["packages"]
            if item["name"] == "dogfacenet_semantic_v3_v1"
        )
        artifacts = {
            Path(item["path"]).name: item for item in package["artifacts"]
        }
        self.assertEqual(metadata["fusion_mode"], "semantic_residual_v3")
        self.assertEqual(metadata["outputs"]["embedding"]["shape"], ["N", 512])
        self.assertEqual(metadata["onnx_sha256"], deployment["onnx"]["sha256"])
        self.assertEqual(
            metadata["source_checkpoint_sha256"],
            deployment["packaged_artifacts"]["model_final.pth"]["sha256"],
        )
        self.assertEqual(
            metadata["config_sha256"],
            deployment["packaged_artifacts"]["config.yaml"]["sha256"],
        )
        self.assertEqual(metadata["onnx_bytes"], (root / "onnx/pet_embedding.onnx").stat().st_size)
        self.assertEqual(
            sha256_file(root / "config.yaml"),
            deployment["packaged_artifacts"]["config.yaml"]["sha256"],
        )
        self.assertEqual(
            sha256_file(root / "onnx/metadata.json"),
            artifacts["metadata.json"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
