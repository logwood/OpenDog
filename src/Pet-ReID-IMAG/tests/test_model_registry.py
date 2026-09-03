"""Structural consistency tests for the portable model registry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path

from pet_id.model_profiles import get_runtime_profile
from pet_id.release_compatibility import normalize_fusion_mode
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
        cls.production = get_runtime_profile("production")
        cls.candidate = get_runtime_profile("candidate")
        cls.legacy_semantic = get_runtime_profile("legacy-semantic")

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

    def test_model_roles_separate_current_development_from_production(self):
        roles = self.registry["model_roles"]
        self.assertEqual(
            roles["current_development"]["capability"],
            "spatial-detail-embedding",
        )
        self.assertEqual(
            roles["current_development"]["model_package"],
            self.candidate.model_package,
        )
        self.assertEqual(
            roles["production_baseline"]["capability"],
            "unified-embedding",
        )
        self.assertEqual(
            roles["production_baseline"]["model_package"],
            self.registry["default_deployment"]["model_package"],
        )
        self.assertEqual(
            roles["rollback"]["model_package"],
            roles["production_baseline"]["model_package"],
        )
        self.assertNotEqual(
            roles["current_development"]["model_package"],
            roles["production_baseline"]["model_package"],
        )
        for role in roles.values():
            self.assertNotIn("model_generation", role)
            self.assertNotIn("model_family", role)

    def test_default_deployment_is_in_production_baseline_package(self):
        default = self.registry["default_deployment"]
        package = next(
            item
            for item in self.registry["packages"]
            if item["name"] == default["model_package"]
        )
        self.assertEqual(package["role"], "production")
        self.assertEqual(package["deployment_role"], "production_baseline")
        artifact_by_path = {
            artifact["path"]: artifact for artifact in package["artifacts"]
        }
        for key in ("onnx",):
            with self.subTest(key=key):
                self.assertIn(default[key], artifact_by_path)
                self.assertTrue((WORKSPACE_ROOT / default[key]).is_file())
        self.assertEqual(default["backend"], "unified-onnx")
        self.assertEqual(
            default["onnx"],
            self.production.onnx.relative_to(WORKSPACE_ROOT).as_posix(),
        )
        self.assertNotIn("checkpoint", default)
        self.assertEqual(default["runtime_required_files"], [default["onnx"]])
        self.assertEqual(default["runtime_external_models"], [])
        self.assertNotIn("seed_gallery", default)
        e2e_metadata = load_json(
            self.production.onnx.parent / "metadata.json"
        )
        self.assertTrue(e2e_metadata["raw_spatial_input"])
        self.assertEqual(
            e2e_metadata["runtime_contract"]["inputs"]["rgb"]["value_range"],
            [0, 255],
        )
        self.assertEqual(
            e2e_metadata["runtime_contract"]["outputs"]["embedding"]["shape"],
            ["N", 512],
        )
        self.assertTrue(
            e2e_metadata["runtime_contract"]["outputs"]["embedding"]["l2_normalized"]
        )
        self.assertEqual(e2e_metadata["runtime_contract"]["external_models"], [])
        self.assertFalse(e2e_metadata["blind_data_used"])
        development = package["metrics"]["development"]
        self.assertTrue(development["noninferiority_passed"])
        self.assertGreaterEqual(
            development["candidate_top1_correct"],
            development["parent_top1_correct"],
        )
        self.assertGreaterEqual(
            development["candidate_top5_correct"],
            development["parent_top5_correct"],
        )
        blind = package["metrics"]["blind"]
        self.assertTrue(blind["passed"])
        self.assertGreaterEqual(
            blind["candidate_top1_correct"],
            blind["minimum_top1_correct"],
        )
        self.assertGreaterEqual(
            blind["candidate_top5_correct"],
            blind["minimum_top5_correct"],
        )
        gallery = Path(default["persistent_gallery"])
        self.assertFalse(gallery.is_absolute())
        self.assertNotIn("..", gallery.parts)

    def test_unified_locked_fixed_square_release_hashes_agree(self):
        root = (
            WORKSPACE_ROOT
            / "models"
            / "selected"
            / self.production.model_package
        )
        metadata = load_json(root / "onnx/metadata.json")
        deployment = load_json(root / "deployment_record.json")
        package = next(
            item
            for item in self.registry["packages"]
            if item["name"] == self.production.model_package
        )
        artifacts = {item["path"]: item for item in package["artifacts"]}
        onnx_file = root / "onnx" / "unified_pet_reid.onnx"
        checkpoint_file = root / "model_final.pth"
        onnx_path = onnx_file.relative_to(WORKSPACE_ROOT).as_posix()
        checkpoint_path = checkpoint_file.relative_to(WORKSPACE_ROOT).as_posix()
        self.assertTrue(deployment["architecture"]["single_onnx_graph"])
        self.assertEqual(deployment["architecture"]["runtime_external_models"], [])
        self.assertEqual(metadata["external_models"], [])
        self.assertEqual(metadata["outputs"]["embedding"]["shape"], ["N", 512])
        self.assertEqual(
            metadata["onnx_sha256"], deployment["selected_artifacts"]["onnx"]["sha256"]
        )
        self.assertEqual(
            metadata["source_checkpoint_sha256"],
            deployment["selected_artifacts"]["checkpoint"]["sha256"],
        )
        self.assertEqual(
            sha256_file(onnx_file),
            artifacts[onnx_path]["sha256"],
        )
        self.assertEqual(
            sha256_file(checkpoint_file),
            artifacts[checkpoint_path]["sha256"],
        )

    def test_unified_release_evidence_is_excluded_from_metadata_refresh(self):
        script_path = WORKSPACE_ROOT / "scripts/generate_workspace_metadata.py"
        spec = importlib.util.spec_from_file_location(
            "generate_workspace_metadata_for_test", script_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        deployment = load_json(WORKSPACE_ROOT / "models/deployment.json")
        package_names = tuple(
            name
            for name, policy in deployment["packages"].items()
            if policy.get("immutable_metadata") is True
        )
        before = {
            path: sha256_file(path)
            for name in package_names
            for path in (WORKSPACE_ROOT / "models/selected" / name).rglob("*.json")
        }

        module.refresh_selected_metadata()

        self.assertEqual(
            {path: sha256_file(path) for path in before},
            before,
        )

    def test_legacy_semantic_metadata_hashes_agree(self):
        root = (
            WORKSPACE_ROOT
            / "models"
            / "selected"
            / self.legacy_semantic.model_package
        )
        metadata = load_json(root / "onnx/metadata.json")
        deployment = load_json(root / "deployment_record.json")
        package = next(
            item
            for item in self.registry["packages"]
            if item["name"] == self.legacy_semantic.model_package
        )
        artifacts = {
            Path(item["path"]).name: item for item in package["artifacts"]
        }
        self.assertEqual(
            normalize_fusion_mode(metadata["fusion_mode"]),
            "semantic_residual",
        )
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
