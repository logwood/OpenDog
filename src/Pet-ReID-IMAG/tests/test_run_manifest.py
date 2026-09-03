"""Tests for standard run layout and lifecycle metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pet_id.run_manifest import (
    configure_standard_run,
    finalize_run_manifest,
    initialize_run_manifest,
    safe_slug,
)


class _Config(SimpleNamespace):
    def dump(self):
        return "SEED: 2022\n"


def _args(config_file: Path, **overrides):
    values = {
        "config_file": str(config_file),
        "run_workstream": "workspace-tests",
        "run_id": "fixed-run",
        "run_purpose": "smoke",
        "eval_only": False,
        "resume": False,
        "allow_checkpoint_cleanup": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RunManifestTest(unittest.TestCase):
    def test_safe_slug_rejects_empty_names(self):
        self.assertEqual(
            safe_slug(" semantic residual ", field="name"),
            "semantic-residual",
        )
        with self.assertRaises(ValueError):
            safe_slug("../", field="name")

    def test_standard_run_manifest_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "source.yaml"
            config_file.write_text("MODEL: {}\n", encoding="utf-8")
            split_dir = root / "processed/splits"
            split_dir.mkdir(parents=True)
            split_manifest = split_dir / "split_manifest.json"
            split_manifest.write_text('{"seed": 2022}\n', encoding="utf-8")
            cfg = _Config(
                MODEL=SimpleNamespace(META_ARCHITECTURE="Baseline"),
                DATASETS=SimpleNamespace(NAMES=("PetID",), TESTS=("PetIDValidation",)),
                SEED=2022,
                OUTPUT_DIR="unused",
            )
            args = _args(config_file)
            with (
                patch("pet_id.run_manifest.RUNS_ROOT", root / "runs"),
                patch("pet_id.run_manifest.PROCESSED_DATA_ROOT", root / "processed"),
                patch("pet_id.run_manifest._git_value", return_value="test-git-value"),
            ):
                self.assertTrue(configure_standard_run(cfg, args))
                run_dir = root / "runs/workspace-tests/fixed-run"
                self.assertEqual(Path(cfg.OUTPUT_DIR), run_dir.resolve())
                manifest_path = initialize_run_manifest(cfg, args)
                checkpoint = run_dir / "checkpoints/model_final.pth"
                checkpoint.write_bytes(b"checkpoint")
                finalize_run_manifest(
                    cfg, status="completed", result={"ROC_AUC": 0.99}
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["purpose"], "smoke")
            self.assertEqual(manifest["result"]["ROC_AUC"], 0.99)
            self.assertEqual(
                manifest["checkpoint_policy"]["selected_checkpoint"],
                str(checkpoint),
            )
            self.assertEqual(manifest["paths"]["stdout"], str(run_dir / "stdout.log"))
            self.assertTrue((run_dir / "resolved_config.yaml").is_file())
            self.assertTrue((run_dir / "reports").is_dir())
            self.assertTrue((run_dir / "tensorboard").is_dir())

    def test_fresh_run_allows_only_launcher_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/workspace-tests/fixed-run"
            run_dir.mkdir(parents=True)
            (run_dir / "stdout.log").write_text("launcher\n", encoding="utf-8")
            cfg = _Config(
                MODEL=SimpleNamespace(META_ARCHITECTURE="Baseline"),
                DATASETS=SimpleNamespace(NAMES=(), TESTS=()),
                SEED=1,
                OUTPUT_DIR="unused",
            )
            args = _args(root / "source.yaml")
            with patch("pet_id.run_manifest.RUNS_ROOT", root / "runs"):
                self.assertTrue(configure_standard_run(cfg, args))
                (run_dir / "unexpected.txt").write_text("stale", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    configure_standard_run(cfg, args)


if __name__ == "__main__":
    unittest.main()
