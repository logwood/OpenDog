"""Regression tests for the cleaned workspace path contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pet_id.workspace_paths import (
    LEGACY_RUNS_ROOT,
    LOCAL_GALLERY_ROOT,
    PRETRAINED_MODELS_ROOT,
    PROCESSED_DATA_ROOT,
    QUERY_INBOX_ROOT,
    SAM2_CONFIG_ROOT,
    SELECTED_MODELS_ROOT,
    SOURCE_ROOT,
    WORKSPACE_ROOT,
    normalize_runtime_config,
    resolve_legacy_path,
    resolve_sam2_config_path,
)


class _Config(SimpleNamespace):
    def __init__(self, **values):
        super().__init__(**values)
        self._frozen = True

    def is_frozen(self):
        return self._frozen

    def defrost(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True


class WorkspacePathsTest(unittest.TestCase):
    def test_legacy_source_paths_map_to_canonical_roots(self):
        self.assertEqual(
            resolve_legacy_path("logs/example/model.pth"),
            (LEGACY_RUNS_ROOT / "example" / "model.pth").resolve(),
        )
        self.assertEqual(
            resolve_legacy_path("models/example/model.pth"),
            (SELECTED_MODELS_ROOT / "example" / "model.pth").resolve(),
        )
        self.assertEqual(
            resolve_legacy_path("pretrain/resnest.pth"),
            (PRETRAINED_MODELS_ROOT / "resnest.pth").resolve(),
        )
        self.assertEqual(
            resolve_legacy_path("data/test/test_data.csv"),
            (PROCESSED_DATA_ROOT / "test" / "test_data.csv").resolve(),
        )
        self.assertEqual(
            resolve_legacy_path("../DogFaceNet_alignment/images/a.jpg"),
            (WORKSPACE_ROOT / "data/raw/DogFaceNet_alignment/images/a.jpg").resolve(),
        )
        self.assertEqual(
            resolve_legacy_path("../new-images/query.jpg"),
            (QUERY_INBOX_ROOT / "query.jpg").resolve(),
        )

    def test_canonical_paths_are_idempotent(self):
        paths = (
            SOURCE_ROOT / "configs/modern_smoke.yaml",
            WORKSPACE_ROOT / "docs/PET_API.md",
            WORKSPACE_ROOT / "data/raw/example.jpg",
            SELECTED_MODELS_ROOT / "example/config.yaml",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(resolve_legacy_path(path), path.resolve())

    def test_absolute_pre_cleanup_paths_remain_recoverable(self):
        old_source = WORKSPACE_ROOT / "upstream/Pet-ReID-IMAG/configs/baseline.yaml"
        old_gallery = WORKSPACE_ROOT / "1/reference.jpg"
        self.assertEqual(
            resolve_legacy_path(old_source),
            (SOURCE_ROOT / "configs/baseline.yaml").resolve(),
        )
        self.assertEqual(
            resolve_legacy_path(old_gallery),
            (LOCAL_GALLERY_ROOT / "local-1/reference.jpg").resolve(),
        )

    def test_sam2_hydra_config_spellings_resolve_to_vendor_package(self):
        expected = (SAM2_CONFIG_ROOT / "sam2.1/sam2.1_hiera_t.yaml").resolve()
        for value in (
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1/sam2.1_hiera_t.yaml",
            "third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
            expected,
        ):
            with self.subTest(value=value):
                self.assertEqual(resolve_sam2_config_path(value), expected)

    def test_runtime_config_is_rewritten_and_refrozen(self):
        multimodal = SimpleNamespace(
            NOSE_CONFIG="configs/baseline.yaml",
            NOSE_WEIGHTS="logs/nose/model.pth",
            IDENTITY_WEIGHTS="models/identity/model.pth",
            ARCFACE_WEIGHTS="dog.pt",
            ANYFACE_ROOT="third_party/AnyFace",
            ANYFACE_WEIGHTS="third_party/AnyFace/yolov5-face/weights/yolo.pt",
            SAM2_CHECKPOINT="third_party/sam2/checkpoints/sam2.pt",
            SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml",
            CACHE_DIR="logs/cache",
        )
        cfg = _Config(
            OUTPUT_DIR="logs/run",
            MODEL=SimpleNamespace(
                WEIGHTS="models/model/model.pth",
                BACKBONE=SimpleNamespace(PRETRAIN_PATH="pretrain/resnest.pth"),
            ),
            MULTIMODAL=multimodal,
        )
        normalize_runtime_config(cfg)
        self.assertTrue(cfg.is_frozen())
        self.assertEqual(cfg.OUTPUT_DIR, str((LEGACY_RUNS_ROOT / "run").resolve()))
        self.assertEqual(
            cfg.MODEL.WEIGHTS,
            str((SELECTED_MODELS_ROOT / "model/model.pth").resolve()),
        )
        self.assertEqual(
            cfg.MODEL.BACKBONE.PRETRAIN_PATH,
            str((PRETRAINED_MODELS_ROOT / "resnest.pth").resolve()),
        )
        self.assertEqual(
            cfg.MULTIMODAL.SAM2_CONFIG,
            str((SAM2_CONFIG_ROOT / "sam2.1/sam2.1_hiera_t.yaml").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
