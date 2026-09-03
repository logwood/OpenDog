"""API selection tests for the spatial-detail candidate runtime."""

from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serve_pet_api_highres_under_test",
    ROOT / "tools" / "serve_pet_api.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load tools/serve_pet_api.py")
SERVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVE)


class UnifiedHighresApiRuntimeTest(unittest.TestCase):
    def test_production_and_candidate_defaults_keep_model_spaces_separate(self):
        production = SERVE.get_runtime_profile("production")
        candidate = SERVE.get_runtime_profile("candidate")
        self.assertEqual(SERVE.default_onnx_model("production"), production.onnx)
        self.assertEqual(SERVE.default_onnx_model("candidate"), candidate.onnx)
        self.assertNotEqual(
            SERVE.default_storage_dir("production"),
            SERVE.default_storage_dir("candidate"),
        )
        self.assertEqual(
            SERVE.default_storage_dir("candidate"),
            candidate.persistent_gallery,
        )

    def test_candidate_backend_resolves_only_its_onnx(self):
        onnx_model = Path("highres-runtime-only.onnx")
        forbidden = {
            Path("must-not-read.yaml"),
            Path("must-not-read.pth"),
            Path("must-not-read-detector.pth"),
        }
        args = argparse.Namespace(
            backend="onnx-highres",
            onnx_model=onnx_model,
            onnx_provider="cpu",
            device="cpu",
            onnx_warmup_batches="1",
            config_file=Path("must-not-read.yaml"),
            identity_weights=Path("must-not-read.pth"),
            body_detector=Path("must-not-read-detector.pth"),
            agent=False,
            megadescriptor_checkpoint=Path("must-not-read-megadescriptor.pth"),
            megadescriptor_device=None,
        )
        resolved: list[Path] = []

        def resolve(path):
            path = Path(path)
            self.assertNotIn(path, forbidden)
            resolved.append(path)
            return path

        pipeline = object()
        encoder = SimpleNamespace()
        with (
            patch.object(SERVE, "resolve_legacy_path", side_effect=resolve),
            patch.object(
                SERVE,
                "UnifiedHighResolutionONNXRuntimePipeline",
                return_value=pipeline,
            ) as runtime,
            patch.object(
                SERVE,
                "MultimodalPipelineEncoder",
                return_value=encoder,
            ),
        ):
            actual_encoder, identity_weights = SERVE.build_runtime_encoder(args)

        self.assertIs(actual_encoder, encoder)
        self.assertIsNone(identity_weights)
        self.assertEqual(resolved, [onnx_model])
        self.assertEqual(
            encoder.profile_info,
            SERVE.get_runtime_profile("candidate").public_metadata(),
        )
        runtime.assert_called_once_with(
            onnx_model,
            provider="cpu",
            device="cpu",
            profile=SERVE.get_runtime_profile("candidate"),
            warmup_batches=(1,),
        )


if __name__ == "__main__":
    unittest.main()
