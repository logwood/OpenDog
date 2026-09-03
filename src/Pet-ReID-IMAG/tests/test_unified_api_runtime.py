"""Regression tests for the one-file UnifiedPetReID API deployment."""

import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serve_pet_api_under_test",
    ROOT / "tools" / "serve_pet_api.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load tools/serve_pet_api.py")
SERVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVE)


class UnifiedApiRuntimeTest(unittest.TestCase):
    def test_parser_exposes_semantic_reference_set_scoring_defaults(self):
        args = SERVE.build_parser().parse_args([])
        self.assertEqual(args.scoring_mode, "centroid")
        self.assertEqual(args.reference_top_k, 3)
        self.assertAlmostEqual(args.reference_score_weight, 0.4)

        selected = SERVE.build_parser().parse_args(
            [
                "--scoring-mode",
                "reference_set",
                "--reference-top-k",
                "5",
                "--reference-score-weight",
                "0.75",
            ]
        )
        self.assertEqual(selected.scoring_mode, "reference_set")
        self.assertEqual(selected.reference_top_k, 5)
        self.assertAlmostEqual(selected.reference_score_weight, 0.75)

    def test_unified_backend_resolves_only_the_onnx_model(self):
        onnx_model = Path("runtime-only.onnx")
        forbidden = {
            Path("must-not-read.yaml"),
            Path("must-not-read.pth"),
            Path("must-not-read-detector.pth"),
        }
        args = argparse.Namespace(
            backend="unified-onnx",
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
        resolved = []

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
                "UnifiedONNXRuntimePipeline",
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
            SERVE.get_runtime_profile("production").public_metadata(),
        )
        runtime.assert_called_once_with(
            onnx_model,
            provider="cpu",
            device="cpu",
            profile=SERVE.get_runtime_profile("production"),
            warmup_batches=(1,),
        )


if __name__ == "__main__":
    unittest.main()
