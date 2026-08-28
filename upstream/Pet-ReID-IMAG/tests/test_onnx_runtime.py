"""Tests for the deployable ONNX Runtime identity adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pet_id.onnx_export import ONNX_INPUT_NAMES, ONNX_OUTPUT_NAMES
from pet_id.onnx_runtime import (
    ONNXRuntimeIdentityModel,
    parse_warmup_batches,
    resolve_execution_provider,
)


class _NodeArg:
    def __init__(self, name, shape, type_name="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_name


class _FakeCPUSession:
    def __init__(self, embedding_dim=3072):
        self.runs = 0
        self.embedding_dim = int(embedding_dim)
        self._inputs = [
            _NodeArg("nose_crop", ["batch", 3, 24, 24]),
            _NodeArg("face_crop", ["batch", 3, 20, 20]),
            _NodeArg("nose_mask", ["batch", 1, 24, 24]),
            _NodeArg("quality_signals", ["batch", 6]),
            _NodeArg("viewpoint_signals", ["batch", 4]),
            _NodeArg("branch_available", ["batch", 2], "tensor(bool)"),
        ]
        self._outputs = [
            _NodeArg("embedding", ["batch", self.embedding_dim]),
            _NodeArg("nose_embedding", ["batch", 2048]),
            _NodeArg("face_embedding", ["batch", 512]),
            _NodeArg("fusion_weights", ["batch", 2]),
            _NodeArg("joint_weights", ["batch", 2]),
            _NodeArg("viewpoint_frontality", ["batch"]),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feeds):
        self.runs += 1
        if tuple(feeds) != ONNX_INPUT_NAMES:
            raise AssertionError(tuple(feeds))
        if tuple(output_names) != ONNX_OUTPUT_NAMES:
            raise AssertionError(tuple(output_names))
        batch = feeds["quality_signals"].shape[0]
        available = feeds["branch_available"].astype(np.float32)
        weights = available / available.sum(axis=1, keepdims=True)
        return [
            np.full(
                (batch, self.embedding_dim),
                1.0 / np.sqrt(self.embedding_dim),
                dtype=np.float32,
            ),
            np.full((batch, 2048), 1.0 / np.sqrt(2048), dtype=np.float32),
            np.full((batch, 512), 1.0 / np.sqrt(512), dtype=np.float32),
            weights,
            weights,
            np.ones((batch,), dtype=np.float32),
        ]


class ONNXRuntimeIdentityTest(unittest.TestCase):
    def _package(
        self,
        root: Path,
        *,
        embedding_dim: int | None = None,
        fusion_mode: str | None = None,
    ) -> Path:
        model_path = root / "pet_embedding.onnx"
        model_path.write_bytes(b"fake-onnx-model")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata = {
            "onnx_sha256": digest,
            "inputs": {
                "nose_crop": {"shape": ["N", 3, 24, 24]},
                "face_crop": {"shape": ["N", 3, 20, 20]},
            },
        }
        if embedding_dim is not None:
            metadata["outputs"] = {
                "embedding": {"shape": ["N", embedding_dim]}
            }
        if fusion_mode is not None:
            metadata["fusion_mode"] = fusion_mode
        (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return model_path

    def test_provider_resolution_never_downgrades_explicit_cuda(self):
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            resolve_execution_provider(
                "cuda",
                ["CPUExecutionProvider"],
                torch_cuda_available=True,
            )
        self.assertEqual(
            resolve_execution_provider(
                "auto",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                torch_cuda_available=True,
            ),
            "cuda",
        )
        self.assertEqual(
            resolve_execution_provider(
                "auto",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                torch_cuda_available=False,
            ),
            "cpu",
        )

    def test_parse_warmup_batches_is_ordered_and_unique(self):
        self.assertEqual(parse_warmup_batches("1,4,1,8"), (1, 4, 8))
        self.assertEqual(parse_warmup_batches(""), ())
        with self.assertRaises(ValueError):
            parse_warmup_batches("1,0")

    def test_cpu_adapter_crops_and_maps_runtime_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._package(Path(directory))
            session = _FakeCPUSession()
            with (
                mock.patch(
                    "onnxruntime.get_available_providers",
                    return_value=["CPUExecutionProvider"],
                ),
                mock.patch("onnxruntime.InferenceSession", return_value=session),
            ):
                model = ONNXRuntimeIdentityModel(
                    model_path,
                    provider="cpu",
                    device="cpu",
                    warmup_batches=(1,),
                )
            images = torch.rand(2, 3, 64, 64) * 255
            rois = torch.tensor(
                [[0, 5, 7, 55, 58], [1, 8, 6, 59, 57]],
                dtype=torch.float32,
            )
            with torch.inference_mode():
                output = model(
                    images,
                    face_rois=rois,
                    nose_rois=rois,
                    roll_angles_radians=torch.tensor([0.0, 0.1]),
                    nose_masks=torch.ones(2, 1, 64, 64),
                    quality_signals=torch.ones(2, 6),
                    viewpoint_signals=torch.zeros(2, 4),
                    branch_available=torch.tensor(
                        [[True, True], [False, True]], dtype=torch.bool
                    ),
                )
            self.assertEqual(tuple(output["features"].shape), (2, 3072))
            torch.testing.assert_close(output["features"].norm(dim=1), torch.ones(2))
            torch.testing.assert_close(
                output["fusion_weights"],
                torch.tensor([[0.5, 0.5], [0.0, 1.0]]),
            )
            self.assertEqual(model.backend_info()["provider"], "CPUExecutionProvider")
            self.assertEqual(session.runs, 2)

    def test_metadata_hash_mismatch_fails_before_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = self._package(root)
            model_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                ONNXRuntimeIdentityModel(
                    model_path,
                    provider="cpu",
                    device="cpu",
                )

    def test_cpu_adapter_accepts_shared_space_v2_embedding_width(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._package(
                Path(directory),
                embedding_dim=512,
                fusion_mode="shared_space_v2",
            )
            session = _FakeCPUSession(embedding_dim=512)
            with (
                mock.patch(
                    "onnxruntime.get_available_providers",
                    return_value=["CPUExecutionProvider"],
                ),
                mock.patch("onnxruntime.InferenceSession", return_value=session),
            ):
                model = ONNXRuntimeIdentityModel(
                    model_path,
                    provider="cpu",
                    device="cpu",
                )
            images = torch.rand(1, 3, 64, 64) * 255
            rois = torch.tensor([[0, 5, 7, 55, 58]], dtype=torch.float32)
            with torch.inference_mode():
                output = model(
                    images,
                    face_rois=rois,
                    nose_rois=rois,
                    roll_angles_radians=torch.zeros(1),
                    nose_masks=torch.ones(1, 1, 64, 64),
                    quality_signals=torch.ones(1, 6),
                    viewpoint_signals=torch.zeros(1, 4),
                    branch_available=torch.ones(1, 2, dtype=torch.bool),
                )
            self.assertEqual(tuple(output["features"].shape), (1, 512))
            self.assertEqual(model.backend_info()["embedding_dim"], 512)
            self.assertEqual(
                model.backend_info()["fusion_mode"],
                "shared_space_v2",
            )

    def test_source_checkpoint_must_match_export_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = self._package(root)
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_checkpoint_sha256"] = hashlib.sha256(
                b"expected-checkpoint"
            ).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            checkpoint = root / "model_final.pth"
            checkpoint.write_bytes(b"different-checkpoint")
            with self.assertRaisesRegex(ValueError, "source checkpoint hash"):
                ONNXRuntimeIdentityModel(
                    model_path,
                    provider="cpu",
                    device="cpu",
                    source_checkpoint=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
