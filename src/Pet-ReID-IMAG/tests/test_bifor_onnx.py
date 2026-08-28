"""Regression tests for the BIFOR ONNX fusion boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from pet_id.bifor_onnx import PreCroppedBIFORPetEmbeddingModel
from pet_id.body_detection import select_target_dog_box
from pet_id.multimodal import LocalEndToEndPetIDModel


class _PixelMeanHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))


class _FakeNoseEncoder(nn.Module):
    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.model = _PixelMeanHolder()
        self.projection = nn.Linear(3, self.feature_dim, bias=False)

    def forward(self, images):
        return F.normalize(self.projection(images.mean(dim=(2, 3))), dim=1)


class _FakeFaceEncoder(nn.Module):
    feature_dim = 3

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.feature_dim, bias=False)

    def forward(self, images):
        return F.normalize(self.projection(images.mean(dim=(2, 3))), dim=1)


class _FakeBodyEncoder(nn.Module):
    feature_dim = 2

    def forward(self, images):
        pooled = images.mean(dim=(2, 3))[:, : self.feature_dim]
        return {"global_features": F.normalize(pooled, dim=1)}


class BIFORONNXBoundaryTest(unittest.TestCase):
    def _wrapper(self, root: Path):
        torch.manual_seed(23)
        identity = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            fusion_mode="semantic_residual_v3",
            joint_dim=3,
            adapter_bottleneck_dim=4,
            branch_priors=(0.10, 0.90),
            semantic_max_nose_weight=0.35,
        ).eval()
        body = _FakeBodyEncoder().eval()
        input_dim = identity.fused_dim + body.feature_dim
        projection = torch.eye(input_dim)[:3]
        checkpoint = root / "fusion.pth"
        torch.save(
            {
                "architecture": {
                    "name": "lowrank_semantic_body_fusion_v1",
                    "output_dim": 3,
                },
                "body_weight": 0.08,
                "semantic_weight": 0.92,
                "mean": torch.zeros(1, input_dim),
                "projection": projection,
            },
            checkpoint,
        )
        return PreCroppedBIFORPetEmbeddingModel(identity, body, checkpoint).eval()

    def test_body_fusion_matches_locked_formula_and_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = self._wrapper(Path(directory))
            inputs = (
                torch.rand(2, 3, 24, 24) * 255,
                torch.rand(2, 3, 20, 20) * 255,
                torch.rand(2, 3, 22, 22) * 255,
                torch.ones(2, 1, 24, 24),
                torch.ones(2, 6),
                torch.zeros(2, 4),
                torch.ones(2, 2, dtype=torch.bool),
            )
            with torch.inference_mode():
                outputs = wrapper(*inputs)
                semantic = wrapper.semantic_model(inputs[0], inputs[1], *inputs[3:])[0]
                body = wrapper._body_features(inputs[2])
                joint = torch.cat((semantic * 0.92**0.5, body * 0.08**0.5), dim=1)
                expected = F.normalize(
                    F.linear(
                        joint - wrapper.fusion_mean,
                        wrapper.fusion_projection,
                    ),
                    dim=1,
                )
            torch.testing.assert_close(outputs[0], expected)
            torch.testing.assert_close(outputs[0].norm(dim=1), torch.ones(2))
            self.assertEqual(tuple(outputs[0].shape), (2, 3))

    def test_body_fusion_is_torch_exportable_with_dynamic_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = self._wrapper(Path(directory))
            inputs = (
                torch.rand(2, 3, 24, 24) * 255,
                torch.rand(2, 3, 20, 20) * 255,
                torch.rand(2, 3, 22, 22) * 255,
                torch.ones(2, 1, 24, 24),
                torch.ones(2, 6),
                torch.zeros(2, 4),
                torch.ones(2, 2, dtype=torch.bool),
            )
            batch = torch.export.Dim("batch", min=1, max=8)
            exported = torch.export.export(
                wrapper,
                inputs,
                dynamic_shapes=tuple({0: batch} for _ in inputs),
                strict=True,
            )
            self.assertGreater(len(list(exported.graph.nodes)), 0)

    def test_target_body_selection_prefers_face_containment(self):
        box, score = select_target_dog_box(
            torch.tensor([[0, 0, 20, 20], [30, 30, 90, 90]], dtype=torch.float32),
            torch.tensor([18, 18]),
            torch.tensor([0.99, 0.75]),
            [40, 40, 55, 55],
            dog_label=18,
            score_threshold=0.5,
        )
        self.assertEqual(box, [30.0, 30.0, 90.0, 90.0])
        self.assertAlmostEqual(score, 0.75)


if __name__ == "__main__":
    unittest.main()
