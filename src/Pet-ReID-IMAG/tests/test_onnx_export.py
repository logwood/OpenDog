# encoding: utf-8
"""Tests for the ONNX deployment boundary without requiring a large model."""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from pet_id.multimodal import LocalEndToEndPetIDModel
from pet_id.onnx_export import (
    PreCroppedPetEmbeddingModel,
    extract_precropped_onnx_inputs,
)


class _PixelMeanHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
        )


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


class ONNXExportBoundaryTest(unittest.TestCase):
    def test_precropped_wrapper_matches_full_joint_model(self):
        torch.manual_seed(7)
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            joint_enabled=True,
            joint_dim=5,
            adapter_bottleneck_dim=4,
            joint_initial_mix=0.1,
        ).eval()
        images = torch.rand(3, 3, 64, 64) * 255
        rois = torch.tensor(
            [[0, 8, 8, 56, 56], [1, 10, 9, 54, 55], [2, 7, 11, 57, 53]],
            dtype=torch.float32,
        )
        full_inputs = {
            "images_0_255": images,
            "face_rois": rois,
            "nose_rois": rois,
            "roll_angles_radians": torch.tensor([0.0, 0.1, -0.2]),
            "nose_masks": torch.rand(3, 1, 64, 64).gt(0.25).float(),
            "quality_signals": torch.tensor(
                [
                    [0.9, 0.8, 0.9, 0.8, 1.0, 1.0],
                    [0.0, 0.8, 0.9, 0.0, 0.0, 1.0],
                    [0.9, 0.0, 0.9, 0.8, 1.0, 0.0],
                ]
            ),
            "viewpoint_signals": torch.tensor(
                [[0.0, 0.0, 0.0, 0.7], [0.4, 0.2, -0.1, 0.8], [-0.5, 0.1, 0.2, 0.6]]
            ),
            "branch_available": torch.tensor(
                [[True, True], [False, True], [True, False]]
            ),
        }
        with torch.inference_mode():
            expected = model(**full_inputs)
            onnx_inputs = extract_precropped_onnx_inputs(model, **full_inputs)
            actual = PreCroppedPetEmbeddingModel(model)(*onnx_inputs)
        expected_values = (
            expected["features"],
            expected["nose_features"],
            expected["face_features"],
            expected["fusion_weights"],
            expected["joint_weights"],
            expected["viewpoint_frontality"],
        )
        for expected_value, actual_value in zip(expected_values, actual):
            torch.testing.assert_close(actual_value, expected_value)

    def test_precropped_wrapper_is_torch_exportable(self):
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            joint_enabled=True,
            joint_dim=5,
            adapter_bottleneck_dim=4,
        ).eval()
        wrapper = PreCroppedPetEmbeddingModel(model)
        inputs = (
            torch.rand(2, 3, 24, 24) * 255,
            torch.rand(2, 3, 20, 20) * 255,
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

    def test_precropped_wrapper_matches_shared_projection(self):
        torch.manual_seed(13)
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            fusion_mode="shared_projection",
            joint_dim=5,
            adapter_bottleneck_dim=4,
        ).eval()
        images = torch.rand(3, 3, 64, 64) * 255
        rois = torch.tensor(
            [[0, 8, 8, 56, 56], [1, 10, 9, 54, 55], [2, 7, 11, 57, 53]],
            dtype=torch.float32,
        )
        full_inputs = {
            "images_0_255": images,
            "face_rois": rois,
            "nose_rois": rois,
            "roll_angles_radians": torch.tensor([0.0, 0.1, -0.2]),
            "nose_masks": torch.rand(3, 1, 64, 64).gt(0.25).float(),
            "quality_signals": torch.ones(3, 6),
            "viewpoint_signals": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.7],
                    [0.4, 0.2, -0.1, 0.8],
                    [-0.5, 0.1, 0.2, 0.6],
                ]
            ),
            "branch_available": torch.tensor(
                [[True, True], [False, True], [True, False]]
            ),
        }
        with torch.inference_mode():
            expected = model(**full_inputs)
            onnx_inputs = extract_precropped_onnx_inputs(model, **full_inputs)
            actual = PreCroppedPetEmbeddingModel(model)(*onnx_inputs)
        expected_values = (
            expected["features"],
            expected["nose_features"],
            expected["face_features"],
            expected["fusion_weights"],
            expected["joint_weights"],
            expected["viewpoint_frontality"],
        )
        for expected_value, actual_value in zip(expected_values, actual):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual(tuple(actual[0].shape), (3, 5))

    def test_precropped_wrapper_matches_semantic_residual(self):
        torch.manual_seed(17)
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            fusion_mode="semantic_residual",
            joint_dim=3,
            adapter_bottleneck_dim=4,
            branch_priors=(0.10, 0.90),
            semantic_max_nose_weight=0.35,
            semantic_residual_scale=0.05,
        ).eval()
        images = torch.rand(3, 3, 64, 64) * 255
        rois = torch.tensor(
            [[0, 8, 8, 56, 56], [1, 10, 9, 54, 55], [2, 7, 11, 57, 53]],
            dtype=torch.float32,
        )
        full_inputs = {
            "images_0_255": images,
            "face_rois": rois,
            "nose_rois": rois,
            "roll_angles_radians": torch.tensor([0.0, 0.1, -0.2]),
            "nose_masks": torch.rand(3, 1, 64, 64).gt(0.25).float(),
            "quality_signals": torch.tensor(
                [
                    [0.9, 0.8, 0.9, 0.8, 1.0, 1.0],
                    [0.0, 0.8, 0.9, 0.0, 0.0, 1.0],
                    [0.9, 0.0, 0.9, 0.8, 1.0, 0.0],
                ]
            ),
            "viewpoint_signals": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.7],
                    [0.4, 0.2, -0.1, 0.8],
                    [-0.5, 0.1, 0.2, 0.6],
                ]
            ),
            "branch_available": torch.tensor(
                [[True, True], [False, True], [True, False]]
            ),
        }
        with torch.inference_mode():
            expected = model(**full_inputs)
            onnx_inputs = extract_precropped_onnx_inputs(model, **full_inputs)
            actual = PreCroppedPetEmbeddingModel(model)(*onnx_inputs)
        expected_values = (
            expected["features"],
            expected["nose_features"],
            expected["face_features"],
            expected["fusion_weights"],
            expected["joint_weights"],
            expected["viewpoint_frontality"],
        )
        for expected_value, actual_value in zip(expected_values, actual):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual(tuple(actual[0].shape), (3, 3))
        torch.testing.assert_close(actual[0][1], actual[2][1])
        self.assertTrue(torch.equal(actual[3][1], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(actual[3][2], torch.tensor([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
