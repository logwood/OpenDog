# encoding: utf-8
"""Unit tests for the one-input UnifiedPetReID building blocks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pet_id.unified import (
    BoundedSemanticResidual,
    GeometryPrediction,
    Layer2Layer3GeometryAdapter,
    NormalizedRotatedCropper,
    PartGeometryCalibration,
    SoftPartGeometryHead,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
)
from pet_id.multimodal import DifferentiableROICropper
from pet_id.unified_data import (
    UnifiedTeacherCache,
    letterbox_rgb,
    transform_box_to_letterbox,
)


class UnifiedModelTest(unittest.TestCase):
    def test_box_round_trip(self):
        boxes = torch.tensor([[0.1, 0.2, 0.8, 0.9]], dtype=torch.float32)
        actual = cxcywh_to_xyxy(xyxy_to_cxcywh(boxes))
        torch.testing.assert_close(actual, boxes)

    def test_part_geometry_calibration_is_identity_and_differentiable(self):
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.3], [0.5, 0.5, 0.1, 0.1]]]
        )
        prediction = GeometryPrediction(
            boxes_cxcywh=boxes,
            angle_radians=torch.zeros(1),
            attention=torch.zeros(1, 2, 1, 1),
            pooled_queries=torch.zeros(1, 2, 4),
            confidence=torch.ones(1, 2),
        )
        calibration = PartGeometryCalibration()
        identity = calibration(prediction)
        torch.testing.assert_close(identity.boxes_cxcywh, boxes)
        calibration.set_part("face", size_scale=(1.20, 1.08))
        adjusted = calibration(prediction)
        torch.testing.assert_close(
            adjusted.boxes_cxcywh[0, 0, 2:],
            torch.tensor([0.24, 0.324]),
        )
        torch.testing.assert_close(
            adjusted.boxes_cxcywh[0, 1], boxes[0, 1]
        )
        adjusted.boxes_cxcywh.sum().backward()
        self.assertTrue(
            torch.isfinite(calibration.center_offset_logits.grad).all()
        )
        self.assertTrue(
            torch.isfinite(calibration.log_size_scales.grad).all()
        )

    def test_normalized_crop_is_differentiable(self):
        images = torch.rand(2, 3, 64, 64)
        boxes = torch.tensor(
            [[0.5, 0.5, 0.4, 0.5], [0.45, 0.55, 0.3, 0.4]],
            requires_grad=True,
        )
        angles = torch.zeros(2, requires_grad=True)
        output = NormalizedRotatedCropper((24, 20))(images, boxes, angles)
        self.assertEqual(tuple(output.shape), (2, 3, 24, 20))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(boxes.grad).all())
        self.assertTrue(torch.isfinite(angles.grad).all())

    def test_normalized_crop_matches_legacy_pixel_crop(self):
        images = torch.rand(2, 3, 96, 128)
        rois = torch.tensor(
            [
                [0.0, 13.25, 17.5, 91.75, 82.25],
                [1.0, 21.0, 9.75, 112.5, 74.25],
            ]
        )
        angles = torch.tensor([0.17, -0.29])
        normalized_xyxy = rois[:, 1:].clone()
        normalized_xyxy[:, [0, 2]] /= images.shape[-1]
        normalized_xyxy[:, [1, 3]] /= images.shape[-2]
        normalized_boxes = xyxy_to_cxcywh(normalized_xyxy)
        expected = DifferentiableROICropper()(
            images, rois, angles, (37, 41)
        )
        actual = NormalizedRotatedCropper((37, 41))(
            images, normalized_boxes, angles
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-5)

    def test_geometry_head_shapes_and_gradients(self):
        head = SoftPartGeometryHead(32, 8, 8, hidden_channels=32)
        features = torch.randn(3, 32, 8, 8, requires_grad=True)
        prediction = head(features)
        self.assertEqual(tuple(prediction.boxes_cxcywh.shape), (3, 2, 4))
        self.assertEqual(tuple(prediction.attention.shape), (3, 2, 8, 8))
        self.assertEqual(tuple(prediction.pooled_queries.shape), (3, 2, 32))
        torch.testing.assert_close(
            prediction.attention.flatten(2).sum(dim=2),
            torch.ones(3, 2),
        )
        prediction.boxes_cxcywh.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_geometry_fpn_preserves_stride8_detail(self):
        adapter = Layer2Layer3GeometryAdapter(32)
        layer2 = torch.randn(2, 512, 12, 10, requires_grad=True)
        layer3 = torch.randn(2, 1024, 6, 5, requires_grad=True)
        output = adapter(layer2, layer3)
        self.assertEqual(tuple(output.shape), (2, 32, 12, 10))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(layer2.grad).all())
        self.assertTrue(torch.isfinite(layer3.grad).all())

    def test_zero_initialized_fusion_preserves_face_descriptor(self):
        fusion = BoundedSemanticResidual(16, 32, hidden_dim=24)
        face = torch.nn.functional.normalize(torch.randn(4, 32), dim=1)
        nose = torch.nn.functional.normalize(torch.randn(4, 32), dim=1)
        queries = torch.randn(4, 3, 16)
        confidence = torch.rand(4, 2)
        embedding, _, scale = fusion(face, nose, queries, confidence)
        torch.testing.assert_close(embedding, face)
        self.assertTrue(((scale >= 0) & (scale <= 0.35)).all())

    def test_initial_reliability_scale_tracks_configured_bound(self):
        fusion = BoundedSemanticResidual(
            16, 32, hidden_dim=24, maximum_residual_scale=0.60
        )
        face = torch.nn.functional.normalize(torch.randn(4, 32), dim=1)
        nose = torch.nn.functional.normalize(torch.randn(4, 32), dim=1)
        queries = torch.randn(4, 3, 16)
        confidence = torch.rand(4, 2)
        _, _, scale = fusion(face, nose, queries, confidence)
        torch.testing.assert_close(scale, torch.full_like(scale, 0.10))

    def test_letterbox_and_box_transform(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxed, scale, padding = letterbox_rgb(image, size=400)
        self.assertEqual(boxed.shape, (400, 400, 3))
        self.assertEqual(scale, 2.0)
        self.assertEqual(padding, (0, 100))
        box = transform_box_to_letterbox(
            [50, 25, 150, 75], scale=scale, padding=padding, size=400
        )
        np.testing.assert_allclose(box, [0.5, 0.5, 0.5, 0.25])

    def test_letterbox_can_preserve_native_resolution(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxed, scale, padding = letterbox_rgb(
            image, size=400, allow_upscale=False
        )
        self.assertEqual(boxed.shape, (400, 400, 3))
        self.assertEqual(scale, 1.0)
        self.assertEqual(padding, (100, 150))

    def test_teacher_cache_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.npz"
            np.savez_compressed(
                path,
                source_sha256=np.asarray(["same", "same"]),
                embedding=np.zeros((2, 4), dtype=np.float32),
                face_embedding=np.zeros((2, 4), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                UnifiedTeacherCache(path)


if __name__ == "__main__":
    unittest.main()

