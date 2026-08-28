# encoding: utf-8
"""Regression tests for locally end-to-end multimodal wiring."""

import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pet_id.localization import (
    FaceDetection,
    NoseSegmentation,
    _sam2_hydra_config_name,
    nose_roi_box,
    viewpoint_signals,
)
from pet_id.localization import configure_sam2_import_path
from pet_id.workspace_paths import SAM2_CONFIG_ROOT
from pet_id.multimodal import (
    DescriptorCache,
    DifferentiableROICropper,
    LocalEndToEndPetIDModel,
    PetDescriptor,
    QualityFusionGate,
    SemanticReliabilityGate,
    compare_descriptors,
    viewpoint_supervised_contrastive_loss,
)


class _PixelMeanHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
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


def _detection():
    return FaceDetection(
        bbox_xyxy=(10.0, 8.0, 50.0, 48.0),
        confidence=0.9,
        landmarks_xy=((20.0, 20.0), (40.0, 20.0), (30.0, 31.0), (24.0, 39.0), (36.0, 39.0)),
    )


class MultimodalIntegrationTest(unittest.TestCase):
    def test_vendored_sam2_package_is_importable(self):
        root = configure_sam2_import_path()
        self.assertTrue((root / "sam2/build_sam.py").is_file())
        self.assertIsNotNone(importlib.util.find_spec("sam2.build_sam"))

    def test_sam2_absolute_config_is_converted_to_hydra_name(self):
        absolute = SAM2_CONFIG_ROOT / "sam2.1/sam2.1_hiera_t.yaml"
        self.assertTrue(absolute.is_file())
        expected = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.assertEqual(_sam2_hydra_config_name(absolute), expected)
        self.assertEqual(_sam2_hydra_config_name(expected), expected)

    def test_anyface_geometry_contract(self):
        detection = _detection()
        self.assertEqual(detection.nose_top, (30.0, 31.0))
        self.assertEqual(nose_roi_box(detection, (64, 64, 3)), (20, 25, 40, 40))

    def test_viewpoint_signals_mirror_continuously(self):
        detection = FaceDetection(
            (5.0, 5.0, 59.0, 59.0),
            0.9,
            ((18.0, 19.0), (43.0, 22.0), (34.0, 33.0), (24.0, 45.0), (42.0, 43.0)),
        )
        mirrored = FaceDetection(
            (5.0, 5.0, 59.0, 59.0),
            0.9,
            tuple(
                (64.0 - detection.landmarks_xy[index][0], detection.landmarks_xy[index][1])
                for index in (1, 0, 2, 4, 3)
            ),
        )
        original = viewpoint_signals(detection)
        flipped = viewpoint_signals(mirrored)
        np.testing.assert_allclose(flipped[:3], -original[:3], atol=1e-5)
        self.assertAlmostEqual(float(flipped[3]), float(original[3]), places=5)

    def test_rotated_crop_is_differentiable(self):
        images = torch.rand(2, 3, 64, 64, requires_grad=True)
        rois = torch.tensor([[0, 8, 8, 56, 56], [1, 12, 10, 52, 54]], dtype=torch.float32)
        angles = torch.tensor([0.0, 0.2])
        crops = DifferentiableROICropper()(images, rois, angles, (24, 20))
        self.assertEqual(tuple(crops.shape), (2, 3, 24, 20))
        crops.square().mean().backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(float(images.grad.abs().sum()), 0.0)

    def test_quality_gate_prior_dynamic_quality_and_missing_branch(self):
        gate = QualityFusionGate(branch_priors=(0.75, 0.25))
        quality = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.1, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        available = torch.tensor([[True, True], [True, True], [True, False]])
        weights = gate(quality, available)
        torch.testing.assert_close(weights[0], torch.tensor([0.75, 0.25]))
        self.assertGreater(float(weights[1, 1].detach()), float(weights[1, 0].detach()))
        torch.testing.assert_close(weights[2], torch.tensor([1.0, 0.0]))

    def test_local_model_backward_reaches_both_encoders_and_gate(self):
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            num_classes=2,
        )
        images = torch.rand(2, 3, 64, 64) * 255
        rois = torch.tensor([[0, 8, 8, 56, 56], [1, 10, 10, 54, 54]], dtype=torch.float32)
        masks = torch.ones(2, 1, 64, 64)
        output = model(
            images,
            face_rois=rois,
            nose_rois=rois,
            roll_angles_radians=torch.tensor([0.0, 0.1]),
            nose_masks=masks,
            quality_signals=torch.ones(2, 6),
            branch_available=torch.ones(2, 2, dtype=torch.bool),
            targets=torch.tensor([0, 1]),
        )
        self.assertEqual(tuple(output["features"].shape), (2, 7))
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertEqual(tuple(output["probabilities"].shape), (2, 2))
        loss = sum(output["losses"].values())
        loss.backward()
        self.assertIsNotNone(model.nose_encoder.projection.weight.grad)
        self.assertIsNotNone(model.face_encoder.projection.weight.grad)
        self.assertIsNotNone(model.gate.residual[-1].weight.grad)

    def test_classifier_logits_are_available_without_targets(self):
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            num_classes=3,
        )
        images = torch.rand(2, 3, 64, 64) * 255
        rois = torch.tensor([[0, 8, 8, 56, 56], [1, 10, 10, 54, 54]], dtype=torch.float32)
        output = model(
            images,
            face_rois=rois,
            nose_rois=rois,
            roll_angles_radians=torch.zeros(2),
            nose_masks=torch.ones(2, 1, 64, 64),
            quality_signals=torch.ones(2, 6),
            branch_available=torch.ones(2, 2, dtype=torch.bool),
        )
        self.assertNotIn("losses", output)
        self.assertEqual(tuple(output["logits"].shape), (2, 3))
        torch.testing.assert_close(
            output["probabilities"].sum(dim=1), torch.ones(2)
        )

    def test_residual_joint_neck_preserves_baseline_and_backpropagates(self):
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            num_classes=2,
            joint_enabled=True,
            joint_dim=5,
            adapter_bottleneck_dim=4,
            joint_initial_mix=0.0025,
            cross_view_weight=0.2,
            cross_modal_weight=0.05,
        )
        images = torch.rand(4, 3, 64, 64) * 255
        rois = torch.tensor(
            [[i, 8, 8, 56, 56] for i in range(4)], dtype=torch.float32
        )
        targets = torch.tensor([0, 0, 1, 1])
        viewpoints = torch.tensor(
            [[0.0, 0.0, 0.0, 0.8], [2.0, 1.0, 1.0, 0.9], [-0.6, -0.2, 0.1, 0.7], [0.6, 0.3, -0.1, 1.0]]
        )
        output = model(
            images,
            face_rois=rois,
            nose_rois=rois,
            roll_angles_radians=torch.zeros(4),
            nose_masks=torch.ones(4, 1, 64, 64),
            quality_signals=torch.ones(4, 6),
            viewpoint_signals=viewpoints,
            branch_available=torch.ones(4, 2, dtype=torch.bool),
            targets=targets,
        )
        self.assertEqual(tuple(output["features"].shape), (4, 12))
        self.assertEqual(tuple(output["joint_features"].shape), (4, 5))
        self.assertAlmostEqual(float(output["joint_mix"].detach()), 0.0025, places=5)
        self.assertGreater(
            float(output["joint_weights"][0, 0].detach()),
            float(output["joint_weights"][1, 0].detach()),
        )
        base_similarity = output["base_features"] @ output["base_features"].T
        final_similarity = output["features"] @ output["features"].T
        self.assertLessEqual(
            float((base_similarity - final_similarity).abs().max().detach()), 0.0051
        )
        self.assertIn("loss_cross_view", output["losses"])
        sum(output["losses"].values()).backward()
        self.assertGreater(float(model.nose_adapter.projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.face_adapter.projection.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.joint_mix_logit.grad)

    def test_shared_space_v2_has_no_legacy_bypass_and_backpropagates(self):
        torch.manual_seed(11)
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            num_classes=2,
            nose_aux_weight=0.0,
            face_aux_weight=0.0,
            fusion_mode="shared_space_v2",
            joint_dim=5,
            adapter_bottleneck_dim=4,
            modality_dropout=0.0,
            branch_consistency_weight=0.25,
        )
        images = torch.rand(4, 3, 64, 64) * 255
        rois = torch.tensor(
            [[index, 8, 8, 56, 56] for index in range(4)],
            dtype=torch.float32,
        )
        output = model(
            images,
            face_rois=rois,
            nose_rois=rois,
            roll_angles_radians=torch.zeros(4),
            nose_masks=torch.ones(4, 1, 64, 64),
            quality_signals=torch.ones(4, 6),
            viewpoint_signals=torch.zeros(4, 4),
            branch_available=torch.tensor(
                [[True, True], [True, False], [False, True], [True, True]]
            ),
            targets=torch.tensor([0, 0, 1, 1]),
        )
        self.assertEqual(model.fused_dim, 5)
        self.assertEqual(tuple(output["features"].shape), (4, 5))
        torch.testing.assert_close(output["features"], output["joint_features"])
        torch.testing.assert_close(output["features"].norm(dim=1), torch.ones(4))
        self.assertEqual(float(output["joint_mix"]), 1.0)
        self.assertFalse(hasattr(model, "joint_mix_logit"))
        self.assertFalse(hasattr(model, "view_gate"))
        torch.testing.assert_close(
            output["fusion_weights"],
            output["joint_weights"],
        )
        torch.testing.assert_close(
            output["fusion_weights"][1],
            torch.tensor([1.0, 0.0]),
        )
        torch.testing.assert_close(
            output["fusion_weights"][2],
            torch.tensor([0.0, 1.0]),
        )
        self.assertIn("loss_branch_consistency", output["losses"])
        output["losses"]["loss_fusion_cls"].backward()
        self.assertGreater(float(model.nose_adapter.projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.face_adapter.projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.gate.residual[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.cross_modal_residual.network[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_semantic_reliability_gate_is_bounded_and_missing_branch_exact(self):
        gate = SemanticReliabilityGate(
            quality_dim=10,
            feature_dim=3,
            branch_priors=(0.10, 0.90),
            max_nose_weight=0.35,
        )
        nose = F.normalize(torch.randn(3, 3), dim=1)
        face = F.normalize(torch.randn(3, 3), dim=1)
        available = torch.tensor(
            [[True, True], [True, False], [False, True]]
        )
        weights = gate(torch.ones(3, 10), nose, face, available)
        self.assertLessEqual(float(weights[0, 0].detach()), 0.35)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        torch.testing.assert_close(weights[1], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(weights[2], torch.tensor([0.0, 1.0]))

    def test_semantic_v3_protects_face_anchor_and_trains_reliability(self):
        torch.manual_seed(19)
        model = LocalEndToEndPetIDModel(
            _FakeNoseEncoder(),
            _FakeFaceEncoder(),
            nose_size=(24, 24),
            face_size=(20, 20),
            num_classes=2,
            nose_aux_weight=0.0,
            face_aux_weight=0.0,
            fusion_mode="semantic_residual_v3",
            joint_dim=3,
            adapter_bottleneck_dim=4,
            modality_dropout=0.0,
            cross_modal_weight=0.05,
            semantic_max_nose_weight=0.35,
            semantic_residual_scale=0.0,
            semantic_conflict_weight=0.5,
            semantic_conflict_margin=0.05,
            dominance_weight=0.5,
            dominance_tolerance=0.02,
            branch_priors=(0.10, 0.90),
        )
        images = torch.rand(4, 3, 64, 64) * 255
        rois = torch.tensor(
            [[index, 8, 8, 56, 56] for index in range(4)],
            dtype=torch.float32,
        )
        common = {
            "images_0_255": images,
            "face_rois": rois,
            "nose_rois": rois,
            "roll_angles_radians": torch.zeros(4),
            "nose_masks": torch.ones(4, 1, 64, 64),
            "quality_signals": torch.ones(4, 6),
            "viewpoint_signals": torch.zeros(4, 4),
        }
        output = model(
            **common,
            branch_available=torch.ones(4, 2, dtype=torch.bool),
            targets=torch.tensor([0, 0, 1, 1]),
        )
        self.assertEqual(model.fused_dim, 3)
        self.assertIsInstance(model.face_adapter, nn.Identity)
        self.assertTrue((output["fusion_weights"][:, 0] <= 0.35).all())
        self.assertIn("loss_semantic_conflict", output["losses"])
        self.assertIn("loss_branch_dominance", output["losses"])
        sum(output["losses"].values()).backward()
        self.assertGreater(
            float(model.nose_adapter.projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.gate.reliability[-1].weight.grad.abs().sum()),
            0.0,
        )

        model.eval()
        available = torch.tensor(
            [[True, True], [False, True], [True, False], [True, True]]
        )
        with torch.inference_mode():
            missing_output = model(**common, branch_available=available)
            expected_nose = model.nose_adapter(missing_output["nose_features"])
        torch.testing.assert_close(
            missing_output["features"][1],
            missing_output["face_features"][1],
        )
        torch.testing.assert_close(
            missing_output["features"][2],
            expected_nose[2],
        )

    def test_viewpoint_supervised_contrastive_loss_is_finite(self):
        features = F.normalize(torch.randn(4, 8), dim=1)
        loss = viewpoint_supervised_contrastive_loss(
            features,
            torch.tensor([0, 0, 1, 1]),
            torch.tensor(
                [[-1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [-0.5, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_descriptor_comparison_and_safe_cache_roundtrip(self):
        detection = _detection()
        mask = np.zeros((64, 64), dtype=bool)
        mask[25:41, 20:40] = True
        segmentation = NoseSegmentation(mask, (20, 25, 40, 41), 0.8, 2)
        nose = F.normalize(torch.tensor([1.0, 2.0, 3.0]), dim=0)
        face = F.normalize(torch.tensor([2.0, 1.0]), dim=0)
        fused = F.normalize(torch.cat((nose * (0.75**0.5), face * (0.25**0.5))), dim=0)
        descriptor = PetDescriptor(
            fused,
            nose,
            face,
            (0.75, 0.25),
            (0.9, 0.8),
            (True, True),
            detection,
            segmentation,
        )
        score = compare_descriptors(descriptor, descriptor)
        self.assertAlmostEqual(score.fused, 1.0, places=6)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.jpg"
            image.write_bytes(b"cache-key-source")
            cache = DescriptorCache(Path(directory) / "cache", "test-namespace")
            cache.save(image, [descriptor])
            loaded = cache.load(image)
            self.assertEqual(len(loaded), 1)
            torch.testing.assert_close(loaded[0].fused_feature, fused)
            self.assertIsNone(loaded[0].segmentation)
            self.assertEqual(
                loaded[0].metadata_dict()["segmentation"],
                descriptor.metadata_dict()["segmentation"],
            )


if __name__ == "__main__":
    unittest.main()
