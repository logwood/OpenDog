# encoding: utf-8
"""Unit tests for UnifiedPetReID training controls."""

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/train_unified_pet_reid.py"
_SPEC = importlib.util.spec_from_file_location(
    "train_unified_pet_reid_for_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_TRAINING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAINING)
configure_trainable = _TRAINING.configure_trainable
CosineIdentityClassifier = _TRAINING.CosineIdentityClassifier


class _TrainableStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_adapter = nn.Linear(1, 1)
        self.geometry = nn.Linear(1, 1)
        self.geometry_calibration = nn.Linear(1, 1)
        self.semantic_fusion = nn.Linear(1, 1)
        self.identity_encoder = nn.Linear(1, 1)

    def configure_identity_trainable(self, parts: tuple[str, ...]) -> None:
        self.identity_encoder.requires_grad_(bool(parts))


def _is_trainable(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


class UnifiedTrainingTest(unittest.TestCase):
    def test_cosine_identity_classifier_is_feature_scale_invariant(self):
        classifier = CosineIdentityClassifier(3, 4, scale=17.0)
        features = torch.tensor(
            [[1.0, 2.0, -1.0, 0.5], [-0.5, 0.25, 2.0, 1.0]]
        )

        logits = classifier(features)
        scaled_logits = classifier(features * torch.tensor([[0.1], [9.0]]))

        torch.testing.assert_close(logits, scaled_logits)

    def test_cosine_identity_classifier_backward_is_finite(self):
        classifier = CosineIdentityClassifier(3, 4)
        features = torch.randn(6, 4, requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1, 2, 2])

        loss = nn.functional.cross_entropy(classifier(features), targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertIsNotNone(classifier.weight.grad)
        self.assertTrue(torch.isfinite(classifier.weight.grad).all())

    def test_geometry_calibration_can_be_frozen_independently(self):
        model = _TrainableStub()
        configure_trainable(
            model,
            "joint",
            freeze_geometry_calibration=True,
        )

        self.assertTrue(_is_trainable(model.geometry_adapter))
        self.assertTrue(_is_trainable(model.geometry))
        self.assertFalse(_is_trainable(model.geometry_calibration))
        self.assertTrue(_is_trainable(model.semantic_fusion))
        self.assertTrue(_is_trainable(model.identity_encoder))

    def test_freeze_geometry_also_freezes_calibration(self):
        model = _TrainableStub()
        configure_trainable(model, "joint", freeze_geometry=True)

        self.assertFalse(_is_trainable(model.geometry_adapter))
        self.assertFalse(_is_trainable(model.geometry))
        self.assertFalse(_is_trainable(model.geometry_calibration))
