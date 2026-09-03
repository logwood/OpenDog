# encoding: utf-8
"""Dog ArcFace checkpoint loading and feature-level fusion helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


class DogArcFaceEncoder(nn.Module):
    """Frozen ResNet-50/512-D encoder restored from ``dog.pt``.

    Inputs must already use the ImageNet normalization expected by the main
    Pet-ReID pipeline.  The pretraining classifier is optional because its
    46,755 identities do not correspond to the local training identities.
    """

    feature_dim = 512

    def __init__(
        self,
        checkpoint_path,
        *,
        freeze=True,
        normalize=True,
        load_classifier=False,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"ArcFace checkpoint must be a mapping: {checkpoint_path}")

        required = {"state_dict_backbone", "state_dict_softmax_fc"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"ArcFace checkpoint is missing keys: {sorted(missing)}")

        classifier = checkpoint["state_dict_softmax_fc"]
        if "weight" not in classifier or classifier["weight"].ndim != 2:
            raise ValueError("ArcFace classifier must contain a 2-D weight tensor")
        classifier_shape = tuple(classifier["weight"].shape)
        if classifier_shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected a {self.feature_dim}-D ArcFace classifier, got {classifier_shape}"
            )
        self.num_pretraining_classes = classifier_shape[0]
        classifier_weight = classifier["weight"].detach().clone()

        backbone = resnet50(weights=None)
        backbone.fc = nn.Sequential(
            nn.Linear(2048, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
        )
        # Strict loading is intentional: a partial load could silently turn this
        # into a different descriptor model.
        backbone.load_state_dict(checkpoint["state_dict_backbone"], strict=True)
        self.backbone = backbone
        self.normalize = bool(normalize)
        self.register_buffer(
            "classifier_weight",
            classifier_weight if load_classifier else None,
            persistent=False,
        )
        self.frozen = bool(freeze)
        self._trainable_parts = ()
        if self.frozen:
            self.requires_grad_(False)
            super().train(False)

    def train(self, mode=True):
        # A frozen teacher must keep BatchNorm statistics fixed even when a
        # containing Pet-ReID model is switched to training mode.
        if self.frozen:
            return super().train(False)
        super().train(mode)
        if self._trainable_parts:
            # Keep the frozen prefix, including its BatchNorm running stats,
            # fixed while allowing only the explicitly selected tail to train.
            self.backbone.eval()
            for name in self._trainable_parts:
                getattr(self.backbone, name).train(mode)
        return self

    def configure_trainable_parts(self, parts=()):
        """Freeze the encoder except for selected top-level ResNet children.

        ``parts=("layer4", "fc")`` is the conservative local end-to-end
        setting. Passing an empty iterable restores a completely frozen
        teacher.
        """

        parts = tuple(parts)
        available = dict(self.backbone.named_children())
        unknown = sorted(set(parts) - set(available))
        if unknown:
            raise ValueError(f"Unknown ArcFace backbone parts: {unknown}")
        self.requires_grad_(False)
        for name in parts:
            available[name].requires_grad_(True)
        self._trainable_parts = parts
        self.frozen = not bool(parts)
        self.train(self.training if parts else False)
        return self

    def forward(self, images):
        features = self.backbone(images)
        if self.normalize:
            features = F.normalize(features, dim=1)
        return features

    def classifier_cosine_logits(self, features):
        """Return cosine similarity to all 46,755 PetFace-Dog class centers."""

        if self.classifier_weight is None:
            raise RuntimeError("Classifier weights were not loaded")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected [batch, {self.feature_dim}] features, got {features.shape}"
            )
        return F.linear(
            F.normalize(features, dim=1),
            F.normalize(self.classifier_weight, dim=1),
        )

    def forward_with_classifier(self, images, *, topk=5):
        """Return normalized features and nearest pretraining class centers."""

        if not 1 <= topk <= self.num_pretraining_classes:
            raise ValueError(
                f"topk must be in [1, {self.num_pretraining_classes}], got {topk}"
            )
        features = self(images)
        logits = self.classifier_cosine_logits(features)
        topk_scores, topk_indices = logits.topk(topk, dim=1)
        return {
            "features": features,
            "classifier_logits": logits,
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
        }


def fuse_normalized_features(features: Sequence[torch.Tensor], weights=None):
    """L2-normalize branches and concatenate them with cosine-space weights.

    Square-root coefficients make the cosine similarity of the returned
    descriptor equal to the requested weighted sum of branch similarities.
    """

    features = tuple(features)
    if not features:
        raise ValueError("At least one feature branch is required")
    batch_size = features[0].shape[0] if features[0].ndim == 2 else None
    for index, feature in enumerate(features):
        if feature.ndim != 2:
            raise ValueError(f"Feature branch {index} must be 2-D, got {feature.shape}")
        if feature.shape[0] != batch_size:
            raise ValueError("All feature branches must have the same batch size")

    if weights is None:
        weights = [1.0] * len(features)
    if len(weights) != len(features):
        raise ValueError("The number of weights must match the number of branches")
    weights = tuple(float(weight) for weight in weights)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Feature weights must be finite and non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("At least one feature weight must be positive")

    scaled = [
        F.normalize(feature, dim=1) * math.sqrt(weight / weight_sum)
        for feature, weight in zip(features, weights)
    ]
    return F.normalize(torch.cat(scaled, dim=1), dim=1)


def fuse_main_and_arcface(main_features, arcface_features, *, arcface_weight=0.25):
    """Fuse a main descriptor with the dog ArcFace descriptor."""

    arcface_weight = float(arcface_weight)
    if not 0.0 <= arcface_weight <= 1.0:
        raise ValueError("arcface_weight must be in [0, 1]")
    return fuse_normalized_features(
        (main_features, arcface_features),
        (1.0 - arcface_weight, arcface_weight),
    )
