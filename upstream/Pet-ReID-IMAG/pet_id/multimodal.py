# encoding: utf-8
"""Locally end-to-end dog face + nose-print identity pipeline.

AnyFace and SAM 2 remain frozen geometry providers.  From the resulting ROIs
onward, cropping, both identity encoders, the quality gate, and an optional
local-identity loss live in one differentiable ``nn.Module``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.modeling.losses import triplet_loss

from .arcface import DogArcFaceEncoder
from .config import add_retri_config
from .localization import (
    VIEWPOINT_DIM,
    AnyFaceDetector,
    FaceDetection,
    NoseSegmentation,
    SAM2NoseSegmenter,
    crop_aligned_face,
    face_quality,
    laplacian_sharpness_quality,
    nose_quality,
    nose_roi_box,
    viewpoint_signals,
)


QUALITY_DIM = 6
LEGACY_CONCAT_FUSION = "legacy_concat"
SHARED_SPACE_FUSION_V2 = "shared_space_v2"
SEMANTIC_RESIDUAL_FUSION_V3 = "semantic_residual_v3"
SHARED_FUSION_MODES = {SHARED_SPACE_FUSION_V2, SEMANTIC_RESIDUAL_FUSION_V3}
FUSION_MODES = {LEGACY_CONCAT_FUSION, *SHARED_FUSION_MODES}


@dataclass(frozen=True)
class PetDescriptor:
    """Identity descriptors and quality metadata for one detected animal."""

    fused_feature: torch.Tensor
    nose_feature: torch.Tensor
    face_feature: torch.Tensor
    fusion_weights: tuple[float, float]
    branch_quality: tuple[float, float]
    branch_available: tuple[bool, bool]
    detection: FaceDetection | None
    segmentation: NoseSegmentation | None = None
    cached_segmentation_metadata: dict | None = None
    identity_scores: tuple[tuple[str, float], ...] = ()
    inference_size: tuple[int, int] | None = None
    viewpoint: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("fused_feature", "nose_feature", "face_feature"):
            value = getattr(self, name)
            if value.ndim != 1 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be one finite descriptor vector")

    def metadata_dict(self) -> dict:
        segmentation_metadata = self.cached_segmentation_metadata
        if self.segmentation:
            segmentation_metadata = {
                "roi_box_xyxy": list(self.segmentation.roi_box_xyxy),
                "predicted_iou": self.segmentation.predicted_iou,
                "selected_candidate": self.segmentation.selected_candidate,
                "roi_mask_fraction": self.segmentation.roi_mask_fraction,
            }
        return {
            "fusion_weights": list(self.fusion_weights),
            "branch_quality": list(self.branch_quality),
            "branch_available": list(self.branch_available),
            "detection": self.detection.to_dict() if self.detection else None,
            "segmentation": segmentation_metadata,
            "identity_scores": [
                {"identity": identity, "probability": probability}
                for identity, probability in self.identity_scores
            ],
            "inference_size": list(self.inference_size) if self.inference_size else None,
            "viewpoint": list(self.viewpoint),
        }


@dataclass(frozen=True)
class PairSimilarity:
    fused: float
    nose: float | None
    face: float | None

    def to_dict(self) -> dict:
        return {"fused": self.fused, "nose": self.nose, "face": self.face}


def compare_descriptors(left: PetDescriptor, right: PetDescriptor) -> PairSimilarity:
    """Compare locally gated descriptors and expose both branch scores."""

    fused = float(F.cosine_similarity(left.fused_feature[None], right.fused_feature[None]))
    nose = None
    if left.branch_available[0] and right.branch_available[0]:
        nose = float(F.cosine_similarity(left.nose_feature[None], right.nose_feature[None]))
    face = None
    if left.branch_available[1] and right.branch_available[1]:
        face = float(F.cosine_similarity(left.face_feature[None], right.face_feature[None]))
    return PairSimilarity(fused=fused, nose=nose, face=face)


class FastReIDDescriptorEncoder(nn.Module):
    """Expose the descriptor path of a trained FastReID Baseline in train mode."""

    def __init__(self, model: nn.Module, feature_dim: int):
        super().__init__()
        self.model = model
        self.feature_dim = int(feature_dim)
        self.frozen = False
        self._trainable_parts = ()

    @classmethod
    def from_files(cls, config_path, weights_path, *, device=None):
        config_path, weights_path = Path(config_path), Path(weights_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Nose model config not found: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Nose model weights not found: {weights_path}")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        cfg = get_cfg()
        add_retri_config(cfg)
        cfg.merge_from_file(str(config_path))
        cfg.defrost()
        cfg.MODEL.DEVICE = str(device)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        # The local fusion owns its new classification heads. The released
        # competition classifier is identity-specific and intentionally skipped.
        cfg.MODEL.HEADS.NUM_CLASSES = 0
        cfg.freeze()
        model = build_model(cfg)

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        model_state = model.state_dict()
        compatible = {}
        for key, value in state.items():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key in model_state and tuple(value.shape) == tuple(model_state[clean_key].shape):
                compatible[clean_key] = value
        if len(compatible) < max(1, int(0.80 * len(model_state))):
            raise RuntimeError(
                f"Only {len(compatible)}/{len(model_state)} nose-model tensors are compatible"
            )
        model.load_state_dict(compatible, strict=False)
        model.to(device)
        feature_dim = int(cfg.MODEL.HEADS.EMBEDDING_DIM or cfg.MODEL.BACKBONE.FEAT_DIM)
        return cls(model, feature_dim)

    def train(self, mode=True):
        if self.frozen:
            return super().train(False)
        super().train(mode)
        if self._trainable_parts:
            self.model.eval()
            for name in self._trainable_parts:
                if name == "heads":
                    self.model.heads.train(mode)
                else:
                    getattr(self.model.backbone, name).train(mode)
        return self

    def configure_trainable_parts(self, parts=()):
        """Freeze the IMAG encoder except for selected backbone stages/head."""

        parts = tuple(parts)
        available = set(dict(self.model.backbone.named_children())) | {"heads"}
        unknown = sorted(set(parts) - available)
        if unknown:
            raise ValueError(f"Unknown IMAG encoder parts: {unknown}")
        self.requires_grad_(False)
        for name in parts:
            module = self.model.heads if name == "heads" else getattr(self.model.backbone, name)
            module.requires_grad_(True)
        self._trainable_parts = parts
        self.frozen = not bool(parts)
        self.train(self.training if parts else False)
        return self

    def forward(self, images_0_255: torch.Tensor) -> torch.Tensor:
        normalized = (images_0_255 - self.model.pixel_mean) / self.model.pixel_std
        spatial_features = self.model.backbone(normalized)
        pooled = self.model.heads.pool_layer(spatial_features)
        neck = self.model.heads.bottleneck(pooled)[..., 0, 0]
        return F.normalize(neck, dim=1)


class DifferentiableROICropper(nn.Module):
    """Rotated crop implemented with ``affine_grid``/``grid_sample``."""

    def forward(
        self,
        images: torch.Tensor,
        rois: torch.Tensor,
        angles_radians: torch.Tensor,
        output_size: Sequence[int],
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] images, got {tuple(images.shape)}")
        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError(f"Expected [K,5] ROIs, got {tuple(rois.shape)}")
        if angles_radians.shape != (rois.shape[0],):
            raise ValueError("One roll angle is required per ROI")
        out_h, out_w = int(output_size[0]), int(output_size[1])
        batch_indices = rois[:, 0].long()
        if (batch_indices < 0).any() or (batch_indices >= images.shape[0]).any():
            raise IndexError("ROI batch index is outside the image batch")

        selected = images.index_select(0, batch_indices)
        height, width = images.shape[-2:]
        x1, y1, x2, y2 = rois[:, 1:].unbind(dim=1)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = (x2 - x1).clamp_min(1.0) / 2
        half_h = (y2 - y1).clamp_min(1.0) / 2
        cosine, sine = torch.cos(angles_radians), torch.sin(angles_radians)

        x_scale = 2.0 / max(width - 1, 1)
        y_scale = 2.0 / max(height - 1, 1)
        theta = torch.zeros((rois.shape[0], 2, 3), dtype=images.dtype, device=images.device)
        theta[:, 0, 0] = x_scale * cosine * half_w
        theta[:, 0, 1] = -x_scale * sine * half_h
        theta[:, 0, 2] = x_scale * center_x - 1.0
        theta[:, 1, 0] = y_scale * sine * half_w
        theta[:, 1, 1] = y_scale * cosine * half_h
        theta[:, 1, 2] = y_scale * center_y - 1.0
        grid = F.affine_grid(
            theta,
            (rois.shape[0], images.shape[1], out_h, out_w),
            align_corners=True,
        )
        return F.grid_sample(
            selected,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )


class QualityFusionGate(nn.Module):
    """Quality-aware per-sample gate with a 75/25 nose/face prior."""

    def __init__(
        self,
        quality_dim=QUALITY_DIM,
        *,
        hidden_dim=16,
        branch_priors=(0.75, 0.25),
    ):
        super().__init__()
        priors = torch.as_tensor(branch_priors, dtype=torch.float32)
        if priors.shape != (2,) or (priors <= 0).any():
            raise ValueError("Two positive branch priors are required")
        self.register_buffer("log_priors", priors.log(), persistent=True)
        self.residual = nn.Sequential(
            nn.Linear(int(quality_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        # Before training, weights equal normalized prior * measured quality.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        quality_signals: torch.Tensor,
        branch_available: torch.Tensor,
    ) -> torch.Tensor:
        if quality_signals.ndim != 2:
            raise ValueError("quality_signals must have shape [batch, quality_dim]")
        if branch_available.shape != (quality_signals.shape[0], 2):
            raise ValueError("branch_available must have shape [batch, 2]")
        if not branch_available.any(dim=1).all():
            raise ValueError("Every sample needs at least one identity branch")
        measured = quality_signals[:, :2].clamp(1e-4, 1.0)
        logits = self.log_priors + measured.log() + self.residual(quality_signals)
        logits = logits.masked_fill(~branch_available.bool(), torch.finfo(logits.dtype).min)
        return logits.softmax(dim=1)


class SemanticReliabilityGate(nn.Module):
    """Bounded nose reliability learned from geometry and cross-modal agreement.

    When both branches are present, face is the safety anchor and the nose may
    contribute at most ``max_nose_weight``. Missing-branch cases still map to
    exact [1, 0] or [0, 1] weights.
    """

    def __init__(
        self,
        quality_dim: int,
        feature_dim: int,
        *,
        hidden_dim=128,
        branch_priors=(0.10, 0.90),
        max_nose_weight=0.35,
    ):
        super().__init__()
        priors = torch.as_tensor(branch_priors, dtype=torch.float32)
        if priors.shape != (2,) or (priors <= 0).any():
            raise ValueError("Two positive branch priors are required")
        self.max_nose_weight = float(max_nose_weight)
        if not 0.0 < self.max_nose_weight < 0.5:
            raise ValueError("max_nose_weight must be strictly between 0 and 0.5")
        initial_nose_weight = float(priors[0] / priors.sum())
        if not 0.0 < initial_nose_weight < self.max_nose_weight:
            raise ValueError(
                "The normalized nose prior must be below max_nose_weight"
            )

        hidden_dim = int(hidden_dim)
        relation_dim = max(16, hidden_dim // 2)
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(2 * int(feature_dim)),
            nn.Linear(2 * int(feature_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, relation_dim),
            nn.GELU(),
        )
        self.quality_norm = nn.LayerNorm(int(quality_dim))
        self.reliability = nn.Sequential(
            nn.Linear(relation_dim + int(quality_dim) + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        probability = initial_nose_weight / self.max_nose_weight
        initial_logit = math.log(probability / (1.0 - probability))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.constant_(self.reliability[-1].bias, initial_logit)

    @staticmethod
    def agreement_signals(
        nose_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> torch.Tensor:
        cosine = F.cosine_similarity(
            nose_features.float(), face_features.float(), dim=1
        )
        mean_absolute_difference = (
            nose_features.float() - face_features.float()
        ).abs().mean(dim=1)
        return torch.stack((cosine, mean_absolute_difference), dim=1)

    def reliability_logits(
        self,
        quality_signals: torch.Tensor,
        nose_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> torch.Tensor:
        if quality_signals.ndim != 2:
            raise ValueError("quality_signals must have shape [batch, quality_dim]")
        if nose_features.shape != face_features.shape or nose_features.ndim != 2:
            raise ValueError("Aligned nose/face features must have the same 2D shape")
        if nose_features.shape[0] != quality_signals.shape[0]:
            raise ValueError("Quality and feature batches must have the same size")
        relational = torch.cat(
            (
                (nose_features - face_features).abs(),
                nose_features * face_features,
            ),
            dim=1,
        )
        encoded_relation = self.relation_encoder(relational.float()).to(
            dtype=quality_signals.dtype
        )
        agreement = self.agreement_signals(nose_features, face_features).to(
            dtype=quality_signals.dtype
        )
        return self.reliability(
            torch.cat(
                (
                    self.quality_norm(quality_signals.float()).to(
                        dtype=quality_signals.dtype
                    ),
                    encoded_relation,
                    agreement,
                ),
                dim=1,
            )
        )

    def forward(
        self,
        quality_signals: torch.Tensor,
        nose_features: torch.Tensor,
        face_features: torch.Tensor,
        branch_available: torch.Tensor,
    ) -> torch.Tensor:
        if branch_available.shape != (quality_signals.shape[0], 2):
            raise ValueError("branch_available must have shape [batch, 2]")
        if not branch_available.any(dim=1).all():
            raise ValueError("Every sample needs at least one identity branch")
        bounded_nose_weight = (
            self.max_nose_weight
            * self.reliability_logits(
                quality_signals,
                nose_features,
                face_features,
            ).sigmoid()
        )
        nose_available = branch_available[:, 0:1].bool()
        face_available = branch_available[:, 1:2].bool()
        nose_weight = torch.where(
            nose_available & ~face_available,
            torch.ones_like(bounded_nose_weight),
            torch.where(
                nose_available & face_available,
                bounded_nose_weight,
                torch.zeros_like(bounded_nose_weight),
            ),
        )
        return torch.cat((nose_weight, 1.0 - nose_weight), dim=1)


class ResidualProjectionAdapter(nn.Module):
    """Project a pretrained descriptor while preserving a low-rank residual path."""

    def __init__(self, input_dim: int, output_dim=512, bottleneck_dim=128):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.projection = nn.Linear(int(input_dim), int(output_dim), bias=False)
        self.residual = nn.Sequential(
            nn.Linear(int(input_dim), int(bottleneck_dim)),
            nn.GELU(),
            nn.Linear(int(bottleneck_dim), int(output_dim), bias=False),
        )
        if int(input_dim) == int(output_dim):
            nn.init.eye_(self.projection.weight)
        else:
            nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(features.float()).to(dtype=features.dtype)
        projected = self.projection(normalized)
        return F.normalize(projected + self.residual(normalized), dim=1)


class CrossModalResidual(nn.Module):
    """Small zero-initialized interaction block for aligned branch descriptors."""

    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4 * int(feature_dim)),
            nn.Linear(4 * int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(feature_dim), bias=False),
        )
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, nose: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((nose, face, (nose - face).abs(), nose * face), dim=1))


def viewpoint_supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    viewpoints: torch.Tensor,
    *,
    temperature=0.10,
    pose_boost=1.0,
) -> torch.Tensor:
    """Supervised contrastive loss that emphasizes different-view positives."""

    features = F.normalize(features.float(), dim=1)
    targets = targets.reshape(-1)
    logits = features @ features.T / float(temperature)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive = targets[:, None].eq(targets[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pose_distance = torch.cdist(viewpoints.float(), viewpoints.float()).clamp_max(4.0)
    weights = positive.float() * (1.0 + float(pose_boost) * pose_distance / 4.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(weights[valid] * log_probabilities[valid]).sum(dim=1).mean()


def cross_modal_supervised_contrastive_loss(
    nose_features: torch.Tensor,
    face_features: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature=0.10,
) -> torch.Tensor:
    """Align modalities by identity without treating same-ID views as negatives."""

    nose_features = F.normalize(nose_features.float(), dim=1)
    face_features = F.normalize(face_features.float(), dim=1)
    targets = targets.reshape(-1)
    logits = nose_features @ face_features.T / float(temperature)
    positives = targets[:, None].eq(targets[None, :])
    row_log_probability = (
        torch.logsumexp(logits.masked_fill(~positives, -float("inf")), dim=1)
        - torch.logsumexp(logits, dim=1)
    )
    column_log_probability = (
        torch.logsumexp(
            logits.masked_fill(~positives, -float("inf")),
            dim=0,
        )
        - torch.logsumexp(logits, dim=0)
    )
    return -0.5 * (
        row_log_probability.mean() + column_log_probability.mean()
    )


def batch_hard_metric_violation(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin=0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample batch-hard violations and their validity mask."""

    features = F.normalize(features.float(), dim=1)
    targets = targets.reshape(-1)
    similarities = features @ features.T
    eye = torch.eye(
        similarities.shape[0], dtype=torch.bool, device=similarities.device
    )
    positives = targets[:, None].eq(targets[None, :]) & ~eye
    negatives = ~targets[:, None].eq(targets[None, :])
    valid = positives.any(dim=1) & negatives.any(dim=1)
    hardest_positive = similarities.masked_fill(
        ~positives, float("inf")
    ).min(dim=1).values
    hardest_negative = similarities.masked_fill(
        ~negatives, -float("inf")
    ).max(dim=1).values
    violation = F.relu(float(margin) + hardest_negative - hardest_positive)
    return torch.where(valid, violation, torch.zeros_like(violation)), valid


def different_identity_permutation(
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one deterministic different-identity partner per valid sample."""

    targets = targets.reshape(-1)
    indices = torch.arange(targets.shape[0], device=targets.device)
    permutation = indices.clone()
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for offset in range(1, targets.shape[0]):
        candidate = (indices + offset) % targets.shape[0]
        take = ~valid & targets[candidate].ne(targets)
        permutation = torch.where(take, candidate, permutation)
        valid |= take
    return permutation, valid


class LocalEndToEndPetIDModel(nn.Module):
    """Differentiable ROI-to-identity model with frozen geometry prompts."""

    def __init__(
        self,
        nose_encoder: FastReIDDescriptorEncoder,
        face_encoder: DogArcFaceEncoder,
        *,
        nose_size=(244, 244),
        face_size=(224, 224),
        num_classes=0,
        quality_dim=QUALITY_DIM,
        branch_priors=(0.75, 0.25),
        classifier_scale=32.0,
        nose_aux_weight=0.25,
        face_aux_weight=0.15,
        joint_enabled=False,
        fusion_mode=LEGACY_CONCAT_FUSION,
        joint_dim=512,
        adapter_bottleneck_dim=128,
        joint_initial_mix=0.0025,
        modality_dropout=0.0,
        cross_view_weight=0.0,
        cross_modal_weight=0.0,
        branch_consistency_weight=0.0,
        semantic_max_nose_weight=0.35,
        semantic_residual_scale=0.05,
        semantic_conflict_weight=0.0,
        semantic_conflict_margin=0.05,
        dominance_weight=0.0,
        dominance_tolerance=0.02,
        contrastive_temperature=0.10,
        contrastive_pose_boost=1.0,
        viewpoint_nose_penalty=0.35,
        viewpoint_nose_floor=0.50,
    ):
        super().__init__()
        self.nose_encoder = nose_encoder
        self.face_encoder = face_encoder
        self.cropper = DifferentiableROICropper()
        self.nose_size = tuple(int(value) for value in nose_size)
        self.face_size = tuple(int(value) for value in face_size)
        self.base_fused_dim = nose_encoder.feature_dim + face_encoder.feature_dim
        self.fusion_mode = str(fusion_mode).casefold()
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {sorted(FUSION_MODES)}, "
                f"got {fusion_mode!r}"
            )
        self.joint_enabled = bool(
            joint_enabled or self.fusion_mode in SHARED_FUSION_MODES
        )
        self.joint_dim = int(joint_dim) if self.joint_enabled else 0
        if (
            self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3
            and self.joint_dim != face_encoder.feature_dim
        ):
            raise ValueError(
                "semantic_residual_v3 requires joint_dim to equal the raw "
                f"face feature dimension ({face_encoder.feature_dim})"
            )
        self.fused_dim = (
            self.joint_dim
            if self.fusion_mode in SHARED_FUSION_MODES
            else self.base_fused_dim + self.joint_dim
        )
        gate_input_dim = (
            quality_dim + VIEWPOINT_DIM
            if self.fusion_mode in SHARED_FUSION_MODES
            else quality_dim
        )
        if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3:
            self.gate = SemanticReliabilityGate(
                gate_input_dim,
                self.joint_dim,
                hidden_dim=adapter_bottleneck_dim,
                branch_priors=branch_priors,
                max_nose_weight=semantic_max_nose_weight,
            )
        else:
            self.gate = QualityFusionGate(
                gate_input_dim,
                branch_priors=branch_priors,
            )
        self.num_classes = int(num_classes)
        self.classifier_scale = float(classifier_scale)
        self.nose_aux_weight = float(nose_aux_weight)
        self.face_aux_weight = float(face_aux_weight)
        self.modality_dropout = float(modality_dropout)
        self.cross_view_weight = float(cross_view_weight)
        self.cross_modal_weight = float(cross_modal_weight)
        self.branch_consistency_weight = float(branch_consistency_weight)
        self.semantic_max_nose_weight = float(semantic_max_nose_weight)
        self.semantic_residual_scale = float(semantic_residual_scale)
        self.semantic_conflict_weight = float(semantic_conflict_weight)
        self.semantic_conflict_margin = float(semantic_conflict_margin)
        self.dominance_weight = float(dominance_weight)
        self.dominance_tolerance = float(dominance_tolerance)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_pose_boost = float(contrastive_pose_boost)
        self.viewpoint_nose_penalty = float(viewpoint_nose_penalty)
        self.viewpoint_nose_floor = float(viewpoint_nose_floor)
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if self.viewpoint_nose_penalty < 0:
            raise ValueError("viewpoint_nose_penalty must be non-negative")
        if not 0.0 <= self.viewpoint_nose_floor <= 1.0:
            raise ValueError("viewpoint_nose_floor must be in [0, 1]")
        if self.branch_consistency_weight < 0:
            raise ValueError("branch_consistency_weight must be non-negative")
        for name, value in (
            ("semantic_residual_scale", self.semantic_residual_scale),
            ("semantic_conflict_weight", self.semantic_conflict_weight),
            ("semantic_conflict_margin", self.semantic_conflict_margin),
            ("dominance_weight", self.dominance_weight),
            ("dominance_tolerance", self.dominance_tolerance),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.joint_enabled:
            self.nose_adapter = ResidualProjectionAdapter(
                nose_encoder.feature_dim,
                self.joint_dim,
                adapter_bottleneck_dim,
            )
            self.face_adapter = (
                nn.Identity()
                if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3
                else ResidualProjectionAdapter(
                    face_encoder.feature_dim,
                    self.joint_dim,
                    adapter_bottleneck_dim,
                )
            )
            self.cross_modal_residual = CrossModalResidual(
                self.joint_dim,
                adapter_bottleneck_dim,
            )
            if self.fusion_mode == LEGACY_CONCAT_FUSION:
                self.view_gate = QualityFusionGate(
                    quality_dim + VIEWPOINT_DIM,
                    branch_priors=branch_priors,
                )
                initial_mix = float(joint_initial_mix)
                if not 0.0 < initial_mix < 1.0:
                    raise ValueError(
                        "joint_initial_mix must be strictly between 0 and 1"
                    )
                self.joint_mix_logit = nn.Parameter(
                    torch.tensor(math.log(initial_mix / (1.0 - initial_mix)))
                )
        self.register_buffer(
            "face_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "face_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.num_classes > 0:
            self.fused_classifier = nn.Parameter(torch.empty(self.num_classes, self.fused_dim))
            self.nose_classifier = nn.Parameter(
                torch.empty(self.num_classes, nose_encoder.feature_dim)
            )
            self.face_classifier = nn.Parameter(
                torch.empty(self.num_classes, face_encoder.feature_dim)
            )
            for parameter in (
                self.fused_classifier,
                self.nose_classifier,
                self.face_classifier,
            ):
                nn.init.normal_(parameter, std=0.01)
        self.identity_to_label: dict[str, int] = {}
        self.label_to_identity: dict[int, str] = {}

    @staticmethod
    def _cosine_logits(features: torch.Tensor, weights: torch.Tensor, scale: float):
        return scale * F.linear(F.normalize(features, dim=1), F.normalize(weights, dim=1))

    def _shared_fusion(
        self,
        adapted_nose: torch.Tensor,
        adapted_face: torch.Tensor,
        weights: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        joint_base = (
            adapted_nose * weights[:, 0:1]
            + adapted_face * weights[:, 1:2]
        )
        interaction = self.cross_modal_residual(adapted_nose, adapted_face)
        both_available = available.all(dim=1, keepdim=True).to(joint_base.dtype)
        return F.normalize(joint_base + interaction * both_available, dim=1)

    def _semantic_residual_fusion(
        self,
        adapted_nose: torch.Tensor,
        raw_face: torch.Tensor,
        weights: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        """Use the raw face descriptor as an exact, protected anchor."""

        nose_weight = weights[:, 0:1]
        base = raw_face * (1.0 - nose_weight) + adapted_nose * nose_weight
        interaction = self.cross_modal_residual(adapted_nose, raw_face)
        bounded_interaction = torch.tanh(interaction) / math.sqrt(self.joint_dim)
        both_available = available.all(dim=1, keepdim=True).to(base.dtype)
        fused = F.normalize(
            base
            + (
                self.semantic_residual_scale
                * nose_weight
                * bounded_interaction
                * both_available
            ),
            dim=1,
        )
        nose_only = available[:, 0:1] & ~available[:, 1:2]
        face_only = available[:, 1:2] & ~available[:, 0:1]
        fused = torch.where(nose_only, adapted_nose, fused)
        return torch.where(face_only, raw_face, fused)

    def forward(
        self,
        images_0_255: torch.Tensor,
        *,
        face_rois: torch.Tensor,
        nose_rois: torch.Tensor,
        roll_angles_radians: torch.Tensor,
        nose_masks: torch.Tensor,
        quality_signals: torch.Tensor,
        branch_available: torch.Tensor,
        viewpoint_signals: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict:
        device, dtype = images_0_255.device, images_0_255.dtype
        face_rois = face_rois.to(device=device, dtype=dtype)
        nose_rois = nose_rois.to(device=device, dtype=dtype)
        angles = roll_angles_radians.to(device=device, dtype=dtype)
        quality_signals = quality_signals.to(device=device, dtype=dtype)
        branch_available = branch_available.to(device=device, dtype=torch.bool)
        if viewpoint_signals is None:
            viewpoint_signals = torch.zeros(
                (quality_signals.shape[0], VIEWPOINT_DIM),
                device=device,
                dtype=dtype,
            )
        else:
            viewpoint_signals = viewpoint_signals.to(device=device, dtype=dtype)
        if viewpoint_signals.shape != (quality_signals.shape[0], VIEWPOINT_DIM):
            raise ValueError(
                f"viewpoint_signals must have shape [batch, {VIEWPOINT_DIM}]"
            )
        nose_masks = nose_masks.to(device=device, dtype=dtype)

        face_crops = self.cropper(images_0_255, face_rois, angles, self.face_size)
        nose_crops = self.cropper(images_0_255, nose_rois, angles, self.nose_size)
        mask_rois = nose_rois.clone()
        mask_rois[:, 0] = torch.arange(mask_rois.shape[0], device=device, dtype=dtype)
        mask_crops = self.cropper(nose_masks, mask_rois, angles, self.nose_size).clamp(0, 1)
        soft_masks = F.avg_pool2d(mask_crops, kernel_size=5, stride=1, padding=2)
        background = self.nose_encoder.model.pixel_mean.to(dtype=dtype)
        masked_nose_crops = nose_crops * soft_masks + background * (1.0 - soft_masks)

        raw_nose_features = self.nose_encoder(nose_crops)
        masked_nose_features = self.nose_encoder(masked_nose_crops)
        nose_features = F.normalize(raw_nose_features + masked_nose_features, dim=1)
        normalized_faces = (face_crops.div(255.0) - self.face_mean) / self.face_std
        face_features = F.normalize(self.face_encoder(normalized_faces), dim=1)

        effective_available = branch_available
        if self.training and self.modality_dropout > 0:
            effective_available = branch_available.clone()
            both = effective_available.all(dim=1)
            drop = torch.rand(effective_available.shape[0], device=device)
            effective_available[both & (drop < self.modality_dropout / 2), 0] = False
            effective_available[
                both
                & (drop >= self.modality_dropout / 2)
                & (drop < self.modality_dropout),
                1,
            ] = False

        viewpoint_frontality = torch.ones(
            quality_signals.shape[0], device=device, dtype=dtype
        )
        joint_inputs = None
        if self.joint_enabled:
            pose_magnitude = viewpoint_signals[:, :3].float().norm(dim=1)
            viewpoint_frontality = (
                self.viewpoint_nose_floor
                + (1.0 - self.viewpoint_nose_floor)
                * torch.exp(-self.viewpoint_nose_penalty * pose_magnitude)
            ).to(dtype=dtype)
            joint_quality = quality_signals.clone()
            joint_quality[:, 0] = joint_quality[:, 0] * viewpoint_frontality
            joint_inputs = torch.cat((joint_quality, viewpoint_signals), dim=1)
        adapted_nose = None
        adapted_face = None
        semantic_agreement = None
        if self.joint_enabled:
            adapted_nose = self.nose_adapter(nose_features)
            adapted_face = self.face_adapter(face_features)
        if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3:
            fusion_weights = self.gate(
                joint_inputs,
                adapted_nose,
                adapted_face,
                effective_available,
            )
            semantic_agreement = self.gate.agreement_signals(
                adapted_nose,
                adapted_face,
            ).to(dtype=dtype)
        else:
            fusion_weights = self.gate(
                joint_inputs
                if self.fusion_mode == SHARED_SPACE_FUSION_V2
                else quality_signals,
                effective_available,
            )
        # A missing branch has exactly zero weight.  Clamp before sqrt so its
        # derivative cannot become infinite (0.5 / sqrt(0)) during modality
        # dropout; the availability mask restores the exact zero afterward.
        fusion_sqrt_weights = torch.where(
            effective_available,
            fusion_weights.clamp_min(1e-8).sqrt(),
            torch.zeros_like(fusion_weights),
        )
        base_fused_features = torch.cat(
            (
                nose_features * fusion_sqrt_weights[:, 0:1],
                face_features * fusion_sqrt_weights[:, 1:2],
            ),
            dim=1,
        )
        base_fused_features = F.normalize(base_fused_features, dim=1)
        joint_features = None
        joint_weights = None
        joint_mix = torch.zeros((), device=device, dtype=dtype)
        if self.joint_enabled:
            joint_weights = (
                fusion_weights
                if self.fusion_mode in SHARED_FUSION_MODES
                else self.view_gate(joint_inputs, effective_available)
            )
            if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3:
                joint_features = self._semantic_residual_fusion(
                    adapted_nose,
                    adapted_face,
                    joint_weights,
                    effective_available,
                )
            else:
                joint_features = self._shared_fusion(
                    adapted_nose,
                    adapted_face,
                    joint_weights,
                    effective_available,
                )
            if self.fusion_mode in SHARED_FUSION_MODES:
                joint_mix = torch.ones((), device=device, dtype=dtype)
                fused_features = joint_features
            else:
                joint_mix = self.joint_mix_logit.sigmoid().to(dtype=dtype)
                fused_features = F.normalize(
                    torch.cat(
                        (
                            base_fused_features * (1.0 - joint_mix).sqrt(),
                            joint_features * joint_mix.sqrt(),
                        ),
                        dim=1,
                    ),
                    dim=1,
                )
        else:
            fused_features = base_fused_features
        output = {
            "features": fused_features,
            "base_features": base_fused_features,
            "joint_features": joint_features,
            "joint_mix": joint_mix,
            "viewpoint_frontality": viewpoint_frontality,
            "nose_features": nose_features,
            "face_features": face_features,
            "fusion_weights": fusion_weights,
            "joint_weights": joint_weights,
            "semantic_agreement": semantic_agreement,
            "conflict_nose_weight": None,
            "effective_branch_available": effective_available,
            "face_crops": face_crops,
            "nose_crops": nose_crops,
            "soft_nose_masks": soft_masks,
        }

        if self.num_classes > 0:
            fused_logits = self._cosine_logits(
                fused_features, self.fused_classifier, self.classifier_scale
            )
            output["logits"] = fused_logits
            output["probabilities"] = fused_logits.softmax(dim=1)

        if targets is not None:
            if self.num_classes <= 0:
                raise RuntimeError("num_classes must be positive for local end-to-end training")
            targets = targets.to(device=device, dtype=torch.long)
            losses = {"loss_fusion_cls": F.cross_entropy(fused_logits, targets)}
            unique, counts = targets.unique(return_counts=True)
            if unique.numel() >= 2 and (counts >= 2).any():
                losses["loss_fusion_triplet"] = triplet_loss(
                    fused_features,
                    targets,
                    margin=0.3,
                    norm_feat=True,
                    hard_mining=True,
                )
                if self.joint_enabled and self.cross_view_weight > 0:
                    losses["loss_cross_view"] = self.cross_view_weight * (
                        viewpoint_supervised_contrastive_loss(
                            joint_features,
                            targets,
                            viewpoint_signals,
                            temperature=self.contrastive_temperature,
                            pose_boost=self.contrastive_pose_boost,
                        )
                    )
            if self.joint_enabled and self.cross_modal_weight > 0:
                both_valid = branch_available.all(dim=1)
                if both_valid.any():
                    if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3:
                        losses["loss_cross_modal"] = (
                            self.cross_modal_weight
                            * cross_modal_supervised_contrastive_loss(
                                adapted_nose[both_valid],
                                adapted_face[both_valid],
                                targets[both_valid],
                                temperature=self.contrastive_temperature,
                            )
                        )
                    else:
                        losses["loss_cross_modal"] = self.cross_modal_weight * (
                            1.0
                            - F.cosine_similarity(
                                adapted_nose[both_valid].float(),
                                adapted_face[both_valid].float(),
                                dim=1,
                            ).mean()
                        )
            if (
                self.fusion_mode == SHARED_SPACE_FUSION_V2
                and self.branch_consistency_weight > 0
            ):
                both_valid = branch_available.all(dim=1)
                if both_valid.any():
                    full_available = branch_available[both_valid]
                    full_weights = self.gate(
                        joint_inputs[both_valid],
                        full_available,
                    )
                    full_features = self._shared_fusion(
                        adapted_nose[both_valid],
                        adapted_face[both_valid],
                        full_weights,
                        full_available,
                    )
                    nose_only = adapted_nose[both_valid]
                    face_only = adapted_face[both_valid]
                    consistency = 0.5 * (
                        1.0
                        - F.cosine_similarity(
                            full_features.float(),
                            nose_only.float(),
                            dim=1,
                        ).mean()
                        + 1.0
                        - F.cosine_similarity(
                            full_features.float(),
                            face_only.float(),
                            dim=1,
                        ).mean()
                    )
                    losses["loss_branch_consistency"] = (
                        self.branch_consistency_weight * consistency
                    )
            if (
                self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3
                and self.semantic_conflict_weight > 0
            ):
                both_indices = branch_available.all(dim=1).nonzero(
                    as_tuple=False
                ).flatten()
                if both_indices.numel() > 1:
                    local_targets = targets.index_select(0, both_indices)
                    permutation, conflict_valid = different_identity_permutation(
                        local_targets
                    )
                    if conflict_valid.any():
                        local_nose = adapted_nose.index_select(0, both_indices)
                        local_face = adapted_face.index_select(0, both_indices)
                        local_inputs = joint_inputs.index_select(0, both_indices)
                        positive_logits = self.gate.reliability_logits(
                            local_inputs,
                            local_nose,
                            local_face,
                        )
                        negative_logits = self.gate.reliability_logits(
                            local_inputs,
                            local_nose.index_select(0, permutation),
                            local_face,
                        )
                        positive_logits = positive_logits[conflict_valid]
                        negative_logits = negative_logits[conflict_valid]
                        classification = 0.5 * (
                            F.binary_cross_entropy_with_logits(
                                positive_logits,
                                torch.ones_like(positive_logits),
                            )
                            + F.binary_cross_entropy_with_logits(
                                negative_logits,
                                torch.zeros_like(negative_logits),
                            )
                        )
                        positive_probability = positive_logits.sigmoid()
                        negative_probability = negative_logits.sigmoid()
                        ranking = F.relu(
                            self.semantic_conflict_margin
                            - positive_probability
                            + negative_probability
                        ).mean()
                        losses["loss_semantic_conflict"] = (
                            self.semantic_conflict_weight
                            * (classification + ranking)
                        )
                        output["conflict_nose_weight"] = (
                            self.semantic_max_nose_weight
                            * negative_probability.detach().mean()
                        )
            if (
                self.fusion_mode == SEMANTIC_RESIDUAL_FUSION_V3
                and self.dominance_weight > 0
            ):
                fused_violation, fused_valid = batch_hard_metric_violation(
                    fused_features,
                    targets,
                )
                nose_violation, nose_metric_valid = batch_hard_metric_violation(
                    nose_features,
                    targets,
                )
                face_violation, face_metric_valid = batch_hard_metric_violation(
                    face_features,
                    targets,
                )
                dominance_valid = (
                    effective_available.all(dim=1)
                    & fused_valid
                    & nose_metric_valid
                    & face_metric_valid
                )
                if dominance_valid.any():
                    best_branch = torch.minimum(
                        nose_violation.detach(),
                        face_violation.detach(),
                    )
                    degradation = F.relu(
                        fused_violation
                        - best_branch
                        - self.dominance_tolerance
                    )
                    losses["loss_branch_dominance"] = (
                        self.dominance_weight
                        * degradation[dominance_valid].mean()
                    )
            nose_valid = branch_available[:, 0]
            if nose_valid.any():
                nose_logits = self._cosine_logits(
                    nose_features[nose_valid], self.nose_classifier, self.classifier_scale
                )
                losses["loss_nose_aux_cls"] = self.nose_aux_weight * F.cross_entropy(
                    nose_logits, targets[nose_valid]
                )
            face_valid = branch_available[:, 1]
            if face_valid.any():
                face_logits = self._cosine_logits(
                    face_features[face_valid], self.face_classifier, self.classifier_scale
                )
                losses["loss_face_aux_cls"] = self.face_aux_weight * F.cross_entropy(
                    face_logits, targets[face_valid]
                )
            output["losses"] = losses
        return output


def _expanded_face_box(
    detection: FaceDetection, image_shape: Sequence[int], padding=0.12
) -> tuple[float, float, float, float]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    pad_x, pad_y = padding * detection.width, padding * detection.height
    return (
        max(x1 - pad_x, 0.0),
        max(y1 - pad_y, 0.0),
        min(x2 + pad_x, float(width)),
        min(y2 + pad_y, float(height)),
    )


def _roll_angle(detection: FaceDetection) -> float:
    left, right = detection.left_eye, detection.right_eye
    return math.atan2(right[1] - left[1], right[0] - left[0])


class MultimodalPetIDPipeline:
    """Run frozen localization, then the locally end-to-end identity model."""

    def __init__(
        self,
        detector: AnyFaceDetector,
        segmenter: SAM2NoseSegmenter,
        identity_model: LocalEndToEndPetIDModel,
        *,
        allow_raw_nose_fallback=True,
        max_long_side=1280,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.identity_model = identity_model
        self.allow_raw_nose_fallback = bool(allow_raw_nose_fallback)
        self.max_long_side = int(max_long_side)

    @property
    def device(self) -> torch.device:
        return next(self.identity_model.parameters()).device

    def _fallback_nose_only(self, image_bgr: np.ndarray) -> list[PetDescriptor]:
        if not self.allow_raw_nose_fallback:
            return []
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images = torch.from_numpy(image_rgb.transpose(2, 0, 1).copy()).float()[None].to(self.device)
        full_roi = torch.tensor([[0, 0, 0, width, height]], dtype=torch.float32)
        masks = torch.ones((1, 1, height, width), dtype=torch.float32)
        resolution = float(np.clip(min(height, width) / 96.0, 0.03, 1.0))
        quality = max(laplacian_sharpness_quality(image_bgr), 0.03) * resolution
        quality_signals = torch.tensor([[quality, 0.0, 0.0, 0.0, resolution, 0.0]])
        view_rows = torch.zeros((1, VIEWPOINT_DIM), dtype=torch.float32)
        available = torch.tensor([[True, False]])
        with torch.inference_mode():
            output = self.identity_model(
                images,
                face_rois=full_roi,
                nose_rois=full_roi,
                roll_angles_radians=torch.zeros(1),
                nose_masks=masks,
                quality_signals=quality_signals,
                viewpoint_signals=view_rows,
                branch_available=available,
            )
        return [
            self._descriptor_from_output(
                output,
                0,
                (quality, 0.0),
                (True, False),
                None,
                None,
                self.identity_model.label_to_identity,
                (width, height),
                tuple(float(value) for value in view_rows[0]),
            )
        ]

    @staticmethod
    def _descriptor_from_output(
        output: dict,
        index: int,
        quality: tuple[float, float],
        available: tuple[bool, bool],
        detection: FaceDetection | None,
        segmentation: NoseSegmentation | None,
        label_to_identity: dict[int, str] | None = None,
        inference_size: tuple[int, int] | None = None,
        viewpoint: tuple[float, ...] = (),
    ) -> PetDescriptor:
        weights = output["fusion_weights"][index].detach().cpu().tolist()
        identity_scores = ()
        if "probabilities" in output and label_to_identity:
            probabilities = output["probabilities"][index].detach().cpu()
            topk = min(5, probabilities.numel())
            values, indices = probabilities.topk(topk)
            identity_scores = tuple(
                (label_to_identity[int(label)], float(probability))
                for probability, label in zip(values, indices)
                if int(label) in label_to_identity
            )
        return PetDescriptor(
            fused_feature=output["features"][index].detach().cpu(),
            nose_feature=output["nose_features"][index].detach().cpu(),
            face_feature=output["face_features"][index].detach().cpu(),
            fusion_weights=(float(weights[0]), float(weights[1])),
            branch_quality=quality,
            branch_available=available,
            detection=detection,
            segmentation=segmentation,
            identity_scores=identity_scores,
            inference_size=inference_size,
            viewpoint=viewpoint,
        )

    def encode_image(self, image) -> list[PetDescriptor]:
        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image}")
        else:
            image_bgr = np.asarray(image)
        height, width = image_bgr.shape[:2]
        if self.max_long_side > 0 and max(height, width) > self.max_long_side:
            scale = self.max_long_side / max(height, width)
            width = max(int(round(width * scale)), 1)
            height = max(int(round(height * scale)), 1)
            image_bgr = cv2.resize(
                image_bgr, (width, height), interpolation=cv2.INTER_AREA
            )
        detections = self.detector.detect(image_bgr)
        if not detections:
            return self._fallback_nose_only(image_bgr)

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images = torch.from_numpy(image_rgb.transpose(2, 0, 1).copy()).float()[None].to(self.device)
        face_rois, nose_rois, angles, masks = [], [], [], []
        quality_rows, view_rows, available_rows, segmentations = [], [], [], []
        for detection in detections:
            aligned_face = crop_aligned_face(image_bgr, detection)
            face_q = face_quality(detection, aligned_face)
            try:
                segmentation = self.segmenter.segment(image_bgr, detection)
                nx1, ny1, nx2, ny2 = segmentation.roi_box_xyxy
                raw_nose = image_bgr[ny1:ny2, nx1:nx2]
                nose_q = nose_quality(segmentation, raw_nose)
                nose_available = True
                mask = segmentation.mask
            except Exception:
                if not self.allow_raw_nose_fallback:
                    nose_available = False
                    segmentation = None
                    nx1, ny1, nx2, ny2 = nose_roi_box(detection, image_bgr.shape)
                    mask = np.zeros((height, width), dtype=bool)
                    nose_q = 0.0
                else:
                    nx1, ny1, nx2, ny2 = nose_roi_box(detection, image_bgr.shape)
                    mask = np.zeros((height, width), dtype=bool)
                    mask[ny1:ny2, nx1:nx2] = True
                    raw_nose = image_bgr[ny1:ny2, nx1:nx2]
                    nose_q = 0.15 * laplacian_sharpness_quality(raw_nose)
                    nose_available = True
                    segmentation = None

            fx1, fy1, fx2, fy2 = _expanded_face_box(detection, image_bgr.shape)
            face_rois.append((0, fx1, fy1, fx2, fy2))
            nose_rois.append((0, nx1, ny1, nx2, ny2))
            angles.append(_roll_angle(detection))
            masks.append(mask)
            nose_resolution = float(np.clip(min(nx2 - nx1, ny2 - ny1) / 96.0, 0.0, 1.0))
            face_resolution = float(
                np.clip(min(detection.width, detection.height) / 160.0, 0.0, 1.0)
            )
            quality_rows.append(
                (
                    nose_q,
                    face_q,
                    detection.confidence,
                    segmentation.predicted_iou if segmentation else 0.0,
                    nose_resolution,
                    face_resolution,
                )
            )
            view_rows.append(tuple(float(value) for value in viewpoint_signals(detection)))
            available_rows.append((nose_available, True))
            segmentations.append(segmentation)

        nose_masks = torch.from_numpy(np.stack(masks)[:, None].astype(np.float32))
        with torch.inference_mode():
            output = self.identity_model(
                images,
                face_rois=torch.tensor(face_rois, dtype=torch.float32),
                nose_rois=torch.tensor(nose_rois, dtype=torch.float32),
                roll_angles_radians=torch.tensor(angles, dtype=torch.float32),
                nose_masks=nose_masks,
                quality_signals=torch.tensor(quality_rows, dtype=torch.float32),
                viewpoint_signals=torch.tensor(view_rows, dtype=torch.float32),
                branch_available=torch.tensor(available_rows, dtype=torch.bool),
            )
        return [
            self._descriptor_from_output(
                output,
                index,
                (float(quality_rows[index][0]), float(quality_rows[index][1])),
                tuple(bool(value) for value in available_rows[index]),
                detections[index],
                segmentations[index],
                self.identity_model.label_to_identity,
                (width, height),
                view_rows[index],
            )
            for index in range(len(detections))
        ]


class DescriptorCache:
    """Safe NumPy/JSON cache keyed by source stat and model namespace."""

    def __init__(self, root, namespace: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)

    def key_for(self, image_path) -> str:
        path = Path(image_path).resolve()
        stat = path.stat()
        payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.namespace}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def save(self, image_path, descriptors: Sequence[PetDescriptor]) -> Path:
        key = self.key_for(image_path)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        metadata = {"source": str(Path(image_path).resolve()), "descriptors": []}
        for index, descriptor in enumerate(descriptors):
            arrays[f"fused_{index}"] = descriptor.fused_feature.numpy()
            arrays[f"nose_{index}"] = descriptor.nose_feature.numpy()
            arrays[f"face_{index}"] = descriptor.face_feature.numpy()
            metadata["descriptors"].append(descriptor.metadata_dict())
        np.savez_compressed(directory / "features.npz", **arrays)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return directory

    def load(self, image_path) -> list[PetDescriptor] | None:
        directory = self.root / self.key_for(image_path)
        feature_path, metadata_path = directory / "features.npz", directory / "metadata.json"
        if not feature_path.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = np.load(feature_path, allow_pickle=False)
        descriptors = []
        for index, item in enumerate(metadata["descriptors"]):
            detection_data = item["detection"]
            detection = None
            if detection_data:
                detection = FaceDetection(
                    tuple(detection_data["bbox_xyxy"]),
                    float(detection_data["confidence"]),
                    tuple(tuple(point) for point in detection_data["landmarks_xy"]),
                    int(detection_data["class_id"]),
                )
            # Cached scoring does not require the full pixel mask.
            descriptors.append(
                PetDescriptor(
                    fused_feature=torch.from_numpy(features[f"fused_{index}"].copy()),
                    nose_feature=torch.from_numpy(features[f"nose_{index}"].copy()),
                    face_feature=torch.from_numpy(features[f"face_{index}"].copy()),
                    fusion_weights=tuple(item["fusion_weights"]),
                    branch_quality=tuple(item["branch_quality"]),
                    branch_available=tuple(item["branch_available"]),
                    detection=detection,
                    segmentation=None,
                    cached_segmentation_metadata=item.get("segmentation"),
                    identity_scores=tuple(
                        (entry["identity"], float(entry["probability"]))
                        for entry in item.get("identity_scores", ())
                    ),
                    inference_size=(
                        tuple(int(value) for value in item["inference_size"])
                        if item.get("inference_size")
                        else None
                    ),
                    viewpoint=tuple(float(value) for value in item.get("viewpoint", ())),
                )
            )
        return descriptors


def pipeline_namespace(paths: Iterable[Path | str], settings: dict) -> str:
    """Build a stable cache namespace from model files and fusion settings."""

    records = []
    for value in paths:
        path = Path(value).resolve()
        stat = path.stat()
        records.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps({"files": records, "settings": settings}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _load_identity_checkpoint(path, device: torch.device) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Multimodal identity checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model", "num_classes", "identity_to_label"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Multimodal identity checkpoint is missing keys: {sorted(missing)}")
    identity_to_label = {
        str(identity): int(label)
        for identity, label in checkpoint["identity_to_label"].items()
    }
    num_classes = int(checkpoint["num_classes"])
    if sorted(identity_to_label.values()) != list(range(num_classes)):
        raise ValueError("Checkpoint identity labels must cover [0, num_classes)")
    checkpoint["identity_to_label"] = identity_to_label
    return checkpoint


def build_local_identity_model(
    cfg, *, device=None, for_training=False, identity_weights=None
):
    """Build only the differentiable ROI-to-identity portion of the system."""

    options = cfg.MULTIMODAL
    device = torch.device(device or cfg.MODEL.DEVICE)
    checkpoint_path = identity_weights
    if checkpoint_path is None:
        checkpoint_path = getattr(options, "IDENTITY_WEIGHTS", "")
    checkpoint = (
        _load_identity_checkpoint(checkpoint_path, device)
        if checkpoint_path
        else None
    )
    nose_encoder = FastReIDDescriptorEncoder.from_files(
        options.NOSE_CONFIG,
        options.NOSE_WEIGHTS,
        device=device,
    )
    face_encoder = DogArcFaceEncoder(options.ARCFACE_WEIGHTS, freeze=True)
    if for_training:
        nose_encoder.configure_trainable_parts(options.NOSE_TRAINABLE_PARTS)
        face_encoder.configure_trainable_parts(options.ARCFACE_TRAINABLE_PARTS)
    else:
        nose_encoder.configure_trainable_parts(())
    checkpoint_state = checkpoint["model"] if checkpoint is not None else {}
    checkpoint_architecture = checkpoint.get("architecture", {}) if checkpoint else {}
    if checkpoint is not None:
        fusion_mode = checkpoint_architecture.get("fusion_mode")
        if fusion_mode is None:
            architecture_name = checkpoint_architecture.get("name")
            fusion_mode = (
                architecture_name
                if architecture_name in SHARED_FUSION_MODES
                else LEGACY_CONCAT_FUSION
            )
    else:
        fusion_mode = options.FUSION_MODE
    checkpoint_has_joint = any(
        key.startswith("nose_adapter.") for key in checkpoint_state
    )
    joint_enabled = (
        bool(checkpoint_architecture.get("joint_enabled", checkpoint_has_joint))
        if checkpoint is not None
        else bool(
            options.JOINT_ENABLED
            or str(fusion_mode).casefold() in SHARED_FUSION_MODES
        )
    )
    identity_model = LocalEndToEndPetIDModel(
        nose_encoder,
        face_encoder,
        nose_size=options.NOSE_SIZE,
        face_size=options.FACE_SIZE,
        num_classes=(
            int(checkpoint["num_classes"])
            if checkpoint is not None
            else options.NUM_CLASSES if for_training else 0
        ),
        branch_priors=(options.NOSE_PRIOR, options.FACE_PRIOR),
        joint_enabled=joint_enabled,
        fusion_mode=fusion_mode,
        joint_dim=int(checkpoint_architecture.get("joint_dim", options.JOINT_DIM)),
        adapter_bottleneck_dim=options.ADAPTER_BOTTLENECK_DIM,
        joint_initial_mix=options.JOINT_INITIAL_MIX,
        modality_dropout=options.MODALITY_DROPOUT if for_training else 0.0,
        cross_view_weight=options.CROSS_VIEW_WEIGHT if for_training else 0.0,
        cross_modal_weight=options.CROSS_MODAL_WEIGHT if for_training else 0.0,
        branch_consistency_weight=(
            options.BRANCH_CONSISTENCY_WEIGHT if for_training else 0.0
        ),
        semantic_max_nose_weight=float(
            checkpoint_architecture.get(
                "semantic_max_nose_weight",
                options.SEMANTIC_MAX_NOSE_WEIGHT,
            )
        ),
        semantic_residual_scale=float(
            checkpoint_architecture.get(
                "semantic_residual_scale",
                options.SEMANTIC_RESIDUAL_SCALE,
            )
        ),
        semantic_conflict_weight=(
            options.SEMANTIC_CONFLICT_WEIGHT if for_training else 0.0
        ),
        semantic_conflict_margin=options.SEMANTIC_CONFLICT_MARGIN,
        dominance_weight=options.DOMINANCE_WEIGHT if for_training else 0.0,
        dominance_tolerance=options.DOMINANCE_TOLERANCE,
        contrastive_temperature=options.CONTRASTIVE_TEMPERATURE,
        contrastive_pose_boost=options.CONTRASTIVE_POSE_BOOST,
        viewpoint_nose_penalty=options.VIEWPOINT_NOSE_PENALTY,
        viewpoint_nose_floor=options.VIEWPOINT_NOSE_FLOOR,
    ).to(device)
    if checkpoint is not None:
        identity_model.load_state_dict(checkpoint["model"], strict=True)
        identity_model.identity_to_label = checkpoint["identity_to_label"]
        identity_model.label_to_identity = {
            label: identity
            for identity, label in identity_model.identity_to_label.items()
        }
    identity_model.train(bool(for_training))
    return identity_model


def build_multimodal_pipeline(cfg, *, device=None, for_training=False):
    """Construct geometry providers and the local end-to-end identity model."""

    options = cfg.MULTIMODAL
    device = torch.device(device or cfg.MODEL.DEVICE)
    identity_model = build_local_identity_model(
        cfg, device=device, for_training=for_training
    )

    detector = AnyFaceDetector(
        options.ANYFACE_WEIGHTS,
        repository_root=options.ANYFACE_ROOT,
        device=device,
        image_size=options.ANYFACE_IMAGE_SIZE,
        confidence_threshold=options.ANYFACE_CONFIDENCE,
    )
    segmenter = SAM2NoseSegmenter(
        options.SAM2_CHECKPOINT,
        config=options.SAM2_CONFIG,
        device=device,
    )
    return MultimodalPetIDPipeline(
        detector,
        segmenter,
        identity_model,
        allow_raw_nose_fallback=options.ALLOW_RAW_NOSE_FALLBACK,
        max_long_side=options.MAX_LONG_SIDE,
    )
