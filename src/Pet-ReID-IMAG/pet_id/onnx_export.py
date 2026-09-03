"""ONNX-friendly deployment boundary for the multimodal identity network.

The production pipeline keeps AnyFace/SAM 2 and rotated ROI extraction outside
the exported graph.  This module consumes their fixed-size crop products and
preserves the exact IMAG + PetFace + learned joint-fusion math used by
``LocalEndToEndPetIDModel``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .multimodal import (
    LEGACY_CONCAT_FUSION,
    SEMANTIC_RESIDUAL_FUSION,
    SHARED_FUSION_MODES,
    SHARED_PROJECTION_FUSION,
    LocalEndToEndPetIDModel,
)


ONNX_INPUT_NAMES = (
    "nose_crop",
    "face_crop",
    "nose_mask",
    "quality_signals",
    "viewpoint_signals",
    "branch_available",
)

ONNX_OUTPUT_NAMES = (
    "embedding",
    "nose_embedding",
    "face_embedding",
    "fusion_weights",
    "joint_weights",
    "viewpoint_frontality",
)


class PreCroppedPetEmbeddingModel(nn.Module):
    """Exportable crop-to-embedding view of a trained joint identity model.

    Inputs are RGB float tensors in the original 0-255 range. ``nose_mask`` is
    the binary/soft mask already transformed into nose-crop coordinates; the
    same 5x5 feathering used by the full PyTorch model remains inside this
    module.
    """

    def __init__(self, identity_model: LocalEndToEndPetIDModel):
        super().__init__()
        if not identity_model.joint_enabled:
            raise ValueError("The ONNX deployment wrapper requires a joint-neck model")

        self.nose_encoder = identity_model.nose_encoder
        self.face_encoder = identity_model.face_encoder
        self.fusion_mode = identity_model.fusion_mode
        self.gate = identity_model.gate
        self.nose_adapter = identity_model.nose_adapter
        self.face_adapter = identity_model.face_adapter
        self.cross_modal_residual = identity_model.cross_modal_residual
        self.joint_dim = int(identity_model.joint_dim)
        self.semantic_residual_scale = float(identity_model.semantic_residual_scale)
        if self.fusion_mode == LEGACY_CONCAT_FUSION:
            self.view_gate = identity_model.view_gate
            self.joint_mix_logit = identity_model.joint_mix_logit
        self.viewpoint_nose_penalty = float(identity_model.viewpoint_nose_penalty)
        self.viewpoint_nose_floor = float(identity_model.viewpoint_nose_floor)
        self.register_buffer(
            "face_mean",
            identity_model.face_mean.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "face_std",
            identity_model.face_std.detach().clone(),
            persistent=False,
        )
        self.eval()

    @staticmethod
    def _apply_gate(gate, quality, available):
        # This is QualityFusionGate.forward without Python-side tensor
        # validation. The validation remains in the application boundary and
        # removing it here prevents data-dependent control flow in ONNX.
        measured = quality[:, :2].clamp(1e-4, 1.0)
        logits = gate.log_priors + measured.log() + gate.residual(quality)
        logits = logits.masked_fill(
            ~available,
            torch.finfo(logits.dtype).min,
        )
        return logits.softmax(dim=1)

    @staticmethod
    def _apply_semantic_gate(
        gate,
        quality,
        adapted_nose,
        adapted_face,
        available,
    ):
        # This mirrors SemanticReliabilityGate without its Python validation,
        # so dynamic ONNX batches retain the exact trained gate math.
        relational = torch.cat(
            (
                (adapted_nose - adapted_face).abs(),
                adapted_nose * adapted_face,
            ),
            dim=1,
        )
        encoded_relation = gate.relation_encoder(relational.float()).to(
            dtype=quality.dtype
        )
        cosine = F.cosine_similarity(adapted_nose.float(), adapted_face.float(), dim=1)
        mean_absolute_difference = (
            (adapted_nose.float() - adapted_face.float()).abs().mean(dim=1)
        )
        agreement = torch.stack((cosine, mean_absolute_difference), dim=1).to(
            dtype=quality.dtype
        )
        reliability_logit = gate.reliability(
            torch.cat(
                (
                    gate.quality_norm(quality.float()).to(dtype=quality.dtype),
                    encoded_relation,
                    agreement,
                ),
                dim=1,
            )
        )
        bounded_nose_weight = gate.max_nose_weight * reliability_logit.sigmoid()
        nose_available = available[:, 0:1]
        face_available = available[:, 1:2]
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

    def forward(
        self,
        nose_crop: torch.Tensor,
        face_crop: torch.Tensor,
        nose_mask: torch.Tensor,
        quality_signals: torch.Tensor,
        viewpoint_signals: torch.Tensor,
        branch_available: torch.Tensor,
    ):
        dtype = nose_crop.dtype
        quality_signals = quality_signals.to(dtype=dtype)
        viewpoint_signals = viewpoint_signals.to(dtype=dtype)
        branch_available = branch_available.to(dtype=torch.bool)

        soft_nose_mask = F.avg_pool2d(
            nose_mask.to(dtype=dtype).clamp(0, 1),
            kernel_size=5,
            stride=1,
            padding=2,
        )
        background = self.nose_encoder.model.pixel_mean.to(dtype=dtype)
        masked_nose = nose_crop * soft_nose_mask + background * (1.0 - soft_nose_mask)
        nose_embedding = F.normalize(
            self.nose_encoder(nose_crop) + self.nose_encoder(masked_nose),
            dim=1,
        )

        normalized_face = (face_crop.div(255.0) - self.face_mean) / self.face_std
        face_embedding = F.normalize(self.face_encoder(normalized_face), dim=1)

        pose_magnitude = viewpoint_signals[:, :3].float().norm(dim=1)
        viewpoint_frontality = (
            self.viewpoint_nose_floor
            + (1.0 - self.viewpoint_nose_floor)
            * torch.exp(-self.viewpoint_nose_penalty * pose_magnitude)
        ).to(dtype=dtype)
        joint_quality = torch.cat(
            (
                quality_signals[:, 0:1] * viewpoint_frontality[:, None],
                quality_signals[:, 1:],
            ),
            dim=1,
        )
        joint_inputs = torch.cat((joint_quality, viewpoint_signals), dim=1)
        adapted_nose = self.nose_adapter(nose_embedding)
        adapted_face = self.face_adapter(face_embedding)
        if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION:
            fusion_weights = self._apply_semantic_gate(
                self.gate,
                joint_inputs,
                adapted_nose,
                adapted_face,
                branch_available,
            )
        else:
            fusion_weights = self._apply_gate(
                self.gate,
                (
                    joint_inputs
                    if self.fusion_mode == SHARED_PROJECTION_FUSION
                    else quality_signals
                ),
                branch_available,
            )
        sqrt_weights = torch.where(
            branch_available,
            fusion_weights.clamp_min(1e-8).sqrt(),
            torch.zeros_like(fusion_weights),
        )
        base_embedding = F.normalize(
            torch.cat(
                (
                    nose_embedding * sqrt_weights[:, 0:1],
                    face_embedding * sqrt_weights[:, 1:2],
                ),
                dim=1,
            ),
            dim=1,
        )

        joint_weights = (
            fusion_weights
            if self.fusion_mode in SHARED_FUSION_MODES
            else self._apply_gate(
                self.view_gate,
                joint_inputs,
                branch_available,
            )
        )
        if self.fusion_mode == SEMANTIC_RESIDUAL_FUSION:
            nose_weight = joint_weights[:, 0:1]
            joint_base = adapted_face * (1.0 - nose_weight) + adapted_nose * nose_weight
            interaction = self.cross_modal_residual(adapted_nose, adapted_face)
            bounded_interaction = torch.tanh(interaction) / (self.joint_dim**0.5)
            both_available = branch_available.all(dim=1, keepdim=True).to(dtype=dtype)
            joint_embedding = F.normalize(
                joint_base
                + (
                    self.semantic_residual_scale
                    * nose_weight
                    * bounded_interaction
                    * both_available
                ),
                dim=1,
            )
            nose_only = branch_available[:, 0:1] & ~branch_available[:, 1:2]
            face_only = branch_available[:, 1:2] & ~branch_available[:, 0:1]
            joint_embedding = torch.where(nose_only, adapted_nose, joint_embedding)
            joint_embedding = torch.where(face_only, adapted_face, joint_embedding)
        else:
            joint_base = (
                adapted_nose * joint_weights[:, 0:1]
                + adapted_face * joint_weights[:, 1:2]
            )
            interaction = self.cross_modal_residual(adapted_nose, adapted_face)
            both_available = branch_available.to(dtype=dtype).prod(
                dim=1,
                keepdim=True,
            )
            joint_embedding = F.normalize(
                joint_base + interaction * both_available,
                dim=1,
            )

        if self.fusion_mode in SHARED_FUSION_MODES:
            embedding = joint_embedding
        else:
            joint_mix = self.joint_mix_logit.sigmoid().to(dtype=dtype)
            embedding = F.normalize(
                torch.cat(
                    (
                        base_embedding * (1.0 - joint_mix).sqrt(),
                        joint_embedding * joint_mix.sqrt(),
                    ),
                    dim=1,
                ),
                dim=1,
            )
        return (
            embedding,
            nose_embedding,
            face_embedding,
            fusion_weights,
            joint_weights,
            viewpoint_frontality,
        )


def extract_precropped_onnx_inputs(
    identity_model: LocalEndToEndPetIDModel,
    *,
    images_0_255: torch.Tensor,
    face_rois: torch.Tensor,
    nose_rois: torch.Tensor,
    roll_angles_radians: torch.Tensor,
    nose_masks: torch.Tensor,
    quality_signals: torch.Tensor,
    viewpoint_signals: torch.Tensor,
    branch_available: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Reproduce the full model's ROI transforms for ONNX parity samples."""

    device, dtype = images_0_255.device, images_0_255.dtype
    face_rois = face_rois.to(device=device, dtype=dtype)
    nose_rois = nose_rois.to(device=device, dtype=dtype)
    angles = roll_angles_radians.to(device=device, dtype=dtype)
    face_crop = identity_model.cropper(
        images_0_255,
        face_rois,
        angles,
        identity_model.face_size,
    )
    nose_crop = identity_model.cropper(
        images_0_255,
        nose_rois,
        angles,
        identity_model.nose_size,
    )
    mask_rois = nose_rois.clone()
    mask_rois[:, 0] = torch.arange(
        mask_rois.shape[0],
        device=device,
        dtype=dtype,
    )
    nose_mask = identity_model.cropper(
        nose_masks.to(device=device, dtype=dtype),
        mask_rois,
        angles,
        identity_model.nose_size,
    ).clamp(0, 1)
    return (
        nose_crop,
        face_crop,
        nose_mask,
        quality_signals.to(device=device, dtype=dtype),
        viewpoint_signals.to(device=device, dtype=dtype),
        branch_available.to(device=device, dtype=torch.bool),
    )
