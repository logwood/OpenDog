"""RGB end-to-end structural nose bridge over a protected spatial-detail model.

The spatial-detail checkpoint remains the protected anchor. This wrapper bypasses the
legacy 2048-to-512 nose adapter for the new evidence path, keeps pre-BN and
post-BN native nose spaces in separate necks, and lets a bounded residual write
the learned structural evidence back into the protected embedding. At
initialization the residual is exactly zero, so the wrapper is bitwise equal
to the parent model (up to deterministic native-feature bookkeeping).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .dual_space_embedding import DualSpaceNoseEmbeddingBridge
from .unified_highres import (
    ShapeDrivenGlobalLetterbox,
    ShapeDrivenRotatedCropper,
    UnifiedHighResolutionPetReID,
    _detail_energy,
    _validate_rgb,
    build_highres_from_checkpoint,
)
from .unified_training import atomic_torch_save


MODEL_TYPE = "unified_highres_structural_nose_pet_reid"
SCHEMA_VERSION = 1


def _native_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Combine two native encoder calls while retaining the autograd graph."""

    def flatten(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4 and value.shape[-2:] == (1, 1):
            value = value[..., 0, 0]
        if value.ndim != 2:
            raise ValueError(f"Expected [batch, channels] native output, got {tuple(value.shape)}")
        return F.normalize(value.float(), dim=1)

    return F.normalize(flatten(left) + flatten(right), dim=1)


class StructuralDetailResidual(nn.Module):
    """Bounded high-resolution residual over a protected model anchor."""

    def __init__(
        self,
        descriptor_dim: int = 512,
        hidden_dim: int = 128,
        maximum_residual_scale: float = 0.12,
        signal_dim: int = 6,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_residual_scale = float(maximum_residual_scale)
        self.signal_dim = int(signal_dim)
        if min(self.descriptor_dim, self.hidden_dim, self.signal_dim) <= 0:
            raise ValueError("residual dimensions must be positive")
        if not 0.0 < self.maximum_residual_scale <= 0.25:
            raise ValueError("maximum_residual_scale must be in (0, 0.25]")
        # anchor, global structural, detail structural, and three pairwise
        # relations: enough capacity to distinguish evidence without a huge
        # 6*512 interaction tensor.
        relation_dim = 6 * self.descriptor_dim + self.signal_dim
        self.signal_norm = nn.LayerNorm(self.signal_dim)
        self.relation_norm = nn.LayerNorm(relation_dim)
        self.delta = nn.Sequential(
            nn.Linear(relation_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.descriptor_dim, bias=False),
        )
        nn.init.zeros_(self.delta[-1].weight)
        self.gain_logit = nn.Parameter(torch.zeros(()))

    def configuration(self) -> dict[str, Any]:
        return {
            "descriptor_dim": self.descriptor_dim,
            "hidden_dim": self.hidden_dim,
            "maximum_residual_scale": self.maximum_residual_scale,
            "signal_dim": self.signal_dim,
            "zero_initialized_exact_anchor": True,
        }

    def forward(
        self,
        anchor: torch.Tensor,
        global_structural: torch.Tensor,
        detail_structural: torch.Tensor,
        signals: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        expected = (anchor.shape[0], self.descriptor_dim)
        for name, value in (
            ("anchor", anchor),
            ("global_structural", global_structural),
            ("detail_structural", detail_structural),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if tuple(signals.shape) != (anchor.shape[0], self.signal_dim):
            raise ValueError(
                f"signals must have shape [{anchor.shape[0]}, {self.signal_dim}]"
            )
        unit_anchor = F.normalize(anchor.float(), dim=1).to(anchor.dtype)
        unit_global = F.normalize(global_structural.float(), dim=1).to(anchor.dtype)
        unit_detail = F.normalize(detail_structural.float(), dim=1).to(anchor.dtype)
        relation = torch.cat(
            (
                unit_anchor,
                unit_global,
                unit_detail,
                (unit_global - unit_anchor).abs(),
                (unit_detail - unit_anchor).abs(),
                unit_global * unit_detail,
                self.signal_norm(signals.float()).to(anchor.dtype),
            ),
            dim=1,
        )
        delta = torch.tanh(self.delta(self.relation_norm(relation.float())))
        delta = delta.to(anchor.dtype) / math.sqrt(self.descriptor_dim)
        gain = self.maximum_residual_scale * self.gain_logit.tanh()
        candidate = anchor + gain.to(anchor.dtype) * delta
        # A zero gain is an exact anchor, including its original norm.  Once
        # active, normalize the candidate for retrieval.
        active = torch.sign(gain.abs()).to(anchor.dtype)
        norm = candidate.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
        normalization = 1.0 + active * (norm.reciprocal().to(anchor.dtype) - 1.0)
        embedding = candidate * normalization
        if not return_aux:
            return embedding
        return {
            "embedding": embedding,
            "anchor": anchor,
            "global_structural": unit_global,
            "detail_structural": unit_detail,
            "signals": signals,
            "delta": delta,
            "global_gain": gain,
        }


class UnifiedHighResolutionStructuralPetReID(nn.Module):
    """Spatial-detail RGB graph with a trainable structural interface."""

    descriptor_dim = 512

    def __init__(
        self,
        detail_model: UnifiedHighResolutionPetReID,
        *,
        bridge_variant: str = "dual_consensus",
        bridge_token_dim: int = 256,
        bridge_bottleneck_dim: int = 128,
        bridge_hidden_dim: int = 256,
        bridge_attention_heads: int = 4,
        bridge_dropout: float = 0.10,
        residual_hidden_dim: int = 128,
        maximum_structural_residual: float = 0.12,
    ) -> None:
        super().__init__()
        if int(detail_model.descriptor_dim) != self.descriptor_dim:
            raise ValueError("Spatial-detail model must emit 512-D descriptors")
        self.detail_model = detail_model
        self.global_sampler = ShapeDrivenGlobalLetterbox(detail_model.global_input_size)
        self.face_detail_cropper = ShapeDrivenRotatedCropper(
            (detail_model.face_detail_size, detail_model.face_detail_size),
            minimum_side=detail_model.global_input_size,
        )
        self.nose_detail_cropper = ShapeDrivenRotatedCropper(
            (detail_model.nose_detail_size, detail_model.nose_detail_size),
            minimum_side=detail_model.global_input_size,
        )
        self.global_bridge = DualSpaceNoseEmbeddingBridge(
            variant=bridge_variant,
            token_dim=bridge_token_dim,
            bottleneck_dim=bridge_bottleneck_dim,
            hidden_dim=bridge_hidden_dim,
            attention_heads=bridge_attention_heads,
            dropout=bridge_dropout,
        )
        self.detail_bridge = DualSpaceNoseEmbeddingBridge(
            variant=bridge_variant,
            token_dim=bridge_token_dim,
            bottleneck_dim=bridge_bottleneck_dim,
            hidden_dim=bridge_hidden_dim,
            attention_heads=bridge_attention_heads,
            dropout=bridge_dropout,
        )
        self.structural_residual = StructuralDetailResidual(
            self.descriptor_dim,
            hidden_dim=residual_hidden_dim,
            maximum_residual_scale=maximum_structural_residual,
        )
        full_mask = torch.ones(
            1,
            1,
            detail_model.nose_detail_size,
            detail_model.nose_detail_size,
        )
        self.register_buffer(
            "nose_detail_feather_mask",
            F.avg_pool2d(full_mask, kernel_size=5, stride=1, padding=2),
            persistent=False,
        )
        self._nose_trainable_parts: tuple[str, ...] = ()
        self.configure_trainable(nose_encoder_parts=(), structural=True)

    @property
    def global_input_size(self) -> int:
        return int(self.detail_model.global_input_size)

    @property
    def face_detail_size(self) -> int:
        return int(self.detail_model.face_detail_size)

    @property
    def nose_detail_size(self) -> int:
        return int(self.detail_model.nose_detail_size)

    @property
    def maximum_input_side(self) -> int:
        return int(self.detail_model.maximum_input_side)

    def configuration(self) -> dict[str, Any]:
        return {
            "global_input_size": self.global_input_size,
            "face_detail_size": self.face_detail_size,
            "nose_detail_size": self.nose_detail_size,
            "maximum_input_side": self.maximum_input_side,
            "bridge": self.global_bridge.configuration(),
            "detail_bridge": self.detail_bridge.configuration(),
            "structural_residual": self.structural_residual.configuration(),
            "detail_configuration": self.detail_model.configuration(),
        }

    def configure_trainable(
        self,
        *,
        nose_encoder_parts: tuple[str, ...] = (),
        structural: bool = True,
    ) -> None:
        """Freeze geometry/ArcFace/legacy heads; optionally unfreeze nose tail."""

        self.detail_model.requires_grad_(False)
        semantic = self.detail_model.parent_model.base_model
        encoder = semantic.nose_encoder
        encoder.configure_trainable_parts(tuple(nose_encoder_parts))
        self._nose_trainable_parts = tuple(nose_encoder_parts)
        self.global_bridge.requires_grad_(bool(structural))
        self.detail_bridge.requires_grad_(bool(structural))
        self.structural_residual.requires_grad_(bool(structural))

    def train(self, mode: bool = True):
        super().train(mode)
        # Everything inherited from the spatial-detail model is a frozen anchor. The
        # FastReID wrapper re-enables only the explicitly selected nose parts.
        self.detail_model.eval()
        semantic = self.detail_model.parent_model.base_model
        semantic.nose_encoder.train(bool(mode))
        for name, module in semantic.nose_encoder.model.backbone.named_children():
            module.train(bool(mode) and name in self._nose_trainable_parts)
        if "heads" in self._nose_trainable_parts:
            semantic.nose_encoder.model.heads.train(bool(mode))
        self.global_bridge.train(mode)
        self.detail_bridge.train(mode)
        self.structural_residual.train(mode)
        return self

    @staticmethod
    def _capture_pool(encoder: nn.Module):
        captured: list[torch.Tensor] = []

        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            captured.append(output)

        handle = encoder.model.heads.pool_layer.register_forward_hook(hook)
        return captured, handle

    def _native_nose_pair(
        self,
        nose_crop: torch.Tensor,
        feathered_nose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.detail_model.parent_model.base_model.nose_encoder
        captured, handle = self._capture_pool(encoder)
        try:
            post_left = encoder(nose_crop)
            post_right = encoder(feathered_nose)
        finally:
            handle.remove()
        if len(captured) != 2:
            raise RuntimeError(f"Expected two native nose calls, got {len(captured)}")
        pre = _native_pair(captured[0], captured[1])
        post = _native_pair(post_left, post_right)
        return pre, post

    def _global_features(
        self,
        global_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        semantic = self.detail_model.parent_model.base_model
        geometry_frontend = semantic.geometry_frontend
        geometry, _ = geometry_frontend._localize(global_rgb)
        stable_boxes, stable_angles = semantic.geometry_discretizer(
            geometry.boxes_cxcywh,
            geometry.angle_radians,
        )
        face_crops = []
        for scale in semantic.face_crop_scales:
            face_box = stable_boxes[:, 0]
            if scale != 1.0:
                face_box = torch.cat((face_box[:, :2], face_box[:, 2:] * scale), dim=1)
            face_crops.append(
                geometry_frontend.cropper(global_rgb, face_box, stable_angles)
            )
        face_batch = torch.cat(face_crops, dim=0)
        face_values = geometry_frontend._backbone_descriptor(
            geometry_frontend._normalize(face_batch)
        ).reshape(len(face_crops), -1, self.descriptor_dim)
        weights = semantic.face_crop_weights_tensor.to(face_values.dtype).view(-1, 1, 1)
        face_descriptor = F.normalize((face_values * weights).sum(dim=0), dim=1)
        nose_crop = semantic.nose_cropper(
            global_rgb,
            stable_boxes[:, 1],
            stable_angles,
        )
        feather = semantic.nose_feather_mask.to(nose_crop.dtype)
        background = semantic.nose_encoder.model.pixel_mean.to(nose_crop.dtype)
        feathered = nose_crop * feather + background * (1.0 - feather)
        nose_pre, nose_post = self._native_nose_pair(nose_crop, feathered)
        return {
            "face_descriptor": face_descriptor,
            "nose_pre": nose_pre,
            "nose_post": nose_post,
            "geometry_confidence": geometry.confidence,
            "boxes": stable_boxes,
            "angles": stable_angles,
            "face_crops": face_crops[0],
            "nose_crop": nose_crop,
        }

    def forward(self, rgb_0_255: torch.Tensor, *, return_aux: bool = False):
        rgb_0_255 = _validate_rgb(rgb_0_255)
        global_rgb, detail_scale, detail_availability = self.global_sampler(rgb_0_255)

        # Obtain the protected embedding as a detached teacher anchor. The
        # structural path below is the only route through the new interface.
        with torch.no_grad():
            protected_anchor = self.detail_model(global_rgb if False else rgb_0_255)
        # The parent model already performs the dynamic global/detail path;
        # recompute native features from the same geometry so gradients can
        # reach the selected nose tail. Calling it under no_grad keeps
        # the baseline anchor inexpensive in the backward graph.
        global_features = self._global_features(global_rgb)
        global_structural = self.global_bridge(
            global_features["face_descriptor"],
            global_features["nose_pre"],
            global_features["nose_post"],
        )

        semantic = self.detail_model.parent_model.base_model
        geometry_frontend = semantic.geometry_frontend
        face_detail_crop = self.face_detail_cropper(
            rgb_0_255,
            global_features["boxes"][:, 0],
            global_features["angles"],
        )
        nose_detail_crop = self.nose_detail_cropper(
            rgb_0_255,
            global_features["boxes"][:, 1],
            global_features["angles"],
        )
        detail_face = geometry_frontend._backbone_descriptor(
            geometry_frontend._normalize(face_detail_crop)
        )
        feather = self.nose_detail_feather_mask.to(nose_detail_crop.dtype)
        background = semantic.nose_encoder.model.pixel_mean.to(nose_detail_crop.dtype)
        feathered_detail = nose_detail_crop * feather + background * (1.0 - feather)
        detail_pre, detail_post = self._native_nose_pair(
            nose_detail_crop,
            feathered_detail,
        )
        detail_structural = self.detail_bridge(detail_face, detail_pre, detail_post)

        batch = rgb_0_255.shape[0]
        scale = detail_scale.reshape(-1).expand(batch)
        available = detail_availability.reshape(-1).expand(batch)
        confidence = global_features["geometry_confidence"][:, 0]
        signals = torch.stack(
            (
                confidence,
                available,
                torch.log2(scale.clamp_min(1.0)).div(2.0).clamp(0.0, 1.0),
                _detail_energy(face_detail_crop),
                _detail_energy(nose_detail_crop),
                F.cosine_similarity(
                    global_structural.float(), detail_structural.float(), dim=1
                ),
            ),
            dim=1,
        )
        structural = self.structural_residual(
            protected_anchor.detach(),
            global_structural,
            detail_structural,
            signals,
            return_aux=return_aux,
        )
        if not return_aux:
            return structural
        return {
            **structural,
            "protected_anchor": protected_anchor,
            "global_rgb": global_rgb,
            "global_face_descriptor": global_features["face_descriptor"],
            "global_nose_pre_descriptor": global_features["nose_pre"],
            "global_nose_post_descriptor": global_features["nose_post"],
            "detail_face_descriptor": detail_face,
            "detail_nose_pre_descriptor": detail_pre,
            "detail_nose_post_descriptor": detail_post,
            "boxes_cxcywh": global_features["boxes"],
            "angle_radians": global_features["angles"],
            "geometry_confidence": global_features["geometry_confidence"],
            "detail_scale": scale,
            "detail_availability": available,
            "detail_face_crop": face_detail_crop,
            "detail_nose_crop": nose_detail_crop,
        }


def build_structural_from_detail_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
    **kwargs: Any,
) -> tuple[UnifiedHighResolutionStructuralPetReID, dict[str, Any]]:
    detail_model, detail_payload = build_highres_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=verify_sources,
    )
    model = UnifiedHighResolutionStructuralPetReID(detail_model, **kwargs)
    return model.to(device).eval(), detail_payload


def create_structural_checkpoint(
    model: UnifiedHighResolutionStructuralPetReID,
    *,
    detail_checkpoint: str | Path,
    training: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(detail_checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    from .unified_highres import sha256_file

    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_config": model.configuration(),
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "sources": {
            "detail_checkpoint": {
                "path": str(source),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        },
        "runtime_contract": {
            "inputs": {
                "rgb": {
                    "dtype": "float32",
                    "shape": ["N", 3, "H", "W"],
                    "raw_spatial_input": True,
                }
            },
            "outputs": {
                "embedding": {
                    "dtype": "float32",
                    "shape": ["N", 512],
                    "l2_normalized": True,
                }
            },
            "external_models": [],
        },
        "training": training,
        "selection": selection,
        "promotion_status": "development_validation_required",
        "default_backend_changed": False,
    }


def save_structural_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    atomic_torch_save(payload, path)
