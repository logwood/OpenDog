"""One-pass implementation of the structural spatial-detail RGB bridge.

The first prototype called the protected graph once for an anchor and then
recomputed geometry and nose crops for the trainable path.  This module builds
the same anchor from those already-computed native features, avoiding a
second expensive backbone traversal while preserving the exact teacher path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .unified_highres_structural import (
    MODEL_TYPE,
    SCHEMA_VERSION,
    UnifiedHighResolutionStructuralPetReID,
    _detail_energy,
    _validate_rgb,
    build_highres_from_checkpoint,
)
from .unified_highres import sha256_file
from .release_compatibility import (
    detail_checkpoint_source,
    migrate_structural_state_dict,
)


class UnifiedHighResolutionStructuralOnePassPetReID(
    UnifiedHighResolutionStructuralPetReID
):
    """Structural model with one geometry/identity traversal per input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Keep the direction exactly zero (protected anchor at initialization), while
        # making the scalar gain differentiable from the first optimizer step.
        with torch.no_grad():
            self.structural_residual.gain_logit.fill_(0.2)

    def _build_detail_hierarchy(
        self,
        global_features: dict[str, torch.Tensor],
        detail_face: torch.Tensor,
        detail_nose: torch.Tensor,
        detail_scale: torch.Tensor,
        detail_availability: torch.Tensor,
        face_detail_crop: torch.Tensor,
        nose_detail_crop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recreate the three frozen identity levels from native endpoints."""

        semantic = self.detail_model.parent_model.base_model
        parent_refiner = self.detail_model.parent_model.refiner
        confidence = global_features["geometry_confidence"][:, 0]
        with torch.no_grad():
            adapted_global = semantic.nose_adapter(global_features["nose_post"])
            semantic_output = semantic.fusion(
                global_features["face_descriptor"],
                adapted_global,
                confidence,
                return_aux=True,
            )
            parent_output = parent_refiner(
                semantic_output["embedding"],
                semantic_output["face_descriptor"],
                semantic_output["adapted_nose_descriptor"],
                confidence,
                return_aux=True,
            )
            adapted_detail = semantic.nose_adapter(detail_nose)
            detail_output = self.detail_model.refiner(
                parent_output["embedding"],
                detail_face,
                adapted_detail,
                confidence,
                detail_scale,
                detail_availability,
                _detail_energy(face_detail_crop),
                _detail_energy(nose_detail_crop),
                return_aux=True,
            )
        return {
            "semantic_embedding": semantic_output["embedding"].detach(),
            "parent_embedding": parent_output["embedding"].detach(),
            "detail_embedding": detail_output["embedding"].detach(),
        }

    def _fuse_structural_hierarchy(
        self,
        hierarchy: dict[str, torch.Tensor],
        global_bridge_output: dict[str, torch.Tensor],
        detail_bridge_output: dict[str, torch.Tensor],
        signals: torch.Tensor,
        continuous_context: torch.Tensor,
        *,
        return_aux: bool,
    ):
        """Legacy control: one correction after the complete protected graph."""

        del continuous_context
        return self.structural_residual(
            hierarchy["detail_embedding"],
            global_bridge_output["embedding"],
            detail_bridge_output["embedding"],
            signals,
            return_aux=return_aux,
        )

    def forward(self, rgb_0_255: torch.Tensor, *, return_aux: bool = False):
        rgb_0_255 = _validate_rgb(rgb_0_255)
        global_rgb, detail_scale, detail_availability = self.global_sampler(rgb_0_255)
        global_features = self._global_features(global_rgb)
        global_bridge_output = self.global_bridge(
            global_features["face_descriptor"],
            global_features["nose_pre"],
            global_features["nose_post"],
            return_aux=True,
        )
        global_structural = global_bridge_output["embedding"]

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
        semantic = self.detail_model.parent_model.base_model
        geometry_frontend = semantic.geometry_frontend
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
        detail_bridge_output = self.detail_bridge(
            detail_face,
            detail_pre,
            detail_post,
            return_aux=True,
        )
        detail_structural = detail_bridge_output["embedding"]

        batch = rgb_0_255.shape[0]
        scale = detail_scale.reshape(-1).expand(batch)
        available = detail_availability.reshape(-1).expand(batch)
        confidence = global_features["geometry_confidence"][:, 0]
        face_energy = _detail_energy(face_detail_crop)
        nose_energy = _detail_energy(nose_detail_crop)
        structural_cosine = F.cosine_similarity(
            global_structural.float(), detail_structural.float(), dim=1
        )
        signals = torch.stack(
            (
                confidence,
                available,
                torch.log2(scale.clamp_min(1.0)).div(2.0).clamp(0.0, 1.0),
                face_energy,
                nose_energy,
                structural_cosine,
            ),
            dim=1,
        )
        # Continuous observations only: no hand-selected threshold or branch
        # weight is applied to the integrated model's token context.
        continuous_context = torch.cat(
            (
                confidence[:, None],
                torch.log2(scale.clamp_min(1.0))[:, None],
                face_energy[:, None],
                nose_energy[:, None],
                structural_cosine[:, None],
                global_features["boxes"].reshape(batch, -1),
                global_features["angles"].sin()[:, None],
                global_features["angles"].cos()[:, None],
            ),
            dim=1,
        )
        detail_hierarchy = self._build_detail_hierarchy(
            global_features,
            detail_face,
            detail_post,
            scale,
            available,
            face_detail_crop,
            nose_detail_crop,
        )
        structural = self._fuse_structural_hierarchy(
            detail_hierarchy,
            global_bridge_output,
            detail_bridge_output,
            signals,
            continuous_context,
            return_aux=return_aux,
        )
        if not return_aux:
            return structural
        return {
            **structural,
            "protected_anchor": detail_hierarchy["detail_embedding"],
            "semantic_embedding": detail_hierarchy["semantic_embedding"],
            "parent_embedding": detail_hierarchy["parent_embedding"],
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


class StructuralTokenBlock(nn.Module):
    """Pre-norm identity-token block with inspectable self-attention."""

    def __init__(
        self,
        descriptor_dim: int,
        *,
        attention_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(descriptor_dim)
        self.attention = nn.MultiheadAttention(
            descriptor_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(descriptor_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(descriptor_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, descriptor_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # Near-identity checkpoint initialization without a runtime residual
        # cap.  These are ordinary trainable weights and can grow freely.
        nn.init.normal_(self.attention.out_proj.weight, mean=0.0, std=2.0e-4)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.normal_(self.feedforward[-1].weight, mean=0.0, std=2.0e-4)
        nn.init.zeros_(self.feedforward[-1].bias)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(tokens.float()).to(tokens.dtype)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        feedforward = self.feedforward(
            self.feedforward_norm(tokens.float()).to(tokens.dtype)
        )
        return tokens + self.feedforward_dropout(feedforward), weights


class EndToEndStructuralIdentityCore(nn.Module):
    """Learn one identity vector from all face, nose, detail, and geometry tokens.

    There is no hand-authored branch threshold, expert prior, bounded residual,
    or mandatory similarity to the protected output. The frozen levels are tokens
    that provide a strong checkpoint initialization; self-attention can rewrite
    the identity token using every native-space nose and high-resolution token.
    """

    token_names = (
        "identity_detail",
        "semantic_anchor",
        "parent_anchor",
        "global_face",
        "global_nose_pre",
        "global_nose_post",
        "global_bridge",
        "detail_face",
        "detail_nose_pre",
        "detail_nose_post",
        "detail_bridge",
        "continuous_geometry",
    )
    nose_token_indices = (4, 5, 6, 8, 9, 10)

    def __init__(
        self,
        descriptor_dim: int = 512,
        *,
        native_token_dim: int = 256,
        context_dim: int = 15,
        depth: int = 2,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.native_token_dim = int(native_token_dim)
        self.context_dim = int(context_dim)
        self.depth = int(depth)
        self.attention_heads = int(attention_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)
        if min(
            self.descriptor_dim,
            self.native_token_dim,
            self.context_dim,
            self.depth,
            self.attention_heads,
            self.feedforward_dim,
        ) <= 0:
            raise ValueError("token fusion dimensions must be positive")
        if self.descriptor_dim % self.attention_heads:
            raise ValueError("descriptor_dim must divide evenly into attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.global_pre_projection = self._native_projection()
        self.global_post_projection = self._native_projection()
        self.detail_pre_projection = self._native_projection()
        self.detail_post_projection = self._native_projection()
        self.geometry_token = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.descriptor_dim, bias=False),
        )
        self.token_type_embedding = nn.Parameter(
            torch.empty(1, len(self.token_names), self.descriptor_dim)
        )
        nn.init.normal_(self.token_type_embedding, mean=0.0, std=0.002)
        self.blocks = nn.ModuleList(
            [
                StructuralTokenBlock(
                    self.descriptor_dim,
                    attention_heads=self.attention_heads,
                    feedforward_dim=self.feedforward_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.identity_projection = nn.Linear(
            self.descriptor_dim,
            self.descriptor_dim,
            bias=False,
        )
        nn.init.eye_(self.identity_projection.weight)

    def _native_projection(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.native_token_dim),
            nn.Linear(self.native_token_dim, self.descriptor_dim, bias=False),
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "end_to_end_structural_identity_transformer",
            "descriptor_dim": self.descriptor_dim,
            "native_token_dim": self.native_token_dim,
            "context_dim": self.context_dim,
            "depth": self.depth,
            "attention_heads": self.attention_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "tokens": list(self.token_names),
            "manual_branch_thresholds": False,
            "bounded_output_residual": False,
            "fixed_expert_weights": False,
            "initialization_role": "trainable_identity_token",
            "near_identity_checkpoint_initialization": True,
            "runtime_update_cap": False,
        }

    def _descriptor(self, name: str, value: torch.Tensor, batch: int) -> torch.Tensor:
        if tuple(value.shape) != (batch, self.descriptor_dim):
            raise ValueError(
                f"{name} must have shape [{batch}, {self.descriptor_dim}], "
                f"got {tuple(value.shape)}"
            )
        return F.normalize(value.float(), dim=1)

    def _native(
        self,
        name: str,
        value: torch.Tensor,
        batch: int,
        projection: nn.Module,
    ) -> torch.Tensor:
        if tuple(value.shape) != (batch, self.native_token_dim):
            raise ValueError(
                f"{name} must have shape [{batch}, {self.native_token_dim}], "
                f"got {tuple(value.shape)}"
            )
        return F.normalize(projection(value.float()), dim=1)

    def forward(
        self,
        *,
        semantic_embedding: torch.Tensor,
        parent_embedding: torch.Tensor,
        detail_embedding: torch.Tensor,
        global_face: torch.Tensor,
        global_nose_pre: torch.Tensor,
        global_nose_post: torch.Tensor,
        global_structural: torch.Tensor,
        detail_face: torch.Tensor,
        detail_nose_pre: torch.Tensor,
        detail_nose_post: torch.Tensor,
        detail_structural: torch.Tensor,
        continuous_context: torch.Tensor,
        return_aux: bool = False,
    ):
        batch = detail_embedding.shape[0]
        if tuple(continuous_context.shape) != (batch, self.context_dim):
            raise ValueError(
                f"continuous_context must have shape [{batch}, {self.context_dim}], "
                f"got {tuple(continuous_context.shape)}"
            )
        token_rows = (
            self._descriptor("detail_embedding", detail_embedding, batch),
            self._descriptor("semantic_embedding", semantic_embedding, batch),
            self._descriptor("parent_embedding", parent_embedding, batch),
            self._descriptor("global_face", global_face, batch),
            self._native(
                "global_nose_pre",
                global_nose_pre,
                batch,
                self.global_pre_projection,
            ),
            self._native(
                "global_nose_post",
                global_nose_post,
                batch,
                self.global_post_projection,
            ),
            self._descriptor("global_structural", global_structural, batch),
            self._descriptor("detail_face", detail_face, batch),
            self._native(
                "detail_nose_pre",
                detail_nose_pre,
                batch,
                self.detail_pre_projection,
            ),
            self._native(
                "detail_nose_post",
                detail_nose_post,
                batch,
                self.detail_post_projection,
            ),
            self._descriptor("detail_structural", detail_structural, batch),
            F.normalize(self.geometry_token(continuous_context.float()), dim=1),
        )
        tokens = torch.stack(token_rows, dim=1)
        tokens = tokens + self.token_type_embedding.to(tokens.dtype)
        attention_rows = []
        for block in self.blocks:
            tokens, attention = block(tokens)
            attention_rows.append(attention)
        embedding = F.normalize(
            self.identity_projection(tokens[:, 0].float()), dim=1
        )
        if not return_aux:
            return embedding

        attention_stack = torch.stack(attention_rows, dim=1)
        final_identity_attention = attention_stack[:, -1, :, 0].mean(dim=1)
        nose_attention_mass = final_identity_attention[
            :, list(self.nose_token_indices)
        ].sum(dim=1).mean()
        return {
            "embedding": embedding,
            "anchor": F.normalize(detail_embedding.float(), dim=1),
            "global_structural": F.normalize(global_structural.float(), dim=1),
            "detail_structural": F.normalize(detail_structural.float(), dim=1),
            "signals": continuous_context,
            "identity_tokens": tokens,
            "token_attention": attention_stack,
            "nose_attention_mass": nose_attention_mass,
            # Compatibility with the existing trainer's scalar diagnostic.
            "global_gain": nose_attention_mass,
        }


class UnifiedHighResolutionIntegratedStructuralPetReID(
    UnifiedHighResolutionStructuralOnePassPetReID
):
    """One RGB graph with native nose tokens inside the identity backbone."""

    def __init__(
        self,
        *args: Any,
        integrated_depth: int = 2,
        integrated_attention_heads: int = 8,
        integrated_feedforward_dim: int = 1024,
        integrated_dropout: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.global_bridge.pre_neck is None or self.detail_bridge.pre_neck is None:
            raise ValueError("integrated fusion requires a dual-space bridge variant")
        if self.global_bridge.token_dim != self.detail_bridge.token_dim:
            raise ValueError("global/detail bridges must use the same token dimension")
        # Keep the historical attribute name so the established trainer and
        # checkpoint schema train/save this module without a third source edit.
        self.structural_residual = EndToEndStructuralIdentityCore(
            self.descriptor_dim,
            native_token_dim=int(self.global_bridge.token_dim),
            depth=integrated_depth,
            attention_heads=integrated_attention_heads,
            feedforward_dim=integrated_feedforward_dim,
            dropout=integrated_dropout,
        )

    def configure_trainable(
        self,
        *,
        nose_encoder_parts: tuple[str, ...] = (),
        structural: bool = True,
    ) -> None:
        super().configure_trainable(
            nose_encoder_parts=nose_encoder_parts,
            structural=structural,
        )
        # score_scale belonged to the bridge's old standalone classifier API.
        # The integrated backbone consumes bridge tokens/embeddings directly,
        # so these two scalars are deliberately absent from the optimizer.
        self.global_bridge.logit_scale_log.requires_grad_(False)
        self.detail_bridge.logit_scale_log.requires_grad_(False)

    @staticmethod
    def _required_native_token(
        output: dict[str, torch.Tensor], name: str
    ) -> torch.Tensor:
        value = output.get(name)
        if value is None:
            raise RuntimeError(f"Integrated bridge did not return {name}")
        return value

    def _fuse_structural_hierarchy(
        self,
        hierarchy: dict[str, torch.Tensor],
        global_bridge_output: dict[str, torch.Tensor],
        detail_bridge_output: dict[str, torch.Tensor],
        signals: torch.Tensor,
        continuous_context: torch.Tensor,
        *,
        return_aux: bool,
    ):
        del signals
        return self.structural_residual(
            semantic_embedding=hierarchy["semantic_embedding"],
            parent_embedding=hierarchy["parent_embedding"],
            detail_embedding=hierarchy["detail_embedding"],
            global_face=global_bridge_output["face_descriptor"],
            global_nose_pre=self._required_native_token(
                global_bridge_output, "nose_pre_token"
            ),
            global_nose_post=self._required_native_token(
                global_bridge_output, "nose_post_token"
            ),
            global_structural=global_bridge_output["embedding"],
            detail_face=detail_bridge_output["face_descriptor"],
            detail_nose_pre=self._required_native_token(
                detail_bridge_output, "nose_pre_token"
            ),
            detail_nose_post=self._required_native_token(
                detail_bridge_output, "nose_post_token"
            ),
            detail_structural=detail_bridge_output["embedding"],
            continuous_context=continuous_context,
            return_aux=return_aux,
        )


def build_onepass_from_detail_checkpoint(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
    **kwargs: Any,
):
    detail_model, payload = build_highres_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=verify_sources,
    )
    model = UnifiedHighResolutionStructuralOnePassPetReID(detail_model, **kwargs)
    return model.to(device).eval(), payload


def build_integrated_from_detail_checkpoint(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
    **kwargs: Any,
):
    detail_model, payload = build_highres_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=verify_sources,
    )
    model = UnifiedHighResolutionIntegratedStructuralPetReID(detail_model, **kwargs)
    return model.to(device).eval(), payload


def build_integrated_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    detail_checkpoint_override: str | Path | None = None,
    verify_sources: bool = True,
):
    """Rebuild a trained integrated model without the legacy residual loader."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported integrated structural checkpoint schema")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint is not a structural high-resolution model")
    configuration = payload["model_config"]
    core = configuration["structural_residual"]
    if core.get("type") != "end_to_end_structural_identity_transformer":
        raise ValueError("Checkpoint does not contain the integrated token backbone")
    source = detail_checkpoint_source(payload["sources"])
    detail_path = (
        Path(detail_checkpoint_override).expanduser().resolve()
        if detail_checkpoint_override is not None
        else Path(source["path"]).expanduser().resolve()
    )
    if not detail_path.is_file():
        raise FileNotFoundError(detail_path)
    if verify_sources and sha256_file(detail_path) != source["sha256"]:
        raise RuntimeError("Spatial-detail source checkpoint hash mismatch")
    bridge = configuration["bridge"]
    model, _ = build_integrated_from_detail_checkpoint(
        detail_path,
        device=device,
        verify_sources=verify_sources,
        bridge_variant=str(bridge["variant"]),
        bridge_token_dim=int(bridge["token_dim"]),
        bridge_bottleneck_dim=int(bridge["bottleneck_dim"]),
        bridge_hidden_dim=int(bridge["hidden_dim"]),
        bridge_attention_heads=int(bridge["attention_heads"]),
        bridge_dropout=float(bridge["dropout"]),
        integrated_depth=int(core["depth"]),
        integrated_attention_heads=int(core["attention_heads"]),
        integrated_feedforward_dim=int(core["feedforward_dim"]),
        integrated_dropout=float(core["dropout"]),
    )
    incompatible = model.load_state_dict(
        migrate_structural_state_dict(payload["model"]), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Integrated structural checkpoint mismatch: {incompatible}")
    model.configure_trainable(nose_encoder_parts=(), structural=False)
    return model.to(device).eval(), payload
