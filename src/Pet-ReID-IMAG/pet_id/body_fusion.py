"""Feature-level fusion that promotes whole-body evidence into the main trunk.

The public identity representation stays a 512-D unit vector.  Face and body
form the primary descriptor; the already-adapted nose descriptor is then added
as the same bounded semantic residual used by the nose+face model.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .multimodal import (
    CrossModalResidual,
    ResidualProjectionAdapter,
    SemanticReliabilityGate,
)


class BoundedBodyReliabilityGate(nn.Module):
    """Predict a bounded body contribution when face and body both exist.

    ``branch_available`` uses ``[body, face]`` order.  The configured upper
    bound applies only when both branches exist; single-branch fallbacks retain
    the available descriptor with weight one.
    """

    def __init__(
        self,
        quality_dim: int,
        feature_dim: int,
        *,
        hidden_dim: int = 128,
        initial_body_weight: float = 0.30,
        max_body_weight: float = 0.55,
    ) -> None:
        super().__init__()
        self.max_body_weight = float(max_body_weight)
        initial_body_weight = float(initial_body_weight)
        if not 0.0 < initial_body_weight < self.max_body_weight < 1.0:
            raise ValueError(
                "Expected 0 < initial_body_weight < max_body_weight < 1"
            )

        hidden_dim = int(hidden_dim)
        relation_dim = max(16, hidden_dim // 2)
        feature_dim = int(feature_dim)
        quality_dim = int(quality_dim)
        self.quality_dim = quality_dim
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, relation_dim),
            nn.GELU(),
        )
        self.quality_norm = nn.LayerNorm(quality_dim)
        self.reliability = nn.Sequential(
            nn.Linear(relation_dim + quality_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        probability = initial_body_weight / self.max_body_weight
        initial_logit = math.log(probability / (1.0 - probability))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.constant_(self.reliability[-1].bias, initial_logit)

    @staticmethod
    def agreement_signals(
        body_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> torch.Tensor:
        cosine = F.cosine_similarity(
            body_features.float(), face_features.float(), dim=1
        )
        mean_absolute_difference = (
            body_features.float() - face_features.float()
        ).abs().mean(dim=1)
        return torch.stack((cosine, mean_absolute_difference), dim=1)

    def reliability_logits(
        self,
        quality_signals: torch.Tensor,
        body_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> torch.Tensor:
        if quality_signals.ndim != 2 or quality_signals.shape[1] != self.quality_dim:
            raise ValueError(
                f"quality_signals must have shape [batch, {self.quality_dim}]"
            )
        if body_features.shape != face_features.shape or body_features.ndim != 2:
            raise ValueError("Aligned body/face features must have the same 2D shape")
        if body_features.shape[0] != quality_signals.shape[0]:
            raise ValueError("Quality and feature batches must have the same size")
        relational = torch.cat(
            (
                (body_features - face_features).abs(),
                body_features * face_features,
            ),
            dim=1,
        )
        encoded_relation = self.relation_encoder(relational.float()).to(
            dtype=quality_signals.dtype
        )
        agreement = self.agreement_signals(body_features, face_features).to(
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
        body_features: torch.Tensor,
        face_features: torch.Tensor,
        branch_available: torch.Tensor,
    ) -> torch.Tensor:
        if branch_available.shape != (quality_signals.shape[0], 2):
            raise ValueError("branch_available must have shape [batch, 2]")
        if not branch_available.any(dim=1).all():
            raise ValueError("Every sample needs a face or body branch")
        bounded_body_weight = self.max_body_weight * self.reliability_logits(
            quality_signals,
            body_features,
            face_features,
        ).sigmoid()
        body_available = branch_available[:, 0:1].bool()
        face_available = branch_available[:, 1:2].bool()
        body_weight = torch.where(
            body_available & ~face_available,
            torch.ones_like(bounded_body_weight),
            torch.where(
                body_available & face_available,
                bounded_body_weight,
                torch.zeros_like(bounded_body_weight),
            ),
        )
        return torch.cat((body_weight, 1.0 - body_weight), dim=1)


class BodyPrimaryFusionNeck(nn.Module):
    """Fuse adapted nose, face, and headless whole-body backbone descriptors.

    Inputs are feature tensors from frozen encoders:

    - ``nose_features``: already adapted to ``embedding_dim``;
    - ``face_features``: the semantic-v3 face descriptor;
    - ``body_features``: pooled descriptor from the headless body backbone;
    - ``branch_available``: boolean columns in ``[nose, face, body]`` order.

    The returned ``features`` tensor has shape ``[batch, embedding_dim]`` and is
    L2 normalized, preserving the existing cosine-gallery interface.
    """

    def __init__(
        self,
        *,
        body_dim: int = 1024,
        embedding_dim: int = 512,
        body_quality_dim: int = 4,
        nose_quality_dim: int = 10,
        adapter_bottleneck_dim: int = 128,
        gate_hidden_dim: int = 128,
        initial_body_weight: float = 0.30,
        max_body_weight: float = 0.55,
        initial_nose_weight: float = 0.10,
        max_nose_weight: float = 0.35,
        body_residual_scale: float = 0.05,
        nose_residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.body_dim = int(body_dim)
        self.embedding_dim = int(embedding_dim)
        self.body_quality_dim = int(body_quality_dim)
        self.nose_quality_dim = int(nose_quality_dim)
        self.initial_body_weight = float(initial_body_weight)
        self.initial_nose_weight = float(initial_nose_weight)
        self.body_residual_scale = float(body_residual_scale)
        self.nose_residual_scale = float(nose_residual_scale)
        if self.body_residual_scale < 0.0 or self.nose_residual_scale < 0.0:
            raise ValueError("Residual scales must be non-negative")

        self.body_adapter = ResidualProjectionAdapter(
            self.body_dim,
            self.embedding_dim,
            adapter_bottleneck_dim,
        )
        self.body_gate = BoundedBodyReliabilityGate(
            self.body_quality_dim,
            self.embedding_dim,
            hidden_dim=gate_hidden_dim,
            initial_body_weight=initial_body_weight,
            max_body_weight=max_body_weight,
        )
        self.body_interaction = CrossModalResidual(
            self.embedding_dim,
            gate_hidden_dim,
        )
        self.nose_gate = SemanticReliabilityGate(
            self.nose_quality_dim,
            self.embedding_dim,
            hidden_dim=gate_hidden_dim,
            branch_priors=(initial_nose_weight, 1.0 - initial_nose_weight),
            max_nose_weight=max_nose_weight,
        )
        self.nose_interaction = CrossModalResidual(
            self.embedding_dim,
            gate_hidden_dim,
        )

    def _check_inputs(
        self,
        nose_features: torch.Tensor,
        face_features: torch.Tensor,
        body_features: torch.Tensor,
        nose_quality_signals: torch.Tensor,
        body_quality_signals: torch.Tensor,
        branch_available: torch.Tensor,
    ) -> None:
        batch = face_features.shape[0]
        if face_features.shape != (batch, self.embedding_dim):
            raise ValueError(
                f"face_features must have shape [batch, {self.embedding_dim}]"
            )
        if nose_features.shape != (batch, self.embedding_dim):
            raise ValueError(
                f"nose_features must have shape [batch, {self.embedding_dim}]"
            )
        if body_features.shape != (batch, self.body_dim):
            raise ValueError(f"body_features must have shape [batch, {self.body_dim}]")
        if nose_quality_signals.shape != (batch, self.nose_quality_dim):
            raise ValueError(
                "nose_quality_signals must have shape "
                f"[batch, {self.nose_quality_dim}]"
            )
        if body_quality_signals.shape != (batch, self.body_quality_dim):
            raise ValueError(
                "body_quality_signals must have shape "
                f"[batch, {self.body_quality_dim}]"
            )
        if branch_available.shape != (batch, 3):
            raise ValueError("branch_available must have shape [batch, 3]")
        if not branch_available.bool().any(dim=1).all():
            raise ValueError("Every sample needs at least one identity branch")

    def forward(
        self,
        *,
        nose_features: torch.Tensor,
        face_features: torch.Tensor,
        body_features: torch.Tensor,
        nose_quality_signals: torch.Tensor,
        body_quality_signals: torch.Tensor,
        branch_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._check_inputs(
            nose_features,
            face_features,
            body_features,
            nose_quality_signals,
            body_quality_signals,
            branch_available,
        )
        available = branch_available.bool()
        nose_features = F.normalize(nose_features.float(), dim=1)
        face_features = F.normalize(face_features.float(), dim=1)
        adapted_body = self.body_adapter(body_features)

        face_body_available = torch.stack(
            (available[:, 2], available[:, 1]),
            dim=1,
        )
        body_weights = self.body_gate(
            body_quality_signals.float(),
            adapted_body,
            face_features,
            face_body_available,
        )
        body_weight = body_weights[:, 0:1]
        primary_base = (
            face_features * (1.0 - body_weight) + adapted_body * body_weight
        )
        body_interaction = self.body_interaction(adapted_body, face_features)
        bounded_body_interaction = torch.tanh(body_interaction) / math.sqrt(
            self.embedding_dim
        )
        both_face_body = face_body_available.all(dim=1, keepdim=True).float()
        primary = F.normalize(
            primary_base
            + self.body_residual_scale
            * body_weight
            * bounded_body_interaction
            * both_face_body,
            dim=1,
        )
        body_only = face_body_available[:, 0:1] & ~face_body_available[:, 1:2]
        face_only = face_body_available[:, 1:2] & ~face_body_available[:, 0:1]
        primary = torch.where(body_only, adapted_body, primary)
        primary = torch.where(face_only, face_features, primary)

        primary_available = face_body_available.any(dim=1, keepdim=True)
        nose_primary_available = torch.cat(
            (available[:, 0:1], primary_available),
            dim=1,
        )
        nose_weights = self.nose_gate(
            nose_quality_signals.float(),
            nose_features,
            primary,
            nose_primary_available,
        )
        nose_weight = nose_weights[:, 0:1]
        final_base = primary * (1.0 - nose_weight) + nose_features * nose_weight
        nose_interaction = self.nose_interaction(nose_features, primary)
        bounded_nose_interaction = torch.tanh(nose_interaction) / math.sqrt(
            self.embedding_dim
        )
        both_nose_primary = nose_primary_available.all(dim=1, keepdim=True).float()
        features = F.normalize(
            final_base
            + self.nose_residual_scale
            * nose_weight
            * bounded_nose_interaction
            * both_nose_primary,
            dim=1,
        )
        nose_only = nose_primary_available[:, 0:1] & ~nose_primary_available[:, 1:2]
        primary_only = nose_primary_available[:, 1:2] & ~nose_primary_available[:, 0:1]
        features = torch.where(nose_only, nose_features, features)
        features = torch.where(primary_only, primary, features)

        return {
            "features": features,
            "primary_features": primary,
            "adapted_body_features": adapted_body,
            "body_weights": body_weights,
            "nose_weights": nose_weights,
            "body_interaction": bounded_body_interaction,
            "nose_interaction": bounded_nose_interaction,
            "effective_branch_available": available,
        }

