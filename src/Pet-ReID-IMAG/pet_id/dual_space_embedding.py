"""Structure-level bridge for native pre/post-BN nose descriptors.

The legacy unified model projects one post-BN 2048-D descriptor directly into
the 512-D face space.  This module keeps the two native nose geometries
separate, turns them into compact tokens, and uses them to propose a bounded
residual update to the protected ArcFace descriptor.  It contains no
identity-specific classifier and therefore remains usable for unseen pets.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


STRUCTURAL_VARIANTS = (
    "post_residual",
    "dual_consensus",
    "dual_cross_attention",
)


def _bounded_logit(value: float, upper: float) -> float:
    value = float(value)
    upper = float(upper)
    if not 0.0 < value < upper:
        raise ValueError("initial value must be strictly inside (0, upper)")
    probability = value / upper
    return math.log(probability / (1.0 - probability))


class NativeNoseTokenNeck(nn.Module):
    """Map one native nose space into a token without a narrow serial choke.

    The full-rank projection is the geometry path.  A separately initialized
    low-rank residual can learn task-specific corrections without being the
    only route from 2048 dimensions to the token.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        token_dim: int = 256,
        *,
        bottleneck_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        if min(self.input_dim, self.token_dim, self.bottleneck_dim) <= 0:
            raise ValueError("neck dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.geometry_projection = nn.Linear(
            self.input_dim,
            self.token_dim,
            bias=False,
        )
        self.task_residual = nn.Sequential(
            nn.Linear(self.input_dim, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.bottleneck_dim, self.token_dim, bias=False),
        )
        nn.init.orthogonal_(self.geometry_projection.weight)
        nn.init.zeros_(self.task_residual[-1].weight)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.input_dim:
            raise ValueError(
                f"descriptor must have shape [batch, {self.input_dim}]"
            )
        normalized = self.input_norm(descriptor.float()).to(descriptor.dtype)
        token = self.geometry_projection(normalized) + self.task_residual(normalized)
        return F.normalize(token, dim=1)


class DualSpaceNoseEmbeddingBridge(nn.Module):
    """Fuse face/pre-BN/post-BN descriptors by changing representation structure.

    ``post_residual`` is a replacement-adapter control. ``dual_consensus``
    explicitly models agreement and disagreement between the two native nose
    spaces. ``dual_cross_attention`` additionally lets the face descriptor
    query both nose tokens before the residual identity update.
    """

    def __init__(
        self,
        *,
        variant: str,
        face_dim: int = 512,
        nose_dim: int = 2048,
        token_dim: int = 256,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.10,
        maximum_residual_scale: float = 0.30,
        initial_residual_scale: float = 0.08,
    ) -> None:
        super().__init__()
        if variant not in STRUCTURAL_VARIANTS:
            raise ValueError(
                f"variant must be one of {STRUCTURAL_VARIANTS}, got {variant!r}"
            )
        self.variant = str(variant)
        self.face_dim = int(face_dim)
        self.nose_dim = int(nose_dim)
        self.token_dim = int(token_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.dropout = float(dropout)
        self.maximum_residual_scale = float(maximum_residual_scale)
        if min(
            self.face_dim,
            self.nose_dim,
            self.token_dim,
            self.bottleneck_dim,
            self.hidden_dim,
            self.attention_heads,
        ) <= 0:
            raise ValueError("bridge dimensions must be positive")
        if self.token_dim % self.attention_heads:
            raise ValueError("token_dim must divide evenly into attention_heads")
        if not 0.0 < self.maximum_residual_scale < 1.0:
            raise ValueError("maximum_residual_scale must be in (0, 1)")

        self.post_neck = NativeNoseTokenNeck(
            self.nose_dim,
            self.token_dim,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
        )
        self.pre_neck = (
            None
            if self.variant == "post_residual"
            else NativeNoseTokenNeck(
                self.nose_dim,
                self.token_dim,
                bottleneck_dim=self.bottleneck_dim,
                dropout=self.dropout,
            )
        )
        self.face_token = nn.Sequential(
            nn.LayerNorm(self.face_dim),
            nn.Linear(self.face_dim, self.token_dim, bias=False),
        )
        nn.init.orthogonal_(self.face_token[-1].weight)

        self.consensus_mixer = None
        self.cross_attention = None
        self.attention_norm = None
        if self.variant == "dual_consensus":
            self.consensus_mixer = nn.Sequential(
                nn.LayerNorm(4 * self.token_dim),
                nn.Linear(4 * self.token_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.token_dim, bias=False),
            )
        elif self.variant == "dual_cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                self.token_dim,
                self.attention_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(self.token_dim)

        self.residual_head = nn.Sequential(
            nn.LayerNorm(4 * self.token_dim),
            nn.Linear(4 * self.token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.face_dim, bias=False),
        )
        # Exact face-only initialization, while the last layer immediately
        # receives gradients from the first optimization step.
        nn.init.zeros_(self.residual_head[-1].weight)
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(
                _bounded_logit(
                    initial_residual_scale,
                    self.maximum_residual_scale,
                ),
                dtype=torch.float32,
            )
        )
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(16.0)))

    def configuration(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "face_dim": self.face_dim,
            "nose_dim": self.nose_dim,
            "token_dim": self.token_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "hidden_dim": self.hidden_dim,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
            "maximum_residual_scale": self.maximum_residual_scale,
            "fixed_identity_classifier": False,
        }

    def score_scale(self) -> torch.Tensor:
        return self.logit_scale_log.clamp(math.log(1.0), math.log(64.0)).exp()

    def _context(
        self,
        face_token: torch.Tensor,
        pre_token: torch.Tensor | None,
        post_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.variant == "post_residual":
            return post_token, None
        if pre_token is None:
            raise RuntimeError("dual variants require a pre-BN token")
        if self.variant == "dual_consensus":
            relation = torch.cat(
                (
                    pre_token,
                    post_token,
                    (pre_token - post_token).abs(),
                    pre_token * post_token,
                ),
                dim=1,
            )
            context = self.consensus_mixer(relation)
            return F.normalize(context + 0.5 * (pre_token + post_token), dim=1), None
        nose_tokens = torch.stack((pre_token, post_token), dim=1)
        attended, weights = self.cross_attention(
            face_token[:, None],
            nose_tokens,
            nose_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.attention_norm(attended[:, 0] + face_token)
        return F.normalize(context, dim=1), weights[:, 0]

    def forward(
        self,
        face_descriptor: torch.Tensor,
        nose_pre_descriptor: torch.Tensor,
        nose_post_descriptor: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        if face_descriptor.ndim != 2 or face_descriptor.shape[1] != self.face_dim:
            raise ValueError(
                f"face_descriptor must have shape [batch, {self.face_dim}]"
            )
        expected_nose = (face_descriptor.shape[0], self.nose_dim)
        if tuple(nose_pre_descriptor.shape) != expected_nose:
            raise ValueError(f"nose_pre_descriptor must have shape {expected_nose}")
        if tuple(nose_post_descriptor.shape) != expected_nose:
            raise ValueError(f"nose_post_descriptor must have shape {expected_nose}")

        face = F.normalize(face_descriptor, dim=1)
        post_token = self.post_neck(nose_post_descriptor)
        pre_token = (
            None
            if self.pre_neck is None
            else self.pre_neck(nose_pre_descriptor)
        )
        face_token = F.normalize(self.face_token(face.float()), dim=1).to(face.dtype)
        context, attention_weights = self._context(
            face_token,
            pre_token,
            post_token,
        )
        relation = torch.cat(
            (
                face_token,
                context,
                (face_token - context).abs(),
                face_token * context,
            ),
            dim=1,
        )
        delta = torch.tanh(self.residual_head(relation.float())).to(face.dtype)
        delta = delta / math.sqrt(self.face_dim)
        residual_scale = (
            self.maximum_residual_scale * self.residual_scale_logit.sigmoid()
        ).to(face.dtype)
        embedding = F.normalize(face + residual_scale * delta, dim=1)
        if not return_aux:
            return embedding
        return {
            "embedding": embedding,
            "face_descriptor": face,
            "face_token": face_token,
            "nose_pre_token": pre_token,
            "nose_post_token": post_token,
            "nose_context_token": context,
            "attention_weights": attention_weights,
            "identity_residual": delta,
            "residual_scale": residual_scale,
            "score_scale": self.score_scale(),
        }

