"""Token-level query/reference matching for complementary pet viewpoints.

The descriptor matcher in :mod:`reference_set_model` is intentionally kept as
the stable, cheap serving path.  This module adds an opt-in research path that
keeps a small grid of features from the image encoder and lets the query
attend to reference tokens *before* they are collapsed to one descriptor.
References are still scored as a set, but a duplicate view is down-weighted by
an explicit coverage gate instead of being counted as independent evidence.

The wrapper accepts the existing single-image encoder contract.  When a
feature-map hook can be found (ResNet ``layer4`` is the common case), pooled
spatial tokens are used.  Small test encoders and third-party encoders that
only expose a descriptor use a deterministic learned token fallback; this
keeps the public API total while making the structural path explicit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .reference_aware_model import _embedding_from_encoder_output


MODEL_FORMAT = "reference-token-aware-pet-reid"
DEFAULT_TOKEN_DIM = 128
DEFAULT_TOKEN_GRID = 4
DEFAULT_HIDDEN_DIM = 128
DEFAULT_MAX_REFERENCES = 16
DEFAULT_REFERENCE_TOP_K = 3
DEFAULT_REFERENCE_SCORE_WEIGHT = 0.4
DEFAULT_ATTENTION_TEMPERATURE = 0.10
DEFAULT_MAXIMUM_RESIDUAL = 0.25
DEFAULT_COVERAGE_WEIGHT = 0.35


def _finite_unit(value: torch.Tensor, *, name: str, ndim: int) -> torch.Tensor:
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    value = value.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")
    return F.normalize(value, dim=-1, eps=1e-12)


def _first_feature_map(value: Any) -> torch.Tensor | None:
    """Find a four-dimensional feature map in a nested module output."""

    if isinstance(value, torch.Tensor):
        return value if value.ndim == 4 else None
    if isinstance(value, Mapping):
        for candidate in value.values():
            found = _first_feature_map(candidate)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for candidate in value:
            found = _first_feature_map(candidate)
            if found is not None:
                return found
    return None


class ImageTokenAdapter(nn.Module):
    """Expose a descriptor and a fixed number of spatial tokens per image.

    A forward hook is registered only on a likely feature-map module.  The
    hook preserves every batch-compatible invocation.  Unified encoders pack
    several crop branches as ``[branch * batch, channels, height, width]``;
    those rows are restored to their source images and the branch channels are
    kept distinct before token projection.  This avoids changing the encoder's
    forward implementation or collapsing genuine multi-scale evidence.
    """

    _FEATURE_SUFFIXES = (
        "identity_encoder.backbone.layer4",
        "geometry_frontend.identity_encoder.backbone.layer4",
        "base_model.geometry_frontend.identity_encoder.backbone.layer4",
        "base_model.identity_encoder.backbone.layer4",
        "parent_model.base_model.geometry_frontend.identity_encoder.backbone.layer4",
        "parent_model.base_model.identity_encoder.backbone.layer4",
        "backbone.layer4",
        "layer4",
    )

    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        token_dim: int = DEFAULT_TOKEN_DIM,
        token_grid: int = DEFAULT_TOKEN_GRID,
    ) -> None:
        super().__init__()
        descriptor_dim = getattr(image_encoder, "descriptor_dim", None)
        if descriptor_dim is None:
            descriptor_dim = getattr(image_encoder, "feature_dim", None)
        if descriptor_dim is None or int(descriptor_dim) <= 0:
            raise ValueError("image encoder must expose a positive descriptor_dim")
        self.encoder = image_encoder
        self.descriptor_dim = int(descriptor_dim)
        self.input_size = getattr(image_encoder, "input_size", None)
        self.input_size = int(self.input_size) if self.input_size is not None else None
        self.token_dim = int(token_dim)
        self.token_grid = int(token_grid)
        if self.token_dim < 1 or self.token_grid < 1:
            raise ValueError("token_dim and token_grid must be positive")
        self.token_count = self.token_grid * self.token_grid

        # LazyLinear handles the 2048-channel ResNet feature map as well as
        # compact research encoders without hard-coding a backbone width.
        self.feature_projection = nn.LazyLinear(self.token_dim)
        self.feature_norm = nn.LayerNorm(self.token_dim)
        self.fallback_projection = nn.Linear(self.descriptor_dim, self.token_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.token_count, self.token_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)

        self._captured: list[torch.Tensor] = []
        self._hook_target_name: str | None = None
        self._hook_handle: Any | None = None
        target = self._find_feature_module()
        if target is not None:
            self._hook_handle = target.register_forward_hook(self._capture_hook)

    def _find_feature_module(self) -> nn.Module | None:
        names = tuple(self.encoder.named_modules())
        for name, module in names:
            if not name:
                continue
            lowered = name.casefold()
            if any(
                lowered == suffix or lowered.endswith("." + suffix)
                for suffix in self._FEATURE_SUFFIXES
            ):
                self._hook_target_name = name
                return module
        return None

    def _capture_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        feature = _first_feature_map(output)
        if feature is not None:
            self._captured.append(feature)

    def _pool_captured_features(self, batch_size: int) -> torch.Tensor | None:
        """Pool and restore hook outputs to one token row per source image.

        Crop branches in the unified encoders are concatenated branch-first,
        so a layer4 output with ``K * batch_size`` rows represents K spatial
        views of every source image.  Each view is pooled independently and
        concatenated on the channel axis.  Keeping the token axis fixed makes
        corresponding spatial cells interact while retaining a distinct set
        of learnable projection weights for every crop branch.
        """

        if int(batch_size) < 1:
            return None
        pooled_groups: list[torch.Tensor] = []
        for feature in self._captured:
            if feature.ndim != 4 or int(feature.shape[0]) < int(batch_size):
                continue
            if int(feature.shape[0]) % int(batch_size):
                continue
            branch_count = int(feature.shape[0]) // int(batch_size)
            pooled = self._pool_spatial_feature(feature)
            if branch_count > 1:
                pooled = (
                    pooled.reshape(
                        branch_count,
                        -1,
                        self.token_count,
                        int(pooled.shape[-1]),
                    )
                    .permute(1, 2, 0, 3)
                    .flatten(2)
                    .contiguous()
                )
            pooled_groups.append(pooled)
        if not pooled_groups:
            return None
        if len(pooled_groups) == 1:
            return pooled_groups[0]
        return torch.cat(pooled_groups, dim=2)

    @property
    def feature_hook_name(self) -> str | None:
        """Return the encoder module used as the spatial feature source."""

        return self._hook_target_name

    def tokens_from_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        descriptors = _finite_unit(descriptors, name="descriptors", ndim=2)
        base = self.fallback_projection(descriptors).unsqueeze(1)
        tokens = base.expand(-1, self.token_count, -1) + self.position_embedding
        return F.normalize(self.feature_norm(tokens), dim=-1, eps=1e-12)

    def _pool_spatial_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] < 1:
            raise ValueError(
                "spatial feature must have shape [batch, channels, height, width]"
            )
        feature = feature.float()
        if not bool(torch.isfinite(feature).all()):
            raise ValueError("spatial feature must contain finite values")
        # The legacy TorchScript ONNX exporter cannot lower adaptive pool when
        # the source spatial dimensions are symbolic. Resize is equivalent for
        # the fixed token grid and keeps the export path usable for real
        # backbones with dynamic feature-map sizes.
        if torch.onnx.is_in_onnx_export():
            pooled = F.interpolate(
                feature,
                size=(self.token_grid, self.token_grid),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pooled = F.adaptive_avg_pool2d(
                feature, (self.token_grid, self.token_grid)
            )
        return pooled.flatten(2).transpose(1, 2).contiguous()

    def encode_cacheable_features(
        self,
        images: torch.Tensor,
        *,
        require_spatial: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode immutable descriptor and pre-projection spatial features.

        The returned spatial rows are pooled to the configured token grid but
        intentionally remain on the encoder channel width. They can therefore
        be cached while the token projection, normalization and positional
        embeddings continue to learn.
        """

        if images.ndim != 4 or images.shape[1] < 1:
            raise ValueError("images must have shape [batch, channels, height, width]")
        self._captured.clear()
        descriptor = _embedding_from_encoder_output(self.encoder(images))
        if descriptor.shape[0] != images.shape[0]:
            raise ValueError("image encoder changed the batch dimension")
        if descriptor.shape[1] != self.descriptor_dim:
            raise ValueError(
                "image encoder descriptor width does not match adapter: "
                f"{descriptor.shape[1]} != {self.descriptor_dim}"
            )
        descriptor = F.normalize(descriptor.float(), dim=1, eps=1e-12)

        pooled = self._pool_captured_features(int(images.shape[0]))
        if pooled is None:
            if require_spatial:
                hook = self._hook_target_name or "<not found>"
                shapes = [
                    tuple(int(size) for size in item.shape)
                    for item in self._captured
                ]
                raise RuntimeError(
                    "a real spatial feature map is required for cached token "
                    f"training; encoder hook {hook!r} produced no batch-compatible "
                    f"feature (captured shapes: {shapes})"
                )
            return descriptor, None
        return descriptor, pooled

    def tokens_from_pooled_features(self, pooled: torch.Tensor) -> torch.Tensor:
        """Apply the trainable token projection to cached spatial features."""

        if pooled.ndim != 3:
            raise ValueError(
                "pooled spatial features must have shape [batch, tokens, channels]"
            )
        if pooled.shape[1] != self.token_count or pooled.shape[2] < 1:
            raise ValueError(
                "pooled spatial feature dimensions do not match the adapter token grid"
            )
        pooled = pooled.float()
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("pooled spatial features must contain finite values")
        projected = self.feature_projection(pooled)
        return F.normalize(
            self.feature_norm(projected + self.position_embedding),
            dim=-1,
            eps=1e-12,
        )

    def forward_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor, pooled = self.encode_cacheable_features(images)
        if pooled is None:
            tokens = self.tokens_from_descriptors(descriptor)
        else:
            tokens = self.tokens_from_pooled_features(pooled)
        return descriptor, tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        descriptor, _tokens = self.forward_features(images)
        return descriptor

    def configuration(self) -> dict[str, Any]:
        encoder_configuration = getattr(self.encoder, "configuration", None)
        if callable(encoder_configuration):
            value = encoder_configuration()
            encoder_config = dict(value) if isinstance(value, Mapping) else {}
        else:
            encoder_config = {
                "type": type(self.encoder).__name__,
                "input_size": self.input_size,
                "descriptor_dim": self.descriptor_dim,
            }
        return {
            "type": type(self).__name__,
            "descriptor_dim": self.descriptor_dim,
            "token_dim": self.token_dim,
            "token_grid": self.token_grid,
            "feature_hook": self._hook_target_name,
            "encoder": encoder_config,
        }

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):  # pragma: no cover - interpreter shutdown ordering
        try:
            self.close()
        except Exception:
            pass


class TokenConditionedReferenceMatcher(nn.Module):
    """Cross-token reference matcher with an explicit diversity/coverage gate."""

    def __init__(
        self,
        descriptor_dim: int,
        *,
        token_dim: int = DEFAULT_TOKEN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_references: int = DEFAULT_MAX_REFERENCES,
        reference_top_k: int = DEFAULT_REFERENCE_TOP_K,
        reference_score_weight: float = DEFAULT_REFERENCE_SCORE_WEIGHT,
        attention_temperature: float = DEFAULT_ATTENTION_TEMPERATURE,
        maximum_residual: float = DEFAULT_MAXIMUM_RESIDUAL,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_references = int(max_references)
        self.reference_top_k = int(reference_top_k)
        self.reference_score_weight = float(reference_score_weight)
        self.attention_temperature = float(attention_temperature)
        self.maximum_residual = float(maximum_residual)
        self.coverage_weight = float(coverage_weight)
        if (
            min(
                self.descriptor_dim,
                self.token_dim,
                self.hidden_dim,
                self.max_references,
                self.reference_top_k,
            )
            <= 0
        ):
            raise ValueError("matcher dimensions and reference counts must be positive")
        if self.reference_top_k > self.max_references:
            raise ValueError("reference_top_k cannot exceed max_references")
        if not 0.0 <= self.reference_score_weight <= 1.0:
            raise ValueError("reference_score_weight must be between 0 and 1")
        if (
            not np.isfinite(self.attention_temperature)
            or self.attention_temperature <= 0
        ):
            raise ValueError("attention_temperature must be positive and finite")
        if not np.isfinite(self.maximum_residual) or self.maximum_residual <= 0:
            raise ValueError("maximum_residual must be positive and finite")
        if not np.isfinite(self.coverage_weight) or self.coverage_weight < 0:
            raise ValueError("coverage_weight must be finite and non-negative")

        self.query_projection = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.reference_projection = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
        )
        # Per-reference learned compatibility is conditioned on the full
        # token interaction score, not only on a 512-D cosine.
        self.pair_head = nn.Sequential(
            nn.Linear(4 * self.hidden_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)
        self.coverage_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.coverage_head[-1].weight)
        nn.init.zeros_(self.coverage_head[-1].bias)
        self.descriptor_token_projection = nn.Linear(
            self.descriptor_dim, self.token_dim
        )
        self.descriptor_position = nn.Parameter(torch.zeros(1, 1, self.token_dim))

        score_feature_dim = 4 * self.hidden_dim + 5
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_feature_dim),
            nn.Linear(score_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def configuration(self) -> dict[str, Any]:
        return {
            "descriptor_dim": self.descriptor_dim,
            "token_dim": self.token_dim,
            "hidden_dim": self.hidden_dim,
            "max_references": self.max_references,
            "reference_top_k": self.reference_top_k,
            "reference_score_weight": self.reference_score_weight,
            "attention_temperature": self.attention_temperature,
            "maximum_residual": self.maximum_residual,
            "coverage_weight": self.coverage_weight,
            "baseline": "centroid_plus_masked_top_k_mean",
            "conditioning": "query_reference_cross_token_attention",
            "coverage": "novelty_gated_reference_attention",
        }

    def _validate_inputs(
        self,
        query_descriptor: torch.Tensor,
        references: torch.Tensor,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            query_descriptor.ndim != 2
            or query_descriptor.shape[1] != self.descriptor_dim
        ):
            raise ValueError(
                f"query_descriptor must have shape [batch, {self.descriptor_dim}]"
            )
        if references.ndim != 3 or references.shape[2] != self.descriptor_dim:
            raise ValueError(
                f"references must have shape [batch, references, {self.descriptor_dim}]"
            )
        if query_tokens.ndim != 3 or query_tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"query_tokens must have shape [batch, tokens, {self.token_dim}]"
            )
        if (
            reference_tokens.ndim != 4
            or reference_tokens.shape[2] != query_tokens.shape[1]
            or reference_tokens.shape[3] != self.token_dim
        ):
            raise ValueError(
                "reference_tokens must have shape [batch, references, tokens, token_dim]"
            )
        if references.shape[:2] != reference_tokens.shape[:2]:
            raise ValueError("reference descriptor/token dimensions do not match")
        if references.shape[0] != query_descriptor.shape[0]:
            raise ValueError("query and reference batch dimensions must match")
        if references.shape[1] < 1 or references.shape[1] > self.max_references:
            raise ValueError(
                f"references must contain between 1 and {self.max_references} rows"
            )
        query_descriptor = _finite_unit(
            query_descriptor, name="query_descriptor", ndim=2
        )
        references = _finite_unit(references, name="references", ndim=3)
        query_tokens = _finite_unit(query_tokens, name="query_tokens", ndim=3)
        reference_tokens = _finite_unit(
            reference_tokens, name="reference_tokens", ndim=4
        )
        if reference_mask is None:
            mask = torch.ones(
                references.shape[:2], dtype=torch.bool, device=references.device
            )
        else:
            if tuple(reference_mask.shape) != tuple(references.shape[:2]):
                raise ValueError("reference_mask must have shape [batch, references]")
            if torch.is_floating_point(reference_mask):
                if not bool(torch.isfinite(reference_mask.float()).all()):
                    raise ValueError("reference_mask must contain finite values")
            mask = (reference_mask != 0).to(device=references.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("each query must have at least one reference")
        return query_descriptor, references, query_tokens, reference_tokens, mask

    @staticmethod
    def _masked_top_k_mean(
        similarities: torch.Tensor, mask: torch.Tensor, top_k: int
    ) -> torch.Tensor:
        masked = similarities.masked_fill(~mask, -2.0)
        selected, indices = torch.topk(
            masked, k=min(int(top_k), similarities.shape[1]), dim=1
        )
        selected_mask = mask.gather(1, indices)
        denominator = selected_mask.sum(dim=1).clamp_min(1).to(selected.dtype)
        return (selected * selected_mask.to(selected.dtype)).sum(dim=1) / denominator

    def tokens_from_descriptors(
        self, descriptors: torch.Tensor, *, token_count: int
    ) -> torch.Tensor:
        descriptors = _finite_unit(descriptors, name="descriptors", ndim=2)
        base = self.descriptor_token_projection(descriptors).unsqueeze(1)
        position = self.descriptor_position.expand(-1, int(token_count), -1)
        return F.normalize(base.expand(-1, int(token_count), -1) + position, dim=-1)

    def _forward_impl(
        self,
        query_descriptor: torch.Tensor,
        references: torch.Tensor,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        mask_float = mask.to(dtype=query_descriptor.dtype)
        query_summary = query_tokens.mean(dim=1)
        reference_summary = reference_tokens.mean(dim=2)
        query_hidden = F.normalize(self.query_projection(query_summary), dim=-1)
        reference_hidden = F.normalize(
            self.reference_projection(reference_summary), dim=-1
        )

        # Full cross-token matching: every query token can select the best
        # reference token, so a side-view can contribute a different local
        # region than a frontal-view rather than being averaged away.
        token_similarity = torch.einsum(
            "btd,bkud->bktu", query_tokens, reference_tokens
        )
        token_attention = F.softmax(
            token_similarity / self.attention_temperature, dim=-1
        )
        aligned_tokens = (token_attention * token_similarity).sum(dim=-1)
        token_scores = aligned_tokens.mean(dim=-1)

        descriptor_similarity = torch.einsum("bd,bkd->bk", query_descriptor, references)
        pair_query = query_hidden.unsqueeze(1).expand_as(reference_hidden)
        pair_features = torch.cat(
            (
                pair_query,
                reference_hidden,
                pair_query * reference_hidden,
                (pair_query - reference_hidden).abs(),
                token_scores.unsqueeze(-1),
                descriptor_similarity.unsqueeze(-1),
            ),
            dim=-1,
        )
        learned_logits = self.pair_head(pair_features).squeeze(-1)

        # Reference novelty is computed from token summaries.  The diagonal is
        # excluded; a singleton set gets neutral novelty instead of a false
        # duplicate penalty.
        summary_similarity = torch.einsum(
            "bkd,bjd->bkj", reference_hidden, reference_hidden
        )
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(
            references.shape[1], dtype=torch.bool, device=references.device
        ).unsqueeze(0)
        other_mask = pair_mask & ~eye
        other_values = summary_similarity.masked_fill(~other_mask, -2.0)
        other_max = other_values.max(dim=2).values
        has_other = other_mask.any(dim=2)
        novelty = torch.where(
            has_other,
            (1.0 - other_max).clamp(0.0, 2.0) * 0.5,
            torch.ones_like(other_max),
        )
        novelty = novelty * mask_float
        coverage_features = torch.cat((reference_hidden, novelty.unsqueeze(-1)), dim=-1)
        learned_coverage = torch.sigmoid(
            self.coverage_head(coverage_features).squeeze(-1)
        )

        logits = descriptor_similarity / self.attention_temperature + learned_logits
        logits = logits + float(self.coverage_weight) * novelty * learned_coverage
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, dim=1) * mask_float
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        gated_attention = attention * (0.5 + 0.5 * learned_coverage) * mask_float
        gated_attention = gated_attention / gated_attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)

        pooled_raw = torch.einsum("bk,bkd->bd", gated_attention, references)
        pooled_reference = F.normalize(pooled_raw, dim=-1)
        centroid_raw = (references * mask_float.unsqueeze(-1)).sum(dim=1)
        centroid = F.normalize(centroid_raw, dim=-1)
        centroid_score = torch.einsum("bd,bd->b", query_descriptor, centroid)
        top_k_score = self._masked_top_k_mean(
            descriptor_similarity, mask, self.reference_top_k
        )
        baseline_score = (
            1.0 - self.reference_score_weight
        ) * centroid_score + self.reference_score_weight * top_k_score

        pooled_tokens = torch.einsum("bk,bktd->btd", gated_attention, reference_tokens)
        pooled_hidden = F.normalize(
            self.reference_projection(pooled_tokens.mean(dim=1)), dim=-1
        )
        coverage_score = (gated_attention * novelty).sum(dim=1)
        duplicate_score = (gated_attention * (1.0 - novelty)).sum(dim=1)
        count = mask_float.sum(dim=1)
        count_feature = (count / float(self.max_references)).unsqueeze(1)
        score_features = torch.cat(
            (
                query_hidden,
                pooled_hidden,
                query_hidden * pooled_hidden,
                (query_hidden - pooled_hidden).abs(),
                centroid_score.unsqueeze(1),
                top_k_score.unsqueeze(1),
                coverage_score.unsqueeze(1),
                duplicate_score.unsqueeze(1),
                count_feature,
            ),
            dim=1,
        )
        residual = self.maximum_residual * torch.tanh(
            self.score_head(score_features).squeeze(1)
        )
        score = baseline_score + residual
        if not return_aux:
            return score
        return {
            "score": score,
            "baseline_score": baseline_score,
            "residual": residual,
            "attention": gated_attention,
            "raw_attention": attention,
            "coverage_gate": learned_coverage,
            "novelty": novelty,
            "token_scores": token_scores,
            "token_attention": token_attention,
            "token_similarity": token_similarity,
            "similarities": descriptor_similarity,
            "pooled_reference": pooled_reference,
            "reference_count": count,
            "coverage_score": coverage_score,
            "duplicate_score": duplicate_score,
            "centroid_score": centroid_score,
            "top_k_score": top_k_score,
        }

    def forward(
        self,
        query_descriptor: torch.Tensor,
        references: torch.Tensor,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        values = self._validate_inputs(
            query_descriptor,
            references,
            query_tokens,
            reference_tokens,
            reference_mask,
        )
        return self._forward_impl(*values, return_aux=return_aux)

    def forward_export(
        self,
        query_descriptor: torch.Tensor,
        references: torch.Tensor,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_descriptor = F.normalize(query_descriptor.float(), dim=-1, eps=1e-12)
        references = F.normalize(references.float(), dim=-1, eps=1e-12)
        query_tokens = F.normalize(query_tokens.float(), dim=-1, eps=1e-12)
        reference_tokens = F.normalize(reference_tokens.float(), dim=-1, eps=1e-12)
        output = self._forward_impl(
            query_descriptor,
            references,
            query_tokens,
            reference_tokens,
            reference_mask != 0,
        )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("token matcher export path returned auxiliary output")
        return output


class TokenReferenceAwarePetReID(nn.Module):
    """Image-set model that performs reference interaction on spatial tokens."""

    def __init__(
        self,
        image_encoder: nn.Module,
        matcher: TokenConditionedReferenceMatcher,
        *,
        token_dim: int = DEFAULT_TOKEN_DIM,
        token_grid: int = DEFAULT_TOKEN_GRID,
        max_references: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(matcher, TokenConditionedReferenceMatcher):
            raise TypeError("matcher must be a TokenConditionedReferenceMatcher")
        if int(matcher.token_dim) != int(token_dim):
            raise ValueError("matcher and adapter token dimensions differ")
        encoder_dim = getattr(image_encoder, "descriptor_dim", None)
        if encoder_dim is not None and int(encoder_dim) != matcher.descriptor_dim:
            raise ValueError("image encoder and matcher descriptor dimensions differ")
        self.image_encoder = (
            image_encoder
            if isinstance(image_encoder, ImageTokenAdapter)
            else ImageTokenAdapter(
                image_encoder, token_dim=token_dim, token_grid=token_grid
            )
        )
        resolved_max = (
            matcher.max_references if max_references is None else int(max_references)
        )
        if resolved_max < 1 or resolved_max > matcher.max_references:
            raise ValueError("max_references must fit within matcher.max_references")
        self.matcher = matcher
        self.max_references = resolved_max
        self.descriptor_dim = matcher.descriptor_dim
        self.token_dim = matcher.token_dim
        self.token_grid = int(self.image_encoder.token_grid)

    @property
    def input_size(self) -> int | None:
        value = getattr(self.image_encoder, "input_size", None)
        return int(value) if value is not None else None

    def configuration(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "descriptor_dim": self.descriptor_dim,
            "token_dim": self.token_dim,
            "token_grid": self.token_grid,
            "max_references": self.max_references,
            "encoder": self.image_encoder.configuration(),
            "matcher": self.matcher.configuration(),
        }

    def encode_image_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image_encoder.forward_features(images)

    def encode_cacheable_image_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen descriptors and real pre-projection spatial features."""

        descriptors, pooled = self.image_encoder.encode_cacheable_features(
            images,
            require_spatial=True,
        )
        if pooled is None:  # Defensive: require_spatial already rejects this.
            raise RuntimeError("spatial feature extraction returned no feature tensor")
        return descriptors, pooled

    def tokens_from_pooled_features(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.image_encoder.tokens_from_pooled_features(pooled)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        descriptor, _tokens = self.encode_image_features(images)
        return descriptor

    def encode_reference_features(
        self, reference_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_rgb.ndim != 5:
            raise ValueError(
                "reference_rgb must have shape [batch, references, channels, height, width]"
            )
        batch, count = reference_rgb.shape[:2]
        if count < 1 or count > self.max_references:
            raise ValueError(
                f"reference_rgb must contain between 1 and {self.max_references} rows"
            )
        flattened = reference_rgb.reshape(batch * count, *reference_rgb.shape[2:])
        descriptors, tokens = self.encode_image_features(flattened)
        return (
            descriptors.reshape(batch, count, self.descriptor_dim),
            tokens.reshape(batch, count, tokens.shape[1], self.token_dim),
        )

    def encode_reference_images(self, reference_rgb: torch.Tensor) -> torch.Tensor:
        descriptors, _tokens = self.encode_reference_features(reference_rgb)
        return descriptors

    def forward_encoded(
        self,
        query_descriptor: torch.Tensor,
        reference_descriptors: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        query_tokens: torch.Tensor | None = None,
        reference_tokens: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if query_tokens is None:
            query_tokens = self.matcher.tokens_from_descriptors(
                query_descriptor, token_count=self.token_grid * self.token_grid
            )
        if reference_tokens is None:
            flat = reference_descriptors.reshape(-1, reference_descriptors.shape[-1])
            flat_tokens = self.matcher.tokens_from_descriptors(
                flat, token_count=query_tokens.shape[1]
            )
            reference_tokens = flat_tokens.reshape(
                reference_descriptors.shape[0],
                reference_descriptors.shape[1],
                query_tokens.shape[1],
                self.token_dim,
            )
        return self.matcher(
            query_descriptor,
            reference_descriptors,
            query_tokens,
            reference_tokens,
            reference_mask,
            return_aux=return_aux,
        )

    def score_descriptors(
        self,
        query_descriptor: torch.Tensor,
        reference_descriptors: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.forward_encoded(
            query_descriptor,
            reference_descriptors,
            reference_mask,
            return_aux=return_aux,
        )

    def _validate_set_inputs(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if query_rgb.ndim != 4 or reference_rgb.ndim != 5:
            raise ValueError("query/reference images have unexpected dimensions")
        if query_rgb.shape[0] != reference_rgb.shape[0]:
            raise ValueError("query and reference batch dimensions must match")
        if query_rgb.shape[1] != reference_rgb.shape[2]:
            raise ValueError("query and reference channel dimensions must match")
        if reference_rgb.shape[1] < 1 or reference_rgb.shape[1] > self.max_references:
            raise ValueError("reference count exceeds model capacity")
        if reference_mask is None:
            return None
        if tuple(reference_mask.shape) != tuple(reference_rgb.shape[:2]):
            raise ValueError("reference_mask must have shape [batch, references]")
        if torch.is_floating_point(reference_mask) and not bool(
            torch.isfinite(reference_mask.float()).all()
        ):
            raise ValueError("reference_mask must contain finite values")
        mask = (reference_mask != 0).to(device=reference_rgb.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("each query must have at least one reference")
        return mask

    def forward(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        reference_mask = self._validate_set_inputs(
            query_rgb, reference_rgb, reference_mask
        )
        query_descriptor, query_tokens = self.encode_image_features(query_rgb)
        reference_descriptors, reference_tokens = self.encode_reference_features(
            reference_rgb
        )
        output = self.forward_encoded(
            query_descriptor,
            reference_descriptors,
            reference_mask,
            query_tokens=query_tokens,
            reference_tokens=reference_tokens,
            return_aux=return_aux,
        )
        if not return_aux:
            return output
        if not isinstance(output, dict):
            raise RuntimeError("token matcher returned no auxiliary output")
        return {
            **output,
            "query_descriptor": query_descriptor,
            "reference_descriptors": reference_descriptors,
            "query_tokens": query_tokens,
            "reference_tokens": reference_tokens,
        }

    def forward_export(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_descriptor, query_tokens = self.encode_image_features(query_rgb)
        reference_descriptors, reference_tokens = self.encode_reference_features(
            reference_rgb
        )
        return self.matcher.forward_export(
            query_descriptor,
            reference_descriptors,
            query_tokens,
            reference_tokens,
            reference_mask,
        )

    def freeze_encoder(self) -> None:
        # Keep the adapter projections trainable so a frozen base can learn a
        # useful token basis during head-only warm-up.
        self.image_encoder.encoder.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        self.image_encoder.encoder.requires_grad_(True)


class TokenReferenceAwarePetReIDExport(nn.Module):
    """Tensor-only export boundary for a fixed-width token reference set."""

    def __init__(self, model: TokenReferenceAwarePetReID) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_export(query_rgb, reference_rgb, reference_mask)


def create_token_reference_aware_checkpoint(
    model: TokenReferenceAwarePetReID,
    *,
    base_encoder_checkpoint: str | Path | None = None,
    encoder_fingerprint: str | None = None,
    training: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": MODEL_FORMAT,
        "model_config": model.configuration(),
        "model": model.state_dict(),
        "base_encoder_checkpoint": (
            str(Path(base_encoder_checkpoint).expanduser().resolve())
            if base_encoder_checkpoint is not None
            else None
        ),
        "encoder_fingerprint": encoder_fingerprint,
        "training": dict(training or {}),
        "optimizer": dict(optimizer_state or {}),
    }


def save_token_reference_aware_model(
    model: TokenReferenceAwarePetReID,
    path: str | Path,
    *,
    base_encoder_checkpoint: str | Path | None = None,
    encoder_fingerprint: str | None = None,
    training: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".writing")
    torch.save(
        create_token_reference_aware_checkpoint(
            model,
            base_encoder_checkpoint=base_encoder_checkpoint,
            encoder_fingerprint=encoder_fingerprint,
            training=training,
            optimizer_state=optimizer_state,
        ),
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def build_token_reference_aware_model_from_checkpoint(
    path: str | Path,
    image_encoder: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TokenReferenceAwarePetReID, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != MODEL_FORMAT:
        raise ValueError(
            f"Unexpected token reference-aware model format: {checkpoint_path}"
        )
    config = payload.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("token reference-aware checkpoint has no model_config")
    matcher_config = config.get("matcher")
    if not isinstance(matcher_config, Mapping):
        raise ValueError(
            "token reference-aware checkpoint has no matcher configuration"
        )
    constructor_keys = {
        "descriptor_dim",
        "token_dim",
        "hidden_dim",
        "max_references",
        "reference_top_k",
        "reference_score_weight",
        "attention_temperature",
        "maximum_residual",
        "coverage_weight",
    }
    matcher = TokenConditionedReferenceMatcher(
        **{
            key: matcher_config[key]
            for key in constructor_keys
            if key in matcher_config
        }
    )
    model = TokenReferenceAwarePetReID(
        image_encoder,
        matcher,
        token_dim=int(config.get("token_dim", matcher.token_dim)),
        token_grid=int(config.get("token_grid", DEFAULT_TOKEN_GRID)),
        max_references=int(config.get("max_references", matcher.max_references)),
    )
    incompatible = model.load_state_dict(payload.get("model", {}), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"token reference-aware checkpoint mismatch: {incompatible}")
    return model.to(device), dict(payload)


__all__ = [
    "MODEL_FORMAT",
    "ImageTokenAdapter",
    "TokenConditionedReferenceMatcher",
    "TokenReferenceAwarePetReID",
    "TokenReferenceAwarePetReIDExport",
    "build_token_reference_aware_model_from_checkpoint",
    "create_token_reference_aware_checkpoint",
    "save_token_reference_aware_model",
]
