"""Learnable query-conditioned matching over an identity's reference set.

The image encoder remains deliberately single-image: one RGB image produces one
L2-normalized descriptor.  This module is the model component that learns how a
query should use several cached descriptors.  References are attended to *after*
conditioning on the query, so different viewpoints do not have to be collapsed
into one identity centroid.

The default output is anchored to the existing centroid/top-reference score.  A
zero-initialized residual makes a freshly-created matcher exactly reproduce that
baseline while still giving the new head a useful gradient during training.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


MODEL_FORMAT = "reference-set-matcher"
DEFAULT_DESCRIPTOR_DIM = 512
DEFAULT_HIDDEN_DIM = 128
DEFAULT_MAX_REFERENCES = 16
DEFAULT_REFERENCE_TOP_K = 3
DEFAULT_REFERENCE_SCORE_WEIGHT = 0.4
DEFAULT_ATTENTION_TEMPERATURE = 0.10
DEFAULT_MAXIMUM_RESIDUAL = 0.25


@dataclass(frozen=True)
class ReferenceSetMatch:
    """Model output plus diagnostics for one batch of query/set pairs."""

    score: torch.Tensor
    baseline_score: torch.Tensor
    residual: torch.Tensor
    attention: torch.Tensor
    similarities: torch.Tensor
    pooled_reference: torch.Tensor
    reference_count: torch.Tensor


def _finite_unit(value: torch.Tensor, *, name: str, ndim: int) -> torch.Tensor:
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not torch.is_floating_point(value):
        value = value.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")
    return F.normalize(value.float(), dim=-1, eps=1e-12)


class QueryConditionedReferenceMatcher(nn.Module):
    """Score a query against a masked, variable-size descriptor set.

    ``references`` is padded to a batch-local width and ``reference_mask`` marks
    real rows.  The module accepts any width up to ``max_references``.  It first
    computes descriptor cosine similarities, then uses a small cross-attention
    network to select the references most useful for the current query.  A
    bounded residual over a deterministic centroid/top-k baseline is the final
    score.
    """

    def __init__(
        self,
        descriptor_dim: int = DEFAULT_DESCRIPTOR_DIM,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_references: int = DEFAULT_MAX_REFERENCES,
        reference_top_k: int = DEFAULT_REFERENCE_TOP_K,
        reference_score_weight: float = DEFAULT_REFERENCE_SCORE_WEIGHT,
        attention_temperature: float = DEFAULT_ATTENTION_TEMPERATURE,
        maximum_residual: float = DEFAULT_MAXIMUM_RESIDUAL,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_references = int(max_references)
        self.reference_top_k = int(reference_top_k)
        self.reference_score_weight = float(reference_score_weight)
        self.attention_temperature = float(attention_temperature)
        self.maximum_residual = float(maximum_residual)
        if (
            min(
                self.descriptor_dim,
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
            raise ValueError("attention_temperature must be a positive finite number")
        if not np.isfinite(self.maximum_residual) or self.maximum_residual <= 0:
            raise ValueError("maximum_residual must be a positive finite number")

        self.token_projection = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.pair_attention = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # At initialization attention is driven only by the transparent cosine
        # term.  The learned part becomes active as soon as training moves it.
        nn.init.zeros_(self.pair_attention[-1].weight)
        nn.init.zeros_(self.pair_attention[-1].bias)

        feature_dim = 4 * self.hidden_dim + 3
        self.score_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # This is the safety anchor: a new head cannot silently change scores
        # before it has learned a residual.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def configuration(self) -> dict[str, Any]:
        """Return serializable architecture and scoring settings."""

        return {
            "descriptor_dim": self.descriptor_dim,
            "hidden_dim": self.hidden_dim,
            "max_references": self.max_references,
            "reference_top_k": self.reference_top_k,
            "reference_score_weight": self.reference_score_weight,
            "attention_temperature": self.attention_temperature,
            "maximum_residual": self.maximum_residual,
            "baseline": "centroid_plus_masked_top_k_mean",
            "conditioning": "query_conditioned_attention",
        }

    def _validate_inputs(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        reference_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 2 or query.shape[1] != self.descriptor_dim:
            raise ValueError(f"query must have shape [batch, {self.descriptor_dim}]")
        if references.ndim != 3 or references.shape[2] != self.descriptor_dim:
            raise ValueError(
                f"references must have shape [batch, references, {self.descriptor_dim}]"
            )
        if references.shape[0] != query.shape[0]:
            raise ValueError("query and references batch dimensions must match")
        if references.shape[1] < 1 or references.shape[1] > self.max_references:
            raise ValueError(
                f"references must contain between 1 and {self.max_references} rows"
            )
        query = _finite_unit(query, name="query", ndim=2)
        references = _finite_unit(references, name="references", ndim=3)
        if reference_mask is None:
            mask = torch.ones(
                references.shape[:2], dtype=torch.bool, device=references.device
            )
        else:
            if tuple(reference_mask.shape) != tuple(references.shape[:2]):
                raise ValueError("reference_mask must have shape [batch, references]")
            if (
                not torch.is_floating_point(reference_mask)
                and reference_mask.dtype != torch.bool
            ):
                mask = reference_mask != 0
            else:
                if not bool(torch.isfinite(reference_mask.float()).all()):
                    raise ValueError("reference_mask must contain finite values")
                mask = reference_mask != 0
            mask = mask.to(device=references.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("each query must have at least one reference")
        return query, references, mask

    @staticmethod
    def _masked_top_k_mean(
        similarities: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        # Cosine similarities are in [-1, 1], so -2 is a safe padding sentinel
        # and remains friendlier to ONNX than -inf.
        masked = similarities.masked_fill(~mask, -2.0)
        selected_values, selected_indices = torch.topk(
            masked, k=min(int(top_k), similarities.shape[1]), dim=1
        )
        selected_mask = mask.gather(1, selected_indices)
        denominator = (
            selected_mask.sum(dim=1).clamp_min(1).to(dtype=selected_values.dtype)
        )
        return (selected_values * selected_mask.to(selected_values.dtype)).sum(
            dim=1
        ) / denominator

    def _forward_impl(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        mask_float = mask.to(dtype=query.dtype)
        query_tokens = F.normalize(self.token_projection(query), dim=-1)
        reference_tokens = F.normalize(self.token_projection(references), dim=-1)

        similarities = torch.einsum("bd,bkd->bk", query, references)
        pair_query = query_tokens.unsqueeze(1).expand_as(reference_tokens)
        pair_features = torch.cat(
            (pair_query, reference_tokens, pair_query * reference_tokens), dim=-1
        )
        learned_logits = self.pair_attention(pair_features).squeeze(-1)
        logits = similarities / self.attention_temperature + learned_logits
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, dim=1) * mask_float
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled_raw = torch.einsum("bk,bkd->bd", attention, references)
        pooled_reference = F.normalize(pooled_raw, dim=-1)
        centroid_raw = (references * mask_float.unsqueeze(-1)).sum(dim=1)
        centroid = F.normalize(centroid_raw, dim=-1)
        centroid_score = torch.einsum("bd,bd->b", query, centroid)
        top_k_score = self._masked_top_k_mean(similarities, mask, self.reference_top_k)
        baseline_score = (
            1.0 - self.reference_score_weight
        ) * centroid_score + self.reference_score_weight * top_k_score

        pooled_tokens = torch.einsum("bk,bkh->bh", attention, reference_tokens)
        count = mask_float.sum(dim=1)
        count_feature = (count / float(self.max_references)).unsqueeze(1)
        score_features = torch.cat(
            (
                query_tokens,
                pooled_tokens,
                query_tokens * pooled_tokens,
                (query_tokens - pooled_tokens).abs(),
                centroid_score.unsqueeze(1),
                top_k_score.unsqueeze(1),
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
            "attention": attention,
            "similarities": similarities,
            "pooled_reference": pooled_reference,
            "reference_count": count,
            "centroid_score": centroid_score,
            "top_k_score": top_k_score,
        }

    def forward(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        query, references, mask = self._validate_inputs(
            query, references, reference_mask
        )
        return self._forward_impl(query, references, mask, return_aux=return_aux)

    def forward_export(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-only path for ONNX export with a dynamic batch dimension.

        Public ``forward`` performs eager-mode shape and finiteness checks.  A
        Python ``bool(tensor.all())`` is intentionally absent here because it
        creates a data-dependent symbolic guard in newer PyTorch exporters.
        The serving boundary validates these inputs before invoking the graph.
        """

        query = F.normalize(query.float(), dim=-1, eps=1e-12)
        references = F.normalize(references.float(), dim=-1, eps=1e-12)
        mask = reference_mask != 0
        output = self._forward_impl(query, references, mask)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("matcher export path returned auxiliary output")
        return output


class ReferenceSetMatcherExport(nn.Module):
    """Strict tensor-only wrapper used when exporting the matcher to ONNX."""

    def __init__(self, model: QueryConditionedReferenceMatcher) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_export(query, references, reference_mask)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_reference_set_matcher_checkpoint(
    model: QueryConditionedReferenceMatcher,
    *,
    encoder_fingerprint: str | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a portable checkpoint payload without embedding image data."""

    return {
        "format": MODEL_FORMAT,
        "model_config": model.configuration(),
        "model": model.state_dict(),
        "encoder_fingerprint": encoder_fingerprint,
        "training": dict(training or {}),
    }


def save_reference_set_matcher(
    model: QueryConditionedReferenceMatcher,
    path: str | Path,
    *,
    encoder_fingerprint: str | None = None,
    training: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a matcher checkpoint and return its resolved path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".writing")
    payload = create_reference_set_matcher_checkpoint(
        model,
        encoder_fingerprint=encoder_fingerprint,
        training=training,
    )
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def build_reference_set_matcher_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[QueryConditionedReferenceMatcher, dict[str, Any]]:
    """Load and strictly validate a matcher checkpoint."""

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"Unexpected reference matcher format: {checkpoint_path}")
    configuration = payload.get("model_config")
    if not isinstance(configuration, dict):
        raise ValueError("reference matcher checkpoint has no model_config")
    constructor_keys = {
        "descriptor_dim",
        "hidden_dim",
        "max_references",
        "reference_top_k",
        "reference_score_weight",
        "attention_temperature",
        "maximum_residual",
    }
    constructor_config = {
        key: configuration[key] for key in constructor_keys if key in configuration
    }
    model = QueryConditionedReferenceMatcher(**constructor_config)
    incompatible = model.load_state_dict(payload.get("model", {}), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"reference matcher checkpoint mismatch: {incompatible}")
    return model.to(device).eval(), payload


def _numpy_unit(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with {ndim} dimensions")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError(f"{name} contains an invalid descriptor norm")
    return np.ascontiguousarray(array / np.maximum(norms, 1e-12), dtype=np.float32)


def _pad_reference_sets(
    reference_sets: Sequence[np.ndarray],
    *,
    descriptor_dim: int,
    max_references: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not reference_sets:
        raise ValueError("at least one reference set is required")
    normalized: list[np.ndarray] = []
    for index, references in enumerate(reference_sets):
        array = _numpy_unit(
            np.asarray(references), name=f"reference_sets[{index}]", ndim=2
        )
        if array.shape[1] != descriptor_dim:
            raise ValueError(
                f"reference_sets[{index}] must have width {descriptor_dim}"
            )
        if array.shape[0] < 1 or array.shape[0] > max_references:
            raise ValueError(
                f"reference_sets[{index}] must contain 1-{max_references} references"
            )
        normalized.append(array)
    width = max(array.shape[0] for array in normalized)
    padded = np.zeros((len(normalized), width, descriptor_dim), dtype=np.float32)
    mask = np.zeros((len(normalized), width), dtype=np.bool_)
    for index, array in enumerate(normalized):
        padded[index, : array.shape[0]] = array
        mask[index, : array.shape[0]] = True
    return padded, mask


class ReferenceSetMatcherRuntime:
    """NumPy-facing runtime adapter for gallery scoring."""

    def __init__(
        self,
        model: QueryConditionedReferenceMatcher,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: str | Path | None = None,
        checkpoint_payload: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else None
        )
        self.checkpoint_payload = dict(checkpoint_payload or {})
        self.model_fingerprint = (
            _sha256_file(self.checkpoint_path)
            if self.checkpoint_path is not None and self.checkpoint_path.is_file()
            else None
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "ReferenceSetMatcherRuntime":
        model, payload = build_reference_set_matcher_from_checkpoint(
            path, device=device
        )
        return cls(
            model,
            device=device,
            checkpoint_path=path,
            checkpoint_payload=payload,
        )

    def backend_info(self) -> dict[str, Any]:
        return {
            "type": MODEL_FORMAT,
            "device": str(self.device),
            "model_sha256": self.model_fingerprint,
            "model_config": self.model.configuration(),
            "encoder_fingerprint": self.checkpoint_payload.get("encoder_fingerprint"),
        }

    def score(
        self,
        query: np.ndarray,
        references: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        """Score one query/reference pair and return diagnostics."""

        scores, details = self.score_many(query, [references])
        return float(scores[0]), details[0]

    def _score_bounded_many(
        self,
        query_unit: np.ndarray,
        reference_sets: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Score sets whose widths fit in one matcher invocation."""

        padded, mask = _pad_reference_sets(
            reference_sets,
            descriptor_dim=self.model.descriptor_dim,
            max_references=self.model.max_references,
        )
        queries = np.repeat(query_unit[None, :], len(reference_sets), axis=0)
        with torch.inference_mode():
            output = self.model(
                torch.from_numpy(queries).to(self.device),
                torch.from_numpy(padded).to(self.device),
                torch.from_numpy(mask).to(self.device),
                return_aux=True,
            )
        assert isinstance(output, dict)
        scores = output["score"].float().cpu().numpy().astype(np.float32)
        details: list[dict[str, Any]] = []
        for index in range(len(reference_sets)):
            count = int(mask[index].sum())
            details.append(
                {
                    "baseline_score": float(output["baseline_score"][index]),
                    "residual": float(output["residual"][index]),
                    "reference_count": count,
                    "attention": output["attention"][index, :count]
                    .float()
                    .cpu()
                    .tolist(),
                    "similarities": output["similarities"][index, :count]
                    .float()
                    .cpu()
                    .tolist(),
                    "score": float(scores[index]),
                }
            )
        return scores, details

    def score_many(
        self,
        query: np.ndarray,
        reference_sets: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Score one query against several identity reference sets.

        A gallery identity can contain more references than the fixed matcher
        width. Such a set is evaluated in deterministic chunks and its two
        strongest chunk scores are averaged, preserving view-specific evidence
        without silently dropping older enrolled images.
        """

        if not reference_sets:
            raise ValueError("at least one reference set is required")
        query_unit = _numpy_unit(np.asarray(query), name="query", ndim=1)
        if query_unit.shape[0] != self.model.descriptor_dim:
            raise ValueError(f"query must have width {self.model.descriptor_dim}")
        chunks: list[np.ndarray] = []
        owners: list[int] = []
        for owner, references in enumerate(reference_sets):
            rows = np.asarray(references)
            if rows.ndim != 2:
                # Let the bounded validator produce the canonical shape error.
                chunks.append(rows)
                owners.append(owner)
                continue
            if rows.shape[0] < 1:
                chunks.append(rows)
                owners.append(owner)
                continue
            for start in range(0, rows.shape[0], self.model.max_references):
                chunks.append(rows[start : start + self.model.max_references])
                owners.append(owner)
        chunk_scores, chunk_details = self._score_bounded_many(query_unit, chunks)
        if len(chunks) == len(reference_sets):
            return chunk_scores, chunk_details

        grouped: list[list[int]] = [[] for _ in reference_sets]
        for index, owner in enumerate(owners):
            grouped[owner].append(index)
        scores = np.empty(len(reference_sets), dtype=np.float32)
        details: list[dict[str, Any]] = []
        for owner, indices in enumerate(grouped):
            values = np.asarray([chunk_scores[index] for index in indices])
            selected_count = min(2, values.size)
            selected = np.argsort(values)[-selected_count:]
            scores[owner] = np.float32(values[selected].mean())
            best_index = indices[int(np.argmax(values))]
            detail = dict(chunk_details[best_index])
            detail["score"] = float(scores[owner])
            detail["baseline_score"] = float(
                np.mean(
                    [
                        float(chunk_details[indices[index]].get("baseline_score", 0.0))
                        for index in selected
                    ]
                )
            )
            detail["residual"] = float(
                np.mean(
                    [
                        float(chunk_details[indices[index]].get("residual", 0.0))
                        for index in selected
                    ]
                )
            )
            detail["reference_count"] = int(
                sum(
                    int(chunk_details[index].get("reference_count", 0))
                    for index in indices
                )
            )
            detail["chunk_count"] = len(indices)
            detail["chunk_scores"] = [float(chunk_scores[index]) for index in indices]
            detail["chunk_aggregation"] = "top2_mean"
            details.append(detail)
        return scores, details

    def score_gallery(
        self,
        query: np.ndarray,
        prototypes: Sequence[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        """Score gallery prototype records containing ``reference_features``."""

        if not prototypes:
            raise ValueError("gallery cannot be empty")
        reference_sets = []
        identities = []
        for item in prototypes:
            if "reference_features" not in item:
                raise ValueError(
                    "learned reference scoring requires reference_features"
                )
            reference_sets.append(np.asarray(item["reference_features"]))
            identities.append(str(item["pet_id"]))
        scores, rows = self.score_many(query, reference_sets)
        details = {identity: row for identity, row in zip(identities, rows)}
        return scores, details


__all__ = [
    "MODEL_FORMAT",
    "ReferenceSetMatch",
    "QueryConditionedReferenceMatcher",
    "ReferenceSetMatcherExport",
    "ReferenceSetMatcherRuntime",
    "build_reference_set_matcher_from_checkpoint",
    "create_reference_set_matcher_checkpoint",
    "save_reference_set_matcher",
]
