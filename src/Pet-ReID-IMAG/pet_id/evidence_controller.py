"""Learned evidence aggregation and defer policy for pet identification.

The identity encoders stay frozen.  This module only consumes identity-agnostic
evidence: ranked gallery scores, reference support, capture diagnostics, and
whether the optional body expert has already been queried.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


CANDIDATE_FEATURE_NAMES = (
    "bifor_score",
    "bifor_reference_mean",
    "bifor_reference_max",
    "bifor_reference_min",
    "bifor_reference_std",
    "bifor_gallery_consistency",
    "bifor_rank_fraction",
    "bifor_is_top1",
    "mega_score",
    "mega_reference_mean",
    "mega_reference_max",
    "mega_reference_min",
    "mega_reference_std",
    "mega_gallery_consistency",
    "mega_rank_fraction",
    "mega_is_top1",
    "cross_expert_score_gap",
    "expert_available",
)

CONTEXT_FEATURE_NAMES = (
    "log_gallery_size",
    "expert_available",
    "bifor_top1",
    "bifor_top2",
    "bifor_margin",
    "bifor_score_mean",
    "bifor_score_std",
    "mega_top1",
    "mega_top2",
    "mega_margin",
    "mega_score_mean",
    "mega_score_std",
    "expert_top1_agreement",
    "nose_available",
    "face_available",
    "nose_quality",
    "face_quality",
    "nose_fusion_weight",
    "face_fusion_weight",
    "detection_confidence",
    "body_detected",
    "body_detection_score",
    "body_crop_coverage",
    "sharpness",
    "exposure",
    "luminance_mean",
    "luminance_std",
    "dark_fraction",
    "bright_fraction",
    "viewpoint_0",
    "viewpoint_1",
    "viewpoint_2",
    "viewpoint_3",
)

OUTPUT_NAMES = (
    "bifor_correct",
    "mega_correct",
    "consult_success",
    "recapture_correct",
    "unknown",
    "gallery_stable",
    "expert_gain",
    "recapture_gain",
    "temporal_consistency",
)


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _pair(values: object, default: float) -> list[float]:
    if not isinstance(values, list):
        return [default, default]
    return [_float(values[index], default) if index < len(values) else default for index in range(2)]


def _top_statistics(scores: np.ndarray) -> tuple[float, float, float, float, float]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not scores.size:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    order = np.argsort(-scores)
    top1 = float(scores[int(order[0])])
    top2 = float(scores[int(order[1])]) if scores.size > 1 else top1
    return top1, top2, top1 - top2, float(scores.mean()), float(scores.std())


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    return ranks


def _support_row(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return (
        float(values.mean()),
        float(values.max()),
        float(values.min()),
        float(values.std()),
    )


def build_evidence_arrays(
    *,
    bifor_scores: np.ndarray,
    mega_scores: np.ndarray,
    bifor_reference_scores: np.ndarray,
    mega_reference_scores: np.ndarray,
    bifor_gallery_consistency: np.ndarray,
    mega_gallery_consistency: np.ndarray,
    metadata: Mapping[str, object],
    expert_available: bool,
    top_candidates: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a permutation-invariant candidate set and query context.

    When ``expert_available`` is false, neither candidate selection nor any
    returned value depends on MegaDescriptor scores.  This prevents the
    pre-consultation controller from seeing future expert evidence.
    """

    bifor_scores = np.asarray(bifor_scores, dtype=np.float32).reshape(-1)
    mega_scores = np.asarray(mega_scores, dtype=np.float32).reshape(-1)
    if bifor_scores.shape != mega_scores.shape or not bifor_scores.size:
        raise ValueError("BIFOR and Mega scores must be non-empty and aligned")
    gallery_size = bifor_scores.size
    bifor_reference_scores = np.asarray(bifor_reference_scores, dtype=np.float32)
    mega_reference_scores = np.asarray(mega_reference_scores, dtype=np.float32)
    expected_prefix = (gallery_size,)
    if bifor_reference_scores.shape[:1] != expected_prefix:
        raise ValueError("BIFOR reference scores do not match gallery size")
    if mega_reference_scores.shape != bifor_reference_scores.shape:
        raise ValueError("BIFOR and Mega reference score shapes differ")

    keep = max(1, min(int(top_candidates), gallery_size))
    bifor_order = np.argsort(-bifor_scores)
    selected = list(int(value) for value in bifor_order[:keep])
    if expert_available:
        for value in np.argsort(-mega_scores)[:keep]:
            index = int(value)
            if index not in selected:
                selected.append(index)

    bifor_rank = _ranks(bifor_scores)
    mega_rank = _ranks(mega_scores)
    rank_denominator = max(gallery_size - 1, 1)
    rows = []
    for index in selected:
        b_mean, b_max, b_min, b_std = _support_row(bifor_reference_scores[index])
        if expert_available:
            m_mean, m_max, m_min, m_std = _support_row(mega_reference_scores[index])
            mega_values = (
                float(mega_scores[index]),
                m_mean,
                m_max,
                m_min,
                m_std,
                float(mega_gallery_consistency[index]),
                float(mega_rank[index] / rank_denominator),
                float(index == int(np.argmax(mega_scores))),
            )
            score_gap = float(bifor_scores[index] - mega_scores[index])
        else:
            mega_values = (0.0,) * 8
            score_gap = 0.0
        rows.append(
            (
                float(bifor_scores[index]),
                b_mean,
                b_max,
                b_min,
                b_std,
                float(bifor_gallery_consistency[index]),
                float(bifor_rank[index] / rank_denominator),
                float(index == int(np.argmax(bifor_scores))),
                *mega_values,
                score_gap,
                float(expert_available),
            )
        )

    primary = metadata.get("primary")
    primary = primary if isinstance(primary, Mapping) else {}
    descriptor = primary.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    available = _pair(descriptor.get("branch_available"), 1.0)
    quality = _pair(descriptor.get("branch_quality"), 0.5)
    fusion = _pair(descriptor.get("fusion_weights"), 0.5)
    detection = descriptor.get("detection")
    detection = detection if isinstance(detection, Mapping) else {}
    runtime = descriptor.get("runtime_diagnostics")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    body = runtime.get("body")
    body = body if isinstance(body, Mapping) else {}
    viewpoint = descriptor.get("viewpoint")
    viewpoint = viewpoint if isinstance(viewpoint, list) else []

    experts = metadata.get("experts")
    experts = experts if isinstance(experts, Mapping) else {}
    mega_metadata = experts.get("megadescriptor_b224")
    mega_metadata = mega_metadata if isinstance(mega_metadata, Mapping) else {}
    image_quality = mega_metadata.get("quality")
    image_quality = image_quality if isinstance(image_quality, Mapping) else {}

    b_stats = _top_statistics(bifor_scores)
    m_stats = _top_statistics(mega_scores) if expert_available else (0.0,) * 5
    agreement = float(
        expert_available and int(np.argmax(bifor_scores)) == int(np.argmax(mega_scores))
    )
    context = np.asarray(
        (
            math.log1p(gallery_size),
            float(expert_available),
            *b_stats,
            *m_stats,
            agreement,
            *available,
            *quality,
            *fusion,
            _float(detection.get("confidence")),
            float(bool(body.get("detected"))),
            _float(body.get("score")),
            _float(mega_metadata.get("crop_coverage")),
            _float(image_quality.get("sharpness"), 0.5),
            _float(image_quality.get("exposure"), 0.5),
            _float(image_quality.get("luminance_mean"), 0.5),
            _float(image_quality.get("luminance_std")),
            _float(image_quality.get("dark_fraction")),
            _float(image_quality.get("bright_fraction")),
            *(_float(viewpoint[index]) if index < len(viewpoint) else 0.0 for index in range(4)),
        ),
        dtype=np.float32,
    )
    candidates = np.asarray(rows, dtype=np.float32)
    mask = np.ones(candidates.shape[0], dtype=np.bool_)
    if candidates.shape[1] != len(CANDIDATE_FEATURE_NAMES):
        raise AssertionError("candidate feature schema mismatch")
    if context.size != len(CONTEXT_FEATURE_NAMES):
        raise AssertionError("context feature schema mismatch")
    return candidates, mask, context


@dataclass(frozen=True)
class EvidenceNormalizer:
    candidate_mean: np.ndarray
    candidate_std: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray

    @classmethod
    def fit(
        cls,
        candidate_rows: Sequence[np.ndarray],
        context_rows: Sequence[np.ndarray],
    ) -> "EvidenceNormalizer":
        candidates = np.concatenate(candidate_rows, axis=0).astype(np.float32)
        contexts = np.stack(context_rows).astype(np.float32)
        candidate_std = candidates.std(axis=0)
        context_std = contexts.std(axis=0)
        return cls(
            candidate_mean=candidates.mean(axis=0),
            candidate_std=np.where(candidate_std < 1e-6, 1.0, candidate_std),
            context_mean=contexts.mean(axis=0),
            context_std=np.where(context_std < 1e-6, 1.0, context_std),
        )

    def normalize(
        self, candidates: np.ndarray, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            (candidates - self.candidate_mean) / self.candidate_std,
            (context - self.context_mean) / self.context_std,
        )

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_std": self.candidate_std.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Sequence[float]]) -> "EvidenceNormalizer":
        return cls(**{key: np.asarray(state[key], dtype=np.float32) for key in (
            "candidate_mean", "candidate_std", "context_mean", "context_std"
        )})


class EvidenceNet(nn.Module):
    """DeepSets evidence encoder with independent one-vs-all output heads."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(len(CANDIDATE_FEATURE_NAMES), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(len(CONTEXT_FEATURE_NAMES), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in OUTPUT_NAMES})

    def forward(
        self,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 3 or candidate_mask.ndim != 2 or context.ndim != 2:
            raise ValueError("expected batched candidates, mask, and context")
        encoded = self.candidate_encoder(candidates)
        mask = candidate_mask.bool().unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        pooled_mean = (encoded * mask).sum(dim=1) / count
        minimum = torch.finfo(encoded.dtype).min
        pooled_max = encoded.masked_fill(~mask, minimum).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        query = self.context_encoder(context)
        shared = self.shared(torch.cat((pooled_mean, pooled_max, query), dim=1))
        return torch.cat([self.heads[name](shared) for name in OUTPUT_NAMES], dim=1)


def calibrated_probabilities(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits / temperatures.clamp_min(1e-3))


DEFAULT_ACTION_COSTS = {
    "accept_bifor": 0.0,
    "accept_mega": 0.0,
    "consult_expert": 0.04,
    "recapture": 0.08,
    "reject_unknown": 0.02,
    "defer_review": 0.25,
}


def choose_action(
    probabilities: Mapping[str, float],
    *,
    expert_available: bool,
    costs: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Choose the maximum expected-utility action without score thresholds."""

    action_costs = {**DEFAULT_ACTION_COSTS, **dict(costs or {})}
    utilities = {
        "accept_bifor": _float(probabilities.get("bifor_correct")) - action_costs["accept_bifor"],
        "recapture": _float(probabilities.get("recapture_correct")) - action_costs["recapture"],
        "reject_unknown": _float(probabilities.get("unknown")) - action_costs["reject_unknown"],
        "defer_review": 1.0 - action_costs["defer_review"],
    }
    if expert_available:
        utilities["accept_mega"] = _float(probabilities.get("mega_correct")) - action_costs["accept_mega"]
    else:
        utilities["consult_expert"] = _float(probabilities.get("consult_success")) - action_costs["consult_expert"]
    priority = (
        "accept_bifor",
        "accept_mega",
        "consult_expert",
        "recapture",
        "reject_unknown",
        "defer_review",
    )
    action = max(utilities, key=lambda name: (utilities[name], -priority.index(name)))
    return {
        "action": action,
        "utility": float(utilities[action]),
        "utilities": {key: float(value) for key, value in utilities.items()},
    }


def learned_judgments(
    probabilities: Mapping[str, float], *, expert_available: bool
) -> dict[str, float]:
    match = _float(probabilities.get("bifor_correct"))
    if expert_available:
        match = max(match, _float(probabilities.get("mega_correct")))
    return {
        "match_reliability": match,
        "novelty_risk": _float(probabilities.get("unknown")),
        "gallery_support": _float(probabilities.get("gallery_stable")),
        "expert_expected_gain": _float(probabilities.get("expert_gain")),
        "recapture_expected_gain": _float(probabilities.get("recapture_gain")),
        "temporal_consistency": _float(probabilities.get("temporal_consistency")),
    }
