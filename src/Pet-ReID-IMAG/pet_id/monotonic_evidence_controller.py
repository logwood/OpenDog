"""Regularized monotonic evidence controller.

This controller treats the raw BIFOR score and margin as ordered evidence. It learns
monotonic calibration on those two quantities and only a linear residual from
the remaining capture/gallery diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .evidence_controller import (
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
)


SCALAR_FEATURE_NAMES = (
    "expert_available",
    "log_gallery_size",
    "bifor_top1",
    "bifor_margin",
    "bifor_score_mean",
    "bifor_score_std",
    "bifor_reference_mean",
    "bifor_reference_min",
    "bifor_reference_std",
    "bifor_gallery_consistency",
    "nose_available",
    "face_available",
    "nose_quality",
    "face_quality",
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
    "mega_top1",
    "mega_margin",
    "mega_score_mean",
    "mega_score_std",
    "mega_reference_mean",
    "mega_reference_min",
    "mega_reference_std",
    "mega_gallery_consistency",
    "expert_top1_agreement",
    "bifor_mega_top_score_gap",
)

OUTPUT_NAMES = (
    "bifor_correct",
    "mega_correct",
    "unknown",
    "expert_gain",
    "recapture_gain",
    "gallery_stable",
    "temporal_consistency",
)

_FEATURE_INDEX = {name: index for index, name in enumerate(SCALAR_FEATURE_NAMES)}
_OUTPUT_INDEX = {name: index for index, name in enumerate(OUTPUT_NAMES)}
_CANDIDATE_INDEX = {
    name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)
}
_CONTEXT_INDEX = {name: index for index, name in enumerate(CONTEXT_FEATURE_NAMES)}


def scalarize_evidence(candidates: np.ndarray, context: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != len(CANDIDATE_FEATURE_NAMES):
        raise ValueError("candidate evidence shape does not match schema")
    if context.shape != (len(CONTEXT_FEATURE_NAMES),):
        raise ValueError("context evidence shape does not match schema")

    bifor_top = candidates[
        int(np.argmax(candidates[:, _CANDIDATE_INDEX["bifor_is_top1"]]))
    ]
    expert_available = bool(
        context[_CONTEXT_INDEX["expert_available"]] >= 0.5
    )
    if expert_available:
        mega_top = candidates[
            int(np.argmax(candidates[:, _CANDIDATE_INDEX["mega_is_top1"]]))
        ]
    else:
        mega_top = np.zeros(candidates.shape[1], dtype=np.float32)

    def ctx(name: str) -> float:
        return float(context[_CONTEXT_INDEX[name]])

    def b(name: str) -> float:
        return float(bifor_top[_CANDIDATE_INDEX[name]])

    def m(name: str) -> float:
        return float(mega_top[_CANDIDATE_INDEX[name]]) if expert_available else 0.0

    values = (
        float(expert_available),
        ctx("log_gallery_size"),
        ctx("bifor_top1"),
        ctx("bifor_margin"),
        ctx("bifor_score_mean"),
        ctx("bifor_score_std"),
        b("bifor_reference_mean"),
        b("bifor_reference_min"),
        b("bifor_reference_std"),
        b("bifor_gallery_consistency"),
        ctx("nose_available"),
        ctx("face_available"),
        ctx("nose_quality"),
        ctx("face_quality"),
        ctx("detection_confidence"),
        ctx("body_detected"),
        ctx("body_detection_score"),
        ctx("body_crop_coverage"),
        ctx("sharpness"),
        ctx("exposure"),
        ctx("luminance_mean"),
        ctx("luminance_std"),
        ctx("dark_fraction"),
        ctx("bright_fraction"),
        ctx("viewpoint_0"),
        ctx("viewpoint_1"),
        ctx("viewpoint_2"),
        ctx("viewpoint_3"),
        ctx("mega_top1") if expert_available else 0.0,
        ctx("mega_margin") if expert_available else 0.0,
        ctx("mega_score_mean") if expert_available else 0.0,
        ctx("mega_score_std") if expert_available else 0.0,
        m("mega_reference_mean"),
        m("mega_reference_min"),
        m("mega_reference_std"),
        m("mega_gallery_consistency"),
        ctx("expert_top1_agreement") if expert_available else 0.0,
        (
            ctx("bifor_top1") - ctx("mega_top1")
            if expert_available
            else 0.0
        ),
    )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (len(SCALAR_FEATURE_NAMES),):
        raise AssertionError("scalar evidence schema mismatch")
    return output


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows: np.ndarray) -> "ScalarNormalizer":
        rows = np.asarray(rows, dtype=np.float32)
        mean = rows.mean(axis=0)
        std = rows.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        # Keep monotonic cores and the availability mask on their raw scale.
        for name in (
            "expert_available",
            "bifor_top1",
            "bifor_margin",
            "mega_top1",
            "mega_margin",
        ):
            index = _FEATURE_INDEX[name]
            mean[index] = 0.0
            std[index] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def normalize(self, rows: np.ndarray) -> np.ndarray:
        return (np.asarray(rows, dtype=np.float32) - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Sequence[float]]
    ) -> "ScalarNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


class MonotonicResidualController(nn.Module):
    """Constrained logistic heads plus a strongly regularized linear residual."""

    def __init__(self):
        super().__init__()
        self.residual = nn.Linear(len(SCALAR_FEATURE_NAMES), len(OUTPUT_NAMES))
        self.bifor_monotonic = nn.Parameter(torch.ones(2))
        self.unknown_monotonic = nn.Parameter(torch.ones(2))
        self.mega_monotonic = nn.Parameter(torch.ones(2))

        mask = torch.ones(len(OUTPUT_NAMES), len(SCALAR_FEATURE_NAMES))
        # Development folds contain a fixed gallery cardinality, so the
        # controller cannot identify a gallery-size effect from its fit split.
        # Leaving this column trainable makes a numerically near-constant
        # feature acquire an arbitrary weight and catastrophically extrapolate
        # when calibration uses a different-sized gallery.
        mask[:, _FEATURE_INDEX["log_gallery_size"]] = 0.0
        bifor_core = (
            _FEATURE_INDEX["bifor_top1"],
            _FEATURE_INDEX["bifor_margin"],
        )
        mega_core = (
            _FEATURE_INDEX["mega_top1"],
            _FEATURE_INDEX["mega_margin"],
        )
        for index in bifor_core:
            mask[_OUTPUT_INDEX["bifor_correct"], index] = 0.0
            mask[_OUTPUT_INDEX["unknown"], index] = 0.0
        for index in mega_core:
            mask[_OUTPUT_INDEX["mega_correct"], index] = 0.0
        self.register_buffer("residual_mask", mask)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or evidence.shape[1] != len(SCALAR_FEATURE_NAMES):
            raise ValueError("expected [batch, scalar_feature] evidence")
        weight = self.residual.weight * self.residual_mask
        logits = nn.functional.linear(evidence, weight, self.residual.bias)
        b_core = evidence[
            :,
            [
                _FEATURE_INDEX["bifor_top1"],
                _FEATURE_INDEX["bifor_margin"],
            ],
        ]
        m_core = evidence[
            :,
            [
                _FEATURE_INDEX["mega_top1"],
                _FEATURE_INDEX["mega_margin"],
            ],
        ]
        available = evidence[:, _FEATURE_INDEX["expert_available"]]
        logits[:, _OUTPUT_INDEX["bifor_correct"]] += (
            b_core * nn.functional.softplus(self.bifor_monotonic)
        ).sum(dim=1)
        logits[:, _OUTPUT_INDEX["unknown"]] -= (
            b_core * nn.functional.softplus(self.unknown_monotonic)
        ).sum(dim=1)
        logits[:, _OUTPUT_INDEX["mega_correct"]] += available * (
            m_core * nn.functional.softplus(self.mega_monotonic)
        ).sum(dim=1)
        return logits


def probabilities_from_logits(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    return torch.sigmoid(logits / temperatures.clamp_min(1e-3))


DEFAULT_ACTION_COSTS = {
    "accept_bifor": 0.0,
    "accept_mega": 0.0,
    "consult_expert": 0.04,
    "reject_unknown": 0.02,
    "defer_review": 0.25,
}


def choose_action(
    probabilities: Mapping[str, float],
    *,
    expert_available: bool,
    costs: Mapping[str, float] | None = None,
) -> dict[str, object]:
    action_costs = {**DEFAULT_ACTION_COSTS, **dict(costs or {})}
    bifor = float(probabilities.get("bifor_correct", 0.0))
    unknown = float(probabilities.get("unknown", 0.0))
    utilities = {
        "accept_bifor": bifor - action_costs["accept_bifor"],
        "reject_unknown": unknown - action_costs["reject_unknown"],
        "defer_review": 1.0 - action_costs["defer_review"],
    }
    if expert_available:
        utilities["accept_mega"] = (
            float(probabilities.get("mega_correct", 0.0))
            - action_costs["accept_mega"]
        )
    else:
        union_success = min(
            1.0, bifor + float(probabilities.get("expert_gain", 0.0))
        )
        utilities["consult_expert"] = (
            union_success - action_costs["consult_expert"]
        )
    priority = (
        "accept_bifor",
        "accept_mega",
        "consult_expert",
        "reject_unknown",
        "defer_review",
    )
    action = max(
        utilities, key=lambda name: (utilities[name], -priority.index(name))
    )
    return {
        "action": action,
        "utility": float(utilities[action]),
        "utilities": {name: float(value) for name, value in utilities.items()},
    }
