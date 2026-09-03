"""Scoring utilities for identities with multiple reference images.

The image encoder produces one descriptor per image.  This module keeps that
contract intact and offers three scoring policies: a centroid anchor, a robust
top-k reference score, and an optional learned query-conditioned matcher.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


CENTROID_SCORING = "centroid"
REFERENCE_SET_SCORING = "reference_set"
LEARNED_REFERENCE_SET_SCORING = "learned_reference_set"
SCORING_MODES = (
    CENTROID_SCORING,
    REFERENCE_SET_SCORING,
    LEARNED_REFERENCE_SET_SCORING,
)
DEFAULT_REFERENCE_TOP_K = 3
DEFAULT_REFERENCE_SCORE_WEIGHT = 0.4
MAX_REFERENCE_TOP_K = 32


class LearnedReferenceScorer(Protocol):
    """Minimal protocol implemented by a trained set matcher runtime."""

    def score(
        self, query: np.ndarray, references: np.ndarray
    ) -> tuple[float, dict[str, Any]]: ...

    def score_gallery(
        self,
        query: np.ndarray,
        prototypes: Sequence[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]: ...


def validate_scoring_mode(value: str) -> str:
    """Normalize and validate a semantic gallery scoring mode."""

    mode = str(value).strip().casefold()
    if mode not in SCORING_MODES:
        choices = ", ".join(SCORING_MODES)
        raise ValueError(f"scoring_mode must be one of: {choices}")
    return mode


def validate_reference_top_k(value: int) -> int:
    """Validate the number of per-reference scores used by set scoring."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("reference_top_k must be an integer")
    try:
        top_k = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("reference_top_k must be an integer") from error
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError("reference_top_k must be an integer")
    if top_k < 1 or top_k > MAX_REFERENCE_TOP_K:
        raise ValueError(
            f"reference_top_k must be between 1 and {MAX_REFERENCE_TOP_K}"
        )
    return top_k


def validate_reference_score_weight(value: float) -> float:
    """Validate the contribution of the reference-set score."""

    try:
        weight = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("reference_score_weight must be a number") from error
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("reference_score_weight must be between 0 and 1")
    return weight


def _unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite descriptor vector")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} has an invalid norm")
    # Unified ONNX descriptors are required to be L2-normalized, but floating
    # point export/runtime differences can leave a small residual (for example
    # 0.999).  Preserve that value instead of applying a second normalization;
    # this keeps the graph-output contract observable by callers.  Legacy or
    # hand-built inputs that are clearly not normalized remain safe to score.
    if np.isclose(norm, 1.0, atol=3e-3, rtol=3e-3):
        return np.ascontiguousarray(array, dtype=np.float32)
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def _unit_matrix(value: np.ndarray, name: str, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix with width {dimension}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains an invalid descriptor norm")
    near_unit = np.isclose(norms, 1.0, atol=3e-3, rtol=3e-3)
    normalized = np.where(near_unit, array, array / norms)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def score_identity(
    query: np.ndarray,
    centroid: np.ndarray,
    references: np.ndarray | None = None,
    *,
    scoring_mode: str = CENTROID_SCORING,
    reference_top_k: int = DEFAULT_REFERENCE_TOP_K,
    reference_score_weight: float = DEFAULT_REFERENCE_SCORE_WEIGHT,
    learned_scorer: LearnedReferenceScorer | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score one identity and return a finite, JSON-friendly explanation.

    ``centroid`` and each row of ``references`` are validated and normalized
    defensively when they are materially off-unit; graph descriptors that are
    already within the export tolerance are preserved. In ``reference_set``
    mode the score is a convex combination of the centroid cosine and the mean
    of the strongest ``reference_top_k`` cosines.  In
    ``learned_reference_set`` mode a trained query-conditioned scorer supplies
    the final score while this function still reports raw cosine diagnostics.
    """

    mode = validate_scoring_mode(scoring_mode)
    top_k = validate_reference_top_k(reference_top_k)
    weight = validate_reference_score_weight(reference_score_weight)
    query_unit = _unit_vector(query, "query")
    centroid_unit = _unit_vector(centroid, "centroid")
    if query_unit.shape != centroid_unit.shape:
        raise ValueError("query and centroid dimensions do not match")
    centroid_score = float(query_unit @ centroid_unit)

    detail: dict[str, Any] = {
        "mode": mode,
        "centroid_score": centroid_score,
        "reference_score": None,
        "reference_best": None,
        "reference_top_k": None,
        "reference_count": 0,
        "reference_score_weight": weight if mode == REFERENCE_SET_SCORING else 0.0,
    }
    if mode == CENTROID_SCORING:
        detail["score"] = centroid_score
        return centroid_score, detail
    if references is None:
        raise ValueError("reference-set scoring requires reference descriptors")

    reference_units = _unit_matrix(references, "references", query_unit.size)
    reference_scores = reference_units @ query_unit
    if mode == LEARNED_REFERENCE_SET_SCORING:
        if learned_scorer is None:
            raise ValueError(
                "learned_reference_set scoring requires a trained reference scorer"
            )
        score_method = getattr(learned_scorer, "score", None)
        if callable(score_method):
            learned_score, learned_detail = score_method(
                query_unit, reference_units
            )
        else:
            score_gallery_method = getattr(learned_scorer, "score_gallery", None)
            if not callable(score_gallery_method):
                raise ValueError(
                    "learned reference scorer must provide score or score_gallery"
                )
            gallery_scores, gallery_details = score_gallery_method(
                query_unit,
                [
                    {
                        "pet_id": "__single_reference_identity__",
                        "prototype": centroid_unit,
                        "reference_features": reference_units,
                    }
                ],
            )
            gallery_scores = np.asarray(gallery_scores, dtype=np.float32).reshape(-1)
            if gallery_scores.shape != (1,) or not np.isfinite(gallery_scores).all():
                raise ValueError("learned reference scorer returned an invalid score")
            learned_score = float(gallery_scores[0])
            learned_detail = (
                gallery_details.get("__single_reference_identity__", {})
                if isinstance(gallery_details, dict)
                else {}
            )
        learned_score = float(learned_score)
        if not np.isfinite(learned_score):
            raise ValueError("learned reference scorer returned a non-finite score")
        if not isinstance(learned_detail, dict):
            raise ValueError("learned reference scorer diagnostics must be a mapping")
        detail.update(
            {
                "reference_score": learned_detail.get("baseline_score"),
                "reference_best": float(np.max(reference_scores)),
                "reference_top_k": min(top_k, int(reference_scores.size)),
                "reference_count": int(reference_scores.size),
                "reference_score_weight": 0.0,
                "learned": dict(learned_detail),
                "score": learned_score,
            }
        )
        return learned_score, detail

    selected_count = min(top_k, int(reference_scores.size))
    selected = np.sort(reference_scores)[-selected_count:]
    reference_score = float(np.mean(selected))
    score = (1.0 - weight) * centroid_score + weight * reference_score
    detail.update(
        {
            "reference_score": reference_score,
            "reference_best": float(np.max(reference_scores)),
            "reference_top_k": selected_count,
            "reference_count": int(reference_scores.size),
            "score": float(score),
        }
    )
    return float(score), detail


def score_gallery(
    query: np.ndarray,
    prototypes: Sequence[dict[str, Any]],
    *,
    scoring_mode: str = CENTROID_SCORING,
    reference_top_k: int = DEFAULT_REFERENCE_TOP_K,
    reference_score_weight: float = DEFAULT_REFERENCE_SCORE_WEIGHT,
    learned_scorer: LearnedReferenceScorer | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Score all gallery identities and return scores plus per-identity details."""

    mode = validate_scoring_mode(scoring_mode)
    if mode == LEARNED_REFERENCE_SET_SCORING:
        if learned_scorer is None:
            raise ValueError(
                "learned_reference_set scoring requires a trained reference scorer"
            )
        # The runtime adapter can score all identities in one tensor batch. Keep
        # a per-identity fallback for small test doubles and third-party scorers.
        score_gallery_method = getattr(learned_scorer, "score_gallery", None)
        if callable(score_gallery_method):
            scores, learned_details = score_gallery_method(query, prototypes)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != len(prototypes) or not np.isfinite(scores).all():
                raise ValueError(
                    "learned reference scorer returned invalid gallery scores"
                )
            if not isinstance(learned_details, dict):
                raise ValueError(
                    "learned reference scorer diagnostics must be a mapping"
                )
            details: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(prototypes):
                pet_id = str(item["pet_id"])
                if pet_id in details:
                    raise ValueError(f"duplicate gallery identity: {pet_id}")
                row = dict(learned_details.get(pet_id, {}))
                row.setdefault("mode", mode)
                row.setdefault("score", float(scores[index]))
                row.setdefault(
                    "reference_count",
                    int(np.asarray(item.get("reference_features", [])).shape[0]),
                )
                details[pet_id] = row
            if not details:
                raise ValueError("gallery cannot be empty")
            return scores, details

    scores: list[float] = []
    details: dict[str, dict[str, Any]] = {}
    for item in prototypes:
        pet_id = str(item["pet_id"])
        if pet_id in details:
            raise ValueError(f"duplicate gallery identity: {pet_id}")
        score, detail = score_identity(
            query,
            np.asarray(item["prototype"]),
            item.get("reference_features"),
            scoring_mode=mode,
            reference_top_k=reference_top_k,
            reference_score_weight=reference_score_weight,
            learned_scorer=learned_scorer,
        )
        scores.append(score)
        details[pet_id] = detail
    if not scores:
        raise ValueError("gallery cannot be empty")
    return np.asarray(scores, dtype=np.float32), details


__all__ = [
    "CENTROID_SCORING",
    "REFERENCE_SET_SCORING",
    "LEARNED_REFERENCE_SET_SCORING",
    "SCORING_MODES",
    "DEFAULT_REFERENCE_TOP_K",
    "DEFAULT_REFERENCE_SCORE_WEIGHT",
    "MAX_REFERENCE_TOP_K",
    "LearnedReferenceScorer",
    "score_identity",
    "score_gallery",
    "validate_scoring_mode",
    "validate_reference_top_k",
    "validate_reference_score_weight",
]
