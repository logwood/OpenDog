"""Two-stage identity retrieval with reference-level evidence.

The coarse stage compares a query descriptor with every enrolled reference
independently. Only the strongest references determine identity recall, so a
useful side or frontal view is not cancelled by averaging it with unrelated
views. Shortlisted identities are then scored in one batch by the
token-conditioned matcher.

The gallery service exposes this path only through explicit construction. Its
stable centroid and descriptor-set modes remain unchanged, while opted-in
galleries persist these reference tokens beside their descriptors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn


IDENTITY_SET_RERANKING = "identity_set_rerank"


@dataclass(frozen=True)
class QueryEvidence:
    """Descriptor and spatial tokens extracted from one query image."""

    descriptor: np.ndarray
    tokens: np.ndarray
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReferenceEvidence:
    """One independently addressable gallery image and its cached evidence."""

    reference_id: str
    descriptor: np.ndarray
    tokens: np.ndarray | None
    viewpoint: str | None = None
    viewpoint_signals: np.ndarray | None = None
    quality: float | None = None


@dataclass(frozen=True)
class IdentityReferenceSet:
    """References belonging to one identity without a collapsed prototype."""

    identity_id: str
    references: Sequence[ReferenceEvidence]
    display_name: str | None = None


class CandidateReferenceSelector(Protocol):
    """Batch scorer used after descriptor-based candidate retrieval."""

    descriptor_dim: int
    token_dim: int
    max_references: int

    def select(
        self,
        query_descriptor: np.ndarray,
        query_tokens: np.ndarray,
        reference_descriptors: np.ndarray,
        reference_tokens: np.ndarray,
        reference_mask: np.ndarray,
    ) -> Mapping[str, np.ndarray]: ...


class IdentitySetRerankerRuntime(Protocol):
    """Structural service contract for Python, Torch, or exported backends."""

    descriptor_dim: int
    token_dim: int
    max_references: int
    candidate_count: int
    coarse_support_count: int

    def configuration(self) -> Mapping[str, Any]: ...

    def rerank(
        self,
        query: QueryEvidence,
        identities: Sequence[IdentityReferenceSet],
        *,
        limit: int | None = None,
        token_loader: (
            Callable[[Sequence[str]], Mapping[str, np.ndarray]] | None
        ) = None,
    ) -> Mapping[str, Any]: ...


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{name} must be a positive integer")
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit_vector(value: np.ndarray, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite vector with width {width}")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} has an invalid norm")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def _unit_rows(value: np.ndarray, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != width
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a non-empty finite matrix with width {width}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains an invalid row norm")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _finite_output(
    output: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    required: bool,
) -> np.ndarray | None:
    value = output.get(name)
    if value is None:
        if required:
            raise ValueError(f"selector output is missing {name}")
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"selector output {name} must have shape {shape}")
    return array


class QueryConditionedReferenceSelector:
    """NumPy-facing adapter around the learned token reference matcher."""

    _RERANK_OUTPUTS = (
        "score",
        "attention",
        "similarities",
        "token_scores",
        "novelty",
        "coverage_gate",
        "baseline_score",
        "residual",
        "coverage_score",
        "duplicate_score",
        "centroid_score",
        "top_k_score",
    )

    def __init__(
        self,
        matcher: nn.Module,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        for name in ("descriptor_dim", "token_dim", "max_references"):
            if not hasattr(matcher, name):
                raise TypeError(f"matcher must expose {name}")
        parameter = next(matcher.parameters(), None)
        resolved_device = (
            torch.device(device)
            if device is not None
            else (parameter.device if parameter is not None else torch.device("cpu"))
        )
        self.matcher = matcher.to(resolved_device).eval()
        self.device = resolved_device
        self.descriptor_dim = int(getattr(matcher, "descriptor_dim"))
        self.token_dim = int(getattr(matcher, "token_dim"))
        self.max_references = int(getattr(matcher, "max_references"))

    def backend_info(self) -> dict[str, Any]:
        configuration = getattr(self.matcher, "configuration", None)
        return {
            "type": "query_conditioned_reference_selector",
            "device": str(self.device),
            "descriptor_dim": self.descriptor_dim,
            "token_dim": self.token_dim,
            "max_references": self.max_references,
            "matcher": configuration() if callable(configuration) else None,
        }

    def select(
        self,
        query_descriptor: np.ndarray,
        query_tokens: np.ndarray,
        reference_descriptors: np.ndarray,
        reference_tokens: np.ndarray,
        reference_mask: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        """Score all shortlisted identities in one matcher invocation."""

        descriptors = np.asarray(reference_descriptors, dtype=np.float32)
        tokens = np.asarray(reference_tokens, dtype=np.float32)
        mask = np.asarray(reference_mask, dtype=np.bool_)
        if descriptors.ndim != 3 or descriptors.shape[2] != self.descriptor_dim:
            raise ValueError("reference_descriptors have an unexpected shape")
        batch, width = descriptors.shape[:2]
        if width < 1 or width > self.max_references:
            raise ValueError("reference_descriptors exceed matcher capacity")
        if tokens.ndim != 4 or tokens.shape[:2] != (batch, width):
            raise ValueError("reference_tokens have an unexpected shape")
        if tokens.shape[3] != self.token_dim:
            raise ValueError("reference token width does not match the matcher")
        if mask.shape != (batch, width) or not mask.any(axis=1).all():
            raise ValueError("reference_mask must select at least one row per identity")
        query = _unit_vector(query_descriptor, "query_descriptor", self.descriptor_dim)
        query_grid = _unit_rows(query_tokens, "query_tokens", self.token_dim)
        if tokens.shape[2] != query_grid.shape[0]:
            raise ValueError("query and reference token counts do not match")

        queries = np.repeat(query[None, :], batch, axis=0)
        query_grids = np.repeat(query_grid[None, :, :], batch, axis=0)
        with torch.inference_mode():
            result = self.matcher(
                torch.from_numpy(queries).to(self.device),
                torch.from_numpy(descriptors).to(self.device),
                torch.from_numpy(query_grids).to(self.device),
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(mask).to(self.device),
                return_aux=True,
            )
        if not isinstance(result, Mapping):
            raise RuntimeError("token matcher returned no auxiliary evidence")
        converted: dict[str, np.ndarray] = {}
        for name in self._RERANK_OUTPUTS:
            value = result.get(name)
            if isinstance(value, torch.Tensor):
                converted[name] = value.detach().float().cpu().numpy()
        return converted


class ModelReferenceEvidenceEncoder:
    """Adapt a token-aware image model to the gallery evidence contract.

    The preprocess callable is supplied by the caller so the serving transform
    exactly matches the transform used while training/exporting the model. It
    must return a CHW or BCHW tensor (or an array with the same layout).
    """

    def __init__(
        self,
        model: nn.Module,
        preprocess: Callable[[Path], Any],
        *,
        model_fingerprint: str,
        device: str | torch.device | None = None,
    ) -> None:
        if not callable(preprocess):
            raise TypeError("preprocess must be callable")
        if not model_fingerprint or not str(model_fingerprint).strip():
            raise ValueError("model_fingerprint is required")
        for name in ("descriptor_dim", "token_dim", "token_grid"):
            if not hasattr(model, name):
                raise TypeError(f"token-aware model must expose {name}")
        parameter = next(model.parameters(), None)
        resolved_device = (
            torch.device(device)
            if device is not None
            else (parameter.device if parameter is not None else torch.device("cpu"))
        )
        self.model = model.to(resolved_device).eval()
        self.preprocess = preprocess
        self.device = resolved_device
        self.model_fingerprint = str(model_fingerprint).strip()
        self.descriptor_dim = int(getattr(model, "descriptor_dim"))
        self.token_dim = int(getattr(model, "token_dim"))
        grid = int(getattr(model, "token_grid"))
        if grid < 1:
            raise ValueError("token-aware model token_grid must be positive")
        self.token_count = grid * grid

    def backend_info(self) -> dict[str, Any]:
        configuration = getattr(self.model, "configuration", None)
        preprocess_configuration = getattr(self.preprocess, "configuration", None)
        preprocess_info = (
            preprocess_configuration()
            if callable(preprocess_configuration)
            else {"type": type(self.preprocess).__name__}
        )
        if not isinstance(preprocess_info, Mapping):
            raise TypeError("preprocess configuration must be a mapping")
        return {
            "type": "model_reference_evidence_encoder",
            "model_sha256": self.model_fingerprint,
            "reference_evidence_fingerprint": self.model_fingerprint,
            "device": str(self.device),
            "descriptor_dim": self.descriptor_dim,
            "token_dim": self.token_dim,
            "token_count": self.token_count,
            "preprocess": dict(preprocess_info),
            "model_config": configuration() if callable(configuration) else None,
        }

    def encode_file(self, path: Path) -> QueryEvidence:
        value = self.preprocess(Path(path))
        image = torch.as_tensor(value).float()
        if image.ndim == 3:
            if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("preprocess must return one CHW or BCHW image tensor")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("preprocess returned non-finite image values")
        encode = getattr(self.model, "encode_image_features", None)
        if not callable(encode):
            raise TypeError("token-aware model must provide encode_image_features")
        with torch.inference_mode():
            descriptor, tokens = encode(image.to(self.device))
        descriptor = descriptor.detach().float().cpu().numpy()
        tokens = tokens.detach().float().cpu().numpy()
        if descriptor.shape != (1, self.descriptor_dim):
            raise ValueError("token-aware model returned an invalid descriptor shape")
        if tokens.shape != (1, self.token_count, self.token_dim):
            raise ValueError("token-aware model returned an invalid token shape")
        return QueryEvidence(
            descriptor=np.ascontiguousarray(descriptor[0]),
            tokens=np.ascontiguousarray(tokens[0]),
        )


class IdentitySetReranker:
    """Recall identities cheaply, then rerank only candidates with tokens."""

    _PER_REFERENCE_OUTPUTS = (
        "similarities",
        "token_scores",
        "novelty",
        "coverage_gate",
    )
    _PER_IDENTITY_OUTPUTS = (
        "baseline_score",
        "residual",
        "coverage_score",
        "duplicate_score",
        "centroid_score",
        "top_k_score",
    )

    def __init__(
        self,
        selector: CandidateReferenceSelector,
        *,
        candidate_count: int = 32,
        coarse_support_count: int = 1,
    ) -> None:
        self.selector = selector
        self.candidate_count = _positive_integer(candidate_count, "candidate_count")
        self.coarse_support_count = _positive_integer(
            coarse_support_count, "coarse_support_count"
        )
        self.descriptor_dim = _positive_integer(
            selector.descriptor_dim, "selector.descriptor_dim"
        )
        self.token_dim = _positive_integer(selector.token_dim, "selector.token_dim")
        self.max_references = _positive_integer(
            selector.max_references, "selector.max_references"
        )

    def configuration(self) -> dict[str, Any]:
        selector_info = getattr(self.selector, "backend_info", None)
        return {
            "type": IDENTITY_SET_RERANKING,
            "candidate_count": self.candidate_count,
            "coarse_support_count": self.coarse_support_count,
            "descriptor_dim": self.descriptor_dim,
            "token_dim": self.token_dim,
            "max_references": self.max_references,
            "selector": selector_info() if callable(selector_info) else None,
        }

    @staticmethod
    def _evidence_diagnostic(
        references: Sequence[dict[str, Any]],
        *,
        coverage_score: float | None,
        duplicate_score: float | None,
    ) -> dict[str, Any]:
        duplicate_pairs: list[list[str]] = []
        for left in range(len(references)):
            for right in range(left + 1, len(references)):
                if np.array_equal(
                    references[left]["descriptor"], references[right]["descriptor"]
                ) and np.array_equal(
                    references[left]["tokens"], references[right]["tokens"]
                ):
                    duplicate_pairs.append(
                        [
                            references[left]["reference_id"],
                            references[right]["reference_id"],
                        ]
                    )

        viewpoints = [
            reference["viewpoint"]
            for reference in references
            if reference["viewpoint"] is not None
        ]
        viewpoint_counts = {
            viewpoint: viewpoints.count(viewpoint)
            for viewpoint in sorted(set(viewpoints))
        }
        repeated_viewpoints = sorted(
            viewpoint for viewpoint, count in viewpoint_counts.items() if count > 1
        )
        continuous_viewpoints = [
            tuple(float(value) for value in reference["viewpoint_signals"])
            for reference in references
            if reference["viewpoint_signals"] is not None
        ]
        if len(references) < 2:
            status = "low_reference_support"
        elif duplicate_pairs:
            status = "duplicate_references"
        elif not viewpoints and not continuous_viewpoints:
            status = "viewpoint_metadata_unavailable"
        elif not viewpoints and len(continuous_viewpoints) != len(references):
            status = "viewpoint_metadata_incomplete"
        elif not viewpoints and len(set(continuous_viewpoints)) < 2:
            status = "missing_complementary_view"
        elif not viewpoints:
            status = "viewpoint_coverage_uncalibrated"
        elif len(viewpoints) != len(references):
            status = "viewpoint_metadata_incomplete"
        elif len(viewpoint_counts) < 2:
            status = "missing_complementary_view"
        else:
            status = "sufficient"
        return {
            "status": status,
            "reference_count": len(references),
            "viewpoint_metadata_count": len(viewpoints),
            "continuous_viewpoint_count": len(continuous_viewpoints),
            "distinct_viewpoints": list(viewpoint_counts),
            "repeated_viewpoints": repeated_viewpoints,
            "duplicate_reference_pairs": duplicate_pairs,
            "model_signals": {
                "coverage_score": coverage_score,
                "duplicate_score": duplicate_score,
                "calibration": "uncalibrated",
            },
        }

    def rerank(
        self,
        query: QueryEvidence,
        identities: Sequence[IdentityReferenceSet],
        *,
        limit: int | None = None,
        token_loader: (
            Callable[[Sequence[str]], Mapping[str, np.ndarray]] | None
        ) = None,
    ) -> dict[str, Any]:
        """Return a JSON-friendly coarse ranking and token-reranked shortlist."""

        if not identities:
            raise ValueError("identities cannot be empty")
        query_descriptor = _unit_vector(
            query.descriptor, "query.descriptor", self.descriptor_dim
        )
        query_tokens = _unit_rows(query.tokens, "query.tokens", self.token_dim)
        prepared: list[dict[str, Any]] = []
        identity_ids: set[str] = set()
        global_reference_ids: set[str] = set()
        for identity_index, identity in enumerate(identities):
            identity_id = _identifier(
                identity.identity_id, f"identities[{identity_index}].identity_id"
            )
            if identity_id in identity_ids:
                raise ValueError(f"duplicate identity_id: {identity_id}")
            identity_ids.add(identity_id)
            if not identity.references:
                raise ValueError(f"identity {identity_id!r} has no references")
            if len(identity.references) > self.max_references:
                raise ValueError(
                    f"identity {identity_id!r} has {len(identity.references)} references; "
                    f"the selector capacity is {self.max_references}"
                )
            references: list[dict[str, Any]] = []
            for reference_index, reference in enumerate(identity.references):
                prefix = f"identity {identity_id!r} reference {reference_index}"
                reference_id = _identifier(reference.reference_id, f"{prefix} id")
                if reference_id in global_reference_ids:
                    raise ValueError(f"duplicate reference_id: {reference_id}")
                global_reference_ids.add(reference_id)
                descriptor = _unit_vector(
                    reference.descriptor, f"{prefix} descriptor", self.descriptor_dim
                )
                tokens = (
                    None
                    if reference.tokens is None
                    else _unit_rows(
                        reference.tokens,
                        f"{prefix} tokens",
                        self.token_dim,
                    )
                )
                if tokens is not None and tokens.shape[0] != query_tokens.shape[0]:
                    raise ValueError(
                        f"{prefix} token count does not match the query token count"
                    )
                viewpoint = (
                    None
                    if reference.viewpoint is None
                    else _identifier(
                        reference.viewpoint, f"{prefix} viewpoint"
                    ).casefold()
                )
                viewpoint_signals = reference.viewpoint_signals
                if viewpoint_signals is not None:
                    viewpoint_signals = np.asarray(viewpoint_signals, dtype=np.float32)
                    if (
                        viewpoint_signals.ndim != 1
                        or viewpoint_signals.size < 1
                        or not np.isfinite(viewpoint_signals).all()
                    ):
                        raise ValueError(
                            f"{prefix} viewpoint_signals must be one finite vector"
                        )
                    viewpoint_signals = np.ascontiguousarray(viewpoint_signals)
                quality = reference.quality
                if quality is not None:
                    quality = float(quality)
                    if not np.isfinite(quality):
                        raise ValueError(f"{prefix} quality must be finite")
                references.append(
                    {
                        "reference_id": reference_id,
                        "descriptor": descriptor,
                        "tokens": tokens,
                        "viewpoint": viewpoint,
                        "viewpoint_signals": viewpoint_signals,
                        "quality": quality,
                    }
                )
            descriptors = np.stack([item["descriptor"] for item in references])
            similarities = descriptors @ query_descriptor
            support_count = min(self.coarse_support_count, len(references))
            support_order = sorted(
                range(len(references)),
                key=lambda index: (
                    -float(similarities[index]),
                    references[index]["reference_id"],
                ),
            )
            support_indices = support_order[:support_count]
            prepared.append(
                {
                    "identity_id": identity_id,
                    "display_name": identity.display_name,
                    "references": references,
                    "coarse_score": float(np.mean(similarities[support_indices])),
                    "coarse_similarities": similarities.astype(np.float32),
                    "coarse_support_ids": [
                        references[index]["reference_id"] for index in support_indices
                    ],
                }
            )

        coarse_order = sorted(
            range(len(prepared)),
            key=lambda index: (
                -prepared[index]["coarse_score"],
                prepared[index]["identity_id"],
            ),
        )
        coarse_ranks = {index: rank + 1 for rank, index in enumerate(coarse_order)}
        shortlist_indices = coarse_order[: min(self.candidate_count, len(coarse_order))]
        shortlisted = [prepared[index] for index in shortlist_indices]
        missing_token_references = [
            reference
            for identity in shortlisted
            for reference in identity["references"]
            if reference["tokens"] is None
        ]
        if missing_token_references:
            if token_loader is None:
                raise ValueError(
                    "shortlisted references have no tokens and no token_loader was provided"
                )
            requested_ids = [
                reference["reference_id"] for reference in missing_token_references
            ]
            loaded_tokens = token_loader(requested_ids)
            if not isinstance(loaded_tokens, Mapping):
                raise ValueError("token_loader must return a reference-id mapping")
            for reference in missing_token_references:
                reference_id = reference["reference_id"]
                if reference_id not in loaded_tokens:
                    raise ValueError(
                        f"token_loader did not return reference {reference_id!r}"
                    )
                tokens = _unit_rows(
                    loaded_tokens[reference_id],
                    f"reference {reference_id!r} tokens",
                    self.token_dim,
                )
                if tokens.shape[0] != query_tokens.shape[0]:
                    raise ValueError(
                        f"reference {reference_id!r} token count does not match "
                        "the query token count"
                    )
                reference["tokens"] = tokens
        batch = len(shortlisted)
        width = max(len(item["references"]) for item in shortlisted)
        token_count = query_tokens.shape[0]
        reference_descriptors = np.zeros(
            (batch, width, self.descriptor_dim), dtype=np.float32
        )
        reference_tokens = np.zeros(
            (batch, width, token_count, self.token_dim), dtype=np.float32
        )
        reference_mask = np.zeros((batch, width), dtype=np.bool_)
        for row, identity in enumerate(shortlisted):
            count = len(identity["references"])
            reference_descriptors[row, :count] = np.stack(
                [item["descriptor"] for item in identity["references"]]
            )
            reference_tokens[row, :count] = np.stack(
                [item["tokens"] for item in identity["references"]]
            )
            reference_mask[row, :count] = True

        selector_output = self.selector.select(
            query_descriptor,
            query_tokens,
            reference_descriptors,
            reference_tokens,
            reference_mask,
        )
        if not isinstance(selector_output, Mapping):
            raise ValueError("selector output must be a mapping")
        scores = _finite_output(selector_output, "score", (batch,), required=True)
        attention = _finite_output(
            selector_output, "attention", (batch, width), required=True
        )
        assert scores is not None and attention is not None
        if np.any(attention < 0.0):
            raise ValueError("selector attention cannot be negative")
        if np.any(np.abs(attention[~reference_mask]) > 1e-6):
            raise ValueError("selector attention must be zero on padded references")
        active_attention = (attention * reference_mask).sum(axis=1)
        if not np.allclose(active_attention, 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError("selector attention must sum to one per identity")

        per_reference = {
            name: _finite_output(selector_output, name, (batch, width), required=False)
            for name in self._PER_REFERENCE_OUTPUTS
        }
        per_identity = {
            name: _finite_output(selector_output, name, (batch,), required=False)
            for name in self._PER_IDENTITY_OUTPUTS
        }
        matches: list[dict[str, Any]] = []
        for row, identity_index in enumerate(shortlist_indices):
            identity = prepared[identity_index]
            contributions = []
            for column, reference in enumerate(identity["references"]):
                contribution: dict[str, Any] = {
                    "reference_id": reference["reference_id"],
                    "contribution_weight": float(attention[row, column]),
                    "coarse_similarity": float(identity["coarse_similarities"][column]),
                    "coarse_support": reference["reference_id"]
                    in identity["coarse_support_ids"],
                    "viewpoint": reference["viewpoint"],
                    "viewpoint_signals": (
                        None
                        if reference["viewpoint_signals"] is None
                        else reference["viewpoint_signals"].tolist()
                    ),
                    "quality": reference["quality"],
                }
                for name, values in per_reference.items():
                    contribution[name] = (
                        None if values is None else float(values[row, column])
                    )
                contributions.append(contribution)
            identity_signals = {
                name: None if values is None else float(values[row])
                for name, values in per_identity.items()
            }
            matches.append(
                {
                    "identity_id": identity["identity_id"],
                    "display_name": identity["display_name"],
                    "score": float(scores[row]),
                    "coarse_score": identity["coarse_score"],
                    "coarse_rank": coarse_ranks[identity_index],
                    "reference_contributions": contributions,
                    "evidence": self._evidence_diagnostic(
                        identity["references"],
                        coverage_score=identity_signals["coverage_score"],
                        duplicate_score=identity_signals["duplicate_score"],
                    ),
                    "model_signals": identity_signals,
                }
            )

        matches.sort(
            key=lambda item: (
                -item["score"],
                -item["coarse_score"],
                item["identity_id"],
            )
        )
        for rank, match in enumerate(matches, start=1):
            match["rank"] = rank
        if limit is not None:
            matches = matches[: _positive_integer(limit, "limit")]

        shortlist_set = set(shortlist_indices)
        coarse_ranking = []
        for rank, identity_index in enumerate(coarse_order, start=1):
            identity = prepared[identity_index]
            coarse_ranking.append(
                {
                    "rank": rank,
                    "identity_id": identity["identity_id"],
                    "display_name": identity["display_name"],
                    "score": identity["coarse_score"],
                    "reference_count": len(identity["references"]),
                    "support_reference_ids": identity["coarse_support_ids"],
                    "references": [
                        {
                            "reference_id": reference["reference_id"],
                            "similarity": float(
                                identity["coarse_similarities"][reference_index]
                            ),
                        }
                        for reference_index, reference in enumerate(
                            identity["references"]
                        )
                    ],
                    "shortlisted": identity_index in shortlist_set,
                }
            )
        return {
            "matches": matches,
            "coarse_ranking": coarse_ranking,
            "total_identities": len(prepared),
            "reranked_identities": batch,
            "coarse_support_count": self.coarse_support_count,
        }


__all__ = [
    "CandidateReferenceSelector",
    "IDENTITY_SET_RERANKING",
    "IdentityReferenceSet",
    "IdentitySetReranker",
    "IdentitySetRerankerRuntime",
    "ModelReferenceEvidenceEncoder",
    "QueryConditionedReferenceSelector",
    "QueryEvidence",
    "ReferenceEvidence",
]
