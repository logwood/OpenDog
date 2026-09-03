"""Descriptor-episode sampling and evaluation for the reference-set matcher.

The image encoder is intentionally kept out of this module.  Training consumes
the descriptor caches already produced by the single-image model, which makes
the experiment cheap, reproducible, and independent of the protected image
model artifact.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .reference_set_model import QueryConditionedReferenceMatcher


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains an invalid descriptor norm")
    return np.ascontiguousarray(array / np.maximum(norms, 1e-12), dtype=np.float32)


@dataclass(frozen=True)
class DescriptorTable:
    """Immutable descriptor cache grouped by normalized identity name."""

    embeddings: np.ndarray
    identities: tuple[str, ...]
    source_paths: tuple[str, ...]
    groups: dict[str, np.ndarray]
    source_path: Path | None = None

    @property
    def descriptor_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def identity_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.groups))

    @property
    def num_identities(self) -> int:
        return len(self.groups)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        embedding_key: str = "embedding",
    ) -> "DescriptorTable":
        source = Path(path).expanduser().resolve()
        payload = np.load(source, allow_pickle=False)
        if embedding_key not in payload:
            raise ValueError(f"descriptor cache is missing {embedding_key!r}")
        embeddings = _normalize_rows(payload[embedding_key], name=embedding_key)
        identity_key = "identities" if "identities" in payload else "identity"
        if identity_key not in payload:
            raise ValueError("descriptor cache must contain an identities array")
        raw_identities = np.asarray(payload[identity_key]).reshape(-1)
        if raw_identities.shape[0] != embeddings.shape[0]:
            raise ValueError("identity and embedding row counts differ")
        identities = tuple(str(value).casefold() for value in raw_identities.tolist())
        if "source_paths" in payload:
            raw_paths = np.asarray(payload["source_paths"]).reshape(-1)
            if raw_paths.shape[0] != embeddings.shape[0]:
                raise ValueError("source_paths and embedding row counts differ")
            source_paths = tuple(str(value) for value in raw_paths.tolist())
        else:
            source_paths = tuple("" for _ in identities)
        groups: dict[str, list[int]] = {}
        for index, identity in enumerate(identities):
            groups.setdefault(identity, []).append(index)
        frozen_groups = {
            identity: np.asarray(indices, dtype=np.int64)
            for identity, indices in groups.items()
        }
        return cls(
            embeddings=embeddings,
            identities=identities,
            source_paths=source_paths,
            groups=frozen_groups,
            source_path=source,
        )

    def require_records(self, minimum: int) -> None:
        minimum = int(minimum)
        if minimum < 1:
            raise ValueError("minimum record count must be positive")
        insufficient = {
            identity: int(indices.size)
            for identity, indices in self.groups.items()
            if indices.size < minimum
        }
        if insufficient:
            raise ValueError(
                f"each identity needs at least {minimum} records: {insufficient}"
            )


@dataclass(frozen=True)
class EpisodeBatch:
    """Flattened query/set pairs suitable for one matcher forward pass."""

    queries: torch.Tensor
    references: torch.Tensor
    reference_mask: torch.Tensor
    targets: torch.Tensor
    identity_names: tuple[str, ...]


class ReferenceEpisodeSampler:
    """Sample P-way episodes without putting a query in its own reference set."""

    def __init__(
        self,
        table: DescriptorTable,
        *,
        identities_per_batch: int = 16,
        reference_count: int = 2,
        queries_per_identity: int = 1,
        max_references: int = 4,
        variable_reference_count: bool = True,
        seed: int = 20260902,
    ) -> None:
        self.table = table
        self.identities_per_batch = int(identities_per_batch)
        self.reference_count = int(reference_count)
        self.queries_per_identity = int(queries_per_identity)
        self.max_references = int(max_references)
        self.variable_reference_count = bool(variable_reference_count)
        self.seed = int(seed)
        if min(
            self.identities_per_batch,
            self.reference_count,
            self.queries_per_identity,
            self.max_references,
        ) <= 0:
            raise ValueError("episode sizes must be positive")
        if self.reference_count > self.max_references:
            raise ValueError("reference_count cannot exceed max_references")
        self.table.require_records(self.reference_count + self.queries_per_identity)
        if self.table.num_identities < self.identities_per_batch:
            raise ValueError(
                "descriptor table has fewer identities than identities_per_batch"
            )

    def _rng(self, epoch: int, step: int) -> np.random.Generator:
        # Large relatively-prime strides keep neighboring workers/epochs from
        # producing the same episode while retaining deterministic replay.
        seed = self.seed + int(epoch) * 1_000_003 + int(step) * 9_176
        return np.random.default_rng(seed)

    def sample(self, *, epoch: int, step: int) -> EpisodeBatch:
        rng = self._rng(epoch, step)
        identities = tuple(
            rng.choice(
                np.asarray(self.table.identity_names),
                size=self.identities_per_batch,
                replace=False,
            ).tolist()
        )
        identity_sets: list[np.ndarray] = []
        query_rows: list[np.ndarray] = []
        for identity in identities:
            rows = self.table.groups[identity]
            reference_count = (
                int(rng.integers(1, self.reference_count + 1))
                if self.variable_reference_count
                else self.reference_count
            )
            total = reference_count + self.queries_per_identity
            selected = rng.choice(rows, size=total, replace=False)
            reference_rows = selected[:reference_count]
            query_rows_for_identity = selected[reference_count:]
            identity_sets.append(self.table.embeddings[reference_rows])
            query_rows.extend(
                self.table.embeddings[index] for index in query_rows_for_identity
            )

        query_count = self.queries_per_identity * self.identities_per_batch
        candidate_count = self.identities_per_batch
        queries = np.stack(query_rows).astype(np.float32, copy=False)
        references = np.zeros(
            (
                query_count * candidate_count,
                self.max_references,
                self.table.descriptor_dim,
            ),
            dtype=np.float32,
        )
        mask = np.zeros(
            (query_count * candidate_count, self.max_references), dtype=np.bool_
        )
        # Each identity contributes the same candidate set to all of its queries.
        for query_index in range(query_count):
            for candidate_index, reference_rows in enumerate(identity_sets):
                offset = query_index * candidate_count + candidate_index
                count = reference_rows.shape[0]
                references[offset, :count] = reference_rows
                mask[offset, :count] = True
        expanded_queries = np.repeat(queries, candidate_count, axis=0)
        targets = np.repeat(
            np.arange(self.identities_per_batch, dtype=np.int64),
            self.queries_per_identity,
        )
        return EpisodeBatch(
            queries=torch.from_numpy(expanded_queries),
            references=torch.from_numpy(references),
            reference_mask=torch.from_numpy(mask),
            targets=torch.from_numpy(targets),
            identity_names=identities,
        )


def episode_retrieval_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Cross-entropy over identities in one P-way episode."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, identities]")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("targets must have one value per query")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    return F.cross_entropy(scores.float() / float(temperature), targets.long())


def _fixed_reference_sets(
    table: DescriptorTable,
    *,
    reference_count: int,
    max_references: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, list[int]]:
    table.require_records(reference_count + 1)
    identities = table.identity_names
    references = np.zeros(
        (len(identities), max_references, table.descriptor_dim), dtype=np.float32
    )
    mask = np.zeros((len(identities), max_references), dtype=np.bool_)
    query_indices: list[int] = []
    for column, identity in enumerate(identities):
        rows = table.groups[identity]
        selected = rows[:reference_count]
        references[column, :reference_count] = table.embeddings[selected]
        mask[column, :reference_count] = True
        query_indices.extend(int(index) for index in rows[reference_count:])
    return identities, references, mask, query_indices


def evaluate_reference_matcher(
    model: QueryConditionedReferenceMatcher,
    table: DescriptorTable,
    *,
    reference_count: int = 2,
    max_references: int | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate identity retrieval with a fixed, held-out reference split."""

    if max_references is None:
        max_references = model.max_references
    if max_references < reference_count:
        raise ValueError("max_references cannot be smaller than reference_count")
    identities, references, mask, query_indices = _fixed_reference_sets(
        table,
        reference_count=int(reference_count),
        max_references=int(max_references),
    )
    if not query_indices:
        raise ValueError("evaluation split has no held-out queries")
    identity_to_column = {identity: index for index, identity in enumerate(identities)}
    model = model.to(device).eval()
    predictions: list[np.ndarray] = []
    baseline_predictions: list[np.ndarray] = []
    query_targets: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(query_indices), int(batch_size)):
            indices = query_indices[start : start + int(batch_size)]
            query_array = table.embeddings[indices]
            query_count = len(indices)
            candidate_count = len(identities)
            expanded_queries = np.repeat(query_array, candidate_count, axis=0)
            expanded_references = np.tile(references, (query_count, 1, 1))
            expanded_mask = np.tile(mask, (query_count, 1))
            output = model(
                torch.from_numpy(expanded_queries).to(device),
                torch.from_numpy(expanded_references).to(device),
                torch.from_numpy(expanded_mask).to(device),
                return_aux=True,
            )
            learned = output["score"].reshape(query_count, candidate_count)
            baseline = output["baseline_score"].reshape(query_count, candidate_count)
            predictions.append(learned.float().cpu().numpy())
            baseline_predictions.append(baseline.float().cpu().numpy())
            query_targets.extend(
                identity_to_column[table.identities[index]] for index in indices
            )
    scores = np.concatenate(predictions, axis=0)
    baseline_scores = np.concatenate(baseline_predictions, axis=0)
    targets = np.asarray(query_targets, dtype=np.int64)

    def metrics(matrix: np.ndarray, *, label: str) -> dict[str, Any]:
        ranking = np.argsort(-matrix, axis=1)
        ranks = np.asarray(
            [int(np.flatnonzero(row == target)[0]) + 1 for row, target in zip(ranking, targets)],
            dtype=np.int64,
        )
        positive = matrix[np.arange(targets.size), targets]
        negative = matrix.copy()
        negative[np.arange(targets.size), targets] = np.nan
        negative_values = negative[~np.isnan(negative)]
        return {
            "scorer": label,
            "gallery_identities": len(identities),
            "gallery_images_per_identity": int(reference_count),
            "query_records": int(targets.size),
            "top1_correct": int((ranks == 1).sum()),
            "top1_accuracy": float((ranks == 1).mean()),
            "top5_correct": int((ranks <= 5).sum()),
            "top5_accuracy": float((ranks <= 5).mean()),
            "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
            "same_score_mean": float(np.mean(positive)),
            "different_score_mean": float(np.mean(negative_values)),
        }

    return {
        "learned": metrics(scores, label="query_conditioned_reference_matcher"),
        "baseline": metrics(baseline_scores, label="centroid_top_k_baseline"),
        "model": model.configuration(),
    }


__all__ = [
    "DescriptorTable",
    "EpisodeBatch",
    "ReferenceEpisodeSampler",
    "episode_retrieval_loss",
    "evaluate_reference_matcher",
    "sha256_file",
]


