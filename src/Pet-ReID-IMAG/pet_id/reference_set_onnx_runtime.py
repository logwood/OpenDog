"""ONNX Runtime adapter for the reference-set matcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_vector(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != width or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite descriptor of width {width}")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} has an invalid norm")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


class ReferenceSetONNXRuntime:
    """Score padded reference sets with a fixed-width ONNX graph."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "cpu",
        metadata_path: str | Path | None = None,
        verify_hash: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("onnxruntime is required for the ONNX matcher") from error
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        candidate_metadata = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        self.metadata_path = candidate_metadata if candidate_metadata.is_file() else None
        self.metadata = (
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata_path is not None
            else {}
        )
        self.model_sha256 = _sha256_file(self.model_path)
        expected_hash = self.metadata.get("onnx_sha256")
        if verify_hash and expected_hash and self.model_sha256.casefold() != str(expected_hash).casefold():
            raise ValueError("reference matcher ONNX hash differs from metadata")

        requested = str(provider).casefold()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("provider must be one of: auto, cpu, cuda")
        available = set(ort.get_available_providers())
        cuda_ready = "CUDAExecutionProvider" in available
        if requested == "auto":
            requested = "cuda" if cuda_ready else "cpu"
        if requested == "cuda" and not cuda_ready:
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        if requested == "cpu" and "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is unavailable")
        self.provider = requested
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if requested == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if tuple(item.name for item in inputs) != (
            "query",
            "references",
            "reference_mask",
        ):
            raise RuntimeError("reference matcher ONNX input names are invalid")
        if tuple(item.name for item in outputs) != ("score",):
            raise RuntimeError("reference matcher ONNX output names are invalid")
        query_shape = tuple(inputs[0].shape)
        reference_shape = tuple(inputs[1].shape)
        mask_shape = tuple(inputs[2].shape)
        if len(query_shape) != 2 or len(reference_shape) != 3 or len(mask_shape) != 2:
            raise RuntimeError("reference matcher ONNX tensor ranks are invalid")
        if not isinstance(query_shape[1], int) or not isinstance(reference_shape[1], int):
            raise RuntimeError("reference matcher descriptor dimensions must be fixed")
        if reference_shape[2] != query_shape[1] or mask_shape[1] != reference_shape[1]:
            raise RuntimeError("reference matcher ONNX dimensions are inconsistent")
        self.descriptor_dim = int(query_shape[1])
        self.max_references = int(reference_shape[1])
        if inputs[2].type != "tensor(bool)":
            raise RuntimeError("reference_mask must be a bool tensor")

    def backend_info(self) -> dict[str, Any]:
        return {
            "type": "reference-set-matcher-onnx",
            "provider": self.provider,
            "model_sha256": self.model_sha256,
            "descriptor_dim": self.descriptor_dim,
            "max_references": self.max_references,
            "metadata": str(self.metadata_path) if self.metadata_path else None,
            "encoder_fingerprint": self.metadata.get("encoder_fingerprint"),
        }

    def _prepare(
        self,
        query: np.ndarray,
        reference_sets: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not reference_sets:
            raise ValueError("at least one reference set is required")
        query_unit = _unit_vector(
            np.asarray(query), name="query", width=self.descriptor_dim
        )
        padded = np.zeros(
            (len(reference_sets), self.max_references, self.descriptor_dim),
            dtype=np.float32,
        )
        mask = np.zeros((len(reference_sets), self.max_references), dtype=np.bool_)
        for index, references in enumerate(reference_sets):
            rows = np.asarray(references, dtype=np.float32)
            if rows.ndim != 2 or rows.shape[1] != self.descriptor_dim:
                raise ValueError(
                    f"reference_sets[{index}] must have shape [K,{self.descriptor_dim}]"
                )
            if rows.shape[0] < 1 or rows.shape[0] > self.max_references:
                raise ValueError(
                    f"reference_sets[{index}] must contain 1-{self.max_references} rows"
                )
            if not np.isfinite(rows).all():
                raise ValueError(f"reference_sets[{index}] must be finite")
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            if np.any(norms <= 0.0) or not np.isfinite(norms).all():
                raise ValueError(f"reference_sets[{index}] contains an invalid norm")
            rows = rows / np.maximum(norms, 1e-12)
            padded[index, : rows.shape[0]] = rows
            mask[index, : rows.shape[0]] = True
        queries = np.repeat(query_unit[None, :], len(reference_sets), axis=0)
        return queries, padded, mask

    def _score_bounded_many(
        self,
        query: np.ndarray,
        reference_sets: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Score sets whose widths fit in one fixed-width ONNX invocation."""

        queries, padded, mask = self._prepare(query, reference_sets)
        scores = self.session.run(
            ["score"],
            {
                "query": queries,
                "references": padded,
                "reference_mask": mask,
            },
        )[0]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(reference_sets) or not np.isfinite(scores).all():
            raise RuntimeError("reference matcher ONNX returned invalid scores")
        details = [
            {
                "mode": "learned_reference_set",
                "score": float(score),
                "reference_count": int(mask[index].sum()),
            }
            for index, score in enumerate(scores)
        ]
        return scores, details

    def score_many(
        self,
        query: np.ndarray,
        reference_sets: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Score one query against several sets, chunking oversized identities."""

        if not reference_sets:
            raise ValueError("at least one reference set is required")
        chunks: list[np.ndarray] = []
        owners: list[int] = []
        for owner, references in enumerate(reference_sets):
            rows = np.asarray(references)
            if rows.ndim != 2 or rows.shape[0] < 1:
                chunks.append(rows)
                owners.append(owner)
                continue
            for start in range(0, rows.shape[0], self.max_references):
                chunks.append(rows[start : start + self.max_references])
                owners.append(owner)
        chunk_scores, chunk_details = self._score_bounded_many(query, chunks)
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

    def score(
        self,
        query: np.ndarray,
        references: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        scores, details = self.score_many(query, [references])
        return float(scores[0]), details[0]

    def score_gallery(
        self,
        query: np.ndarray,
        prototypes: Sequence[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        reference_sets = []
        identities = []
        for item in prototypes:
            if "reference_features" not in item:
                raise ValueError("reference_features are required for learned scoring")
            reference_sets.append(np.asarray(item["reference_features"]))
            identities.append(str(item["pet_id"]))
        scores, rows = self.score_many(query, reference_sets)
        return scores, {identity: row for identity, row in zip(identities, rows)}


__all__ = ["ReferenceSetONNXRuntime"]
