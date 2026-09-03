"""ONNX Runtime adapter for the raw image-set scoring graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReferenceAwareONNXRuntime:
    """Run a fixed-width raw RGB query/reference graph with strict validation."""

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
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is required for the image-set graph"
            ) from error
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.metadata_path = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        self.metadata: dict[str, Any] = {}
        if self.metadata_path.is_file():
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("reference-aware ONNX metadata must be an object")
            self.metadata = value
            expected = self.metadata.get("onnx_sha256")
            if verify_hash and expected:
                actual = sha256_file(self.model_path)
                if actual.casefold() != str(expected).casefold():
                    raise ValueError(
                        "reference-aware ONNX hash mismatch: "
                        f"expected {expected}, got {actual}"
                    )
        requested = str(provider).casefold()
        if requested not in {"cpu", "cuda"}:
            raise ValueError("provider must be 'cpu' or 'cuda'")
        expected_provider = (
            "CUDAExecutionProvider" if requested == "cuda" else "CPUExecutionProvider"
        )
        available = ort.get_available_providers()
        if expected_provider not in available:
            raise RuntimeError(f"{expected_provider} is unavailable")
        self.session = ort.InferenceSession(
            str(self.model_path), providers=[expected_provider]
        )
        active = self.session.get_providers()
        if not active or active[0] != expected_provider:
            raise RuntimeError(
                f"requested {expected_provider}, activated {active}; refusing fallback"
            )
        self.provider = requested
        self.model_sha256 = sha256_file(self.model_path)
        self._validate_contract()

    def _validate_contract(self) -> None:
        names = [item.name for item in self.session.get_inputs()]
        outputs = [item.name for item in self.session.get_outputs()]
        expected_inputs = ["query_rgb", "reference_rgb", "reference_mask"]
        if names != expected_inputs or outputs[:1] != ["score"]:
            raise ValueError(
                "unexpected reference-aware ONNX contract: "
                f"inputs={names}, outputs={outputs}"
            )
        contract = self.metadata.get("input_contract", {})
        references = (
            contract.get("reference_rgb") if isinstance(contract, dict) else None
        )
        if isinstance(references, list) and len(references) >= 2:
            self.max_references = int(references[1])
        else:
            shape = self.session.get_inputs()[1].shape
            self.max_references = int(shape[1]) if isinstance(shape[1], int) else 0
        if self.max_references < 1:
            raise ValueError("reference-aware ONNX graph has no fixed reference width")

    @staticmethod
    def _pixels(value: Any, *, name: str, ndim: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != ndim or not np.isfinite(array).all():
            raise ValueError(
                f"{name} must be a finite float32 tensor with {ndim} dimensions"
            )
        if np.any(array < 0.0) or np.any(array > 255.0):
            raise ValueError(f"{name} pixels must be in the inclusive range 0..255")
        return np.ascontiguousarray(array)

    def predict(
        self,
        query_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        reference_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        query = self._pixels(query_rgb, name="query_rgb", ndim=4)
        references = self._pixels(reference_rgb, name="reference_rgb", ndim=5)
        if references.shape[0] != query.shape[0]:
            raise ValueError("query/reference batch dimensions must match")
        if references.shape[1] != self.max_references:
            raise ValueError(
                f"reference_rgb must have exactly {self.max_references} rows"
            )
        if references.shape[2:] != query.shape[1:]:
            raise ValueError("query and reference image shapes must match")
        if reference_mask is None:
            mask = np.ones((query.shape[0], self.max_references), dtype=np.bool_)
        else:
            mask = np.asarray(reference_mask)
            if mask.shape != (query.shape[0], self.max_references):
                raise ValueError("reference_mask has an unexpected shape")
            if not np.isfinite(mask.astype(np.float32)).all():
                raise ValueError("reference_mask must be finite")
            mask = np.ascontiguousarray(mask != 0, dtype=np.bool_)
        if not mask.any(axis=1).all():
            raise ValueError("each query must have at least one reference")
        scores = self.session.run(
            ["score"],
            {
                "query_rgb": query,
                "reference_rgb": references,
                "reference_mask": mask,
            },
        )[0]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != query.shape[0] or not np.isfinite(scores).all():
            raise RuntimeError("reference-aware ONNX graph returned invalid scores")
        return scores

    def backend_info(self) -> dict[str, Any]:
        return {
            "type": "reference-aware-pet-reid-onnx",
            "provider": self.provider,
            "model_sha256": self.model_sha256,
            "max_references": self.max_references,
            "model_config": self.metadata.get("model_config"),
        }


__all__ = ["ReferenceAwareONNXRuntime", "sha256_file"]
