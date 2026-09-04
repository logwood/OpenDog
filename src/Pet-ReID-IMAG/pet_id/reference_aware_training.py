"""Image-episode sampling and losses for the reference-aware model.

Unlike the descriptor-cache experiment, these helpers materialize RGB images
and run the shared encoder before the set matcher.  The resulting gradients
therefore reach the image model when it is unfrozen.  Reference sets are
sampled without using the query image itself, which prevents an easy leakage
path during training and validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .reference_aware_model import ReferenceAwarePetReID
from .reference_token_model import catalog_confidence_gate_from_scores


REFERENCE_IMAGE_MANIFEST_FIELDS = (
    "identity",
    "source_path",
    "resized_size",
    "face_roi_xyxy",
    "nose_roi_xyxy",
    "roll_angle_radians",
)

# Geometry manifests expose continuous viewpoint and quality observations. They
# are weak supervision, not hard front/side/back labels. Raw high-resolution
# manifests may omit them; the structural loss then safely becomes a no-op.
VIEW_FEATURE_KEY = "viewpoint_signals"
QUALITY_FEATURE_KEY = "quality_signals"
VIEW_FEATURE_DIM = 4
QUALITY_FEATURE_DIM = 6
SPATIAL_FEATURE_CACHE_FORMAT = "reference-spatial-feature-cache"
REFERENCE_SELECTION_TOLERANCES = (1.0e-12, 1.0e-6, 1.0e-6)


def validate_reference_image_manifest(
    manifest_path: str | Path,
    *,
    reject_blind: bool = True,
) -> dict[str, Any]:
    """Validate the fixed-size image episode manifest before model loading."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    split = str(payload.get("protocol_split", "")).casefold()
    if reject_blind and ("blind" in split or "blind" in path.stem.casefold()):
        raise ValueError(f"blind manifests are not allowed here: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"manifest has no image records: {path}")
    missing = sorted(
        {
            field
            for record in records
            if not isinstance(record, dict)
            for field in ("<record-object>",)
        }
        | {
            field
            for record in records
            if isinstance(record, dict)
            for field in REFERENCE_IMAGE_MANIFEST_FIELDS
            if field not in record
        }
    )
    if missing:
        raise ValueError(
            f"manifest is not a fixed-size reference image manifest; "
            f"missing fields: {missing}: {path}"
        )
    return payload


@dataclass(frozen=True)
class ReferenceImageEpisode:
    """Indices for one P-way episode."""

    identity_names: tuple[str, ...]
    query_indices: tuple[int, ...]
    reference_indices: tuple[tuple[int, ...], ...]
    targets: torch.Tensor


@dataclass(frozen=True)
class ReferenceImageBatch:
    """Unique query/reference images used to construct episode scores."""

    query_images: torch.Tensor
    reference_images: torch.Tensor
    reference_mask: torch.Tensor
    targets: torch.Tensor
    identity_names: tuple[str, ...]
    query_view_features: torch.Tensor | None = None
    reference_view_features: torch.Tensor | None = None
    query_view_valid: torch.Tensor | None = None
    reference_view_valid: torch.Tensor | None = None
    query_quality_features: torch.Tensor | None = None
    reference_quality_features: torch.Tensor | None = None
    query_quality_valid: torch.Tensor | None = None
    reference_quality_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class CachedReferenceFeatureBatch:
    """Cached encoder outputs used to construct an all-identity score matrix."""

    query_descriptors: torch.Tensor
    query_pooled_spatial_features: torch.Tensor
    reference_descriptors: torch.Tensor
    reference_pooled_spatial_features: torch.Tensor
    reference_mask: torch.Tensor
    targets: torch.Tensor
    identity_names: tuple[str, ...]
    query_view_features: torch.Tensor | None = None
    reference_view_features: torch.Tensor | None = None
    query_view_valid: torch.Tensor | None = None
    reference_view_valid: torch.Tensor | None = None
    query_quality_features: torch.Tensor | None = None
    reference_quality_features: torch.Tensor | None = None
    query_quality_valid: torch.Tensor | None = None
    reference_quality_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class ReferenceSpatialFeatureCache:
    """Persistent frozen-encoder features indexed in manifest record order."""

    descriptors: torch.Tensor
    pooled_spatial_features: torch.Tensor
    source_sha256s: tuple[str, ...]
    manifest_sha256: str
    base_checkpoint_sha256: str
    feature_hook: str
    token_grid: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.descriptors.dtype != torch.float32:
            raise ValueError("cached descriptors must use float32")
        if self.pooled_spatial_features.dtype != torch.float32:
            raise ValueError("cached pooled spatial features must use float32")
        if self.descriptors.ndim != 2 or self.descriptors.shape[1] < 1:
            raise ValueError("cached descriptors must have shape [records, dimension]")
        if (
            self.pooled_spatial_features.ndim != 3
            or self.pooled_spatial_features.shape[1] != int(self.token_grid) ** 2
            or self.pooled_spatial_features.shape[2] < 1
        ):
            raise ValueError(
                "cached pooled spatial features must have shape "
                "[records, token_grid^2, channels]"
            )
        rows = int(self.descriptors.shape[0])
        if rows < 1 or int(self.pooled_spatial_features.shape[0]) != rows:
            raise ValueError("cached descriptor and spatial feature rows differ")
        if len(self.source_sha256s) != rows:
            raise ValueError("cached source_sha256 order does not match tensor rows")
        if int(self.token_grid) < 1:
            raise ValueError("cached token_grid must be positive")
        if not self.feature_hook:
            raise ValueError("cached feature hook name must be non-empty")
        for name, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("base_checkpoint_sha256", self.base_checkpoint_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"cached {name} must be a lowercase SHA-256 digest")
        if any(not str(value) for value in self.source_sha256s):
            raise ValueError("cached source_sha256 values must be non-empty")
        if not bool(torch.isfinite(self.descriptors).all()):
            raise ValueError("cached descriptors contain non-finite values")
        if not bool(torch.isfinite(self.pooled_spatial_features).all()):
            raise ValueError("cached pooled spatial features contain non-finite values")
        norms = torch.linalg.vector_norm(self.descriptors, dim=1)
        if not bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1.0e-4, rtol=1.0e-4)
        ):
            raise ValueError("cached descriptors must be unit normalized")

    def to(self, device: str | torch.device) -> "ReferenceSpatialFeatureCache":
        target = torch.device(device)
        return ReferenceSpatialFeatureCache(
            descriptors=self.descriptors.to(target),
            pooled_spatial_features=self.pooled_spatial_features.to(target),
            source_sha256s=self.source_sha256s,
            manifest_sha256=self.manifest_sha256,
            base_checkpoint_sha256=self.base_checkpoint_sha256,
            feature_hook=self.feature_hook,
            token_grid=self.token_grid,
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_source_sha256s(dataset: Any) -> tuple[str, ...]:
    records = getattr(dataset, "records", None)
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must expose non-empty manifest records")
    values = tuple(
        str(record.get("source_sha256", "")) if isinstance(record, dict) else ""
        for record in records
    )
    if any(not value for value in values):
        raise ValueError("every cached manifest record must contain source_sha256")
    return values


def _identity_groups(dataset: Any) -> dict[str, np.ndarray]:
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("dataset must expose manifest records")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "identity" not in record:
            raise ValueError("dataset records must contain identity fields")
        identity = str(record["identity"]).casefold()
        groups.setdefault(identity, []).append(index)
    if not groups:
        raise ValueError("dataset has no identities")
    return {
        identity: np.asarray(indices, dtype=np.int64)
        for identity, indices in groups.items()
    }


def save_reference_spatial_feature_cache(
    cache: ReferenceSpatialFeatureCache,
    path: str | Path,
) -> Path:
    """Persist cache tensors and provenance without serializing Python objects."""

    cache.validate()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SPATIAL_FEATURE_CACHE_FORMAT,
        "descriptors": cache.descriptors.detach().cpu().contiguous(),
        "pooled_spatial_features": (
            cache.pooled_spatial_features.detach().cpu().contiguous()
        ),
        "source_sha256s": list(cache.source_sha256s),
        "manifest_sha256": cache.manifest_sha256,
        "base_checkpoint_sha256": cache.base_checkpoint_sha256,
        "feature_hook": cache.feature_hook,
        "token_grid": int(cache.token_grid),
    }
    partial = destination.with_name(destination.name + ".partial")
    torch.save(payload, partial)
    partial.replace(destination)
    return destination


def load_reference_spatial_feature_cache(
    path: str | Path,
    *,
    dataset: Any,
    manifest_path: str | Path,
    base_checkpoint_sha256: str,
    feature_hook: str,
    token_grid: int,
    descriptor_dim: int,
) -> ReferenceSpatialFeatureCache:
    """Load a cache only when every data/model provenance field still matches."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != SPATIAL_FEATURE_CACHE_FORMAT
    ):
        raise ValueError(f"not a reference spatial feature cache: {source}")
    required = (
        "descriptors",
        "pooled_spatial_features",
        "source_sha256s",
        "manifest_sha256",
        "base_checkpoint_sha256",
        "feature_hook",
        "token_grid",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"spatial feature cache is missing fields {missing}: {source}")
    if not isinstance(payload["descriptors"], torch.Tensor) or not isinstance(
        payload["pooled_spatial_features"], torch.Tensor
    ):
        raise ValueError("spatial feature cache tensors have invalid types")
    source_values = payload["source_sha256s"]
    if not isinstance(source_values, (tuple, list)):
        raise ValueError("cached source_sha256s must be a sequence")
    cache = ReferenceSpatialFeatureCache(
        descriptors=payload["descriptors"],
        pooled_spatial_features=payload["pooled_spatial_features"],
        source_sha256s=tuple(str(value) for value in source_values),
        manifest_sha256=str(payload["manifest_sha256"]),
        base_checkpoint_sha256=str(payload["base_checkpoint_sha256"]),
        feature_hook=str(payload["feature_hook"]),
        token_grid=int(payload["token_grid"]),
    )
    expected_manifest = _sha256_file(Path(manifest_path).expanduser().resolve())
    expected_sources = _record_source_sha256s(dataset)
    mismatches: list[str] = []
    if cache.manifest_sha256 != expected_manifest:
        mismatches.append("manifest SHA-256")
    if cache.base_checkpoint_sha256 != str(base_checkpoint_sha256):
        mismatches.append("base checkpoint SHA-256")
    if cache.source_sha256s != expected_sources:
        mismatches.append("manifest record source_sha256 order")
    if cache.feature_hook != str(feature_hook):
        mismatches.append("spatial feature hook")
    if cache.token_grid != int(token_grid):
        mismatches.append("token grid")
    if int(cache.descriptors.shape[1]) != int(descriptor_dim):
        mismatches.append("descriptor dimension")
    if mismatches:
        raise ValueError(
            "spatial feature cache provenance mismatch: " + ", ".join(mismatches)
        )
    return cache


def build_reference_spatial_feature_cache(
    model: Any,
    dataset: Any,
    *,
    manifest_path: str | Path,
    base_checkpoint_sha256: str,
    device: str | torch.device,
    batch_size: int = 4,
) -> ReferenceSpatialFeatureCache:
    """Encode every manifest record once through a frozen spatial encoder."""

    if int(batch_size) < 1:
        raise ValueError("feature cache batch_size must be positive")
    if bool(getattr(dataset, "training", False)):
        raise ValueError("feature caching requires augmentation-free dataset loading")
    adapter = getattr(model, "image_encoder", None)
    encoder = getattr(adapter, "encoder", None)
    if encoder is None:
        raise TypeError("token model must expose its wrapped image encoder")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("feature caching requires a fully frozen base encoder")
    feature_hook = getattr(adapter, "feature_hook_name", None)
    if not isinstance(feature_hook, str) or not feature_hook.casefold().endswith(
        "layer4"
    ):
        raise RuntimeError(
            "feature caching requires a real layer4 spatial feature hook"
        )
    encode = getattr(model, "encode_cacheable_image_features", None)
    if not callable(encode):
        raise TypeError("token model does not expose cacheable spatial features")
    source_sha256s = _record_source_sha256s(dataset)
    target_device = torch.device(device)
    descriptor_chunks: list[torch.Tensor] = []
    spatial_chunks: list[torch.Tensor] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(source_sha256s), int(batch_size)):
                stop = min(start + int(batch_size), len(source_sha256s))
                images = torch.stack(
                    [_sample_rgb(dataset, index) for index in range(start, stop)]
                ).to(target_device)
                descriptors, pooled = encode(images)
                if (
                    descriptors.shape[0] != stop - start
                    or pooled.shape[0] != stop - start
                ):
                    raise ValueError(
                        "encoder changed the feature cache batch dimension"
                    )
                descriptor_chunks.append(
                    descriptors.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                spatial_chunks.append(
                    pooled.detach().to(device="cpu", dtype=torch.float32).contiguous()
                )
    finally:
        model.train(was_training)
    return ReferenceSpatialFeatureCache(
        descriptors=torch.cat(descriptor_chunks, dim=0),
        pooled_spatial_features=torch.cat(spatial_chunks, dim=0),
        source_sha256s=source_sha256s,
        manifest_sha256=_sha256_file(Path(manifest_path).expanduser().resolve()),
        base_checkpoint_sha256=str(base_checkpoint_sha256),
        feature_hook=feature_hook,
        token_grid=int(adapter.token_grid),
    )


class ReferenceImageEpisodeSampler:
    """Sample P identities and disjoint reference/query images."""

    def __init__(
        self,
        dataset: Any,
        *,
        identities_per_batch: int = 8,
        reference_count: int = 2,
        queries_per_identity: int = 1,
        max_references: int = 4,
        variable_reference_count: bool = True,
        seed: int = 20260903,
    ) -> None:
        self.dataset = dataset
        self.groups = _identity_groups(dataset)
        self.identity_names = tuple(sorted(self.groups))
        self.identities_per_batch = int(identities_per_batch)
        self.reference_count = int(reference_count)
        self.queries_per_identity = int(queries_per_identity)
        self.max_references = int(max_references)
        self.variable_reference_count = bool(variable_reference_count)
        self.seed = int(seed)
        if (
            min(
                self.identities_per_batch,
                self.reference_count,
                self.queries_per_identity,
                self.max_references,
            )
            <= 0
        ):
            raise ValueError("episode sizes must be positive")
        if self.reference_count > self.max_references:
            raise ValueError("reference_count cannot exceed max_references")
        if len(self.identity_names) < self.identities_per_batch:
            raise ValueError("dataset has fewer identities than identities_per_batch")
        minimum = self.reference_count + self.queries_per_identity
        insufficient = {
            identity: int(rows.size)
            for identity, rows in self.groups.items()
            if rows.size < minimum
        }
        if insufficient:
            raise ValueError(
                f"each identity needs at least {minimum} images: {insufficient}"
            )

    def sample(self, *, epoch: int = 0, step: int = 0) -> ReferenceImageEpisode:
        # Distinct large strides keep adjacent episodes independent while
        # remaining deterministic across workers and resumed runs.
        seed = self.seed + int(epoch) * 1_000_003 + int(step) * 9_176
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            np.asarray(self.identity_names, dtype=object),
            size=self.identities_per_batch,
            replace=False,
        )
        names = tuple(str(value) for value in chosen.tolist())
        query_indices: list[int] = []
        reference_indices: list[tuple[int, ...]] = []
        for identity in names:
            rows = self.groups[identity]
            count = (
                int(rng.integers(1, self.reference_count + 1))
                if self.variable_reference_count
                else self.reference_count
            )
            selected = rng.choice(
                rows,
                size=count + self.queries_per_identity,
                replace=False,
            )
            references = tuple(int(value) for value in selected[:count].tolist())
            queries = selected[count:]
            reference_indices.append(references)
            query_indices.extend(int(value) for value in queries.tolist())
        targets = torch.arange(
            self.identities_per_batch, dtype=torch.long
        ).repeat_interleave(self.queries_per_identity)
        return ReferenceImageEpisode(
            identity_names=names,
            query_indices=tuple(query_indices),
            reference_indices=tuple(reference_indices),
            targets=targets,
        )


class AllIdentityReferenceEpisodeSampler(ReferenceImageEpisodeSampler):
    """Sample a few queries while retaining every train identity as a candidate."""

    def sample(self, *, epoch: int = 0, step: int = 0) -> ReferenceImageEpisode:
        seed = self.seed + int(epoch) * 1_000_003 + int(step) * 9_176
        rng = np.random.default_rng(seed)
        selected_query_identities = rng.choice(
            np.asarray(self.identity_names, dtype=object),
            size=self.identities_per_batch,
            replace=False,
        )
        query_names = tuple(str(value) for value in selected_query_identities.tolist())
        query_rows: dict[str, tuple[int, ...]] = {}
        for identity in query_names:
            selected = rng.choice(
                self.groups[identity],
                size=self.queries_per_identity,
                replace=False,
            )
            query_rows[identity] = tuple(int(value) for value in selected.tolist())

        reference_indices: list[tuple[int, ...]] = []
        for identity in self.identity_names:
            count = (
                int(rng.integers(1, self.reference_count + 1))
                if self.variable_reference_count
                else self.reference_count
            )
            rows = self.groups[identity]
            excluded = set(query_rows.get(identity, ()))
            available = np.asarray(
                [int(value) for value in rows.tolist() if int(value) not in excluded],
                dtype=np.int64,
            )
            if available.size < count:
                raise ValueError(
                    f"identity {identity!r} has fewer than {count} non-query references"
                )
            selected = rng.choice(available, size=count, replace=False)
            reference_indices.append(tuple(int(value) for value in selected.tolist()))

        positions = {
            identity: index for index, identity in enumerate(self.identity_names)
        }
        targets = torch.tensor(
            [
                positions[identity]
                for identity in query_names
                for _ in range(self.queries_per_identity)
            ],
            dtype=torch.long,
        )
        query_indices = tuple(
            index for identity in query_names for index in query_rows[identity]
        )
        return ReferenceImageEpisode(
            identity_names=self.identity_names,
            query_indices=query_indices,
            reference_indices=tuple(reference_indices),
            targets=targets,
        )


def _coerce_sample_feature(
    value: Any,
    *,
    dimension: int,
) -> tuple[torch.Tensor, bool]:
    """Convert an optional manifest feature to a finite tensor and validity bit."""

    if value is None:
        return torch.zeros(dimension, dtype=torch.float32), False
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return torch.zeros(dimension, dtype=torch.float32), False
    if array.size != int(dimension) or not bool(np.isfinite(array).all()):
        return torch.zeros(dimension, dtype=torch.float32), False
    return torch.from_numpy(array.copy()), True


def _sample_with_metadata(
    dataset: Any,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor, bool]:
    sample = dataset[int(index)]
    if not isinstance(sample, dict) or not isinstance(sample.get("rgb"), torch.Tensor):
        raise ValueError("dataset samples must contain an rgb tensor")
    image = sample["rgb"].float()
    if image.ndim != 3 or image.shape[0] < 1:
        raise ValueError(
            "dataset rgb tensors must have shape [channels, height, width]"
        )
    if not bool(torch.isfinite(image).all()):
        raise ValueError("dataset rgb tensor contains non-finite values")
    record = sample.get("record")
    if not isinstance(record, dict):
        record = {}
    view_value = sample.get(VIEW_FEATURE_KEY, record.get(VIEW_FEATURE_KEY))
    quality_value = sample.get(QUALITY_FEATURE_KEY, record.get(QUALITY_FEATURE_KEY))
    view, view_valid = _coerce_sample_feature(
        view_value,
        dimension=VIEW_FEATURE_DIM,
    )
    quality, quality_valid = _coerce_sample_feature(
        quality_value,
        dimension=QUALITY_FEATURE_DIM,
    )
    return image, view, view_valid, quality, quality_valid


def _sample_rgb(dataset: Any, index: int) -> torch.Tensor:
    return _sample_with_metadata(dataset, index)[0]


def _record_with_metadata(
    dataset: Any,
    index: int,
) -> tuple[torch.Tensor, bool, torch.Tensor, bool]:
    """Read weak supervision directly from a manifest record without RGB I/O."""

    records = getattr(dataset, "records", None)
    if not isinstance(records, list) or not 0 <= int(index) < len(records):
        raise ValueError("cached feature index is outside dataset manifest records")
    record = records[int(index)]
    if not isinstance(record, dict):
        raise ValueError("dataset manifest records must be objects")
    view, view_valid = _coerce_sample_feature(
        record.get(VIEW_FEATURE_KEY),
        dimension=VIEW_FEATURE_DIM,
    )
    quality, quality_valid = _coerce_sample_feature(
        record.get(QUALITY_FEATURE_KEY),
        dimension=QUALITY_FEATURE_DIM,
    )
    return view, view_valid, quality, quality_valid


def materialize_reference_image_episode(
    dataset: Any,
    episode: ReferenceImageEpisode,
    *,
    device: str | torch.device | None = None,
) -> ReferenceImageBatch:
    """Load one sampled episode into padded tensors."""

    query_samples = [
        _sample_with_metadata(dataset, index) for index in episode.query_indices
    ]
    query_images = torch.stack([sample[0] for sample in query_samples])
    if not episode.reference_indices:
        raise ValueError("episode must contain at least one identity")
    max_count = max(len(rows) for rows in episode.reference_indices)
    if max_count < 1:
        raise ValueError("each identity must have at least one reference")
    reference_rows = [
        _sample_with_metadata(dataset, index)
        for rows in episode.reference_indices
        for index in rows
    ]
    first = reference_rows[0][0]
    if any(tuple(sample[0].shape) != tuple(first.shape) for sample in reference_rows):
        raise ValueError("all episode images must have the same tensor shape")
    channels, height, width = first.shape
    references = torch.zeros(
        (len(episode.reference_indices), max_count, channels, height, width),
        dtype=first.dtype,
    )
    reference_views = torch.zeros(
        (
            len(episode.reference_indices),
            max_count,
            VIEW_FEATURE_DIM,
        ),
        dtype=torch.float32,
    )
    reference_view_valid = torch.zeros(
        (len(episode.reference_indices), max_count),
        dtype=torch.bool,
    )
    reference_quality = torch.zeros(
        (
            len(episode.reference_indices),
            max_count,
            QUALITY_FEATURE_DIM,
        ),
        dtype=torch.float32,
    )
    reference_quality_valid = torch.zeros(
        (len(episode.reference_indices), max_count),
        dtype=torch.bool,
    )
    mask = torch.zeros((len(episode.reference_indices), max_count), dtype=torch.bool)
    cursor = 0
    for identity_index, rows in enumerate(episode.reference_indices):
        for reference_index in range(len(rows)):
            image, view, view_valid, quality, quality_valid = reference_rows[cursor]
            references[identity_index, reference_index] = image
            reference_views[identity_index, reference_index] = view
            reference_view_valid[identity_index, reference_index] = view_valid
            reference_quality[identity_index, reference_index] = quality
            reference_quality_valid[identity_index, reference_index] = quality_valid
            mask[identity_index, reference_index] = True
            cursor += 1
    query_views = torch.stack([sample[1] for sample in query_samples])
    query_view_valid = torch.tensor(
        [sample[2] for sample in query_samples],
        dtype=torch.bool,
    )
    query_quality = torch.stack([sample[3] for sample in query_samples])
    query_quality_valid = torch.tensor(
        [sample[4] for sample in query_samples],
        dtype=torch.bool,
    )
    if not bool(query_view_valid.any()) and not bool(reference_view_valid.any()):
        query_views = None
        reference_views = None
        query_view_valid = None
        reference_view_valid = None
    if not bool(query_quality_valid.any()) and not bool(reference_quality_valid.any()):
        query_quality = None
        reference_quality = None
        query_quality_valid = None
        reference_quality_valid = None
    if device is not None:
        target_device = torch.device(device)
        query_images = query_images.to(target_device)
        references = references.to(target_device)
        mask = mask.to(target_device)
        targets = episode.targets.to(target_device)
        if query_views is not None:
            query_views = query_views.to(target_device)
            reference_views = reference_views.to(target_device)
            query_view_valid = query_view_valid.to(target_device)
            reference_view_valid = reference_view_valid.to(target_device)
        if query_quality is not None:
            query_quality = query_quality.to(target_device)
            reference_quality = reference_quality.to(target_device)
            query_quality_valid = query_quality_valid.to(target_device)
            reference_quality_valid = reference_quality_valid.to(target_device)
    else:
        targets = episode.targets
    return ReferenceImageBatch(
        query_images=query_images,
        reference_images=references,
        reference_mask=mask,
        targets=targets,
        identity_names=episode.identity_names,
        query_view_features=query_views,
        reference_view_features=reference_views,
        query_view_valid=query_view_valid,
        reference_view_valid=reference_view_valid,
        query_quality_features=query_quality,
        reference_quality_features=reference_quality,
        query_quality_valid=query_quality_valid,
        reference_quality_valid=reference_quality_valid,
    )


def materialize_cached_reference_episode(
    cache: ReferenceSpatialFeatureCache,
    dataset: Any,
    episode: ReferenceImageEpisode,
    *,
    device: str | torch.device | None = None,
) -> CachedReferenceFeatureBatch:
    """Gather a cached episode without decoding or re-encoding any RGB image."""

    if cache.descriptors.shape[0] != cache.pooled_spatial_features.shape[0]:
        raise ValueError("feature cache tensor rows differ")
    if cache.source_sha256s != _record_source_sha256s(dataset):
        raise ValueError("feature cache rows no longer match manifest record order")
    identity_count = len(episode.identity_names)
    if identity_count < 1 or len(episode.reference_indices) != identity_count:
        raise ValueError("episode reference rows must match its identity names")
    if not episode.query_indices:
        raise ValueError("episode must contain at least one query")
    targets = episode.targets.to(dtype=torch.long)
    if targets.ndim != 1 or targets.shape[0] != len(episode.query_indices):
        raise ValueError("episode targets must have one entry per query")
    if bool((targets < 0).any()) or bool((targets >= identity_count).any()):
        raise ValueError("episode targets contain an identity outside candidates")

    records = dataset.records
    for query_index, target in zip(episode.query_indices, targets.tolist()):
        query_identity = str(records[int(query_index)]["identity"]).casefold()
        if query_identity != episode.identity_names[int(target)]:
            raise ValueError("query identity does not match its episode target")
        if int(query_index) in episode.reference_indices[int(target)]:
            raise ValueError("positive query image cannot appear in its reference set")
    for identity, indices in zip(episode.identity_names, episode.reference_indices):
        if not indices:
            raise ValueError("each candidate identity must have at least one reference")
        for index in indices:
            record_identity = str(records[int(index)]["identity"]).casefold()
            if record_identity != identity:
                raise ValueError("reference index is assigned to the wrong identity")

    max_count = max(len(indices) for indices in episode.reference_indices)
    cache_device = cache.descriptors.device
    query_index_tensor = torch.tensor(
        episode.query_indices,
        dtype=torch.long,
        device=cache_device,
    )
    query_descriptors = cache.descriptors.index_select(0, query_index_tensor)
    query_spatial = cache.pooled_spatial_features.index_select(0, query_index_tensor)
    padded_indices = torch.zeros(
        (identity_count, max_count),
        dtype=torch.long,
        device=cache_device,
    )
    reference_mask = torch.zeros(
        (identity_count, max_count),
        dtype=torch.bool,
        device=cache_device,
    )
    for identity_index, indices in enumerate(episode.reference_indices):
        count = len(indices)
        padded_indices[identity_index, :count] = torch.tensor(
            indices,
            dtype=torch.long,
            device=cache_device,
        )
        reference_mask[identity_index, :count] = True
    reference_descriptors = cache.descriptors[padded_indices]
    reference_spatial = cache.pooled_spatial_features[padded_indices]

    query_metadata = [
        _record_with_metadata(dataset, index) for index in episode.query_indices
    ]
    query_views = torch.stack([item[0] for item in query_metadata])
    query_view_valid = torch.tensor(
        [item[1] for item in query_metadata],
        dtype=torch.bool,
    )
    query_quality = torch.stack([item[2] for item in query_metadata])
    query_quality_valid = torch.tensor(
        [item[3] for item in query_metadata],
        dtype=torch.bool,
    )
    reference_views = torch.zeros(
        (identity_count, max_count, VIEW_FEATURE_DIM),
        dtype=torch.float32,
    )
    reference_view_valid = torch.zeros(
        (identity_count, max_count),
        dtype=torch.bool,
    )
    reference_quality = torch.zeros(
        (identity_count, max_count, QUALITY_FEATURE_DIM),
        dtype=torch.float32,
    )
    reference_quality_valid = torch.zeros(
        (identity_count, max_count),
        dtype=torch.bool,
    )
    for identity_index, indices in enumerate(episode.reference_indices):
        for reference_index, index in enumerate(indices):
            view, view_valid, quality, quality_valid = _record_with_metadata(
                dataset, index
            )
            reference_views[identity_index, reference_index] = view
            reference_view_valid[identity_index, reference_index] = view_valid
            reference_quality[identity_index, reference_index] = quality
            reference_quality_valid[identity_index, reference_index] = quality_valid

    if not bool(query_view_valid.any()) and not bool(reference_view_valid.any()):
        query_views = None
        reference_views = None
        query_view_valid = None
        reference_view_valid = None
    if not bool(query_quality_valid.any()) and not bool(reference_quality_valid.any()):
        query_quality = None
        reference_quality = None
        query_quality_valid = None
        reference_quality_valid = None

    target_device = cache_device if device is None else torch.device(device)
    query_descriptors = query_descriptors.to(target_device)
    query_spatial = query_spatial.to(target_device)
    reference_descriptors = reference_descriptors.to(target_device)
    reference_spatial = reference_spatial.to(target_device)
    reference_mask = reference_mask.to(target_device)
    targets = targets.to(target_device)
    if query_views is not None:
        query_views = query_views.to(target_device)
        reference_views = reference_views.to(target_device)
        query_view_valid = query_view_valid.to(target_device)
        reference_view_valid = reference_view_valid.to(target_device)
    if query_quality is not None:
        query_quality = query_quality.to(target_device)
        reference_quality = reference_quality.to(target_device)
        query_quality_valid = query_quality_valid.to(target_device)
        reference_quality_valid = reference_quality_valid.to(target_device)
    return CachedReferenceFeatureBatch(
        query_descriptors=query_descriptors,
        query_pooled_spatial_features=query_spatial,
        reference_descriptors=reference_descriptors,
        reference_pooled_spatial_features=reference_spatial,
        reference_mask=reference_mask,
        targets=targets,
        identity_names=episode.identity_names,
        query_view_features=query_views,
        reference_view_features=reference_views,
        query_view_valid=query_view_valid,
        reference_view_valid=reference_view_valid,
        query_quality_features=query_quality,
        reference_quality_features=reference_quality,
        query_quality_valid=query_quality_valid,
        reference_quality_valid=reference_quality_valid,
    )


def _score_encoded_reference_episode(
    model: Any,
    batch: ReferenceImageBatch | CachedReferenceFeatureBatch,
    query_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    *,
    query_tokens: torch.Tensor | None,
    reference_tokens: torch.Tensor | None,
    return_aux: bool,
) -> torch.Tensor | dict[str, torch.Tensor]:
    identity_count = len(batch.identity_names)
    if identity_count < 1:
        raise ValueError("episode must contain at least one identity")
    if reference_descriptors.shape[0] != identity_count:
        raise ValueError("reference feature rows do not match identity count")
    query_count = query_descriptors.shape[0]
    reference_count = reference_descriptors.shape[1]
    expanded_queries = query_descriptors.repeat_interleave(identity_count, dim=0)
    expanded_references = (
        reference_descriptors.unsqueeze(0)
        .expand(query_count, -1, -1, -1)
        .reshape(query_count * identity_count, reference_count, model.descriptor_dim)
    )
    expanded_mask = (
        batch.reference_mask.unsqueeze(0)
        .expand(query_count, -1, -1)
        .reshape(query_count * identity_count, reference_count)
    )
    forward_kwargs: dict[str, Any] = {"return_aux": return_aux}
    if query_tokens is not None and reference_tokens is not None:
        guard_queries = F.normalize(
            query_descriptors.detach().float(),
            dim=-1,
            eps=1.0e-12,
        )
        guard_references = F.normalize(
            reference_descriptors.detach().float(),
            dim=-1,
            eps=1.0e-12,
        )
        guard_mask = batch.reference_mask.to(
            device=guard_references.device,
            dtype=guard_references.dtype,
        )
        guard_centroids = F.normalize(
            (guard_references * guard_mask.unsqueeze(-1)).sum(dim=1),
            dim=-1,
            eps=1.0e-12,
        )
        baseline_catalog_scores = torch.einsum(
            "qd,id->qi",
            guard_queries,
            guard_centroids,
        )
        query_catalog_gate = catalog_confidence_gate_from_scores(
            baseline_catalog_scores
        )
        forward_kwargs["catalog_confidence_gate"] = (
            query_catalog_gate.repeat_interleave(identity_count, dim=0)
        )
        forward_kwargs["query_tokens"] = query_tokens.repeat_interleave(
            identity_count, dim=0
        )
        forward_kwargs["reference_tokens"] = (
            reference_tokens.unsqueeze(0)
            .expand(query_count, -1, -1, -1, -1)
            .reshape(
                query_count * identity_count,
                reference_count,
                reference_tokens.shape[2],
                reference_tokens.shape[3],
            )
        )
    output = model.forward_encoded(
        expanded_queries,
        expanded_references,
        expanded_mask,
        **forward_kwargs,
    )
    if not return_aux:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("model returned unexpected episode output")
        return output.reshape(query_count, identity_count)
    if not isinstance(output, dict):
        raise RuntimeError("model returned no episode diagnostics")
    reshaped: dict[str, torch.Tensor] = {}
    for key, value in output.items():
        if value.ndim > 0 and value.shape[0] == query_count * identity_count:
            reshaped[key] = value.reshape(query_count, identity_count, *value.shape[1:])
        else:
            reshaped[key] = value
    return reshaped


def score_reference_image_episode(
    model: Any,
    batch: ReferenceImageBatch,
    *,
    return_aux: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Score every query against every identity in a live image episode.

    References are encoded once per identity and then expanded across the
    candidate identities. Descriptor models expand only descriptors; token
    models also expand per-image token grids so cross-view interaction happens
    before descriptor pooling.
    """

    identity_count = len(batch.identity_names)
    if batch.reference_images.shape[0] != identity_count:
        raise ValueError("reference image rows do not match identity count")
    encode_features = getattr(model, "encode_image_features", None)
    if callable(encode_features):
        query_descriptors, query_tokens = encode_features(batch.query_images)
        reference_descriptors, reference_tokens = model.encode_reference_features(
            batch.reference_images
        )
    else:
        query_descriptors = model.encode_images(batch.query_images)
        reference_descriptors = model.encode_reference_images(batch.reference_images)
        query_tokens = None
        reference_tokens = None
    return _score_encoded_reference_episode(
        model,
        batch,
        query_descriptors,
        reference_descriptors,
        query_tokens=query_tokens,
        reference_tokens=reference_tokens,
        return_aux=return_aux,
    )


def score_cached_reference_episode(
    model: Any,
    batch: CachedReferenceFeatureBatch,
    *,
    return_aux: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Score a cached episode while retaining gradients through token heads."""

    project_tokens = getattr(model, "tokens_from_pooled_features", None)
    if not callable(project_tokens):
        raise TypeError("cached spatial features require a token reference model")
    if batch.query_descriptors.ndim != 2:
        raise ValueError("cached query descriptors must have two dimensions")
    if batch.reference_descriptors.ndim != 3:
        raise ValueError("cached reference descriptors must have three dimensions")
    if batch.query_pooled_spatial_features.ndim != 3:
        raise ValueError("cached query spatial features must have three dimensions")
    if batch.reference_pooled_spatial_features.ndim != 4:
        raise ValueError("cached reference spatial features must have four dimensions")
    query_tokens = project_tokens(batch.query_pooled_spatial_features)
    reference_shape = batch.reference_pooled_spatial_features.shape
    flat_spatial = batch.reference_pooled_spatial_features.reshape(
        reference_shape[0] * reference_shape[1],
        reference_shape[2],
        reference_shape[3],
    )
    reference_tokens = project_tokens(flat_spatial).reshape(
        reference_shape[0],
        reference_shape[1],
        reference_shape[2],
        model.token_dim,
    )
    return _score_encoded_reference_episode(
        model,
        batch,
        batch.query_descriptors,
        batch.reference_descriptors,
        query_tokens=query_tokens,
        reference_tokens=reference_tokens,
        return_aux=return_aux,
    )


def _zero_like_scores(scores: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero on the score device/dtype."""

    return scores.float().sum() * 0.0


def hard_negative_margin_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize the strongest non-target identity in each episode row.

    The helper returns ``(loss, observed_margin)``.  A one-identity episode has
    no negative and therefore contributes a differentiable zero rather than a
    fabricated value.  This makes the objective safe for tiny smoke episodes
    while adding a real hard-negative signal whenever the episode has at least
    two identities.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, identities]")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("targets must have one entry per score row")
    if not math.isfinite(float(margin)) or margin <= 0.0:
        raise ValueError("margin must be positive and finite")
    if scores.shape[1] <= 1:
        return _zero_like_scores(scores), _zero_like_scores(scores)
    targets = targets.to(device=scores.device, dtype=torch.long)
    if bool((targets < 0).any()) or bool((targets >= scores.shape[1]).any()):
        raise ValueError("targets contain an identity index outside scores")
    rows = torch.arange(scores.shape[0], device=scores.device)
    positive = scores[rows, targets]
    negative_mask = torch.ones_like(scores, dtype=torch.bool)
    negative_mask[rows, targets] = False
    negative = scores.masked_fill(~negative_mask, torch.finfo(scores.dtype).min)
    hardest_negative = negative.max(dim=1).values
    observed_margin = positive - hardest_negative
    violations = F.relu(float(margin) - observed_margin)
    finite = torch.isfinite(observed_margin)
    if not bool(finite.any()):
        return _zero_like_scores(scores), _zero_like_scores(scores)
    return violations[finite].float().mean(), observed_margin[finite].float().mean()


def baseline_no_harm_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> torch.Tensor:
    """Protect every baseline pairwise margin with near negatives weighted most."""

    scores = output["score"]
    baseline = output.get("baseline_score")
    if not isinstance(baseline, torch.Tensor):
        raise ValueError("matcher diagnostics must contain baseline_score")
    if scores.ndim != 2 or tuple(baseline.shape) != tuple(scores.shape):
        raise ValueError(
            "score and baseline_score must have shape [queries, identities]"
        )
    targets = targets.to(device=scores.device, dtype=torch.long)
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("targets must have one entry per score row")
    if scores.shape[1] <= 1:
        return _zero_like_scores(scores)
    if bool((targets < 0).any()) or bool((targets >= scores.shape[1]).any()):
        raise ValueError("targets contain an identity index outside scores")
    rows = torch.arange(scores.shape[0], device=scores.device)
    negative_mask = torch.ones_like(scores, dtype=torch.bool)
    negative_mask[rows, targets] = False
    # Every negative whose correction exceeds the positive correction reduces
    # that pair's protected baseline margin. Baseline scores only determine a
    # detached neighbor weighting, so the constraint cannot improve itself by
    # moving the descriptor space it is meant to protect.
    correction = scores - baseline
    positive_correction = correction[rows, targets].unsqueeze(1)
    violations = F.relu(correction - positive_correction) * negative_mask.to(
        dtype=correction.dtype
    )
    protected_baseline = baseline.detach().float()
    neighbor_scale = protected_baseline.std(
        dim=1,
        unbiased=False,
        keepdim=True,
    ).clamp_min(torch.finfo(torch.float32).eps)
    neighbor_logits = (
        (protected_baseline - protected_baseline.mean(dim=1, keepdim=True))
        / neighbor_scale
    ).masked_fill(
        ~negative_mask,
        -float("inf"),
    )
    neighbor_weights = F.softmax(neighbor_logits, dim=1)
    return (violations.float() * neighbor_weights).sum(dim=1).mean()


def _normalise_distribution(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = values.float().clamp_min(0.0) * mask.to(dtype=torch.float32)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _view_coverage_targets(
    batch: ReferenceImageBatch,
) -> dict[str, torch.Tensor] | None:
    """Build soft relevance/novelty targets from continuous view metadata."""

    if (
        batch.query_view_features is None
        or batch.reference_view_features is None
        or batch.query_view_valid is None
        or batch.reference_view_valid is None
    ):
        return None
    query = batch.query_view_features.float()
    references = batch.reference_view_features.float()
    query_valid = batch.query_view_valid.to(device=references.device, dtype=torch.bool)
    reference_valid = batch.reference_view_valid.to(
        device=references.device,
        dtype=torch.bool,
    )
    if query.ndim != 2 or query.shape[1] != VIEW_FEATURE_DIM:
        raise ValueError("query_view_features must have shape [queries,4]")
    if references.ndim != 3 or references.shape[2] != VIEW_FEATURE_DIM:
        raise ValueError(
            "reference_view_features must have shape [identities,references,4]"
        )
    if tuple(query_valid.shape) != (query.shape[0],):
        raise ValueError("query_view_valid must have one entry per query")
    if tuple(reference_valid.shape) != tuple(references.shape[:2]):
        raise ValueError("reference_view_valid must match reference view rows")
    mask = batch.reference_mask.to(device=references.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(references.shape[:2]):
        raise ValueError("reference_mask must match reference view rows")
    valid_references = mask & reference_valid
    query_unit = F.normalize(query, dim=-1, eps=1.0e-6)
    reference_unit = F.normalize(references, dim=-1, eps=1.0e-6)
    query_reference_similarity = torch.einsum(
        "qd,ikd->qikd", query_unit, reference_unit
    ).sum(dim=-1)
    # Cosine is mapped to [0,1], retaining a non-zero neutral relevance when a
    # query has no reliable viewpoint signal.
    relevance = ((query_reference_similarity + 1.0) * 0.5).clamp(0.05, 1.0)

    reference_similarity = torch.einsum("ikd,ijd->ikj", reference_unit, reference_unit)
    pair_mask = valid_references.unsqueeze(1) & valid_references.unsqueeze(2)
    eye = torch.eye(
        references.shape[1], dtype=torch.bool, device=references.device
    ).unsqueeze(0)
    other_mask = pair_mask & ~eye
    other_max = reference_similarity.masked_fill(~other_mask, -1.0).max(dim=2).values
    has_other = other_mask.any(dim=2)
    novelty = torch.where(
        has_other,
        ((1.0 - other_max) * 0.5).clamp(0.0, 1.0),
        torch.ones_like(other_max),
    )

    if (
        batch.reference_quality_features is not None
        and batch.reference_quality_valid is not None
    ):
        quality = batch.reference_quality_features.float().to(device=references.device)
        quality_valid = batch.reference_quality_valid.to(
            device=quality.device,
            dtype=torch.bool,
        )
        if quality.ndim != 3 or quality.shape[2] != QUALITY_FEATURE_DIM:
            raise ValueError(
                "reference_quality_features must have shape [identities,references,6]"
            )
        if tuple(quality_valid.shape) != tuple(quality.shape[:2]):
            raise ValueError("reference_quality_valid must match quality rows")
        quality = quality.mean(dim=-1).clamp(0.05, 1.0)
        quality = torch.where(quality_valid, quality, torch.ones_like(quality))
    else:
        quality = torch.ones_like(novelty)

    reliability = quality * (0.5 + 0.5 * novelty)
    weights = relevance * reliability.unsqueeze(0)
    weights = weights * valid_references.unsqueeze(0).to(dtype=weights.dtype)
    targets = _normalise_distribution(
        weights,
        valid_references.unsqueeze(0).expand(query.shape[0], -1, -1),
    )
    query_valid_rows = query_valid[:, None] & valid_references.any(dim=1)[None, :]
    return {
        "attention": targets,
        "reference_valid": valid_references,
        "query_identity_valid": query_valid_rows,
        "novelty": novelty,
        "novelty_valid": valid_references & has_other,
        "reliability": reliability,
    }


def view_coverage_loss(
    output: dict[str, torch.Tensor],
    batch: ReferenceImageBatch | CachedReferenceFeatureBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align learned reference attention with continuous view coverage targets.

    The positive identity's references receive a soft target proportional to
    query-view relevance, pairwise novelty, and (when available) image quality.
    Both the reference gate and token-level evidence are supervised when the
    matcher exposes token scores.  No discrete viewpoint class is introduced.
    """

    attention = output.get("attention")
    if not isinstance(attention, torch.Tensor):
        raise ValueError("matcher diagnostics must contain attention")
    target_pack = _view_coverage_targets(batch)
    if target_pack is None:
        zero = _zero_like_scores(output["score"])
        return zero, {
            "attention_alignment_loss": zero.detach(),
            "token_alignment_loss": zero.detach(),
            "novelty_alignment_loss": zero.detach(),
            "reliability_alignment_loss": zero.detach(),
            "coverage_target_entropy": zero.detach(),
            "coverage_pred_entropy": zero.detach(),
            "coverage_valid_fraction": zero.detach(),
        }
    targets = target_pack["attention"]
    valid_references = target_pack["reference_valid"]
    valid_rows = target_pack["query_identity_valid"]
    scores = output["score"]
    targets = targets.to(device=scores.device)
    valid_references = valid_references.to(device=scores.device)
    valid_rows = valid_rows.to(device=scores.device)
    if attention.ndim != 3 or attention.shape[:2] != scores.shape:
        raise ValueError("attention must have shape [queries,identities,references]")
    if attention.shape[2] != targets.shape[2]:
        raise ValueError("attention/reference metadata widths differ")
    query_count = scores.shape[0]
    target_indices = batch.targets.to(device=scores.device, dtype=torch.long)
    if target_indices.shape[0] != query_count:
        raise ValueError("batch target count differs from matcher output")
    if bool((target_indices < 0).any()) or bool(
        (target_indices >= scores.shape[1]).any()
    ):
        raise ValueError(
            "batch targets contain an identity index outside matcher output"
        )
    rows = torch.arange(query_count, device=scores.device)
    positive_attention = attention[rows, target_indices].float()
    positive_targets = targets[rows, target_indices]
    positive_valid = valid_references[target_indices]
    usable = valid_rows[rows, target_indices]
    positive_attention = _normalise_distribution(positive_attention, positive_valid)
    positive_targets = _normalise_distribution(positive_targets, positive_valid)
    log_prediction = positive_attention.clamp_min(1.0e-6).log()
    attention_kl = (
        positive_targets * (positive_targets.clamp_min(1.0e-6).log() - log_prediction)
    ).sum(dim=1)

    token_scores = output.get("token_scores")
    has_token_scores = isinstance(token_scores, torch.Tensor)
    if has_token_scores:
        if token_scores.ndim != 3 or token_scores.shape[:2] != scores.shape:
            raise ValueError(
                "token_scores must have shape [queries,identities,references]"
            )
        token_prediction = token_scores[rows, target_indices].float()
        token_prediction = token_prediction.masked_fill(~positive_valid, -20.0)
        token_prediction = F.softmax(token_prediction, dim=1)
        token_kl = (
            positive_targets
            * (
                positive_targets.clamp_min(1.0e-6).log()
                - token_prediction.clamp_min(1.0e-6).log()
            )
        ).sum(dim=1)
        predicted = 0.5 * (positive_attention + token_prediction)
    else:
        token_kl = None
        predicted = positive_attention
    usable_float = usable.to(dtype=attention_kl.dtype)
    if bool(usable.any()):
        attention_loss = (
            attention_kl * usable_float
        ).sum() / usable_float.sum().clamp_min(1.0)
        token_loss = (
            (token_kl * usable_float).sum() / usable_float.sum().clamp_min(1.0)
            if token_kl is not None
            else _zero_like_scores(scores)
        )
        target_entropy = (
            -(positive_targets.clamp_min(1.0e-6).log() * positive_targets).sum(dim=1)
            * usable_float
        ).sum() / usable_float.sum().clamp_min(1.0)
        prediction_entropy = (
            -(predicted.clamp_min(1.0e-6).log() * predicted).sum(dim=1) * usable_float
        ).sum() / usable_float.sum().clamp_min(1.0)
    else:
        attention_loss = _zero_like_scores(scores)
        token_loss = _zero_like_scores(scores)
        target_entropy = attention_loss.detach()
        prediction_entropy = attention_loss.detach()

    novelty = output.get("novelty")
    if isinstance(novelty, torch.Tensor):
        if novelty.ndim != 3 or novelty.shape[:2] != scores.shape:
            raise ValueError("novelty must have shape [queries,identities,references]")
        positive_novelty = novelty[rows, target_indices].float()
        novelty_targets = target_pack["novelty"].to(device=scores.device)[
            target_indices
        ]
        novelty_valid = target_pack["novelty_valid"].to(device=scores.device)[
            target_indices
        ]
        novelty_errors = F.smooth_l1_loss(
            positive_novelty,
            novelty_targets,
            reduction="none",
        )
        novelty_valid_float = novelty_valid.to(dtype=novelty_errors.dtype)
        if bool(novelty_valid.any()):
            novelty_loss = (
                novelty_errors * novelty_valid_float
            ).sum() / novelty_valid_float.sum().clamp_min(1.0)
        else:
            novelty_loss = _zero_like_scores(scores)
    else:
        # The descriptor-only matcher predates explicit view novelty. It keeps
        # its attention supervision without pretending to expose this target.
        novelty_loss = _zero_like_scores(scores)

    coverage_gate = output.get("coverage_gate")
    if isinstance(coverage_gate, torch.Tensor):
        if coverage_gate.ndim != 3 or coverage_gate.shape[:2] != scores.shape:
            raise ValueError(
                "coverage_gate must have shape [queries,identities,references]"
            )
        positive_gate = coverage_gate[rows, target_indices].float()
        reliability_targets = target_pack["reliability"].to(device=scores.device)[
            target_indices
        ]
        reliability_errors = F.smooth_l1_loss(
            positive_gate,
            reliability_targets,
            reduction="none",
        )
        reliability_valid_float = positive_valid.to(dtype=reliability_errors.dtype)
        reliability_loss = (
            reliability_errors * reliability_valid_float
        ).sum() / reliability_valid_float.sum().clamp_min(1.0)
    else:
        reliability_loss = _zero_like_scores(scores)

    loss = attention_loss + token_loss + novelty_loss + reliability_loss
    return loss, {
        "attention_alignment_loss": attention_loss,
        "token_alignment_loss": token_loss,
        "novelty_alignment_loss": novelty_loss,
        "reliability_alignment_loss": reliability_loss,
        "coverage_target_entropy": target_entropy.detach(),
        "coverage_pred_entropy": prediction_entropy.detach(),
        "coverage_valid_fraction": usable_float.mean().detach(),
    }


def _reference_episode_loss_from_output(
    output: dict[str, torch.Tensor],
    batch: ReferenceImageBatch | CachedReferenceFeatureBatch,
    *,
    temperature: float = 0.10,
    residual_regularization: float = 0.01,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.15,
    view_coverage_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if (
        not math.isfinite(float(residual_regularization))
        or residual_regularization < 0.0
    ):
        raise ValueError("residual_regularization must be finite and non-negative")
    for name, value in (
        ("hard_negative_weight", hard_negative_weight),
        ("view_coverage_weight", view_coverage_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(hard_negative_margin)) or hard_negative_margin <= 0.0:
        raise ValueError("hard_negative_margin must be positive and finite")
    scores = output["score"]
    targets = batch.targets.to(device=scores.device, dtype=torch.long)
    retrieval = F.cross_entropy(scores.float() / float(temperature), targets)
    residual = output["residual"]
    raw_residual = output.get("raw_residual", residual)
    penalty = raw_residual.float().square().mean()
    hard_negative, observed_margin = hard_negative_margin_loss(
        scores,
        targets,
        margin=hard_negative_margin,
    )
    no_harm = baseline_no_harm_loss(output, targets)
    coverage, coverage_details = view_coverage_loss(output, batch)
    total = (
        retrieval
        + float(residual_regularization) * penalty
        + float(hard_negative_weight) * hard_negative
        + float(view_coverage_weight) * coverage
        + no_harm
    )
    return total, {
        "loss": total,
        "retrieval_loss": retrieval,
        "residual_penalty": penalty,
        "hard_negative_loss": hard_negative,
        "observed_hard_negative_margin": observed_margin,
        "baseline_no_harm_loss": no_harm,
        "view_coverage_loss": coverage,
        **coverage_details,
    }


def _reference_prefix_batch(
    batch: ReferenceImageBatch | CachedReferenceFeatureBatch,
    reference_count: int,
) -> ReferenceImageBatch | CachedReferenceFeatureBatch:
    """Mask one nested reference prefix without copying image/features."""

    width = int(batch.reference_mask.shape[1])
    count = int(reference_count)
    if count < 1 or count > width:
        raise ValueError("reference_count must fit within the batch width")
    columns = torch.arange(
        width,
        device=batch.reference_mask.device,
    ).unsqueeze(0)
    mask = batch.reference_mask & (columns < count)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every nested reference prefix must remain non-empty")
    return replace(batch, reference_mask=mask)


def _average_loss_details(
    rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    *,
    reference_counts: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not rows:
        raise ValueError("nested reference training produced no losses")
    if len(rows) != len(reference_counts):
        raise ValueError("reference count diagnostics do not match loss rows")
    loss = torch.stack([item[0] for item in rows]).mean()
    names = rows[0][1].keys()
    details = {
        name: torch.stack([item[1][name] for item in rows]).mean() for name in names
    }
    details["loss"] = loss
    details["nested_reference_counts"] = torch.tensor(
        reference_counts,
        dtype=torch.long,
        device=loss.device,
    )
    return loss, details


def reference_episode_loss(
    model: ReferenceAwarePetReID,
    batch: ReferenceImageBatch,
    *,
    temperature: float = 0.10,
    residual_regularization: float = 0.01,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.15,
    view_coverage_weight: float = 0.0,
    nested_reference_counts: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return retrieval loss for a live image episode."""

    counts = (
        tuple(range(1, int(batch.reference_mask.shape[1]) + 1))
        if nested_reference_counts
        else (int(batch.reference_mask.shape[1]),)
    )
    rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for count in counts:
        prefix = _reference_prefix_batch(batch, count)
        output = score_reference_image_episode(model, prefix, return_aux=True)
        if not isinstance(output, dict):
            raise RuntimeError("model returned no episode diagnostics")
        rows.append(
            _reference_episode_loss_from_output(
                output,
                prefix,
                temperature=temperature,
                residual_regularization=residual_regularization,
                hard_negative_weight=hard_negative_weight,
                hard_negative_margin=hard_negative_margin,
                view_coverage_weight=view_coverage_weight,
            )
        )
    return _average_loss_details(rows, reference_counts=counts)


def cached_reference_episode_loss(
    model: Any,
    batch: CachedReferenceFeatureBatch,
    *,
    temperature: float = 0.10,
    residual_regularization: float = 0.01,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.15,
    view_coverage_weight: float = 0.0,
    nested_reference_counts: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the same objectives over an all-identity cached score matrix."""

    counts = (
        tuple(range(1, int(batch.reference_mask.shape[1]) + 1))
        if nested_reference_counts
        else (int(batch.reference_mask.shape[1]),)
    )
    rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for count in counts:
        prefix = _reference_prefix_batch(batch, count)
        output = score_cached_reference_episode(model, prefix, return_aux=True)
        if not isinstance(output, dict):
            raise RuntimeError("model returned no cached episode diagnostics")
        rows.append(
            _reference_episode_loss_from_output(
                output,
                prefix,
                temperature=temperature,
                residual_regularization=residual_regularization,
                hard_negative_weight=hard_negative_weight,
                hard_negative_margin=hard_negative_margin,
                view_coverage_weight=view_coverage_weight,
            )
        )
    return _average_loss_details(rows, reference_counts=counts)


def build_full_catalog_validation_episodes(
    dataset: Any,
    *,
    reference_count: int,
    queries_per_identity: int = 1,
    query_identities_per_batch: int = 8,
    seed: int = 20260903,
) -> tuple[ReferenceImageEpisode, ...]:
    """Cover every query identity once against one fixed full candidate catalog.

    Each identity contributes one deterministic maximum reference set and a
    disjoint query set. The same candidate references are reused in every
    query batch, which lets validation compare nested prefixes without changing
    either the images or the negative identities underneath the metric.
    """

    reference_count = int(reference_count)
    queries_per_identity = int(queries_per_identity)
    query_identities_per_batch = int(query_identities_per_batch)
    if min(reference_count, queries_per_identity, query_identities_per_batch) < 1:
        raise ValueError("validation episode sizes must be positive")
    groups = _identity_groups(dataset)
    identity_names = tuple(sorted(groups))
    minimum = reference_count + queries_per_identity
    insufficient = {
        identity: int(rows.size)
        for identity, rows in groups.items()
        if int(rows.size) < minimum
    }
    if insufficient:
        raise ValueError(
            f"each validation identity needs at least {minimum} images: {insufficient}"
        )

    rng = np.random.default_rng(int(seed))
    reference_rows: dict[str, tuple[int, ...]] = {}
    query_rows: dict[str, tuple[int, ...]] = {}
    for identity in identity_names:
        selected = rng.choice(
            groups[identity],
            size=minimum,
            replace=False,
        ).tolist()
        reference_rows[identity] = tuple(
            int(value) for value in selected[:reference_count]
        )
        query_rows[identity] = tuple(int(value) for value in selected[reference_count:])

    references = tuple(reference_rows[identity] for identity in identity_names)
    positions = {identity: position for position, identity in enumerate(identity_names)}
    episodes: list[ReferenceImageEpisode] = []
    for start in range(0, len(identity_names), query_identities_per_batch):
        query_names = identity_names[start : start + query_identities_per_batch]
        query_indices = tuple(
            index for identity in query_names for index in query_rows[identity]
        )
        targets = torch.tensor(
            [
                positions[identity]
                for identity in query_names
                for _ in range(queries_per_identity)
            ],
            dtype=torch.long,
        )
        episodes.append(
            ReferenceImageEpisode(
                identity_names=identity_names,
                query_indices=query_indices,
                reference_indices=references,
                targets=targets,
            )
        )
    return tuple(episodes)


def _ranking_observations(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("catalog scores must contain at least two identities")
    targets = targets.to(device=scores.device, dtype=torch.long)
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("catalog targets must have one entry per score row")
    if bool((targets < 0).any()) or bool((targets >= scores.shape[1]).any()):
        raise ValueError("catalog targets contain an identity outside scores")
    rows = torch.arange(scores.shape[0], device=scores.device)
    ranking = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = (ranking == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    negative_mask = torch.ones_like(scores, dtype=torch.bool)
    negative_mask[rows, targets] = False
    hardest_negative = (
        scores.masked_fill(
            ~negative_mask,
            torch.finfo(scores.dtype).min,
        )
        .max(dim=1)
        .values
    )
    margins = scores[rows, targets] - hardest_negative
    return ranks.to(dtype=torch.long), margins.float()


def _ranking_summary(
    ranks: torch.Tensor,
    margins: torch.Tensor,
) -> dict[str, float | int]:
    if ranks.ndim != 1 or margins.ndim != 1 or ranks.shape != margins.shape:
        raise ValueError("rank and margin observations must be aligned vectors")
    if ranks.numel() < 1:
        raise ValueError("catalog validation produced no query records")
    return {
        "query_records": int(ranks.numel()),
        "top1_correct": int((ranks == 1).sum().item()),
        "top1_accuracy": float((ranks == 1).float().mean().item()),
        "top5_correct": int((ranks <= 5).sum().item()),
        "top5_accuracy": float((ranks <= 5).float().mean().item()),
        "mean_reciprocal_rank": float(ranks.float().reciprocal().mean().item()),
        "mean_positive_margin": float(margins.mean().item()),
    }


def reference_validation_selection_summary(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("catalog validation has no reference-count reports")
    if "1" not in reports:
        raise ValueError("catalog validation has no singleton reference report")
    singleton_exact = bool(reports["1"].get("exact_centroid_match", False))
    multi_reference = [
        reports[key] for key in sorted(reports, key=int) if int(key) > 1
    ]
    query_records = sum(int(row["learned"]["query_records"]) for row in multi_reference)
    learned_top1_correct = sum(
        int(row["learned"]["top1_correct"]) for row in multi_reference
    )
    centroid_top1_correct = sum(
        int(row["centroid"]["top1_correct"]) for row in multi_reference
    )

    def weighted_mean(section: str, metric: str) -> float:
        if query_records < 1:
            return 0.0
        return float(
            sum(
                float(row[section][metric]) * int(row[section]["query_records"])
                for row in multi_reference
            )
            / query_records
        )

    learned_top1 = learned_top1_correct / query_records if query_records else 0.0
    centroid_top1 = centroid_top1_correct / query_records if query_records else 0.0
    learned_reciprocal = weighted_mean("learned", "mean_reciprocal_rank")
    centroid_reciprocal = weighted_mean("centroid", "mean_reciprocal_rank")
    learned_margin = weighted_mean("learned", "mean_positive_margin")
    centroid_margin = weighted_mean("centroid", "mean_positive_margin")
    top1_noninferior = (
        bool(multi_reference) and learned_top1_correct >= centroid_top1_correct
    )
    eligible = singleton_exact and top1_noninferior
    key = (learned_top1, learned_reciprocal, learned_margin)
    return {
        "policy": (
            "require exact singleton centroid behavior and noninferior aggregate "
            "multi-reference top1; rank learned checkpoints by multi-reference "
            "top1, reciprocal rank, then positive margin"
        ),
        "eligible_for_best_learned": eligible,
        "singleton_exact_centroid_match": singleton_exact,
        "multi_reference_top1_noninferior": top1_noninferior,
        "all_multi_reference_counts_top1_noninferior": bool(multi_reference)
        and all(
            int(row["learned"]["top1_correct"])
            >= int(row["centroid"]["top1_correct"])
            for row in multi_reference
        ),
        "multi_reference": {
            "reference_counts": [
                int(row["reference_count"]) for row in multi_reference
            ],
            "query_records": query_records,
            "learned_top1_accuracy": learned_top1,
            "centroid_top1_accuracy": centroid_top1,
            "top1_accuracy_delta": learned_top1 - centroid_top1,
            "learned_mean_reciprocal_rank": learned_reciprocal,
            "centroid_mean_reciprocal_rank": centroid_reciprocal,
            "mean_reciprocal_rank_delta": learned_reciprocal
            - centroid_reciprocal,
            "learned_mean_positive_margin": learned_margin,
            "centroid_mean_positive_margin": centroid_margin,
            "mean_positive_margin_delta": learned_margin - centroid_margin,
        },
        "key": [round(float(value), 12) for value in key],
        "comparison_tolerances": list(REFERENCE_SELECTION_TOLERANCES),
        "tie_policy": "keep_earliest_within_tolerance",
        "per_query_margin_non_degradation": "diagnostic_only",
    }


def reference_validation_selection_key(
    validation: dict[str, Any],
) -> tuple[float, ...]:
    """Return the stable lexicographic key stored by catalog validation."""

    if validation.get("protocol") != "full_identity_catalog_nested_references":
        raise ValueError("validation report does not use the full catalog protocol")
    selection = validation.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("validation report has no selection summary")
    key = selection.get("key")
    if not isinstance(key, list) or not key:
        raise ValueError("validation report has no selection key")
    values = tuple(float(value) for value in key)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("validation selection key contains non-finite values")
    return values


def reference_validation_checkpoint_eligible(validation: dict[str, Any]) -> bool:
    """Return whether a full-catalog report may become the best learned model."""

    if validation.get("protocol") != "full_identity_catalog_nested_references":
        return True
    selection = validation.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("validation report has no selection summary")
    eligible = selection.get("eligible_for_best_learned")
    if not isinstance(eligible, bool):
        raise ValueError("validation report has no learned-checkpoint eligibility")
    return eligible


def reference_validation_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> bool:
    """Compare retrieval reports while treating insignificant float drift as a tie."""

    if not reference_validation_checkpoint_eligible(candidate):
        return False
    if incumbent is None:
        return True
    candidate_key = reference_validation_selection_key(candidate)
    incumbent_key = reference_validation_selection_key(incumbent)
    if len(candidate_key) != len(incumbent_key):
        raise ValueError("validation selection keys have different lengths")
    tolerances = (
        REFERENCE_SELECTION_TOLERANCES
        if candidate.get("protocol") == "full_identity_catalog_nested_references"
        else (1.0e-12,) * len(candidate_key)
    )
    if len(tolerances) != len(candidate_key):
        raise ValueError("validation selection key has an unsupported length")
    for candidate_value, incumbent_value, tolerance in zip(
        candidate_key, incumbent_key, tolerances
    ):
        if candidate_value > incumbent_value + tolerance:
            return True
        if candidate_value < incumbent_value - tolerance:
            return False
    return False


def evaluate_cached_reference_catalog(
    model: Any,
    cache: ReferenceSpatialFeatureCache,
    dataset: Any,
    *,
    reference_count: int,
    queries_per_identity: int = 1,
    query_identities_per_batch: int = 8,
    seed: int = 20260903,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate each nested reference prefix against every candidate identity."""

    reference_count = int(reference_count)
    episodes = build_full_catalog_validation_episodes(
        dataset,
        reference_count=reference_count,
        queries_per_identity=queries_per_identity,
        query_identities_per_batch=query_identities_per_batch,
        seed=seed,
    )
    counts = tuple(range(1, reference_count + 1))
    collected: dict[int, dict[str, list[torch.Tensor]]] = {
        count: {
            "score": [],
            "baseline": [],
            "targets": [],
            "catalog_gate": [],
        }
        for count in counts
    }
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for episode in episodes:
                batch = materialize_cached_reference_episode(
                    cache,
                    dataset,
                    episode,
                    device=device,
                )
                for count in counts:
                    prefix = _reference_prefix_batch(batch, count)
                    output = score_cached_reference_episode(
                        model,
                        prefix,
                        return_aux=True,
                    )
                    if not isinstance(output, dict):
                        raise RuntimeError(
                            "token matcher returned no catalog diagnostics"
                        )
                    scores = output.get("score")
                    baseline = output.get("baseline_score")
                    catalog_gate = output.get("catalog_confidence_gate")
                    if not isinstance(scores, torch.Tensor) or not isinstance(
                        baseline, torch.Tensor
                    ):
                        raise ValueError(
                            "catalog diagnostics must contain score and baseline_score"
                        )
                    if tuple(scores.shape) != tuple(baseline.shape):
                        raise ValueError("catalog score and baseline shapes differ")
                    if not isinstance(catalog_gate, torch.Tensor) or tuple(
                        catalog_gate.shape
                    ) != tuple(scores.shape):
                        raise ValueError(
                            "catalog diagnostics must contain a gate for every "
                            "query/identity row"
                        )
                    if not torch.equal(
                        catalog_gate,
                        catalog_gate[:, :1].expand_as(catalog_gate),
                    ):
                        raise RuntimeError(
                            "catalog confidence gate must be shared by every "
                            "candidate for one query"
                        )
                    if count == 1 and not torch.equal(scores, baseline):
                        raise RuntimeError(
                            "singleton token matching must equal the centroid "
                            "baseline exactly"
                        )
                    collected[count]["score"].append(scores.detach().cpu())
                    collected[count]["baseline"].append(baseline.detach().cpu())
                    collected[count]["targets"].append(prefix.targets.detach().cpu())
                    collected[count]["catalog_gate"].append(
                        catalog_gate[:, 0].detach().cpu()
                    )
    finally:
        model.train(was_training)

    reports: dict[str, dict[str, Any]] = {}
    for count in counts:
        scores = torch.cat(collected[count]["score"], dim=0)
        baseline = torch.cat(collected[count]["baseline"], dim=0)
        targets = torch.cat(collected[count]["targets"], dim=0)
        catalog_gate = torch.cat(collected[count]["catalog_gate"], dim=0).float()
        learned_ranks, learned_margins = _ranking_observations(scores, targets)
        baseline_ranks, baseline_margins = _ranking_observations(
            baseline,
            targets,
        )
        learned = _ranking_summary(learned_ranks, learned_margins)
        centroid = _ranking_summary(baseline_ranks, baseline_margins)
        margin_delta = learned_margins - baseline_margins
        rank_non_degradation = learned_ranks <= baseline_ranks
        margin_non_degradation = margin_delta >= -1.0e-7
        reports[str(count)] = {
            "reference_count": count,
            "learned": learned,
            "centroid": centroid,
            "exact_centroid_match": bool(torch.equal(scores, baseline)),
            "delta": {
                "top1_accuracy": float(
                    learned["top1_accuracy"] - centroid["top1_accuracy"]
                ),
                "top5_accuracy": float(
                    learned["top5_accuracy"] - centroid["top5_accuracy"]
                ),
                "mean_reciprocal_rank": float(
                    learned["mean_reciprocal_rank"] - centroid["mean_reciprocal_rank"]
                ),
                "mean_positive_margin": float(margin_delta.mean().item()),
            },
            "no_harm": {
                "rank_non_degradation_rate": float(
                    rank_non_degradation.float().mean().item()
                ),
                "margin_non_degradation_rate": float(
                    margin_non_degradation.float().mean().item()
                ),
                "harmed_rank_query_count": int((~rank_non_degradation).sum().item()),
                "harmed_margin_query_count": int(
                    (~margin_non_degradation).sum().item()
                ),
            },
            "catalog_confidence_gate": {
                "mean": float(catalog_gate.mean().item()),
                "closed_fraction": float((catalog_gate <= 0.0).float().mean().item()),
                "active_fraction": float((catalog_gate > 0.0).float().mean().item()),
            },
        }

    return {
        "protocol": "full_identity_catalog_nested_references",
        "baseline": "centroid",
        "candidate_identities": len(episodes[0].identity_names),
        "query_records": sum(len(episode.query_indices) for episode in episodes),
        "query_batches": len(episodes),
        "reference_counts": reports,
        "selection": reference_validation_selection_summary(reports),
    }


__all__ = [
    "AllIdentityReferenceEpisodeSampler",
    "CachedReferenceFeatureBatch",
    "REFERENCE_IMAGE_MANIFEST_FIELDS",
    "ReferenceImageBatch",
    "ReferenceImageEpisode",
    "ReferenceImageEpisodeSampler",
    "ReferenceSpatialFeatureCache",
    "baseline_no_harm_loss",
    "SPATIAL_FEATURE_CACHE_FORMAT",
    "build_full_catalog_validation_episodes",
    "build_reference_spatial_feature_cache",
    "cached_reference_episode_loss",
    "evaluate_cached_reference_catalog",
    "hard_negative_margin_loss",
    "load_reference_spatial_feature_cache",
    "materialize_cached_reference_episode",
    "materialize_reference_image_episode",
    "reference_episode_loss",
    "reference_validation_checkpoint_eligible",
    "reference_validation_is_better",
    "reference_validation_selection_key",
    "reference_validation_selection_summary",
    "save_reference_spatial_feature_cache",
    "score_cached_reference_episode",
    "score_reference_image_episode",
    "validate_reference_image_manifest",
    "view_coverage_loss",
]
