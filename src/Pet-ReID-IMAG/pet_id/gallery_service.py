"""Incremental, model-bound pet gallery storage and identification service.

The training classifier is intentionally not involved here.  New identities are
represented by the L2-normalized mean of their enrolled image descriptors.
SQLite protects metadata and descriptor updates transactionally; original
images are stored by content hash so duplicate uploads are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from .gallery import encode_primary, load_gallery_model, normalized_array
from .reference_scoring import (
    CENTROID_SCORING,
    DEFAULT_REFERENCE_SCORE_WEIGHT,
    DEFAULT_REFERENCE_TOP_K,
    LEARNED_REFERENCE_SET_SCORING,
    LearnedReferenceScorer,
    REFERENCE_SET_SCORING,
    score_gallery,
    validate_reference_score_weight,
    validate_reference_top_k,
    validate_scoring_mode,
)
from .workspace_paths import resolve_legacy_path
from .workspace_store import WorkspaceStore


PET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
IMAGE_FORMAT_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GalleryServiceError(RuntimeError):
    """Base error carrying an HTTP-friendly status and stable error code."""

    status_code = 500
    code = "gallery_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class InvalidGalleryRequest(GalleryServiceError):
    status_code = 400
    code = "invalid_request"


class InvalidPetImage(GalleryServiceError):
    status_code = 422
    code = "invalid_pet_image"


class GalleryConflict(GalleryServiceError):
    status_code = 409
    code = "gallery_conflict"


class GalleryNotFound(GalleryServiceError):
    status_code = 404
    code = "not_found"


class GalleryEmpty(GalleryServiceError):
    status_code = 409
    code = "gallery_empty"


class GalleryModelMismatch(GalleryServiceError):
    status_code = 409
    code = "model_mismatch"


class _ClosingConnection(sqlite3.Connection):
    """Make sqlite's transaction context also release its Windows file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


@dataclass(frozen=True)
class UploadPayload:
    filename: str
    content_type: str | None
    data: bytes


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    content_type: str
    data: bytes
    sha256: str
    image_format: str
    suffix: str
    width: int
    height: int


@dataclass(frozen=True)
class EncodedPetImage:
    fused: np.ndarray
    nose: np.ndarray
    face: np.ndarray
    metadata: dict[str, Any]
    expert_features: dict[str, np.ndarray] = field(default_factory=dict)
    expert_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EnrollmentRecord:
    upload: ValidatedUpload
    encoded: EncodedPetImage


class PetFeatureEncoder(Protocol):
    def encode_file(self, path: Path) -> EncodedPetImage: ...

    def backend_info(self) -> dict[str, Any]: ...


def normalize_feature(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise InvalidPetImage(f"{name} must be one finite descriptor vector")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise InvalidPetImage(f"{name} has an invalid norm")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def validate_normalized_feature(value: np.ndarray, name: str) -> np.ndarray:
    """Validate a graph-normalized descriptor without changing its values."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise InvalidPetImage(f"{name} must be one finite descriptor vector")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise InvalidPetImage(f"{name} has an invalid norm")
    if not np.isclose(norm, 1.0, atol=3e-3, rtol=3e-3):
        raise InvalidPetImage(
            f"{name} must already be L2-normalized by the unified ONNX graph"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def validate_pet_id(pet_id: str) -> str:
    value = pet_id.strip()
    if not PET_ID_PATTERN.fullmatch(value):
        raise InvalidGalleryRequest(
            "pet_id must be 1-64 characters using letters, digits, '.', '_' or '-'"
        )
    return value


def validate_display_name(display_name: str | None) -> str | None:
    if display_name is None:
        return None
    value = display_name.strip()
    if not value:
        raise InvalidGalleryRequest("display_name cannot be blank")
    if len(value) > 128:
        raise InvalidGalleryRequest("display_name cannot exceed 128 characters")
    return value


def validate_upload(
    payload: UploadPayload,
    *,
    maximum_bytes: int,
    maximum_pixels: int,
) -> ValidatedUpload:
    if not payload.data:
        raise InvalidPetImage(f"{payload.filename or 'upload'} is empty")
    if len(payload.data) > maximum_bytes:
        raise InvalidPetImage(
            f"{payload.filename or 'upload'} exceeds the {maximum_bytes} byte limit",
            details={"maximum_bytes": maximum_bytes, "actual_bytes": len(payload.data)},
        )
    try:
        with Image.open(BytesIO(payload.data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            if image_format not in IMAGE_FORMAT_SUFFIXES:
                raise InvalidPetImage(
                    f"unsupported image format {image_format or 'unknown'}; "
                    f"expected one of {sorted(IMAGE_FORMAT_SUFFIXES)}"
                )
            if width <= 0 or height <= 0 or width * height > maximum_pixels:
                raise InvalidPetImage(
                    f"image dimensions {width}x{height} exceed the pixel limit",
                    details={"maximum_pixels": maximum_pixels},
                )
            image.verify()
    except InvalidPetImage:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidPetImage(
            f"{payload.filename or 'upload'} is not a valid supported image"
        ) from error
    digest = hashlib.sha256(payload.data).hexdigest()
    content_type = payload.content_type or f"image/{image_format.casefold()}"
    return ValidatedUpload(
        filename=Path(
            payload.filename or f"upload{IMAGE_FORMAT_SUFFIXES[image_format]}"
        ).name,
        content_type=content_type,
        data=payload.data,
        sha256=digest,
        image_format=image_format,
        suffix=IMAGE_FORMAT_SUFFIXES[image_format],
        width=width,
        height=height,
    )


def is_unified_single_graph_descriptor(descriptor: Any) -> bool:
    """Return whether descriptor metadata came from the one-graph runtime.

    The explicit runtime diagnostic is authoritative.  Unified descriptors do
    not carry branch fields; the predicate also accepts older records that
    used the compatibility container so galleries can be upgraded in place.
    """

    if not isinstance(descriptor, dict):
        return False
    diagnostics = descriptor.get("runtime_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    unified = diagnostics.get("unified")
    return isinstance(unified, dict) and unified.get("single_graph") is True


def reference_quality(inference: dict[str, Any]) -> dict[str, Any]:
    """Return stable, model-derived quality flags for one gallery reference."""

    descriptor = inference.get("descriptor")
    if not isinstance(descriptor, dict):
        return {
            "status": "unknown",
            "reasons": ["diagnostics_unavailable"],
            "branch_available": None,
            "branch_quality": None,
        }
    if is_unified_single_graph_descriptor(descriptor):
        return {
            "status": "good",
            "reasons": [],
            "branch_available": None,
            "branch_quality": None,
            "architecture": "unified_single_graph",
        }
    available = descriptor.get("branch_available")
    qualities = descriptor.get("branch_quality")
    normalized_available = (
        [bool(value) for value in available[:2]]
        if isinstance(available, list) and len(available) >= 2
        else None
    )
    normalized_quality = (
        [float(value) for value in qualities[:2]]
        if isinstance(qualities, list) and len(qualities) >= 2
        else None
    )
    reasons: list[str] = []
    if normalized_available is not None and not all(normalized_available):
        reasons.append("single_branch")
    if normalized_quality is not None:
        for index, quality in enumerate(normalized_quality):
            if (
                normalized_available is None or normalized_available[index]
            ) and quality < 0.35:
                reasons.append("low_nose_quality" if index == 0 else "low_face_quality")
    return {
        "status": "warning" if reasons else "good",
        "reasons": reasons,
        "branch_available": normalized_available,
        "branch_quality": normalized_quality,
    }


class UnifiedGraphEncoder:
    """Encode one canonical embedding from a single unified ONNX graph.

    The gallery schema still has legacy ``nose`` and ``face`` columns so old
    backups remain readable.  For a unified graph those columns are populated
    with the canonical vector only as a storage compatibility detail; no
    branch metadata or branch scoring is exposed to callers.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        info = self._read_backend_info()
        self.single_graph = bool(info.get("single_graph"))

    def _read_backend_info(self) -> dict[str, Any]:
        identity_model = getattr(self.pipeline, "identity_model", self.pipeline)
        backend_info = getattr(identity_model, "backend_info", None)
        if callable(backend_info):
            try:
                value = backend_info()
            except Exception:
                value = {}
            return dict(value) if isinstance(value, dict) else {}
        return {}

    @staticmethod
    def _graph_embedding_array(feature: Any) -> np.ndarray:
        """Copy the graph output without applying a second normalization.

        L2 normalization is a hard ONNX output contract.  The service checks
        that contract here and only converts device storage to a contiguous
        NumPy array; it deliberately does not call ``normalized_array`` (which
        would hide a malformed graph behind Python post-processing).
        """

        value = feature.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("unified graph returned an invalid embedding")
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("unified graph returned a zero embedding")
        if not np.isclose(norm, 1.0, atol=3e-3, rtol=3e-3):
            raise ValueError(f"unified graph returned a non-normalized embedding: {norm}")
        return np.ascontiguousarray(value)

    def encode_file(self, path: Path) -> EncodedPetImage:
        if self.single_graph:
            # ``encode_primary`` applies EXIF orientation and then hands a BGR
            # array to the runtime.  The runtime converts to RGB and, for the
            # raw-input graph, performs every geometric transform in ONNX.
            descriptor, _ = encode_primary(self.pipeline, path)
            embedding = self._graph_embedding_array(descriptor.fused_feature)
            metadata = {"descriptor": descriptor.metadata_dict()}
            return EncodedPetImage(
                fused=embedding,
                nose=embedding,
                face=embedding,
                metadata=metadata,
            )
        raise TypeError("UnifiedGraphEncoder requires a single-graph pipeline")

    def backend_info(self) -> dict[str, Any]:
        return self._read_backend_info()


class MultimodalPipelineEncoder(UnifiedGraphEncoder):
    """Compatibility adapter for legacy and unified pipelines.

    The historical class name is retained for third-party imports.  Unified
    pipelines take the :class:`UnifiedGraphEncoder` path above and never
    instantiate a detector, segmenter, or synthetic branch tail.
    """

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.profile_info: dict[str, Any] = {}

    def encode_file(self, path: Path) -> EncodedPetImage:
        if self.single_graph:
            return super().encode_file(path)
        descriptor, inference = encode_primary(self.pipeline, path)
        return EncodedPetImage(
            fused=normalized_array(descriptor.fused_feature),
            nose=normalized_array(descriptor.nose_feature),
            face=normalized_array(descriptor.face_feature),
            metadata=inference,
        )

    def backend_info(self) -> dict[str, Any]:
        if self.single_graph:
            base = super().backend_info()
        else:
            identity_model = self.pipeline.identity_model
            if hasattr(identity_model, "backend_info"):
                base = dict(identity_model.backend_info())
            else:
                base = {"backend": "pytorch", "device": str(self.pipeline.device)}
        return {**base, **self.profile_info}


class PetGalleryStore:
    """SQLite-backed incremental gallery with content-addressed image files."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_root = self.root / "images"
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.staging_root = self.root / "staging"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "gallery.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS service_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pets (
                    pet_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reference_images (
                    image_id TEXT PRIMARY KEY,
                    pet_id TEXT NOT NULL REFERENCES pets(pet_id) ON DELETE CASCADE,
                    sha256 TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    image_format TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    byte_size INTEGER NOT NULL,
                    stored_path TEXT NOT NULL,
                    fused_dim INTEGER NOT NULL,
                    fused BLOB NOT NULL,
                    nose_dim INTEGER NOT NULL,
                    nose BLOB NOT NULL,
                    face_dim INTEGER NOT NULL,
                    face BLOB NOT NULL,
                    inference_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reference_images_pet
                    ON reference_images(pet_id);
                CREATE TABLE IF NOT EXISTS expert_models (
                    expert_id TEXT PRIMARY KEY,
                    model_fingerprint TEXT NOT NULL,
                    backend_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS expert_features (
                    expert_id TEXT NOT NULL
                        REFERENCES expert_models(expert_id) ON DELETE CASCADE,
                    image_id TEXT NOT NULL
                        REFERENCES reference_images(image_id) ON DELETE CASCADE,
                    feature_dim INTEGER NOT NULL,
                    feature BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(expert_id, image_id)
                );
                CREATE INDEX IF NOT EXISTS idx_expert_features_image
                    ON expert_features(image_id);
                """
            )
            row = connection.execute(
                "SELECT value FROM service_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO service_metadata(key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION)),
                )
            elif int(row["value"]) != self.SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported gallery schema {row['value']}; "
                    f"expected {self.SCHEMA_VERSION}"
                )

    def bind_model(self, fingerprint: str, backend_info: dict[str, Any]) -> None:
        if not fingerprint:
            raise GalleryModelMismatch("the embedding model has no stable fingerprint")
        payload = json.dumps(backend_info, ensure_ascii=False, sort_keys=True)
        with self._lock, self._connect() as connection:
            current = connection.execute(
                "SELECT value FROM service_metadata WHERE key = 'model_fingerprint'"
            ).fetchone()
            if current is not None and current["value"] != fingerprint:
                raise GalleryModelMismatch(
                    "gallery descriptors were created by a different embedding model",
                    details={"stored": current["value"], "requested": fingerprint},
                )
            connection.execute(
                "INSERT OR REPLACE INTO service_metadata(key, value) VALUES (?, ?)",
                ("model_fingerprint", fingerprint),
            )
            connection.execute(
                "INSERT OR REPLACE INTO service_metadata(key, value) VALUES (?, ?)",
                ("backend_info", payload),
            )

    def bind_expert_models(self, experts: dict[str, dict[str, Any]]) -> None:
        """Bind independent expert namespaces without changing the primary space."""

        with self._lock, self._connect() as connection:
            for expert_id, backend_info in sorted(experts.items()):
                fingerprint = str(backend_info.get("model_sha256") or "")
                if not expert_id or not fingerprint:
                    raise GalleryModelMismatch(
                        "every gallery expert needs an id and stable model fingerprint"
                    )
                current = connection.execute(
                    "SELECT model_fingerprint FROM expert_models WHERE expert_id = ?",
                    (expert_id,),
                ).fetchone()
                if current is not None and current["model_fingerprint"] != fingerprint:
                    raise GalleryModelMismatch(
                        f"expert {expert_id!r} was created by a different model",
                        details={
                            "expert_id": expert_id,
                            "stored": current["model_fingerprint"],
                            "requested": fingerprint,
                        },
                    )
                connection.execute(
                    """
                    INSERT INTO expert_models(
                        expert_id, model_fingerprint, backend_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(expert_id) DO UPDATE SET
                        model_fingerprint = excluded.model_fingerprint,
                        backend_json = excluded.backend_json
                    """,
                    (
                        expert_id,
                        fingerprint,
                        json.dumps(backend_info, ensure_ascii=False, sort_keys=True),
                        utc_now(),
                    ),
                )

    def expert_models(self) -> dict[str, dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT expert_id, model_fingerprint, backend_json FROM expert_models "
                "ORDER BY expert_id"
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                backend = json.loads(row["backend_json"])
            except json.JSONDecodeError:
                backend = {}
            backend["model_sha256"] = str(row["model_fingerprint"])
            result[str(row["expert_id"])] = backend
        return result

    def metadata(self) -> dict[str, str]:
        with self._lock, self._connect() as connection:
            return {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM service_metadata")
            }

    def image_owner(self, sha256: str) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT pet_id FROM reference_images WHERE sha256 = ?", (sha256,)
            ).fetchone()
            return None if row is None else str(row["pet_id"])

    @staticmethod
    def _pack_feature(
        feature: np.ndarray,
        name: str,
        *,
        already_normalized: bool = False,
    ) -> tuple[int, bytes]:
        value = (
            validate_normalized_feature(feature, name)
            if already_normalized
            else normalize_feature(feature, name)
        )
        return int(value.size), value.astype("<f4", copy=False).tobytes()

    @staticmethod
    def _unpack_feature(blob: bytes, dimension: int) -> np.ndarray:
        value = np.frombuffer(blob, dtype="<f4", count=dimension).copy()
        if value.size != dimension:
            raise RuntimeError("stored gallery feature has an invalid byte length")
        return value

    def _write_image(self, upload: ValidatedUpload) -> tuple[str, bool]:
        relative = (
            Path("images") / upload.sha256[:2] / f"{upload.sha256}{upload.suffix}"
        )
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual != upload.sha256:
                raise GalleryConflict(
                    f"content-addressed image is corrupt: {destination}",
                    details={"expected": upload.sha256, "actual": actual},
                )
            return relative.as_posix(), False
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(upload.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return relative.as_posix(), True

    def enroll(
        self,
        pet_id: str,
        display_name: str | None,
        records: Sequence[EnrollmentRecord],
    ) -> dict[str, Any]:
        pet_id = validate_pet_id(pet_id)
        display_name = validate_display_name(display_name)
        created_files: list[Path] = []
        added: list[str] = []
        duplicates: list[str] = []
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing_pet = connection.execute(
                    "SELECT display_name, created_at FROM pets WHERE pet_id = ?",
                    (pet_id,),
                ).fetchone()
                resolved_name = (
                    display_name
                    if display_name is not None
                    else (str(existing_pet["display_name"]) if existing_pet else pet_id)
                )
                now = utc_now()
                if existing_pet is None:
                    connection.execute(
                        "INSERT INTO pets(pet_id, display_name, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (pet_id, resolved_name, now, now),
                    )
                else:
                    connection.execute(
                        "UPDATE pets SET display_name = ?, updated_at = ? WHERE pet_id = ?",
                        (resolved_name, now, pet_id),
                    )

                seen_batch: set[str] = set()
                for record in records:
                    upload = record.upload
                    if upload.sha256 in seen_batch:
                        duplicates.append(upload.sha256)
                        continue
                    seen_batch.add(upload.sha256)
                    owner = connection.execute(
                        "SELECT pet_id FROM reference_images WHERE sha256 = ?",
                        (upload.sha256,),
                    ).fetchone()
                    if owner is not None:
                        if owner["pet_id"] != pet_id:
                            raise GalleryConflict(
                                "the same image is already enrolled for another pet",
                                details={
                                    "sha256": upload.sha256,
                                    "existing_pet_id": owner["pet_id"],
                                    "requested_pet_id": pet_id,
                                },
                            )
                        duplicates.append(upload.sha256)
                        continue

                    descriptor = record.encoded.metadata.get("descriptor")
                    unified_single_graph = is_unified_single_graph_descriptor(
                        descriptor
                    )
                    fused_dim, fused_blob = self._pack_feature(
                        record.encoded.fused,
                        "fused",
                        already_normalized=unified_single_graph,
                    )
                    nose_dim, nose_blob = self._pack_feature(
                        record.encoded.nose,
                        "nose",
                        already_normalized=unified_single_graph,
                    )
                    face_dim, face_blob = self._pack_feature(
                        record.encoded.face,
                        "face",
                        already_normalized=unified_single_graph,
                    )
                    dimensions = connection.execute(
                        "SELECT fused_dim, nose_dim, face_dim FROM reference_images LIMIT 1"
                    ).fetchone()
                    if dimensions is not None and (
                        int(dimensions["fused_dim"]) != fused_dim
                        or int(dimensions["nose_dim"]) != nose_dim
                        or int(dimensions["face_dim"]) != face_dim
                    ):
                        raise GalleryModelMismatch(
                            "descriptor dimensions do not match the existing gallery"
                        )
                    registered_experts = {
                        str(row["expert_id"])
                        for row in connection.execute(
                            "SELECT expert_id FROM expert_models"
                        )
                    }
                    if set(record.encoded.expert_features) != registered_experts:
                        raise GalleryModelMismatch(
                            "encoded expert features do not match the gallery namespaces",
                            details={
                                "registered": sorted(registered_experts),
                                "encoded": sorted(record.encoded.expert_features),
                            },
                        )
                    packed_experts: dict[str, tuple[int, bytes]] = {}
                    for expert_id, feature in record.encoded.expert_features.items():
                        expert_dim, expert_blob = self._pack_feature(
                            feature, f"expert:{expert_id}"
                        )
                        existing_dim = connection.execute(
                            "SELECT feature_dim FROM expert_features "
                            "WHERE expert_id = ? LIMIT 1",
                            (expert_id,),
                        ).fetchone()
                        if existing_dim is not None and int(existing_dim[0]) != expert_dim:
                            raise GalleryModelMismatch(
                                f"expert {expert_id!r} feature dimension changed"
                            )
                        packed_experts[expert_id] = (expert_dim, expert_blob)
                    relative_path, was_created = self._write_image(upload)
                    if was_created:
                        created_files.append(self.root / relative_path)
                    connection.execute(
                        """
                        INSERT INTO reference_images(
                            image_id, pet_id, sha256, original_filename, content_type,
                            image_format, width, height, byte_size, stored_path,
                            fused_dim, fused, nose_dim, nose, face_dim, face,
                            inference_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            upload.sha256,
                            pet_id,
                            upload.sha256,
                            upload.filename,
                            upload.content_type,
                            upload.image_format,
                            upload.width,
                            upload.height,
                            len(upload.data),
                            relative_path,
                            fused_dim,
                            fused_blob,
                            nose_dim,
                            nose_blob,
                            face_dim,
                            face_blob,
                            json.dumps(record.encoded.metadata, ensure_ascii=False),
                            now,
                        ),
                    )
                    for expert_id, (expert_dim, expert_blob) in packed_experts.items():
                        connection.execute(
                            """
                            INSERT INTO expert_features(
                                expert_id, image_id, feature_dim, feature, metadata_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                expert_id,
                                upload.sha256,
                                expert_dim,
                                expert_blob,
                                json.dumps(
                                    record.encoded.expert_metadata.get(expert_id, {}),
                                    ensure_ascii=False,
                                ),
                            ),
                        )
                    added.append(upload.sha256)
                if added:
                    connection.execute(
                        "UPDATE pets SET updated_at = ? WHERE pet_id = ?", (now, pet_id)
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                for path in created_files:
                    path.unlink(missing_ok=True)
                raise
        return {
            "pet": self.get_pet(pet_id),
            "added_image_ids": added,
            "duplicate_image_ids": duplicates,
        }

    def list_pets(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.pet_id, p.display_name, p.created_at, p.updated_at,
                       COUNT(r.image_id) AS reference_count
                FROM pets p
                LEFT JOIN reference_images r ON r.pet_id = p.pet_id
                GROUP BY p.pet_id
                ORDER BY p.created_at, p.pet_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pet(self, pet_id: str) -> dict[str, Any]:
        pet_id = validate_pet_id(pet_id)
        with self._lock, self._connect() as connection:
            pet = connection.execute(
                "SELECT pet_id, display_name, created_at, updated_at FROM pets WHERE pet_id = ?",
                (pet_id,),
            ).fetchone()
            if pet is None:
                raise GalleryNotFound(f"pet {pet_id!r} is not enrolled")
            images = connection.execute(
                """
                SELECT image_id, original_filename, content_type, width, height,
                       byte_size, sha256, inference_json, created_at
                FROM reference_images WHERE pet_id = ? ORDER BY created_at, image_id
                """,
                (pet_id,),
            ).fetchall()
        result = dict(pet)
        result["reference_count"] = len(images)
        result["images"] = []
        for row in images:
            image = dict(row)
            inference_json = image.pop("inference_json", "{}")
            try:
                inference = json.loads(inference_json)
            except (TypeError, json.JSONDecodeError):
                inference = {}
            image["quality"] = reference_quality(inference)
            result["images"].append(image)
        return result

    def update_pet(self, pet_id: str, display_name: str) -> dict[str, Any]:
        pet_id = validate_pet_id(pet_id)
        normalized_name = validate_display_name(display_name)
        assert normalized_name is not None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE pets SET display_name = ?, updated_at = ? WHERE pet_id = ?",
                (normalized_name, utc_now(), pet_id),
            )
            if cursor.rowcount == 0:
                raise GalleryNotFound(f"pet {pet_id!r} is not enrolled")
        return self.get_pet(pet_id)

    def image_path(self, pet_id: str, image_id: str) -> tuple[Path, str, str]:
        pet_id = validate_pet_id(pet_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT stored_path, content_type, original_filename
                FROM reference_images WHERE pet_id = ? AND image_id = ?
                """,
                (pet_id, image_id),
            ).fetchone()
        if row is None:
            raise GalleryNotFound(
                f"image {image_id!r} is not enrolled for pet {pet_id!r}"
            )
        path = (self.root / row["stored_path"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("stored image path escaped the gallery root") from error
        if not path.is_file():
            raise GalleryNotFound(f"stored image file is missing for {image_id!r}")
        return path, str(row["content_type"]), str(row["original_filename"])

    def delete_image(self, pet_id: str, image_id: str) -> dict[str, Any]:
        pet_id = validate_pet_id(pet_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT stored_path FROM reference_images WHERE pet_id = ? AND image_id = ?",
                (pet_id, image_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise GalleryNotFound(
                    f"image {image_id!r} is not enrolled for pet {pet_id!r}"
                )
            connection.execute(
                "DELETE FROM reference_images WHERE pet_id = ? AND image_id = ?",
                (pet_id, image_id),
            )
            remaining = int(
                connection.execute(
                    "SELECT COUNT(*) FROM reference_images WHERE pet_id = ?", (pet_id,)
                ).fetchone()[0]
            )
            pet_deleted = remaining == 0
            if pet_deleted:
                connection.execute("DELETE FROM pets WHERE pet_id = ?", (pet_id,))
            else:
                connection.execute(
                    "UPDATE pets SET updated_at = ? WHERE pet_id = ?",
                    (utc_now(), pet_id),
                )
            connection.commit()
        (self.root / row["stored_path"]).unlink(missing_ok=True)
        return {
            "pet_id": pet_id,
            "deleted_image_id": image_id,
            "remaining_references": remaining,
            "pet_deleted": pet_deleted,
        }

    def delete_pet(self, pet_id: str) -> dict[str, Any]:
        pet_id = validate_pet_id(pet_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pet = connection.execute(
                "SELECT pet_id FROM pets WHERE pet_id = ?", (pet_id,)
            ).fetchone()
            if pet is None:
                connection.rollback()
                raise GalleryNotFound(f"pet {pet_id!r} is not enrolled")
            paths = [
                str(row["stored_path"])
                for row in connection.execute(
                    "SELECT stored_path FROM reference_images WHERE pet_id = ?",
                    (pet_id,),
                )
            ]
            connection.execute("DELETE FROM pets WHERE pet_id = ?", (pet_id,))
            connection.commit()
        for relative in paths:
            (self.root / relative).unlink(missing_ok=True)
        return {"deleted_pet_id": pet_id, "deleted_images": len(paths)}

    def prototypes(
        self, *, include_references: bool = False
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            expert_ids = [
                str(row["expert_id"])
                for row in connection.execute(
                    "SELECT expert_id FROM expert_models ORDER BY expert_id"
                )
            ]
            pets = connection.execute(
                """
                SELECT p.pet_id, p.display_name, COUNT(r.image_id) AS reference_count
                FROM pets p JOIN reference_images r ON r.pet_id = p.pet_id
                GROUP BY p.pet_id ORDER BY p.pet_id
                """
            ).fetchall()
            result: list[dict[str, Any]] = []
            for pet in pets:
                rows = connection.execute(
                    """
                    SELECT fused, fused_dim, nose, nose_dim, face, face_dim
                    FROM reference_images WHERE pet_id = ? ORDER BY image_id
                    """,
                    (pet["pet_id"],),
                ).fetchall()
                fused_references = np.stack(
                    [
                        self._unpack_feature(row["fused"], int(row["fused_dim"]))
                        for row in rows
                    ]
                )
                nose_references = np.stack(
                    [
                        self._unpack_feature(row["nose"], int(row["nose_dim"]))
                        for row in rows
                    ]
                )
                face_references = np.stack(
                    [
                        self._unpack_feature(row["face"], int(row["face_dim"]))
                        for row in rows
                    ]
                )
                prototype = {
                    "pet_id": str(pet["pet_id"]),
                    "display_name": str(pet["display_name"]),
                    "reference_count": int(pet["reference_count"]),
                    "prototype": normalize_feature(
                        fused_references.mean(axis=0), "prototype"
                    ),
                    "nose_prototype": normalize_feature(
                        nose_references.mean(axis=0), "nose_prototype"
                    ),
                    "face_prototype": normalize_feature(
                        face_references.mean(axis=0), "face_prototype"
                    ),
                    "expert_prototypes": {},
                }
                if include_references:
                    # Keep the per-image evidence available to the scorer while
                    # leaving the historical prototype payload unchanged by
                    # default.  The arrays never cross the API boundary.
                    prototype["reference_features"] = np.ascontiguousarray(
                        fused_references, dtype=np.float32
                    )
                result.append(prototype)
                for expert_id in expert_ids:
                    expert_rows = connection.execute(
                        """
                        SELECT ef.feature, ef.feature_dim
                        FROM expert_features ef
                        JOIN reference_images ri ON ri.image_id = ef.image_id
                        WHERE ef.expert_id = ? AND ri.pet_id = ?
                        ORDER BY ri.image_id
                        """,
                        (expert_id, pet["pet_id"]),
                    ).fetchall()
                    if len(expert_rows) != len(rows):
                        raise GalleryModelMismatch(
                            f"expert {expert_id!r} is incomplete for pet {pet['pet_id']!r}",
                            details={
                                "expert_id": expert_id,
                                "pet_id": str(pet["pet_id"]),
                                "expected": len(rows),
                                "actual": len(expert_rows),
                            },
                        )
                    expert_references = np.stack(
                        [
                            self._unpack_feature(
                                row["feature"], int(row["feature_dim"])
                            )
                            for row in expert_rows
                        ]
                    )
                    result[-1]["expert_prototypes"][expert_id] = normalize_feature(
                        expert_references.mean(axis=0),
                        f"expert_prototype:{expert_id}",
                    )
        return result

    def summary(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            pet_count = int(
                connection.execute("SELECT COUNT(*) FROM pets").fetchone()[0]
            )
            reference_count = int(
                connection.execute("SELECT COUNT(*) FROM reference_images").fetchone()[
                    0
                ]
            )
        return {
            "root": str(self.root),
            "database": str(self.database_path),
            "pets": pet_count,
            "reference_images": reference_count,
            "experts": sorted(self.expert_models()),
        }


class PetIdentificationService:
    """High-level enrollment and reference-set identification operations."""

    def __init__(
        self,
        store: PetGalleryStore,
        encoder: PetFeatureEncoder,
        *,
        model_fingerprint: str | None = None,
        maximum_upload_bytes: int = 15 * 1024 * 1024,
        maximum_image_pixels: int = 25_000_000,
        maximum_images_per_request: int = 8,
        maximum_batch_images: int = 1000,
        maximum_batch_bytes: int = 512 * 1024 * 1024,
        maximum_backup_bytes: int = 1024 * 1024 * 1024,
        require_single_pet_for_enrollment: bool = True,
        default_match_threshold: float | None = None,
        default_minimum_margin: float = 0.0,
        default_scoring_mode: str = CENTROID_SCORING,
        reference_top_k: int = DEFAULT_REFERENCE_TOP_K,
        reference_score_weight: float = DEFAULT_REFERENCE_SCORE_WEIGHT,
        reference_matcher: LearnedReferenceScorer | None = None,
    ):
        self.store = store
        self.encoder = encoder
        self.maximum_upload_bytes = int(maximum_upload_bytes)
        self.maximum_image_pixels = int(maximum_image_pixels)
        self.maximum_images_per_request = int(maximum_images_per_request)
        self.maximum_batch_images = int(maximum_batch_images)
        self.maximum_batch_bytes = int(maximum_batch_bytes)
        self.maximum_backup_bytes = int(maximum_backup_bytes)
        self.require_single_pet_for_enrollment = bool(require_single_pet_for_enrollment)
        self.default_match_threshold = default_match_threshold
        self.default_minimum_margin = float(default_minimum_margin)
        try:
            self.default_scoring_mode = validate_scoring_mode(default_scoring_mode)
            self.reference_top_k = validate_reference_top_k(reference_top_k)
            self.reference_score_weight = validate_reference_score_weight(
                reference_score_weight
            )
        except ValueError as error:
            raise InvalidGalleryRequest(str(error)) from error
        self.reference_matcher = reference_matcher
        if (
            self.default_scoring_mode == LEARNED_REFERENCE_SET_SCORING
            and self.reference_matcher is None
        ):
            raise InvalidGalleryRequest(
                "learned_reference_set scoring requires a trained reference matcher"
            )
        if self.reference_matcher is not None:
            if not callable(getattr(self.reference_matcher, "score", None)) and not callable(
                getattr(self.reference_matcher, "score_gallery", None)
            ):
                raise InvalidGalleryRequest(
                    "reference_matcher must provide score or score_gallery"
                )
        self._inference_lock = threading.Lock()
        self._backend_info = dict(encoder.backend_info())
        matcher_backend: dict[str, Any] | None = None
        if self.reference_matcher is not None:
            matcher_info = getattr(self.reference_matcher, "backend_info", None)
            if callable(matcher_info):
                try:
                    matcher_backend = matcher_info()
                except Exception as error:
                    raise InvalidGalleryRequest(
                        f"reference matcher backend_info failed: {error}"
                    ) from error
                if not isinstance(matcher_backend, dict):
                    raise InvalidGalleryRequest(
                        "reference matcher backend_info must return a mapping"
                    )
                self._backend_info["reference_matcher"] = dict(matcher_backend)
            else:
                self._backend_info["reference_matcher"] = {
                    "type": type(self.reference_matcher).__name__
                }
            encoder_dim = self._backend_info.get("embedding_dim")
            matcher_dim = (
                matcher_backend.get("descriptor_dim")
                if isinstance(matcher_backend, dict)
                else None
            )
            if matcher_dim is None and isinstance(matcher_backend, dict):
                config = matcher_backend.get("model_config")
                if isinstance(config, dict):
                    matcher_dim = config.get("descriptor_dim")
            if encoder_dim is not None and matcher_dim is not None:
                try:
                    dimensions_match = int(encoder_dim) == int(matcher_dim)
                except (TypeError, ValueError):
                    dimensions_match = False
                if not dimensions_match:
                    raise GalleryModelMismatch(
                        "reference matcher descriptor dimension does not match the encoder",
                        details={
                            "encoder_dimension": encoder_dim,
                            "matcher_dimension": matcher_dim,
                        },
                    )
        fingerprint = model_fingerprint or self._backend_info.get("model_sha256")
        if not fingerprint:
            fingerprint = self._backend_info.get("source_checkpoint_sha256")
        if not fingerprint:
            raise GalleryModelMismatch(
                "provide model_fingerprint when the encoder backend has no model hash"
            )
        self.model_fingerprint = str(fingerprint)
        self.store.bind_model(self.model_fingerprint, self._backend_info)
        expert_models = self._backend_info.get("experts")
        if isinstance(expert_models, dict):
            self.store.bind_expert_models(expert_models)
        self.operations = WorkspaceStore(self.store.root)
        self._batch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pet-reid-batch",
        )
        self._batch_futures: dict[str, Future] = {}
        self._batch_lock = threading.RLock()

    def backend_info(self) -> dict[str, Any]:
        return dict(self._backend_info)

    def _validate_payload(self, payload: UploadPayload) -> ValidatedUpload:
        return validate_upload(
            payload,
            maximum_bytes=self.maximum_upload_bytes,
            maximum_pixels=self.maximum_image_pixels,
        )

    def _encode_upload(
        self,
        upload: ValidatedUpload,
        *,
        enforce_single_pet: bool = False,
    ) -> EncodedPetImage:
        with tempfile.TemporaryDirectory(
            prefix="encode-", dir=self.store.staging_root
        ) as directory:
            path = Path(directory) / f"input{upload.suffix}"
            path.write_bytes(upload.data)
            with self._inference_lock:
                try:
                    encoded = self.encoder.encode_file(path)
                except GalleryServiceError:
                    raise
                except Exception as error:
                    raise InvalidPetImage(
                        f"pet descriptor extraction failed for {upload.filename}: {error}"
                    ) from error
        detections = encoded.metadata.get("detections")
        if enforce_single_pet and detections is not None and int(detections) != 1:
            raise InvalidPetImage(
                f"enrollment image {upload.filename} must contain exactly one detected pet",
                details={"detections": int(detections)},
            )
        metadata = dict(encoded.metadata)
        descriptor = metadata.get("descriptor")
        unified_single_graph = is_unified_single_graph_descriptor(descriptor)
        prepare_feature = (
            validate_normalized_feature
            if unified_single_graph
            else normalize_feature
        )
        return EncodedPetImage(
            fused=prepare_feature(encoded.fused, "fused"),
            nose=prepare_feature(encoded.nose, "nose"),
            face=prepare_feature(encoded.face, "face"),
            metadata=metadata,
            expert_features={
                expert_id: normalize_feature(feature, f"expert:{expert_id}")
                for expert_id, feature in encoded.expert_features.items()
            },
            expert_metadata={
                expert_id: dict(metadata)
                for expert_id, metadata in encoded.expert_metadata.items()
            },
        )

    def enroll(
        self,
        pet_id: str,
        uploads: Sequence[UploadPayload],
        *,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        pet_id = validate_pet_id(pet_id)
        display_name = validate_display_name(display_name)
        if not uploads:
            raise InvalidGalleryRequest("at least one image is required")
        if len(uploads) > self.maximum_images_per_request:
            raise InvalidGalleryRequest(
                f"at most {self.maximum_images_per_request} images may be enrolled per request"
            )
        validated: list[ValidatedUpload] = []
        seen: set[str] = set()
        request_duplicates: list[str] = []
        already_enrolled: list[str] = []
        for payload in uploads:
            upload = self._validate_payload(payload)
            if upload.sha256 in seen:
                request_duplicates.append(upload.sha256)
                continue
            seen.add(upload.sha256)
            owner = self.store.image_owner(upload.sha256)
            if owner is not None and owner != pet_id:
                raise GalleryConflict(
                    "the same image is already enrolled for another pet",
                    details={
                        "sha256": upload.sha256,
                        "existing_pet_id": owner,
                        "requested_pet_id": pet_id,
                    },
                )
            if owner == pet_id:
                already_enrolled.append(upload.sha256)
                continue
            validated.append(upload)
        records = [
            EnrollmentRecord(
                upload=upload,
                encoded=self._encode_upload(
                    upload,
                    enforce_single_pet=self.require_single_pet_for_enrollment,
                ),
            )
            for upload in validated
        ]
        result = self.store.enroll(pet_id, display_name, records)
        result["duplicate_image_ids"] = list(
            dict.fromkeys(
                result["duplicate_image_ids"] + request_duplicates + already_enrolled
            )
        )
        return result

    def identify(
        self,
        payload: UploadPayload,
        *,
        top_k: int = 5,
        match_threshold: float | None = None,
        minimum_margin: float | None = None,
        scoring_mode: str | None = None,
        reference_top_k: int | None = None,
        reference_score_weight: float | None = None,
        source: str = "single",
        batch_id: str | None = None,
        expected_pet_id: str | None = None,
        record_history: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        gallery = self.store.summary()
        try:
            if top_k < 1 or top_k > 50:
                raise InvalidGalleryRequest("top_k must be between 1 and 50")
            try:
                resolved_scoring_mode = (
                    self.default_scoring_mode
                    if scoring_mode is None
                    else validate_scoring_mode(scoring_mode)
                )
                resolved_reference_top_k = validate_reference_top_k(
                    self.reference_top_k
                    if reference_top_k is None
                    else reference_top_k
                )
                resolved_reference_score_weight = validate_reference_score_weight(
                    self.reference_score_weight
                    if reference_score_weight is None
                    else reference_score_weight
                )
            except ValueError as error:
                raise InvalidGalleryRequest(str(error)) from error
            if (
                resolved_scoring_mode == LEARNED_REFERENCE_SET_SCORING
                and self.reference_matcher is None
            ):
                raise InvalidGalleryRequest(
                    "learned_reference_set scoring requires a trained reference matcher"
                )
            if expected_pet_id is not None:
                expected_pet_id = validate_pet_id(expected_pet_id)
            upload = self._validate_payload(payload)
            encoded = self._encode_upload(upload)
            prototypes = self.store.prototypes(
                include_references=resolved_scoring_mode
                in (REFERENCE_SET_SCORING, LEARNED_REFERENCE_SET_SCORING)
            )
            if not prototypes:
                raise GalleryEmpty("enroll at least one pet image before identification")
            descriptor = encoded.metadata.get("descriptor")
            descriptor = descriptor if isinstance(descriptor, dict) else {}
            unified_single_graph = is_unified_single_graph_descriptor(descriptor)
            query = (
                validate_normalized_feature(encoded.fused, "query")
                if unified_single_graph
                else normalize_feature(encoded.fused, "query")
            )
            scores, scoring_details = score_gallery(
                query,
                prototypes,
                scoring_mode=resolved_scoring_mode,
                reference_top_k=resolved_reference_top_k,
                reference_score_weight=resolved_reference_score_weight,
                learned_scorer=self.reference_matcher,
            )
            threshold = (
                self.default_match_threshold
                if match_threshold is None
                else float(match_threshold)
            )
            required_margin = (
                self.default_minimum_margin
                if minimum_margin is None
                else float(minimum_margin)
            )
            agent_result = None
            if encoded.expert_features:
                from .recognition_agent import build_agent_decision

                expert_scores: dict[str, np.ndarray] = {}
                for expert_id, feature in sorted(encoded.expert_features.items()):
                    expert_query = normalize_feature(feature, f"query:{expert_id}")
                    try:
                        expert_scores[expert_id] = np.asarray(
                            [
                                float(
                                    expert_query
                                    @ item["expert_prototypes"][expert_id]
                                )
                                for item in prototypes
                            ],
                            dtype=np.float32,
                        )
                    except KeyError as error:
                        raise GalleryModelMismatch(
                            f"gallery is missing expert features for {expert_id!r}"
                        ) from error
                agent_result = build_agent_decision(
                    prototypes=prototypes,
                    encoded=encoded,
                    bifor_scores=scores,
                    expert_scores=expert_scores,
                    top_k=top_k,
                    requested_threshold=threshold,
                    requested_margin=required_margin,
                )
                candidates = agent_result["candidates"]
                best = agent_result["best"]
                margin = agent_result["margin"]
                accepted = bool(agent_result["accepted"])
                threshold = agent_result["agent"]["thresholds"]["match_score"]
                required_margin = agent_result["agent"]["thresholds"]["minimum_margin"]
                decision_mode = "agent_evidence"
            else:
                order = np.argsort(-scores)
                candidates = [
                    {
                        "pet_id": prototypes[int(index)]["pet_id"],
                        "display_name": prototypes[int(index)]["display_name"],
                        "score": float(scores[int(index)]),
                        "reference_count": prototypes[int(index)]["reference_count"],
                    }
                    for index in order[: min(top_k, len(order))]
                ]
                best = candidates[0]
                runner_up_score = (
                    candidates[1]["score"] if len(candidates) > 1 else None
                )
                margin = (
                    None
                    if runner_up_score is None
                    else best["score"] - runner_up_score
                )
                threshold_ok = threshold is None or best["score"] >= threshold
                margin_ok = margin is None or margin >= required_margin
                accepted = bool(threshold_ok and margin_ok)
                decision_mode = (
                    "closed_set_top1"
                    if threshold is None and required_margin <= 0
                    else "thresholded"
                )

            scoring_diagnostics = {
                "mode": resolved_scoring_mode,
                "reference_top_k": (
                    resolved_reference_top_k
                    if resolved_scoring_mode
                    in (REFERENCE_SET_SCORING, LEARNED_REFERENCE_SET_SCORING)
                    else None
                ),
                "reference_score_weight": (
                    resolved_reference_score_weight
                    if resolved_scoring_mode == REFERENCE_SET_SCORING
                    else None
                ),
                "top1": scoring_details.get(str(best["pet_id"])),
            }
            available_value = descriptor.get("branch_available")
            available = (
                [bool(value) for value in available_value[:2]]
                if isinstance(available_value, list) and len(available_value) >= 2
                else [True, True]
            )
            quality_value = descriptor.get("branch_quality")
            branch_quality = (
                [float(value) for value in quality_value[:2]]
                if isinstance(quality_value, list) and len(quality_value) >= 2
                else None
            )

            def branch_best(feature: np.ndarray, key: str) -> dict[str, Any] | None:
                branch_scores = np.asarray(
                    [float(feature @ item[key]) for item in prototypes],
                    dtype=np.float32,
                )
                index = int(np.argmax(branch_scores))
                return {
                    "pet_id": prototypes[index]["pet_id"],
                    "display_name": prototypes[index]["display_name"],
                    "score": float(branch_scores[index]),
                }

            nose_best = (
                branch_best(normalize_feature(encoded.nose, "query_nose"), "nose_prototype")
                if not unified_single_graph and available[0]
                else None
            )
            face_best = (
                branch_best(normalize_feature(encoded.face, "query_face"), "face_prototype")
                if not unified_single_graph and available[1]
                else None
            )
            branch_conflict = bool(
                nose_best
                and face_best
                and nose_best["pet_id"] != face_best["pet_id"]
            )
            diagnostics = (
                {
                    "mode": "unified_single_graph",
                    "single_graph": True,
                    "branch_available": None,
                    "branch_quality": None,
                    "branch_conflict": False,
                }
                if unified_single_graph
                else {
                    "mode": "multibranch",
                    "single_graph": False,
                    "branch_available": available,
                    "branch_quality": branch_quality,
                    "branch_top1": {"nose": nose_best, "face": face_best},
                    "branch_conflict": branch_conflict,
                }
            )
            diagnostics["scoring"] = scoring_diagnostics

            hard_case_reasons: list[str] = []
            if not accepted:
                hard_case_reasons.append("rejected")
            if margin is not None and margin < max(required_margin, 0.05):
                hard_case_reasons.append("low_margin")
            if branch_conflict:
                hard_case_reasons.append("branch_conflict")
            if agent_result is not None:
                hard_case_reasons.extend(agent_result["agent"]["reasons"])
            if not unified_single_graph and not all(available):
                hard_case_reasons.append("single_branch")
            if not unified_single_graph and branch_quality is not None and any(
                available[index] and branch_quality[index] < 0.35
                for index in range(2)
            ):
                hard_case_reasons.append("low_quality")
            if expected_pet_id is not None and best["pet_id"] != expected_pet_id:
                hard_case_reasons.append("incorrect_top1")

            latency_ms = (time.perf_counter() - started) * 1000.0
            result = {
                "decision": decision_mode,
                "accepted": accepted,
                "predicted_pet_id": best["pet_id"] if accepted else None,
                "predicted_display_name": best["display_name"] if accepted else None,
                "top1_score": best["score"],
                "margin": margin,
                "match_threshold": threshold,
                "minimum_margin": required_margin,
                "candidates": candidates,
                "scoring": scoring_diagnostics,
                "query": {
                    "filename": upload.filename,
                    "sha256": upload.sha256,
                    "width": upload.width,
                    "height": upload.height,
                    "inference": encoded.metadata,
                },
                "latency_ms": latency_ms,
                "model_fingerprint": self.model_fingerprint,
                "gallery_snapshot": {
                    "pets": gallery["pets"],
                    "reference_images": gallery["reference_images"],
                },
                "diagnostics": diagnostics,
                "hard_case_reasons": list(dict.fromkeys(hard_case_reasons)),
            }
            if agent_result is not None:
                result["agent"] = agent_result["agent"]
            if record_history:
                result["history_id"] = self.operations.record_success(
                    upload=upload,
                    result=result,
                    source=source,
                    batch_id=batch_id,
                    expected_pet_id=expected_pet_id,
                    latency_ms=latency_ms,
                    model_fingerprint=self.model_fingerprint,
                    gallery=gallery,
                )
            return result
        except GalleryServiceError as error:
            if record_history:
                try:
                    self.operations.record_failure(
                        payload=payload,
                        source=source,
                        batch_id=batch_id,
                        expected_pet_id=expected_pet_id,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        model_fingerprint=self.model_fingerprint,
                        gallery=gallery,
                        error_code=error.code,
                        error_message=str(error),
                    )
                except Exception:
                    pass
            raise
        except Exception as error:
            if record_history:
                try:
                    self.operations.record_failure(
                        payload=payload,
                        source=source,
                        batch_id=batch_id,
                        expected_pet_id=expected_pet_id,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        model_fingerprint=self.model_fingerprint,
                        gallery=gallery,
                        error_code="internal_error",
                        error_message=str(error),
                    )
                except Exception:
                    pass
            raise

    def create_batch(
        self,
        *,
        name: str,
        uploads: Sequence[UploadPayload],
        expected_pet_ids: Sequence[str | None] | None = None,
        top_k: int = 5,
        match_threshold: float | None = None,
        minimum_margin: float | None = None,
        scoring_mode: str | None = None,
        reference_top_k: int | None = None,
        reference_score_weight: float | None = None,
    ) -> dict[str, Any]:
        normalized_name = name.strip() or "批量测试"
        if len(normalized_name) > 128:
            raise InvalidGalleryRequest("batch name cannot exceed 128 characters")
        if not uploads:
            raise InvalidGalleryRequest("at least one batch image is required")
        if len(uploads) > self.maximum_batch_images:
            raise InvalidGalleryRequest(
                f"at most {self.maximum_batch_images} images may be submitted per batch"
            )
        total_bytes = sum(len(upload.data) for upload in uploads)
        if total_bytes > self.maximum_batch_bytes:
            raise InvalidGalleryRequest(
                f"batch exceeds the {self.maximum_batch_bytes} byte limit"
            )
        labels = list(expected_pet_ids or [None] * len(uploads))
        if len(labels) != len(uploads):
            raise InvalidGalleryRequest(
                "expected_pet_ids must align one-to-one with uploaded files"
            )
        normalized_labels = [
            None if label is None or not str(label).strip() else validate_pet_id(str(label))
            for label in labels
        ]
        try:
            resolved_scoring_mode = (
                self.default_scoring_mode
                if scoring_mode is None
                else validate_scoring_mode(scoring_mode)
            )
            resolved_reference_top_k = validate_reference_top_k(
                self.reference_top_k if reference_top_k is None else reference_top_k
            )
            resolved_reference_score_weight = validate_reference_score_weight(
                self.reference_score_weight
                if reference_score_weight is None
                else reference_score_weight
            )
        except ValueError as error:
            raise InvalidGalleryRequest(str(error)) from error
        if (
            resolved_scoring_mode == LEARNED_REFERENCE_SET_SCORING
            and self.reference_matcher is None
        ):
            raise InvalidGalleryRequest(
                "learned_reference_set scoring requires a trained reference matcher"
            )
        parameters = {
            "top_k": int(top_k),
            "match_threshold": match_threshold,
            "minimum_margin": minimum_margin,
            "scoring_mode": resolved_scoring_mode,
            "reference_top_k": resolved_reference_top_k,
            "reference_score_weight": resolved_reference_score_weight,
            "labelled": sum(label is not None for label in normalized_labels),
        }
        batch_id = self.operations.create_batch(
            name=normalized_name,
            total=len(uploads),
            model_fingerprint=self.model_fingerprint,
            parameters=parameters,
        )
        future = self._batch_executor.submit(
            self._run_batch,
            batch_id,
            list(uploads),
            normalized_labels,
            top_k,
            match_threshold,
            minimum_margin,
            resolved_scoring_mode,
            resolved_reference_top_k,
            resolved_reference_score_weight,
        )
        with self._batch_lock:
            self._batch_futures[batch_id] = future
            if future.done():
                self._batch_futures.pop(batch_id, None)
        return self.operations.get_batch(batch_id, include_results=False)

    def _run_batch(
        self,
        batch_id: str,
        uploads: list[UploadPayload],
        expected_pet_ids: list[str | None],
        top_k: int,
        match_threshold: float | None,
        minimum_margin: float | None,
        scoring_mode: str,
        reference_top_k: int,
        reference_score_weight: float,
    ) -> None:
        completed = succeeded = failed = 0
        labelled = top1_correct = accepted_correct = rejected = hard_cases = 0
        latencies: list[float] = []

        def metrics() -> dict[str, Any]:
            return {
                "labelled": labelled,
                "top1_correct": top1_correct,
                "top1_accuracy": None if labelled == 0 else top1_correct / labelled,
                "accepted_correct": accepted_correct,
                "accepted_accuracy": None if labelled == 0 else accepted_correct / labelled,
                "rejected": rejected,
                "hard_cases": hard_cases,
                "average_latency_ms": None
                if not latencies
                else float(np.mean(latencies)),
                "p95_latency_ms": None
                if not latencies
                else float(np.percentile(latencies, 95)),
            }

        try:
            self.operations.update_batch(batch_id, status="running", metrics=metrics())
            for payload, expected_pet_id in zip(uploads, expected_pet_ids):
                if self.operations.batch_cancel_requested(batch_id):
                    self.operations.update_batch(
                        batch_id,
                        status="cancelled",
                        completed=completed,
                        succeeded=succeeded,
                        failed=failed,
                        metrics=metrics(),
                    )
                    return
                try:
                    result = self.identify(
                        payload,
                        top_k=top_k,
                        match_threshold=match_threshold,
                        minimum_margin=minimum_margin,
                        scoring_mode=scoring_mode,
                        reference_top_k=reference_top_k,
                        reference_score_weight=reference_score_weight,
                        source="batch",
                        batch_id=batch_id,
                        expected_pet_id=expected_pet_id,
                    )
                except Exception:
                    failed += 1
                else:
                    succeeded += 1
                    latencies.append(float(result["latency_ms"]))
                    if not result["accepted"]:
                        rejected += 1
                    if result.get("hard_case_reasons"):
                        hard_cases += 1
                    if expected_pet_id is not None:
                        labelled += 1
                        top1_pet_id = (
                            result["candidates"][0]["pet_id"]
                            if result.get("candidates")
                            else None
                        )
                        if top1_pet_id == expected_pet_id:
                            top1_correct += 1
                        if result.get("predicted_pet_id") == expected_pet_id:
                            accepted_correct += 1
                completed += 1
                self.operations.update_batch(
                    batch_id,
                    completed=completed,
                    succeeded=succeeded,
                    failed=failed,
                    metrics=metrics(),
                )
            self.operations.update_batch(
                batch_id,
                status="completed",
                completed=completed,
                succeeded=succeeded,
                failed=failed,
                metrics=metrics(),
            )
        except Exception as error:
            self.operations.update_batch(
                batch_id,
                status="failed",
                completed=completed,
                succeeded=succeeded,
                failed=failed,
                metrics=metrics(),
                error_message=str(error),
            )
        finally:
            with self._batch_lock:
                self._batch_futures.pop(batch_id, None)

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        self.operations.request_batch_cancel(batch_id)
        with self._batch_lock:
            future = self._batch_futures.get(batch_id)
        if future is not None and future.cancel():
            self.operations.update_batch(batch_id, status="cancelled")
            with self._batch_lock:
                self._batch_futures.pop(batch_id, None)
        return self.operations.get_batch(batch_id, include_results=False)

    def create_gallery_backup(self) -> tuple[str, bytes]:
        manifest: dict[str, Any] = {
            "format": "pet-reid-gallery-backup",
            "version": 1,
            "created_at": utc_now(),
            "model_fingerprint": self.model_fingerprint,
            "pets": [],
        }
        output = BytesIO()
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for summary in self.store.list_pets():
                pet = self.store.get_pet(summary["pet_id"])
                pet_manifest = {
                    "pet_id": pet["pet_id"],
                    "display_name": pet["display_name"],
                    "images": [],
                }
                for image in pet["images"]:
                    path, content_type, original_filename = self.store.image_path(
                        pet["pet_id"], image["image_id"]
                    )
                    archive_path = (
                        f"images/{pet['pet_id']}/{image['image_id']}{path.suffix.casefold()}"
                    )
                    archive.write(path, archive_path)
                    pet_manifest["images"].append(
                        {
                            "path": archive_path,
                            "filename": original_filename,
                            "content_type": content_type,
                            "sha256": image["sha256"],
                        }
                    )
                manifest["pets"].append(pet_manifest)
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
        filename = f"pet-reid-gallery-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
        return filename, output.getvalue()

    def restore_gallery_backup(self, data: bytes) -> dict[str, Any]:
        if not data:
            raise InvalidGalleryRequest("gallery backup is empty")
        if len(data) > self.maximum_backup_bytes:
            raise InvalidGalleryRequest(
                f"gallery backup exceeds the {self.maximum_backup_bytes} byte limit"
            )
        try:
            archive = zipfile.ZipFile(BytesIO(data), mode="r")
        except (OSError, zipfile.BadZipFile) as error:
            raise InvalidGalleryRequest("gallery backup is not a valid ZIP archive") from error
        with archive:
            infos = archive.infolist()
            if len(infos) > 10001:
                raise InvalidGalleryRequest("gallery backup contains too many files")
            if sum(info.file_size for info in infos) > self.maximum_backup_bytes:
                raise InvalidGalleryRequest("gallery backup expands beyond the size limit")
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
                raise InvalidGalleryRequest("gallery backup manifest is missing or invalid") from error
            if manifest.get("format") != "pet-reid-gallery-backup" or manifest.get("version") != 1:
                raise InvalidGalleryRequest("unsupported gallery backup format")
            if manifest.get("model_fingerprint") != self.model_fingerprint:
                raise GalleryModelMismatch(
                    "gallery backup was created by a different embedding model",
                    details={
                        "backup": manifest.get("model_fingerprint"),
                        "service": self.model_fingerprint,
                    },
                )
            added = duplicates = restored_pets = 0
            for pet in manifest.get("pets", []):
                pet_id = validate_pet_id(str(pet.get("pet_id", "")))
                display_name = validate_display_name(pet.get("display_name")) or pet_id
                payloads: list[UploadPayload] = []
                for image in pet.get("images", []):
                    archive_path = str(image.get("path", ""))
                    if not archive_path.startswith(f"images/{pet_id}/"):
                        raise InvalidGalleryRequest("gallery backup contains an invalid image path")
                    try:
                        image_data = archive.read(archive_path)
                    except KeyError as error:
                        raise InvalidGalleryRequest(
                            f"gallery backup image is missing: {archive_path}"
                        ) from error
                    expected_hash = str(image.get("sha256", ""))
                    actual_hash = hashlib.sha256(image_data).hexdigest()
                    if expected_hash and expected_hash != actual_hash:
                        raise GalleryConflict(
                            "gallery backup image hash mismatch",
                            details={"expected": expected_hash, "actual": actual_hash},
                        )
                    payloads.append(
                        UploadPayload(
                            filename=str(image.get("filename") or Path(archive_path).name),
                            content_type=image.get("content_type"),
                            data=image_data,
                        )
                    )
                if not payloads:
                    continue
                restored_pets += 1
                for offset in range(0, len(payloads), self.maximum_images_per_request):
                    result = self.enroll(
                        pet_id,
                        payloads[offset : offset + self.maximum_images_per_request],
                        display_name=display_name,
                    )
                    added += len(result["added_image_ids"])
                    duplicates += len(result["duplicate_image_ids"])
        return {
            "pets": restored_pets,
            "added_images": added,
            "duplicate_images": duplicates,
            "mode": "merge",
        }

    def import_gallery_model(self, model_json: str | Path) -> dict[str, Any]:
        model_path = Path(model_json).expanduser().resolve()
        metadata, arrays = load_gallery_model(model_path)
        selected_backend = metadata.get("selected_backend") or {}
        recorded_fingerprint = selected_backend.get("model_sha256")
        if recorded_fingerprint and recorded_fingerprint != self.model_fingerprint:
            raise GalleryModelMismatch(
                "seed gallery was created by a different ONNX model",
                details={
                    "seed": recorded_fingerprint,
                    "service": self.model_fingerprint,
                },
            )
        required = (
            "selected_fused_references",
            "selected_nose_references",
            "selected_face_references",
            "reference_identity_indices",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise InvalidGalleryRequest(
                f"seed gallery is missing production arrays: {missing}"
            )
        identities = list(metadata["identities"])
        references = list(metadata["references"])
        identity_indices = arrays["reference_identity_indices"].astype(np.int64)
        if len(references) != len(identity_indices):
            raise InvalidGalleryRequest("seed gallery reference metadata is misaligned")
        added = 0
        duplicates = 0
        for index, reference in enumerate(references):
            identity = identities[int(identity_indices[index])]
            source = resolve_legacy_path(reference["path"])
            if not source.is_file():
                raise GalleryNotFound(f"seed gallery image is missing: {source}")
            payload = UploadPayload(
                filename=source.name,
                content_type=None,
                data=source.read_bytes(),
            )
            upload = self._validate_payload(payload)
            expected_hash = reference.get("sha256")
            if expected_hash and upload.sha256 != expected_hash:
                raise GalleryConflict(
                    f"seed gallery image hash mismatch: {source}",
                    details={"expected": expected_hash, "actual": upload.sha256},
                )
            encoded = EncodedPetImage(
                fused=arrays["selected_fused_references"][index],
                nose=arrays["selected_nose_references"][index],
                face=arrays["selected_face_references"][index],
                metadata=reference.get("selected_inference", {}),
            )
            result = self.store.enroll(
                identity,
                identity,
                [EnrollmentRecord(upload=upload, encoded=encoded)],
            )
            added += len(result["added_image_ids"])
            duplicates += len(result["duplicate_image_ids"])
        return {
            "source_gallery_model": str(model_path),
            "identities": len(identities),
            "references": len(references),
            "added": added,
            "duplicates": duplicates,
        }

    def health(self) -> dict[str, Any]:
        public_backend = self.backend_info()
        for key in ("model", "metadata", "source_checkpoint"):
            public_backend.pop(key, None)
        for expert in (public_backend.get("experts") or {}).values():
            if isinstance(expert, dict):
                expert.pop("checkpoint", None)
        gallery = self.store.summary()
        operations = self.operations.summary()
        return {
            "status": "ok",
            "model_fingerprint": self.model_fingerprint,
            "backend": public_backend,
            "gallery": {
                "pets": gallery["pets"],
                "reference_images": gallery["reference_images"],
            },
            "operations": operations,
        }
