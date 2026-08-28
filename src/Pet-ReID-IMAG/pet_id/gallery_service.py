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
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from .gallery import encode_primary, load_gallery_model, normalized_array
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


class MultimodalPipelineEncoder:
    """Adapt the existing multimodal pipeline to the gallery service contract."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def encode_file(self, path: Path) -> EncodedPetImage:
        descriptor, inference = encode_primary(self.pipeline, path)
        return EncodedPetImage(
            fused=normalized_array(descriptor.fused_feature),
            nose=normalized_array(descriptor.nose_feature),
            face=normalized_array(descriptor.face_feature),
            metadata=inference,
        )

    def backend_info(self) -> dict[str, Any]:
        identity_model = self.pipeline.identity_model
        if hasattr(identity_model, "backend_info"):
            return dict(identity_model.backend_info())
        return {"backend": "pytorch", "device": str(self.pipeline.device)}


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
    def _pack_feature(feature: np.ndarray, name: str) -> tuple[int, bytes]:
        value = normalize_feature(feature, name)
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

                    fused_dim, fused_blob = self._pack_feature(
                        record.encoded.fused, "fused"
                    )
                    nose_dim, nose_blob = self._pack_feature(
                        record.encoded.nose, "nose"
                    )
                    face_dim, face_blob = self._pack_feature(
                        record.encoded.face, "face"
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

    def prototypes(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
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
                    FROM reference_images WHERE pet_id = ?
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
                result.append(
                    {
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
                    }
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
        }


class PetIdentificationService:
    """High-level enrollment and cosine-prototype identification operations."""

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
        self._inference_lock = threading.Lock()
        self._backend_info = dict(encoder.backend_info())
        fingerprint = model_fingerprint or self._backend_info.get("model_sha256")
        if not fingerprint:
            fingerprint = self._backend_info.get("source_checkpoint_sha256")
        if not fingerprint:
            raise GalleryModelMismatch(
                "provide model_fingerprint when the encoder backend has no model hash"
            )
        self.model_fingerprint = str(fingerprint)
        self.store.bind_model(self.model_fingerprint, self._backend_info)
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
        return EncodedPetImage(
            fused=normalize_feature(encoded.fused, "fused"),
            nose=normalize_feature(encoded.nose, "nose"),
            face=normalize_feature(encoded.face, "face"),
            metadata=dict(encoded.metadata),
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
            if expected_pet_id is not None:
                expected_pet_id = validate_pet_id(expected_pet_id)
            upload = self._validate_payload(payload)
            encoded = self._encode_upload(upload)
            prototypes = self.store.prototypes()
            if not prototypes:
                raise GalleryEmpty("enroll at least one pet image before identification")
            query = normalize_feature(encoded.fused, "query")
            scores = np.asarray(
                [float(query @ item["prototype"]) for item in prototypes],
                dtype=np.float32,
            )
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
            runner_up_score = candidates[1]["score"] if len(candidates) > 1 else None
            margin = None if runner_up_score is None else best["score"] - runner_up_score
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
            threshold_ok = threshold is None or best["score"] >= threshold
            margin_ok = margin is None or margin >= required_margin
            accepted = bool(threshold_ok and margin_ok)
            decision_mode = (
                "closed_set_top1"
                if threshold is None and required_margin <= 0
                else "thresholded"
            )

            descriptor = encoded.metadata.get("descriptor")
            descriptor = descriptor if isinstance(descriptor, dict) else {}
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
                if available[0]
                else None
            )
            face_best = (
                branch_best(normalize_feature(encoded.face, "query_face"), "face_prototype")
                if available[1]
                else None
            )
            branch_conflict = bool(
                nose_best
                and face_best
                and nose_best["pet_id"] != face_best["pet_id"]
            )
            diagnostics = {
                "branch_available": available,
                "branch_quality": branch_quality,
                "branch_top1": {"nose": nose_best, "face": face_best},
                "branch_conflict": branch_conflict,
            }

            hard_case_reasons: list[str] = []
            if not accepted:
                hard_case_reasons.append("rejected")
            if margin is not None and margin < max(required_margin, 0.05):
                hard_case_reasons.append("low_margin")
            if branch_conflict:
                hard_case_reasons.append("branch_conflict")
            if not all(available):
                hard_case_reasons.append("single_branch")
            if branch_quality is not None and any(
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
        parameters = {
            "top_k": int(top_k),
            "match_threshold": match_threshold,
            "minimum_margin": minimum_margin,
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
        result = self.operations.request_batch_cancel(batch_id)
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
