"""Atomic gallery re-encoding between incompatible embedding models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .gallery_service import (
    MultimodalPipelineEncoder,
    PetFeatureEncoder,
    PetGalleryStore,
    PetIdentificationService,
    UploadPayload,
    utc_now,
)


class GalleryMigrationError(RuntimeError):
    """Raised when a gallery cannot be published without losing integrity."""


def _read_backend_metadata(store: PetGalleryStore) -> dict[str, Any]:
    metadata = store.metadata()
    raw_backend = metadata.get("backend_info")
    try:
        backend = json.loads(raw_backend) if raw_backend else {}
    except json.JSONDecodeError:
        backend = {"unparsed": raw_backend}
    return {
        "model_fingerprint": metadata.get("model_fingerprint"),
        "backend": backend,
    }


def _inventory(store: PetGalleryStore, *, verify_files: bool) -> dict[str, Any]:
    pets: dict[str, Any] = {}
    verified_files = 0
    for summary in store.list_pets():
        pet = store.get_pet(summary["pet_id"])
        hashes: list[str] = []
        for image in pet["images"]:
            image_hash = str(image["sha256"])
            hashes.append(image_hash)
            if verify_files:
                path, _, _ = store.image_path(pet["pet_id"], image["image_id"])
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != image_hash:
                    raise GalleryMigrationError(
                        f"gallery image hash mismatch for {path}: "
                        f"expected {image_hash}, got {actual}"
                    )
                verified_files += 1
        pets[str(pet["pet_id"])] = {
            "display_name": str(pet["display_name"]),
            "reference_hashes": sorted(hashes),
        }
    return {
        "pets": pets,
        "pet_count": len(pets),
        "reference_count": sum(
            len(item["reference_hashes"]) for item in pets.values()
        ),
        "verified_files": verified_files,
    }


def _safe_migration_paths(
    source_root: str | Path, target_root: str | Path
) -> tuple[Path, Path]:
    source = Path(source_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    if not (source / "gallery.sqlite3").is_file():
        raise GalleryMigrationError(f"source gallery database is missing: {source}")
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise GalleryMigrationError(
            "source and target galleries must be separate, non-nested directories"
        )
    if target.exists():
        raise GalleryMigrationError(
            f"target gallery already exists; refusing to overwrite it: {target}"
        )
    return source, target


def migrate_gallery(
    source_root: str | Path,
    target_root: str | Path,
    encoder: PetFeatureEncoder,
    *,
    require_single_pet: bool = False,
) -> dict[str, Any]:
    """Re-encode every source reference and atomically publish a new gallery.

    The source is only read. The target path must not exist. Work is performed in
    a sibling staging directory, which is removed on failure and renamed to the
    requested target only after model, identity and image inventories all match.
    """

    source, target = _safe_migration_paths(source_root, target_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.migration-{uuid.uuid4().hex}.tmp")
    if staging.exists():
        raise GalleryMigrationError(f"migration staging path already exists: {staging}")

    started_at = utc_now()
    try:
        source_store = PetGalleryStore(source)
        source_inventory = _inventory(source_store, verify_files=True)
        source_database_sha256 = hashlib.sha256(
            source_store.database_path.read_bytes()
        ).hexdigest()
        if not source_inventory["pet_count"] or not source_inventory["reference_count"]:
            raise GalleryMigrationError("source gallery is empty")
        source_model = _read_backend_metadata(source_store)
        if not source_model["model_fingerprint"]:
            raise GalleryMigrationError("source gallery is not bound to a model fingerprint")

        target_store = PetGalleryStore(staging)
        target_service = PetIdentificationService(
            target_store,
            encoder,
            require_single_pet_for_enrollment=require_single_pet,
        )
        migrated_pets: list[dict[str, Any]] = []
        for summary in source_store.list_pets():
            pet = source_store.get_pet(summary["pet_id"])
            added: list[str] = []
            for image in pet["images"]:
                path, content_type, original_filename = source_store.image_path(
                    pet["pet_id"], image["image_id"]
                )
                data = path.read_bytes()
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_hash != image["sha256"]:
                    raise GalleryMigrationError(
                        f"source image changed during migration: {path}"
                    )
                result = target_service.enroll(
                    pet["pet_id"],
                    [
                        UploadPayload(
                            filename=original_filename,
                            content_type=content_type,
                            data=data,
                        )
                    ],
                    display_name=pet["display_name"],
                )
                if result["duplicate_image_ids"]:
                    raise GalleryMigrationError(
                        "unexpected duplicate while constructing an empty target gallery"
                    )
                added.extend(result["added_image_ids"])
            migrated_pets.append(
                {
                    "pet_id": pet["pet_id"],
                    "display_name": pet["display_name"],
                    "references": len(added),
                }
            )

        target_inventory = _inventory(target_store, verify_files=True)
        if target_inventory["pets"] != source_inventory["pets"]:
            raise GalleryMigrationError(
                "target identity, display-name or reference-hash inventory does not "
                "match the source"
            )
        source_inventory_after = _inventory(source_store, verify_files=True)
        source_database_sha256_after = hashlib.sha256(
            source_store.database_path.read_bytes()
        ).hexdigest()
        if source_inventory_after["pets"] != source_inventory["pets"]:
            raise GalleryMigrationError("source gallery inventory changed during migration")
        if source_database_sha256_after != source_database_sha256:
            raise GalleryMigrationError("source gallery database changed during migration")
        target_model = _read_backend_metadata(target_store)
        expected_fingerprint = target_service.model_fingerprint
        if target_model["model_fingerprint"] != expected_fingerprint:
            raise GalleryMigrationError(
                "target gallery model fingerprint was not persisted correctly"
            )

        prototypes = target_store.prototypes()
        prototype_dims = sorted({int(item["prototype"].size) for item in prototypes})
        if prototype_dims != [512]:
            raise GalleryMigrationError(
                f"target gallery has unexpected fused dimensions: {prototype_dims}"
            )
        expert_dimensions = {
            expert_id: sorted(
                {
                    int(item["expert_prototypes"][expert_id].size)
                    for item in prototypes
                }
            )
            for expert_id in target_store.expert_models()
        }
        with target_store._connect() as connection:
            expert_feature_counts = {
                str(row["expert_id"]): int(row["feature_count"])
                for row in connection.execute(
                    """
                    SELECT expert_id, COUNT(*) AS feature_count
                    FROM expert_features GROUP BY expert_id ORDER BY expert_id
                    """
                )
            }
        if any(
            count != target_inventory["reference_count"]
            for count in expert_feature_counts.values()
        ):
            raise GalleryMigrationError(
                "one or more expert namespaces are incomplete after migration"
            )

        report = {
            "schema_version": 1,
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "source": {
                "root": str(source),
                **source_model,
                "database_sha256": source_database_sha256,
                "pets": source_inventory["pet_count"],
                "reference_images": source_inventory["reference_count"],
            },
            "target": {
                "root": str(target),
                **target_model,
                "pets": target_inventory["pet_count"],
                "reference_images": target_inventory["reference_count"],
                "prototype_dimensions": prototype_dims,
                "expert_dimensions": expert_dimensions,
                "expert_feature_counts": expert_feature_counts,
            },
            "verification": {
                "source_files_sha256_verified": source_inventory["verified_files"],
                "target_files_sha256_verified": target_inventory["verified_files"],
                "pet_ids_match": True,
                "display_names_match": True,
                "reference_hashes_match": True,
                "model_fingerprint_persisted": True,
                "expert_namespaces_complete": all(
                    count == target_inventory["reference_count"]
                    for count in expert_feature_counts.values()
                ),
                "source_inventory_unchanged": True,
                "source_database_sha256_unchanged": True,
                "atomic_publish": True,
            },
            "pets": migrated_pets,
        }
        (staging / "migration_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staging, target)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def migrate_pipeline_gallery(
    source_root: str | Path,
    target_root: str | Path,
    pipeline,
    *,
    require_single_pet: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for the production multimodal pipeline."""

    return migrate_gallery(
        source_root,
        target_root,
        MultimodalPipelineEncoder(pipeline),
        require_single_pet=require_single_pet,
    )
