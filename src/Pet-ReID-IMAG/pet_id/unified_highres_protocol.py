"""Deterministic real-high-resolution protocol for spatial-detail training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .dogfacenet_alignment import AlignmentIndexRecord
from .unified_fresh_protocol import sha256_file, stable_order_key
from .release_compatibility import high_resolution_protocol_name


PROTOCOL_NAME = high_resolution_protocol_name()
SPLIT_NAMES = ("training_extension", "development", "blind_test")


def _rank_record(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["eye_distance"]),
        -int(row["max_side"]),
        str(row["canonical_filename"]).casefold(),
        str(row["source_path"]).casefold(),
    )


def _source_payload(record: AlignmentIndexRecord) -> dict[str, Any]:
    source = record.source_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        width, height = image.size
    return {
        "schema_version": 1,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "canonical_filename": record.canonical_filename,
        "identity": record.identity,
        "width": int(width),
        "height": int(height),
        "max_side": int(max(width, height)),
        "eye_distance": float(record.eye_distance),
        "left_eye": list(record.left_eye),
        "right_eye": list(record.right_eye),
        "nose": list(record.nose),
    }


def build_highres_protocol(
    alignment_records: Sequence[AlignmentIndexRecord],
    *,
    training_identities: int = 20,
    development_identities: int = 8,
    blind_identities: int = 8,
    images_per_identity: int = 4,
    minimum_max_side: int = 1280,
    minimum_eye_distance: float = 96.0,
    seed: int = 20260901,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Create identity- and exact-image-disjoint high-resolution splits.

    This protocol evaluates the incremental detail branch rather than claiming
    that the locked parent has never seen the source identities. The candidate's
    training, development, and blind identities are fully disjoint, and it may
    only consume ``training_extension`` before it is frozen.
    """

    counts = {
        "training_extension": int(training_identities),
        "development": int(development_identities),
        "blind_test": int(blind_identities),
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError("Every split needs at least one identity")
    images_per_identity = int(images_per_identity)
    minimum_max_side = int(minimum_max_side)
    minimum_eye_distance = float(minimum_eye_distance)
    if images_per_identity < 3:
        raise ValueError("At least three images per identity are required")
    if minimum_max_side < 1280:
        raise ValueError("minimum_max_side must be at least 1280")
    if minimum_eye_distance <= 0.0:
        raise ValueError("minimum_eye_distance must be positive")

    filtered: list[dict[str, Any]] = []
    source_resolution_rejected = 0
    face_resolution_rejected = 0
    for record in alignment_records:
        source = record.source_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            width, height = image.size
        if max(width, height) < minimum_max_side:
            source_resolution_rejected += 1
            continue
        if record.eye_distance < minimum_eye_distance:
            face_resolution_rejected += 1
            continue
        filtered.append(_source_payload(record))

    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        by_digest[str(row["source_sha256"])].append(row)
    conflicting_identities: set[str] = set()
    cross_identity_duplicates: list[dict[str, Any]] = []
    for digest, rows in by_digest.items():
        identities = sorted({str(row["identity"]).casefold() for row in rows})
        if len(identities) > 1:
            conflicting_identities.update(identities)
            cross_identity_duplicates.append(
                {
                    "sha256": digest,
                    "identities": identities,
                    "source_paths": sorted(str(row["source_path"]) for row in rows),
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    within_identity_duplicates: list[dict[str, Any]] = []
    for digest, rows in by_digest.items():
        identity = str(rows[0]["identity"]).casefold()
        if identity in conflicting_identities:
            continue
        ranked = sorted(rows, key=_rank_record)
        groups[identity].append(ranked[0])
        for duplicate in ranked[1:]:
            within_identity_duplicates.append(
                {
                    "identity": identity,
                    "sha256": digest,
                    "kept_path": str(ranked[0]["source_path"]),
                    "excluded_path": str(duplicate["source_path"]),
                }
            )
    groups = {
        identity: sorted(rows, key=_rank_record)
        for identity, rows in groups.items()
    }
    eligible = sorted(
        (
            identity
            for identity, rows in groups.items()
            if len(rows) >= images_per_identity
        ),
        key=lambda identity: stable_order_key(
            seed, "spatial-detail-highres-identity", identity
        ),
    )
    requested = sum(counts.values())
    if len(eligible) < requested:
        raise ValueError(
            f"Only {len(eligible)} high-resolution identities remain after "
            f"deduplication; {requested} are required"
        )

    selected: dict[str, list[str]] = {}
    cursor = 0
    for split in SPLIT_NAMES:
        next_cursor = cursor + counts[split]
        selected[split] = eligible[cursor:next_cursor]
        cursor = next_cursor

    manifests: dict[str, dict[str, Any]] = {}
    for split, identities in selected.items():
        records: list[dict[str, Any]] = []
        for identity in identities:
            records.extend(groups[identity][:images_per_identity])
        manifests[split] = {
            "schema_version": 1,
            "protocol_name": PROTOCOL_NAME,
            "protocol_split": split,
            "protocol_seed": int(seed),
            "selection": {
                "minimum_max_side": minimum_max_side,
                "minimum_eye_distance": minimum_eye_distance,
                "identity_source": "canonical DogFaceNet labels.csv filename",
                "ranking": "eye_distance_desc,max_side_desc,canonical_filename",
            },
            "images_per_identity": images_per_identity,
            "records": records,
        }

    identity_sets = {split: set(values) for split, values in selected.items()}
    digest_sets = {
        split: {str(row["source_sha256"]) for row in manifest["records"]}
        for split, manifest in manifests.items()
    }
    pairwise: dict[str, dict[str, list[str]]] = {}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "identity_overlap": sorted(identity_sets[left] & identity_sets[right]),
                "source_sha256_overlap": sorted(digest_sets[left] & digest_sets[right]),
            }
    audit = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "seed": int(seed),
        "source_records": len(alignment_records),
        "source_resolution_rejected": source_resolution_rejected,
        "face_resolution_rejected_after_source_filter": face_resolution_rejected,
        "filtered_records": len(filtered),
        "filtered_identities": len(
            {str(row["identity"]).casefold() for row in filtered}
        ),
        "criteria": {
            "minimum_max_side": minimum_max_side,
            "minimum_eye_distance": minimum_eye_distance,
            "images_per_identity": images_per_identity,
        },
        "cross_identity_conflicting_identities_excluded": sorted(
            conflicting_identities
        ),
        "cross_identity_exact_duplicates": sorted(
            cross_identity_duplicates,
            key=lambda row: row["sha256"],
        ),
        "within_identity_exact_duplicates_excluded": sorted(
            within_identity_duplicates,
            key=lambda row: (row["identity"], row["sha256"]),
        ),
        "eligible_identities": len(eligible),
        "reserve_identities": eligible[requested:],
        "splits": {
            split: {
                "identities": len(selected[split]),
                "records": len(manifests[split]["records"]),
                "identity_names": sorted(selected[split]),
            }
            for split in SPLIT_NAMES
        },
        "pairwise_disjointness": pairwise,
        "parent_identity_exposure_not_claimed": True,
        "candidate_split_identity_disjoint": True,
        "candidate_split_exact_image_disjoint": True,
    }
    return manifests, audit


def manifest_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    records = list(payload["records"])
    identities = {str(row["identity"]).casefold() for row in records}
    return {"records": len(records), "identities": len(identities)}
