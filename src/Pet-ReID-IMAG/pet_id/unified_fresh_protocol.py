"""Deterministic fresh-data protocol construction for UnifiedPetReID.

The DogFaceNet alignment release stores the individual identity in the
canonical ``labels.csv`` filename.  This module deliberately consumes
``AlignmentIndexRecord.identity`` (which is derived from that canonical
annotation) instead of attempting to infer identities from the sometimes
mojibake-renamed extracted image files.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dogfacenet_alignment import AlignmentIndexRecord


SPLIT_NAMES = ("training_extension", "development", "blind_test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order_key(seed: int, scope: str, value: str) -> tuple[str, str]:
    normalized = str(value).casefold()
    payload = f"{int(seed)}:{scope}:{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), normalized


def discover_history_manifests(root: str | Path) -> list[Path]:
    """Return only protocol-like JSON files, never per-image cache records."""

    root = Path(root).expanduser().resolve()
    paths = set(root.rglob("*manifest*.json"))
    paths.update(root.rglob("protocol.json"))
    return sorted(paths, key=lambda path: str(path).casefold())


def collect_historical_identities(
    manifest_paths: Iterable[str | Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Collect every top-level record identity from historical protocols.

    Files without a top-level record list are ignored and reported.  Invalid
    JSON is a hard error: silently skipping a damaged historical protocol
    could leak one of its identities into the new blind split.
    """

    identities: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for raw_path in manifest_paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records") if isinstance(payload, dict) else None
        row_identities: set[str] = set()
        if isinstance(rows, list):
            row_identities = {
                str(row["identity"]).casefold()
                for row in rows
                if isinstance(row, dict) and row.get("identity")
            }
            identities.update(row_identities)
        evidence.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "top_level_record_identities": len(row_identities),
                "has_top_level_records": isinstance(rows, list),
            }
        )
    return identities, evidence


def _record_payload(
    record: AlignmentIndexRecord, source_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_path": str(record.source_path.resolve()),
        "source_sha256": source_sha256,
        "canonical_filename": record.canonical_filename,
        "identity": record.identity,
        "left_eye": list(record.left_eye),
        "right_eye": list(record.right_eye),
        "nose": list(record.nose),
        "eye_distance": float(record.eye_distance),
    }


def _rank_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            -float(row["eye_distance"]),
            str(row["canonical_filename"]).casefold(),
            str(row["source_path"]).casefold(),
        ),
    )


def build_fresh_protocol(
    alignment_records: Sequence[AlignmentIndexRecord],
    *,
    historical_identities: set[str],
    training_identities: int,
    development_identities: int,
    blind_identities: int,
    minimum_images_per_identity: int = 4,
    evaluation_images_per_identity: int = 4,
    seed: int = 20260831,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build an exact-deduplicated, identity-disjoint fresh protocol.

    Every resolved source image is hashed before the split is chosen.  Any
    identity participating in a cross-identity exact duplicate is excluded
    completely.  Within-identity duplicates retain one deterministic record.
    """

    counts = {
        "training_extension": int(training_identities),
        "development": int(development_identities),
        "blind_test": int(blind_identities),
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError("Every split needs at least one identity")
    minimum = int(minimum_images_per_identity)
    evaluation_count = int(evaluation_images_per_identity)
    if minimum < evaluation_count or evaluation_count < 3:
        raise ValueError(
            "minimum_images_per_identity must be at least the evaluation "
            "image count, which must leave a held-out query"
        )

    digest_cache: dict[str, str] = {}
    by_digest: dict[str, list[tuple[AlignmentIndexRecord, str]]] = defaultdict(list)
    for record in alignment_records:
        resolved = record.source_path.expanduser().resolve()
        cache_key = str(resolved).casefold()
        digest = digest_cache.get(cache_key)
        if digest is None:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            digest = sha256_file(resolved)
            digest_cache[cache_key] = digest
        by_digest[digest].append((record, digest))

    conflicting_identities: set[str] = set()
    cross_identity_duplicates: list[dict[str, Any]] = []
    for digest, rows in by_digest.items():
        names = sorted({record.identity.casefold() for record, _ in rows})
        if len(names) > 1:
            conflicting_identities.update(names)
            cross_identity_duplicates.append(
                {
                    "sha256": digest,
                    "identities": names,
                    "source_paths": sorted(
                        str(record.source_path.resolve()) for record, _ in rows
                    ),
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    within_identity_duplicates: list[dict[str, Any]] = []
    for digest, rows in by_digest.items():
        identity = rows[0][0].identity.casefold()
        if identity in conflicting_identities:
            continue
        ranked = sorted(
            rows,
            key=lambda item: (
                -float(item[0].eye_distance),
                item[0].canonical_filename.casefold(),
                str(item[0].source_path.resolve()).casefold(),
            ),
        )
        kept = ranked[0][0]
        groups[identity].append(_record_payload(kept, digest))
        for duplicate, _ in ranked[1:]:
            within_identity_duplicates.append(
                {
                    "identity": identity,
                    "sha256": digest,
                    "kept_path": str(kept.source_path.resolve()),
                    "excluded_path": str(duplicate.source_path.resolve()),
                }
            )

    historical = {identity.casefold() for identity in historical_identities}
    unseen_groups = {
        identity: _rank_records(rows)
        for identity, rows in groups.items()
        if identity not in historical and identity not in conflicting_identities
    }
    eligible = [
        identity
        for identity, rows in unseen_groups.items()
        if len(rows) >= minimum
    ]
    eligible.sort(key=lambda value: stable_order_key(seed, "identity", value))
    requested = sum(counts.values())
    if len(eligible) < requested:
        raise ValueError(
            f"Only {len(eligible)} never-used identities have at least {minimum} "
            f"unique images; {requested} are required"
        )

    selected_by_split: dict[str, list[str]] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        next_cursor = cursor + counts[split_name]
        selected_by_split[split_name] = eligible[cursor:next_cursor]
        cursor = next_cursor

    manifests: dict[str, dict[str, Any]] = {}
    for split_name, identities in selected_by_split.items():
        records: list[dict[str, Any]] = []
        for identity in identities:
            rows = unseen_groups[identity]
            if split_name != "training_extension":
                rows = rows[:evaluation_count]
            records.extend(rows)
        manifests[split_name] = {
            "schema_version": 1,
            "protocol_name": "dogfacenet_unified_fresh_identity_disjoint",
            "protocol_split": split_name,
            "protocol_seed": int(seed),
            "identity_source": (
                "canonical labels.csv filename parsed by "
                "dogfacenet_identity_from_filename"
            ),
            "minimum_images_per_identity_after_exact_dedup": minimum,
            "evaluation_images_per_identity": evaluation_count,
            "records": records,
        }

    identity_sets = {
        name: set(identities) for name, identities in selected_by_split.items()
    }
    digest_sets = {
        name: {row["source_sha256"] for row in payload["records"]}
        for name, payload in manifests.items()
    }
    pairwise: dict[str, dict[str, list[str]]] = {}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "identity_overlap": sorted(identity_sets[left] & identity_sets[right]),
                "source_sha256_overlap": sorted(digest_sets[left] & digest_sets[right]),
            }

    distribution = {
        str(threshold): sum(
            len(rows) >= threshold for rows in unseen_groups.values()
        )
        for threshold in range(2, 9)
    }
    audit = {
        "schema_version": 1,
        "seed": int(seed),
        "source_records": len(alignment_records),
        "source_identities": len(
            {record.identity.casefold() for record in alignment_records}
        ),
        "source_unique_files": len(digest_cache),
        "historical_identities_excluded": len(historical),
        "cross_identity_conflicting_identities_excluded": sorted(
            conflicting_identities
        ),
        "cross_identity_exact_duplicates": sorted(
            cross_identity_duplicates, key=lambda row: row["sha256"]
        ),
        "within_identity_exact_duplicates_excluded": sorted(
            within_identity_duplicates,
            key=lambda row: (row["identity"], row["sha256"]),
        ),
        "never_used_identities_after_dedup": len(unseen_groups),
        "never_used_identity_count_distribution": distribution,
        "eligible_identities": len(eligible),
        "reserve_identities": eligible[requested:],
        "splits": {
            name: {
                "identities": len(selected_by_split[name]),
                "records": len(manifests[name]["records"]),
                "identity_names": sorted(selected_by_split[name]),
            }
            for name in SPLIT_NAMES
        },
        "pairwise_disjointness": pairwise,
    }
    return manifests, audit


def protocol_manifest_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    rows = list(payload["records"])
    identities = {str(row["identity"]).casefold() for row in rows}
    return {"records": len(rows), "identities": len(identities)}
