"""Identity-disjoint external protocol helpers for UnifiedPetReID.

The original DogFaceNet pool is exhausted for another honest blind split.
This module builds a new protocol from an independently labelled ReID dataset
and checks exact image overlap against historical training sources.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
PROTOCOL_SPLITS = ("training_extension", "development", "blind_test")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order_key(seed: int, namespace: str, value: str) -> str:
    payload = f"{int(seed)}\0{namespace}\0{value.casefold()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    """Sort frame-like names numerically without assuming one naming scheme."""

    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: _natural_key(str(path.relative_to(root))),
    )


def hash_image_roots(roots: Sequence[str | Path]) -> dict[str, Any]:
    """Hash image roots used for exact cross-dataset leakage checks."""

    digest_to_paths: dict[str, list[str]] = defaultdict(list)
    root_rows: list[dict[str, Any]] = []
    for value in roots:
        root = Path(value).expanduser().resolve()
        paths = _image_paths(root)
        for path in paths:
            digest_to_paths[sha256_file(path)].append(str(path))
        root_rows.append({"path": str(root), "images": len(paths)})
    return {
        "roots": root_rows,
        "unique_sha256": set(digest_to_paths),
        "duplicate_sha256": {
            digest: sorted(paths)
            for digest, paths in digest_to_paths.items()
            if len(paths) > 1
        },
    }


def collect_identity_images(
    root: str | Path,
    *,
    dataset_namespace: str,
    source_split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read ``root/<identity>/<image>`` and exact-deduplicate every identity."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    identity_dirs = sorted(
        (path for path in resolved.iterdir() if path.is_dir()),
        key=lambda path: _natural_key(path.name),
    )
    raw_by_digest: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    source_images = 0
    for directory in identity_dirs:
        identity = f"{dataset_namespace}:{directory.name}".casefold()
        paths = sorted(
            (
                path.resolve()
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: _natural_key(str(path.relative_to(directory))),
        )
        source_images += len(paths)
        for path in paths:
            raw_by_digest[sha256_file(path)].append((identity, path))

    cross_identity_digests: dict[str, list[dict[str, str]]] = {}
    conflicting_identities: set[str] = set()
    for digest, rows in raw_by_digest.items():
        identities = {identity for identity, _ in rows}
        if len(identities) > 1:
            conflicting_identities.update(identities)
            cross_identity_digests[digest] = [
                {"identity": identity, "source_path": str(path)}
                for identity, path in sorted(rows)
            ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    within_identity_duplicates: list[dict[str, Any]] = []
    for digest, rows in raw_by_digest.items():
        if any(identity in conflicting_identities for identity, _ in rows):
            continue
        identity = rows[0][0]
        ranked = sorted(rows, key=lambda row: _natural_key(row[1].name))
        kept = ranked[0][1]
        grouped[identity].append(
            {
                "identity": identity,
                "source_path": str(kept),
                "source_sha256": digest,
                "source_split": str(source_split),
                "source_filename": kept.name,
            }
        )
        for _, duplicate in ranked[1:]:
            within_identity_duplicates.append(
                {
                    "identity": identity,
                    "source_sha256": digest,
                    "kept_path": str(kept),
                    "excluded_path": str(duplicate),
                }
            )
    normalized = {
        identity: sorted(rows, key=lambda row: _natural_key(row["source_filename"]))
        for identity, rows in grouped.items()
    }
    audit = {
        "root": str(resolved),
        "source_split": str(source_split),
        "source_identity_directories": len(identity_dirs),
        "source_images": source_images,
        "unique_sha256": len(raw_by_digest),
        "eligible_identities_before_history_check": len(normalized),
        "cross_identity_conflicting_identities_excluded": sorted(
            conflicting_identities
        ),
        "cross_identity_exact_duplicates": cross_identity_digests,
        "within_identity_exact_duplicates_excluded": sorted(
            within_identity_duplicates,
            key=lambda row: (row["identity"], row["source_sha256"]),
        ),
    }
    return normalized, audit


def _spaced_records(
    records: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Choose deterministic temporally spread records from one identity."""

    count = int(count)
    if count < 1 or len(records) < count:
        raise ValueError("Not enough records for the requested identity sample")
    if count == 1:
        return [dict(records[len(records) // 2])]
    indices = [
        round(index * (len(records) - 1) / (count - 1))
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError("Spaced record selection produced duplicate indices")
    return [dict(records[index]) for index in indices]


def _digest_set(manifest: Mapping[str, Any]) -> set[str]:
    return {str(row["source_sha256"]) for row in manifest["records"]}


def _identity_set(manifest: Mapping[str, Any]) -> set[str]:
    return {str(row["identity"]).casefold() for row in manifest["records"]}


def build_external_protocol(
    *,
    training_groups: Mapping[str, Sequence[dict[str, Any]]],
    evaluation_groups: Mapping[str, Sequence[dict[str, Any]]],
    historical_sha256: set[str],
    training_identities: int,
    development_identities: int,
    blind_identities: int,
    training_images_per_identity: int = 4,
    evaluation_images_per_identity: int = 4,
    seed: int = 20260831,
    protocol_name: str = "unified_pet_reid_external_noninferiority",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build fixed train/development/blind manifests with no exact leakage."""

    requested = {
        "training_extension": int(training_identities),
        "development": int(development_identities),
        "blind_test": int(blind_identities),
    }
    if any(value < 1 for value in requested.values()):
        raise ValueError("Every protocol split needs at least one identity")
    training_images_per_identity = int(training_images_per_identity)
    evaluation_images_per_identity = int(evaluation_images_per_identity)
    if training_images_per_identity < 2 or evaluation_images_per_identity < 4:
        raise ValueError(
            "Training needs at least two images and evaluation needs at least four"
        )

    historical = {str(value).casefold() for value in historical_sha256}

    def eligible(
        groups: Mapping[str, Sequence[dict[str, Any]]], minimum: int
    ) -> tuple[list[str], dict[str, list[str]]]:
        names: list[str] = []
        overlaps: dict[str, list[str]] = {}
        for identity, rows in groups.items():
            matched = sorted(
                {
                    str(row["source_sha256"])
                    for row in rows
                    if str(row["source_sha256"]).casefold() in historical
                }
            )
            if matched:
                overlaps[str(identity)] = matched
                continue
            if len(rows) >= minimum:
                names.append(str(identity).casefold())
        names.sort(key=lambda name: stable_order_key(seed, "identity", name))
        return names, overlaps

    training_names, training_overlap = eligible(
        training_groups, training_images_per_identity
    )
    evaluation_names, evaluation_overlap = eligible(
        evaluation_groups, evaluation_images_per_identity
    )
    if len(training_names) < requested["training_extension"]:
        raise ValueError(
            f"Only {len(training_names)} eligible training identities; "
            f"{requested['training_extension']} requested"
        )
    evaluation_requested = requested["development"] + requested["blind_test"]
    if len(evaluation_names) < evaluation_requested:
        raise ValueError(
            f"Only {len(evaluation_names)} eligible evaluation identities; "
            f"{evaluation_requested} requested"
        )

    selected = {
        "training_extension": training_names[: requested["training_extension"]],
        "development": evaluation_names[: requested["development"]],
        "blind_test": evaluation_names[
            requested["development"] : evaluation_requested
        ],
    }
    manifests: dict[str, dict[str, Any]] = {}
    for split in PROTOCOL_SPLITS:
        source = (
            training_groups
            if split == "training_extension"
            else evaluation_groups
        )
        image_count = (
            training_images_per_identity
            if split == "training_extension"
            else evaluation_images_per_identity
        )
        rows: list[dict[str, Any]] = []
        for identity in selected[split]:
            rows.extend(_spaced_records(source[identity], image_count))
        manifests[split] = {
            "schema_version": 1,
            "protocol_name": str(protocol_name),
            "protocol_split": split,
            "protocol_seed": int(seed),
            "identity_source": "YT-BB-Dog official identity directory",
            "selection": (
                "deterministic identity hash order; temporally spaced frames"
            ),
            "images_per_identity": image_count,
            "records": rows,
        }

    pairwise: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(PROTOCOL_SPLITS):
        for right in PROTOCOL_SPLITS[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "identity_overlap": sorted(
                    _identity_set(manifests[left])
                    & _identity_set(manifests[right])
                ),
                "source_sha256_overlap": sorted(
                    _digest_set(manifests[left])
                    & _digest_set(manifests[right])
                ),
            }
    if any(
        values["identity_overlap"] or values["source_sha256_overlap"]
        for values in pairwise.values()
    ):
        raise RuntimeError("Constructed protocol splits are not disjoint")
    manifest_hashes = set().union(
        *(_digest_set(value) for value in manifests.values())
    )
    if manifest_hashes & historical:
        raise RuntimeError("Constructed protocol still overlaps historical images")

    training_distribution = Counter(
        len(rows) for rows in training_groups.values()
    )
    evaluation_distribution = Counter(
        len(rows) for rows in evaluation_groups.values()
    )
    audit = {
        "schema_version": 1,
        "protocol_name": str(protocol_name),
        "seed": int(seed),
        "historical_unique_sha256": len(historical),
        "historical_overlap": {
            "training_identities_excluded": training_overlap,
            "evaluation_identities_excluded": evaluation_overlap,
        },
        "available": {
            "training_identities": len(training_groups),
            "training_eligible_identities": len(training_names),
            "evaluation_identities": len(evaluation_groups),
            "evaluation_eligible_identities": len(evaluation_names),
            "training_image_count_distribution": {
                str(count): identities
                for count, identities in sorted(training_distribution.items())
            },
            "evaluation_image_count_distribution": {
                str(count): identities
                for count, identities in sorted(evaluation_distribution.items())
            },
        },
        "splits": {
            name: {
                "identities": len(selected[name]),
                "records": len(manifests[name]["records"]),
                "identity_names": sorted(selected[name]),
            }
            for name in PROTOCOL_SPLITS
        },
        "reserves": {
            "training_identities": training_names[
                requested["training_extension"] :
            ],
            "evaluation_identities": evaluation_names[evaluation_requested:],
        },
        "pairwise_disjointness": pairwise,
    }
    return manifests, audit


def validate_raw_manifest(
    payload: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> dict[str, int]:
    """Validate the immutable contract used by raw-image evaluators."""

    split = str(payload.get("protocol_split", ""))
    if expected_split is not None and split != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, got {split!r}")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest records must be a non-empty list")
    required = ("identity", "source_path", "source_sha256")
    for index, row in enumerate(rows):
        if any(not str(row.get(name, "")) for name in required):
            raise ValueError(f"Manifest record {index} is incomplete")
        source = Path(str(row["source_path"])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256_file(source) != str(row["source_sha256"]).casefold():
            raise RuntimeError(f"Source hash mismatch: {source}")
    digests = [str(row["source_sha256"]).casefold() for row in rows]
    if len(digests) != len(set(digests)):
        raise ValueError("Manifest contains duplicate source images")
    counts = Counter(str(row["identity"]).casefold() for row in rows)
    declared = int(payload.get("images_per_identity", 0))
    if declared < 1 or any(count != declared for count in counts.values()):
        raise ValueError(
            "Manifest identity counts differ from images_per_identity"
        )
    return {"records": len(rows), "identities": len(counts)}
