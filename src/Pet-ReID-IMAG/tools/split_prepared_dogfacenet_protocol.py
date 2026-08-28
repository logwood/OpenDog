#!/usr/bin/env python3
"""Create an identity-disjoint DogFaceNet train/validation/blind protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_protocol(
    manifest: dict,
    *,
    train_identities: int,
    validation_identities: int,
    blind_identities: int,
    min_images_per_identity: int,
    seed: int,
) -> tuple[dict[str, dict], dict]:
    records = list(manifest.get("records", ()))
    if not records:
        raise ValueError("Prepared manifest contains no records")

    by_digest: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        source_path = Path(record["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Prepared source image is missing: {source_path}")
        enriched = dict(record)
        enriched["source_sha256"] = sha256_file(source_path)
        by_digest[enriched["source_sha256"]].append(enriched)

    conflicting_identities: set[str] = set()
    cross_identity_duplicates = []
    for digest, items in by_digest.items():
        identities = sorted({item["identity"].casefold() for item in items})
        if len(identities) > 1:
            conflicting_identities.update(identities)
            cross_identity_duplicates.append(
                {
                    "sha256": digest,
                    "identities": identities,
                    "source_paths": sorted(item["source_path"] for item in items),
                }
            )

    groups: dict[str, list[dict]] = defaultdict(list)
    within_identity_duplicates = []
    for digest, items in by_digest.items():
        identity = items[0]["identity"].casefold()
        if identity in conflicting_identities:
            continue
        ranked = sorted(
            items,
            key=lambda item: (
                -float(item.get("eye_distance", 0.0)),
                item.get("canonical_filename", "").casefold(),
                item["source_path"].casefold(),
            ),
        )
        groups[identity].append(ranked[0])
        for duplicate in ranked[1:]:
            within_identity_duplicates.append(
                {
                    "identity": identity,
                    "sha256": digest,
                    "kept_path": ranked[0]["source_path"],
                    "excluded_path": duplicate["source_path"],
                }
            )

    eligible = sorted(
        identity
        for identity, items in groups.items()
        if len(items) >= min_images_per_identity
    )
    requested = train_identities + validation_identities + blind_identities
    if len(eligible) < requested:
        raise ValueError(
            f"Only {len(eligible)} identities remain after exact deduplication; "
            f"{requested} are required"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:requested]
    split_names = (
        ("train", train_identities),
        ("validation", validation_identities),
        ("blind_test", blind_identities),
    )
    identities_by_split: dict[str, list[str]] = {}
    cursor = 0
    for name, count in split_names:
        identities_by_split[name] = sorted(selected[cursor : cursor + count])
        cursor += count

    base = {
        key: value
        for key, value in manifest.items()
        if key not in {"records", "failures", "prepared_identities", "trainable_identities"}
    }
    split_manifests = {}
    for split_name, identities in identities_by_split.items():
        split_records = [
            record
            for identity in identities
            for record in sorted(
                groups[identity],
                key=lambda item: (
                    item.get("canonical_filename", "").casefold(),
                    item["source_path"].casefold(),
                ),
            )
        ]
        split_manifests[split_name] = {
            **base,
            "records": split_records,
            "failures": [],
            "prepared_identities": len(identities),
            "trainable_identities": len(identities),
            "derived_from": str(manifest.get("manifest_path", "")),
            "protocol_split": split_name,
            "protocol_seed": seed,
            "min_images_per_identity_after_dedup": min_images_per_identity,
        }

    identity_sets = {name: set(values) for name, values in identities_by_split.items()}
    digest_sets = {
        name: {record["source_sha256"] for record in payload["records"]}
        for name, payload in split_manifests.items()
    }
    pairwise = {}
    names = list(identities_by_split)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "identity_overlap": sorted(identity_sets[left] & identity_sets[right]),
                "sha256_overlap": sorted(digest_sets[left] & digest_sets[right]),
            }

    audit = {
        "schema_version": 1,
        "seed": seed,
        "source_records": len(records),
        "source_identities": len({record["identity"].casefold() for record in records}),
        "min_images_per_identity_after_dedup": min_images_per_identity,
        "eligible_identities": len(eligible),
        "reserve_identities": sorted(eligible[requested:]),
        "conflicting_identities_excluded": sorted(conflicting_identities),
        "cross_identity_exact_duplicates": cross_identity_duplicates,
        "within_identity_exact_duplicates_excluded": within_identity_duplicates,
        "splits": {
            name: {
                "identities": len(identities),
                "records": len(split_manifests[name]["records"]),
                "identity_names": identities,
            }
            for name, identities in identities_by_split.items()
        },
        "pairwise_disjointness": pairwise,
    }
    return split_manifests, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=100)
    parser.add_argument("--validation-identities", type=int, default=20)
    parser.add_argument("--blind-identities", type=int, default=20)
    parser.add_argument("--min-images-per-identity", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    split_manifests, audit = split_protocol(
        manifest,
        train_identities=args.train_identities,
        validation_identities=args.validation_identities,
        blind_identities=args.blind_identities,
        min_images_per_identity=args.min_images_per_identity,
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split_name, payload in split_manifests.items():
        path = output_dir / f"{split_name}_manifest.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing split: {path}")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[split_name] = str(path)
    audit_path = output_dir / "split_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {audit_path}")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifests": outputs, "audit": str(audit_path), **audit["splits"]}, indent=2))


if __name__ == "__main__":
    main()
