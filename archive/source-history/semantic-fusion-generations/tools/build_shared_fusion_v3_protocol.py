#!/usr/bin/env python3
"""Build a leak-resistant development and fresh-blind protocol for fusion v3.

The original joint-800 protocol used 800 identities for training, 200 identities
for its blind test, and left 64 eligible identities unused.  Once that 200-way
test has been inspected it is no longer suitable for model selection.  This
tool preserves the original split exactly, partitions only the original 800
training identities into development train/validation sets, and promotes the
64 untouched reserve identities to a new final blind test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(record: dict) -> str:
    return str(record["identity"]).casefold()


def _record_sort_key(record: dict) -> tuple[str, str]:
    return (
        str(record.get("canonical_filename", "")).casefold(),
        str(record["source_path"]).casefold(),
    )


def _deduplicate_source_records(
    records: Iterable[dict],
) -> tuple[dict[str, list[dict]], dict]:
    """Reproduce the exact-deduplication policy of the original split tool."""
    records = list(records)
    if not records:
        raise ValueError("Source manifest contains no records")

    by_digest: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        source_path = Path(record["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Prepared source image is missing: {source_path}")
        digest = sha256_file(source_path)
        recorded_digest = record.get("source_sha256")
        if recorded_digest is not None and recorded_digest != digest:
            raise ValueError(
                f"Source SHA-256 changed for {source_path}: "
                f"manifest={recorded_digest}, actual={digest}"
            )
        enriched = dict(record)
        enriched["source_sha256"] = digest
        by_digest[digest].append(enriched)

    conflicting_identities: set[str] = set()
    cross_identity_duplicates = []
    for digest, items in by_digest.items():
        identities = sorted({_identity(item) for item in items})
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
        identity = _identity(items[0])
        if identity in conflicting_identities:
            continue
        ranked = sorted(
            items,
            key=lambda item: (
                -float(item.get("eye_distance", 0.0)),
                *_record_sort_key(item),
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

    for items in groups.values():
        items.sort(key=_record_sort_key)
    return dict(groups), {
        "source_records": len(records),
        "source_identities": len({_identity(record) for record in records}),
        "conflicting_identities_excluded": sorted(conflicting_identities),
        "cross_identity_exact_duplicates": cross_identity_duplicates,
        "within_identity_exact_duplicates_excluded": within_identity_duplicates,
    }


def _records_by_identity(manifest: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in manifest.get("records", ()):
        if "source_sha256" not in record:
            raise ValueError("Previous protocol record is missing source_sha256")
        groups[_identity(record)].append(dict(record))
    for items in groups.values():
        items.sort(key=_record_sort_key)
    return dict(groups)


def _digest_map(groups: dict[str, list[dict]]) -> dict[str, set[str]]:
    return {
        identity: {str(record["source_sha256"]) for record in records}
        for identity, records in groups.items()
    }


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"Original protocol reconstruction mismatch for {label}")


def _manifest_base(manifest: dict) -> dict:
    excluded = {
        "records",
        "failures",
        "prepared_identities",
        "trainable_identities",
        "manifest_path",
        "derived_from",
        "protocol_split",
        "protocol_seed",
        "usage_policy",
    }
    return {key: value for key, value in manifest.items() if key not in excluded}


def _make_manifest(
    base_manifest: dict,
    groups: dict[str, list[dict]],
    identities: list[str],
    *,
    split_name: str,
    protocol_seed: int,
    derived_from: str,
    usage_policy: str,
) -> dict:
    records = [
        dict(record)
        for identity in sorted(identities)
        for record in groups[identity]
    ]
    return {
        **_manifest_base(base_manifest),
        "records": records,
        "failures": [],
        "prepared_identities": len(identities),
        "trainable_identities": len(identities),
        "derived_from": derived_from,
        "protocol_generation": "shared_fusion_v3",
        "protocol_split": split_name,
        "protocol_seed": protocol_seed,
        "usage_policy": usage_policy,
    }


def _pairwise_disjointness(split_manifests: dict[str, dict]) -> dict[str, dict]:
    identity_sets = {
        name: {_identity(record) for record in payload["records"]}
        for name, payload in split_manifests.items()
    }
    digest_sets = {
        name: {record["source_sha256"] for record in payload["records"]}
        for name, payload in split_manifests.items()
    }
    result = {}
    names = list(split_manifests)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[f"{left}__{right}"] = {
                "identity_overlap": sorted(identity_sets[left] & identity_sets[right]),
                "sha256_overlap": sorted(digest_sets[left] & digest_sets[right]),
            }
    return result


def build_v3_protocol(
    source_manifest: dict,
    previous_train_manifest: dict,
    previous_blind_manifest: dict,
    previous_audit: dict,
    *,
    dev_train_identities: int,
    dev_validation_identities: int,
    dev_seed: int,
    source_manifest_label: str = "",
    previous_protocol_label: str = "",
) -> tuple[dict[str, dict], dict]:
    """Return v3 manifests and an audit after reconstructing the old protocol."""
    source_groups, dedup_audit = _deduplicate_source_records(source_manifest["records"])
    minimum = int(previous_audit["min_images_per_identity_after_dedup"])
    eligible = sorted(
        identity for identity, records in source_groups.items() if len(records) >= minimum
    )

    old_seed = int(previous_audit["seed"])
    old_splits = previous_audit["splits"]
    old_train_count = int(old_splits["train"]["identities"])
    old_validation_count = int(old_splits["validation"]["identities"])
    old_blind_count = int(old_splits["blind_test"]["identities"])
    old_requested = old_train_count + old_validation_count + old_blind_count

    shuffled = list(eligible)
    random.Random(old_seed).shuffle(shuffled)
    reconstructed_train = sorted(shuffled[:old_train_count])
    validation_end = old_train_count + old_validation_count
    reconstructed_validation = sorted(shuffled[old_train_count:validation_end])
    reconstructed_blind = sorted(shuffled[validation_end:old_requested])
    reconstructed_reserve = sorted(shuffled[old_requested:])

    _assert_equal("source_records", dedup_audit["source_records"], previous_audit["source_records"])
    _assert_equal(
        "source_identities",
        dedup_audit["source_identities"],
        previous_audit["source_identities"],
    )
    _assert_equal("eligible_identities", len(eligible), previous_audit["eligible_identities"])
    _assert_equal(
        "conflicting_identities_excluded",
        dedup_audit["conflicting_identities_excluded"],
        previous_audit["conflicting_identities_excluded"],
    )
    _assert_equal("train identities", reconstructed_train, old_splits["train"]["identity_names"])
    _assert_equal(
        "validation identities",
        reconstructed_validation,
        old_splits["validation"]["identity_names"],
    )
    _assert_equal(
        "blind identities",
        reconstructed_blind,
        old_splits["blind_test"]["identity_names"],
    )
    _assert_equal("reserve identities", reconstructed_reserve, previous_audit["reserve_identities"])

    previous_train_groups = _records_by_identity(previous_train_manifest)
    previous_blind_groups = _records_by_identity(previous_blind_manifest)
    _assert_equal("train manifest identities", sorted(previous_train_groups), reconstructed_train)
    _assert_equal("blind manifest identities", sorted(previous_blind_groups), reconstructed_blind)

    source_digests = _digest_map(source_groups)
    previous_train_digests = _digest_map(previous_train_groups)
    previous_blind_digests = _digest_map(previous_blind_groups)
    _assert_equal(
        "train manifest record hashes",
        previous_train_digests,
        {identity: source_digests[identity] for identity in reconstructed_train},
    )
    _assert_equal(
        "blind manifest record hashes",
        previous_blind_digests,
        {identity: source_digests[identity] for identity in reconstructed_blind},
    )

    if dev_train_identities + dev_validation_identities != old_train_count:
        raise ValueError(
            "Development train/validation counts must partition all original "
            f"training identities ({old_train_count})"
        )
    development_pool = list(reconstructed_train)
    random.Random(dev_seed).shuffle(development_pool)
    dev_train = sorted(development_pool[:dev_train_identities])
    dev_validation = sorted(development_pool[dev_train_identities:])

    split_manifests = {
        "dev_train": _make_manifest(
            previous_train_manifest,
            previous_train_groups,
            dev_train,
            split_name="dev_train",
            protocol_seed=dev_seed,
            derived_from=previous_protocol_label,
            usage_policy="training_and_architecture_development",
        ),
        "dev_validation": _make_manifest(
            previous_train_manifest,
            previous_train_groups,
            dev_validation,
            split_name="dev_validation",
            protocol_seed=dev_seed,
            derived_from=previous_protocol_label,
            usage_policy="model_selection_only_no_gradient_updates",
        ),
        "spent_test": _make_manifest(
            previous_blind_manifest,
            previous_blind_groups,
            reconstructed_blind,
            split_name="spent_test",
            protocol_seed=old_seed,
            derived_from=previous_protocol_label,
            usage_policy="postmortem_only_prohibited_for_v3_model_selection",
        ),
        "fresh_blind": _make_manifest(
            source_manifest,
            source_groups,
            reconstructed_reserve,
            split_name="fresh_blind",
            protocol_seed=old_seed,
            derived_from=source_manifest_label,
            usage_policy="single_final_evaluation_after_model_lock",
        ),
    }
    pairwise = _pairwise_disjointness(split_manifests)
    for pair, overlap in pairwise.items():
        if overlap["identity_overlap"] or overlap["sha256_overlap"]:
            raise ValueError(f"Protocol leakage detected in {pair}: {overlap}")

    identities_by_split = {
        "dev_train": dev_train,
        "dev_validation": dev_validation,
        "spent_test": reconstructed_blind,
        "fresh_blind": reconstructed_reserve,
    }
    audit = {
        "schema_version": 1,
        "protocol_name": "dogfacenet_shared_fusion_v3_protocol_v1",
        "development_seed": dev_seed,
        "original_protocol_seed": old_seed,
        "source_records": dedup_audit["source_records"],
        "source_identities": dedup_audit["source_identities"],
        "eligible_identities": len(eligible),
        "min_images_per_identity_after_dedup": minimum,
        "reconstruction_checks": {
            "original_train_identity_list_exact": True,
            "original_blind_identity_list_exact": True,
            "original_reserve_identity_list_exact": True,
            "original_train_record_hashes_exact": True,
            "original_blind_record_hashes_exact": True,
        },
        "policies": {
            "dev_train": "training_and_architecture_development",
            "dev_validation": "model_selection_only_no_gradient_updates",
            "spent_test": "postmortem_only_prohibited_for_v3_model_selection",
            "fresh_blind": "single_final_evaluation_after_model_lock",
        },
        "fresh_blind_status": "LOCKED_UNSCORED",
        "splits": {
            name: {
                "identities": len(identities),
                "records": len(split_manifests[name]["records"]),
                "identity_names": identities,
            }
            for name, identities in identities_by_split.items()
        },
        "pairwise_disjointness": pairwise,
        "deduplication": dedup_audit,
    }
    return split_manifests, audit


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--previous-protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-train-identities", type=int, default=700)
    parser.add_argument("--dev-validation-identities", type=int, default=100)
    parser.add_argument("--dev-seed", type=int, default=20260827)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    previous_dir = args.previous_protocol_dir.resolve()
    output_dir = args.output_dir.resolve()
    previous_train_path = previous_dir / "train_manifest.json"
    previous_blind_path = previous_dir / "blind_test_manifest.json"
    previous_audit_path = previous_dir / "split_audit.json"

    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    previous_train = json.loads(previous_train_path.read_text(encoding="utf-8"))
    previous_blind = json.loads(previous_blind_path.read_text(encoding="utf-8"))
    previous_audit = json.loads(previous_audit_path.read_text(encoding="utf-8"))
    split_manifests, audit = build_v3_protocol(
        source_manifest,
        previous_train,
        previous_blind,
        previous_audit,
        dev_train_identities=args.dev_train_identities,
        dev_validation_identities=args.dev_validation_identities,
        dev_seed=args.dev_seed,
        source_manifest_label=str(source_path),
        previous_protocol_label=str(previous_dir),
    )

    output_paths = {
        name: output_dir / f"{name}_manifest.json" for name in split_manifests
    }
    audit_path = output_dir / "split_audit.json"
    lock_path = output_dir / "protocol_lock.json"
    for path in [*output_paths.values(), audit_path, lock_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing protocol artifact: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_hashes = {}
    for name, payload in split_manifests.items():
        payload["manifest_path"] = str(output_paths[name])
        encoded = _json_bytes(payload)
        output_paths[name].write_bytes(encoded)
        manifest_hashes[name] = hashlib.sha256(encoded).hexdigest()

    audit["inputs"] = {
        "source_manifest": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "previous_train_manifest": {
            "path": str(previous_train_path),
            "sha256": sha256_file(previous_train_path),
        },
        "previous_blind_manifest": {
            "path": str(previous_blind_path),
            "sha256": sha256_file(previous_blind_path),
        },
        "previous_split_audit": {
            "path": str(previous_audit_path),
            "sha256": sha256_file(previous_audit_path),
        },
    }
    audit["manifest_sha256"] = manifest_hashes
    audit_bytes = _json_bytes(audit)
    audit_path.write_bytes(audit_bytes)

    lock = {
        "schema_version": 1,
        "protocol_name": audit["protocol_name"],
        "status": "LOCKED_UNSCORED",
        "split_audit": str(audit_path),
        "split_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "fresh_blind_manifest": str(output_paths["fresh_blind"]),
        "fresh_blind_manifest_sha256": manifest_hashes["fresh_blind"],
        "fresh_blind_identities": audit["splits"]["fresh_blind"]["identities"],
        "fresh_blind_records": audit["splits"]["fresh_blind"]["records"],
        "rule": "Do not score fresh_blind until the v3 architecture, checkpoint, and thresholds are locked.",
    }
    lock_path.write_bytes(_json_bytes(lock))
    print(
        json.dumps(
            {
                "manifests": {name: str(path) for name, path in output_paths.items()},
                "audit": str(audit_path),
                "lock": str(lock_path),
                "splits": audit["splits"],
                "fresh_blind_status": audit["fresh_blind_status"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
