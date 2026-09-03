#!/usr/bin/env python3
"""Assemble spent historical data and the locked fresh split into v2 manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_fresh_protocol import sha256_file  # noqa: E402


HISTORICAL_TRAINING_SOURCES = (
    WORKSPACE
    / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/train_manifest.json",
    WORKSPACE
    / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/blind_test_manifest.json",
    WORKSPACE
    / "artifacts/runs/legacy/dogfacenet_shared_v3_protocol_v1/fresh_blind_manifest.json",
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_complete_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    completion = payload.get("completion")
    if completion is not None and int(completion["prepared"]) < 1:
        raise RuntimeError(f"Prepared split contains no records: {path}")
    return payload


def deduplicate_training(records: list[dict]) -> tuple[list[dict], dict]:
    by_digest: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        digest = str(record.get("source_sha256", ""))
        if len(digest) != 64:
            source = Path(record["source_path"]).expanduser().resolve()
            digest = sha256_file(source)
        row = dict(record)
        row["source_sha256"] = digest
        by_digest[digest].append(row)
    cross_identity = []
    deduplicated = []
    within_identity = 0
    for digest, rows in sorted(by_digest.items()):
        identities = sorted({str(row["identity"]).casefold() for row in rows})
        if len(identities) > 1:
            cross_identity.append({"source_sha256": digest, "identities": identities})
            continue
        ranked = sorted(
            rows,
            key=lambda row: (
                str(row.get("canonical_filename", "")).casefold(),
                str(row["source_path"]).casefold(),
            ),
        )
        deduplicated.append(ranked[0])
        within_identity += len(ranked) - 1
    if cross_identity:
        raise RuntimeError(
            f"Combined training sources contain cross-identity duplicates: {cross_identity}"
        )
    deduplicated.sort(
        key=lambda row: (
            str(row["identity"]).casefold(),
            str(row.get("canonical_filename", "")).casefold(),
            str(row["source_path"]).casefold(),
        )
    )
    return deduplicated, {
        "input_records": len(records),
        "output_records": len(deduplicated),
        "within_identity_exact_duplicates_removed": within_identity,
    }


def split_sets(manifest: dict) -> tuple[set[str], set[str]]:
    identities = {
        str(row["identity"]).casefold() for row in manifest["records"]
    }
    digests = {str(row["source_sha256"]) for row in manifest["records"]}
    return identities, digests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh-protocol-dir",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_unified_fresh_v2_protocol_20260831",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_unified_v2_protocol_20260831",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fresh_root = args.fresh_protocol_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v2 protocol: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fresh_lock_path = fresh_root / "protocol_lock.json"
    fresh_lock = json.loads(fresh_lock_path.read_text(encoding="utf-8"))
    if fresh_lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("Fresh protocol is not in the expected locked state")
    prepared_paths = {
        name: fresh_root / "prepared" / name / "manifest.json"
        for name in ("training_extension", "development", "blind_test")
    }
    prepared = {
        name: load_complete_manifest(path) for name, path in prepared_paths.items()
    }
    for name, payload in prepared.items():
        if payload.get("protocol_split") != name:
            raise RuntimeError(f"Prepared {name} manifest has the wrong split label")
        if name in {"development", "blind_test"} and not payload[
            "completion"
        ]["all_records_prepared"]:
            raise RuntimeError(f"Protected split {name} was not fully prepared")

    source_manifests = []
    combined_records = []
    for path in HISTORICAL_TRAINING_SOURCES:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        combined_records.extend(payload["records"])
        source_manifests.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(payload["records"]),
                "identities": len(
                    {str(row["identity"]).casefold() for row in payload["records"]}
                ),
                "historical_status": "SPENT_REPURPOSED_FOR_V2_TRAINING",
            }
        )
    combined_records.extend(prepared["training_extension"]["records"])
    source_manifests.append(
        {
            "path": str(prepared_paths["training_extension"]),
            "sha256": sha256_file(prepared_paths["training_extension"]),
            "records": len(prepared["training_extension"]["records"]),
            "identities": len(
                {
                    str(row["identity"]).casefold()
                    for row in prepared["training_extension"]["records"]
                }
            ),
            "historical_status": "FRESH_V2_TRAINING_EXTENSION",
        }
    )
    training_records, deduplication = deduplicate_training(combined_records)
    training_counts = Counter(
        str(row["identity"]).casefold() for row in training_records
    )
    if min(training_counts.values(), default=0) < 2:
        raise RuntimeError("Every v2 training identity needs at least two records")
    training = {
        "schema_version": 2,
        "protocol_name": "dogfacenet_unified_v2",
        "protocol_split": "training_v2",
        "policy": (
            "All earlier blind sets are permanently spent and may train v2; "
            "the fresh development and blind identities remain excluded."
        ),
        "source_manifests": source_manifests,
        "deduplication": deduplication,
        "records": training_records,
    }
    training_path = output_dir / "training_manifest.json"
    atomic_json(training_path, training)

    split_payloads = {
        "training": training,
        "development": prepared["development"],
        "blind_test": prepared["blind_test"],
    }
    sets = {name: split_sets(payload) for name, payload in split_payloads.items()}
    pairwise = {}
    names = tuple(split_payloads)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            identity_overlap = sets[left][0] & sets[right][0]
            digest_overlap = sets[left][1] & sets[right][1]
            if identity_overlap or digest_overlap:
                raise RuntimeError(f"v2 split overlap: {left} vs {right}")
            pairwise[f"{left}__{right}"] = {
                "identity_overlap": [],
                "source_sha256_overlap": [],
            }

    manifest_paths = {
        "training": training_path,
        "development": prepared_paths["development"],
        "blind_test": prepared_paths["blind_test"],
    }
    lock = {
        "schema_version": 1,
        "protocol_name": "dogfacenet_unified_v2",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "status": "BASELINE_PENDING",
        "fresh_protocol_lock": {
            "path": str(fresh_lock_path),
            "sha256": sha256_file(fresh_lock_path),
        },
        "policy": {
            "development_only_for_model_selection": True,
            "blind_single_baseline_measurement": True,
            "blind_single_candidate_attempt": True,
            "candidate_must_match_or_exceed_baseline_top1_and_top5_counts": True,
            "failed_candidate_keeps_semantic_v3_default": True,
        },
        "splits": {
            name: {
                "path": str(manifest_paths[name]),
                "sha256": sha256_file(manifest_paths[name]),
                "records": len(payload["records"]),
                "identities": len(sets[name][0]),
                "queries": (
                    len(payload["records"]) - 2 * len(sets[name][0])
                    if name != "training"
                    else None
                ),
            }
            for name, payload in split_payloads.items()
        },
        "pairwise_disjointness": pairwise,
        "training_deduplication": deduplication,
        "baseline_attempt_marker": str(output_dir / "blind_baseline.attempt.json"),
        "candidate_attempt_marker": str(output_dir / "blind_candidate.attempt.json"),
    }
    lock_path = output_dir / "protocol_lock.json"
    atomic_json(lock_path, lock)
    print(
        json.dumps(
            {
                "protocol_lock": str(lock_path),
                "protocol_lock_sha256": sha256_file(lock_path),
                "status": lock["status"],
                "splits": lock["splits"],
                "pairwise_disjointness": pairwise,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
