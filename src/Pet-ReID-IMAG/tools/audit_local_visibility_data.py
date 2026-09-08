"""Inventory visibility supervision and split overlaps without relabeling data.

Reads only the listed training/development manifests. Spent blind data is
represented by protocol-lock metadata, never opened for images or labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
SOURCES = {
    "v4_raw_train": ("train_pool", "artifacts/protocols/unified_v4_full_standard35/train.manifest.json"),
    "external_train": ("train_pool", "artifacts/runs/unified/v3/external_bifor_protocol_20260831/training_extension.manifest.json"),
    "dogface_dev_train": ("train_pool", "artifacts/runs/legacy/dogfacenet_shared_v3_protocol_v1/dev_train_manifest.json"),
    "dogface_dev_validation": ("protected_development", "artifacts/runs/legacy/dogfacenet_shared_v3_protocol_v1/dev_validation_manifest.json"),
    "external_development": ("protected_development", "artifacts/runs/unified/v3/external_bifor_protocol_20260831/development.manifest.json"),
    "highres_development": ("protected_development", "artifacts/runs/unified/v4/highres_protocol_20260901/development.manifest.json"),
}
FIELDS = (
    "detection", "detection.landmarks_xy", "annotation_points", "segmentation",
    "quality_signals", "viewpoint_signals", "branch_available", "local_visibility",
    "face_roi_xyxy", "nose_roi_xyxy", "left_eye", "right_eye", "nose",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def nested_value(record, name):
    value = record
    for key in name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def present(value):
    return value is not None and value != [] and value != {} and value != ""


def identity_set(records):
    return {r["identity"].strip().casefold() for r in records}


def source_set(records):
    return {r["source_sha256"].lower() for r in records}


def inventory_manifest(payload):
    records = payload["records"]
    labels = {part: Counter() for part in ("face", "nose")}
    for record in records:
        for part in labels:
            label = (record.get("local_visibility") or {}).get(part)
            if not isinstance(label, dict):
                labels[part]["unknown"] += 1
            else:
                # Inventory, not validation or automatic conversion of proxies.
                key = f"{label.get('provenance', 'unspecified')}:{label.get('usable')}"
                labels[part][key] += 1
    return {
        "protocol_split": payload.get("protocol_split"),
        "usage_policy": payload.get("usage_policy"),
        "records": len(records),
        "identities": len(identity_set(records)),
        "unique_source_sha256": len(source_set(records)),
        "nonempty_fields": {
            field: sum(present(nested_value(r, field)) for r in records)
            for field in FIELDS
        },
        "branch_available_values": dict(Counter(
            json.dumps(r.get("branch_available")) for r in records
        )),
        "explicit_visibility_labels": {part: dict(counts) for part, counts in labels.items()},
    }


def audit(workspace):
    manifests, report = {}, {}
    for name, (role, relative) in SOURCES.items():
        path = workspace / relative
        payload = read_json(path)
        manifests[name] = payload
        report[name] = {
            "path": relative, "sha256": sha256_file(path), "role": role,
            **inventory_manifest(payload),
        }
    lock_path = workspace / "artifacts/protocols/unified_v4_full_standard35/protocol_lock.json"
    lock = read_json(lock_path)
    if lock["splits"]["train"]["sha256"] != report["v4_raw_train"]["sha256"]:
        raise ValueError("V4 train manifest differs from its source lock")
    # We intentionally do not open the validation manifest or its blind source.
    quarantined = {
        "path": lock["splits"]["validation"]["path"],
        "sha256": lock["splits"]["validation"]["sha256"],
        "source_manifest_sha256": lock["source_manifests"]["validation_sha256"],
        "reason": "historically_consumed_joint800_blind_source_do_not_tune_or_relabel",
        "records_read": False,
    }
    pairs = []
    for first, second in combinations(manifests, 2):
        a, b = manifests[first]["records"], manifests[second]["records"]
        pairs.append({
            "first": first, "second": second,
            "identity_overlap": len(identity_set(a) & identity_set(b)),
            "source_sha256_overlap": len(source_set(a) & source_set(b)),
        })
    protected = [
        r for name, payload in manifests.items()
        if SOURCES[name][0] == "protected_development" for r in payload["records"]
    ]
    protected_ids, protected_hashes = identity_set(protected), source_set(protected)
    eligibility, eligible = {}, {}
    for name, payload in manifests.items():
        if SOURCES[name][0] != "train_pool":
            continue
        records = payload["records"]
        kept = [r for r in records if (
            r["identity"].strip().casefold() not in protected_ids
            and r["source_sha256"].lower() not in protected_hashes
        )]
        eligibility[name] = {
            "available_records_after_development_exclusion": len(kept),
            "excluded_records": len(records) - len(kept),
            "identities_after_exclusion": len(identity_set(kept)),
            "explicit_visibility_records": sum(bool(r.get("local_visibility")) for r in kept),
        }
        for record in kept:
            digest = record["source_sha256"].lower()
            old = eligible.get(digest)
            if old and old["identity"].strip().casefold() != record["identity"].strip().casefold():
                raise ValueError("Same source image has conflicting identity labels")
            eligible[digest] = record
    return {
        "schema_version": 1,
        "status": "data_blocked_for_visibility_training",
        "manifests": report,
        "pairwise_intersections": pairs,
        "training_pool_eligibility": eligibility,
        "unique_eligible_images": len(eligible),
        "unique_eligible_identities": len(identity_set(list(eligible.values()))),
        "quarantined_manifest": quarantined,
        "blockers": [
            "No explicit crop-usability annotations with human/teacher_weak/synthetic provenance.",
            "Existing detector, landmark, segmentation and quality outputs are proxies, not usability labels.",
            "No independent holdout with real positive AND negative usability labels for each part.",
            "External training images have no local-part annotations; missing detections are unknown.",
        ],
        "policy": {
            "development_used_to_generate_labels": False,
            "detection_failure_is_negative": False,
            "synthetic_negative_requires_complete_part_exclusion": True,
            "manifest_identity_and_exact_hash_checks_only": True,
            "upstream_pretraining_overlap": "unknown",
            "blind_images_read": False,
            "training_started": False,
            "production_changed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(WORKSPACE)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "data_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "status": result["status"],
                      "eligible_images": result["unique_eligible_images"],
                      "eligible_identities": result["unique_eligible_identities"]}), flush=True)


if __name__ == "__main__":
    main()
