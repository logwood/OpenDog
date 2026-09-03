#!/usr/bin/env python3
"""Create a portable, locked full-resolution training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PROTOCOL_NAME = "unified_full_resolution_standard35"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_records(payload: dict, *, raw_root: Path) -> list[dict]:
    records = []
    for source in payload["records"]:
        filename = str(source.get("canonical_filename") or Path(source["source_path"]).name)
        image = raw_root / "images" / filename
        if not image.is_file():
            raise FileNotFoundError(image)
        expected = str(source.get("source_sha256", ""))
        actual = sha256_file(image)
        if expected and actual != expected:
            raise RuntimeError(f"Source hash mismatch: {image}")
        records.append(
            {
                "identity": str(source["identity"]).casefold(),
                "canonical_filename": filename,
                "source_path": image.relative_to(WORKSPACE).as_posix(),
                "source_sha256": actual,
                "original_size": source.get("original_size"),
            }
        )
    records.sort(key=lambda row: (row["identity"], row["canonical_filename"]))
    return records


def manifest(split: str, records: list[dict], *, seed: int) -> dict:
    identities = sorted({row["identity"] for row in records})
    counts = {identity: 0 for identity in identities}
    for row in records:
        counts[row["identity"]] += 1
    if any(count != 4 for count in counts.values()):
        raise RuntimeError(f"{split} does not contain exactly four images per identity")
    return {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "protocol_split": split,
        "protocol_seed": seed,
        "identities": len(identities),
        "records_count": len(records),
        "images_per_identity": 4,
        "records": records,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=WORKSPACE / "data/raw/DogFaceNet_alignment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "artifacts/protocols/unified_full_resolution_standard35",
    )
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_train = args.source_train.expanduser().resolve()
    source_validation = args.source_validation.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train = manifest(
        "train",
        portable_records(read_json(source_train), raw_root=raw_root),
        seed=args.seed,
    )
    validation = manifest(
        "validation",
        portable_records(read_json(source_validation), raw_root=raw_root),
        seed=args.seed,
    )
    train_ids = {row["identity"] for row in train["records"]}
    validation_ids = {row["identity"] for row in validation["records"]}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"Identity leakage: {overlap[:5]}")
    if len(train_ids) != 800 or len(validation_ids) != 200:
        raise RuntimeError(
            f"Expected 800/200 identities, got {len(train_ids)}/{len(validation_ids)}"
        )
    train_path = output_dir / "train.manifest.json"
    validation_path = output_dir / "validation.manifest.json"
    write_json(train_path, train)
    write_json(validation_path, validation)
    lock = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "status": "LOCKED_UNSCORED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "policy": {
            "identity_disjoint": True,
            "exact_image_disjoint": True,
            "validation_training_forbidden": True,
            "validation_model_selection_only": True,
        },
        "splits": {
            "train": {
                "path": train_path.relative_to(WORKSPACE).as_posix(),
                "sha256": sha256_file(train_path),
                "identities": 800,
                "records": 3200,
            },
            "validation": {
                "path": validation_path.relative_to(WORKSPACE).as_posix(),
                "sha256": sha256_file(validation_path),
                "identities": 200,
                "records": 800,
            },
        },
        "source_manifests": {
            "train_sha256": sha256_file(source_train),
            "validation_sha256": sha256_file(source_validation),
        },
    }
    lock_path = output_dir / "protocol_lock.json"
    write_json(lock_path, lock)
    print(json.dumps({"lock": str(lock_path), "train": 3200, "validation": 800}, indent=2))


if __name__ == "__main__":
    main()
