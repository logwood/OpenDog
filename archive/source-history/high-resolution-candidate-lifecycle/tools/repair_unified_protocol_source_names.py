#!/usr/bin/env python3
"""Repair cross-platform protocol filenames by matching locked SHA-256 values."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    protocol = workspace / "artifacts/protocols/unified_v4_full_standard35"
    records: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        payload = json.loads(
            (protocol / f"{split}.manifest.json").read_text(encoding="utf-8")
        )
        records.extend(payload["records"])

    image_dir = workspace / "data/raw/DogFaceNet_alignment/images"
    actual_files = sorted(path for path in image_dir.iterdir() if path.is_file())
    actual_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in actual_files:
        actual_by_hash[sha256_file(path)].append(path)

    repairs: list[tuple[Path, Path, str]] = []
    verified = 0
    expected_paths = {
        (workspace / row["source_path"]).resolve()
        for row in records
    }
    for row in records:
        expected = (workspace / row["source_path"]).resolve()
        expected_hash = str(row["source_sha256"])
        if expected.is_file():
            if sha256_file(expected) != expected_hash:
                raise RuntimeError(f"Existing expected path has the wrong hash: {expected}")
            verified += 1
            continue
        candidates = actual_by_hash.get(expected_hash, [])
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one hash match for {expected}, found {len(candidates)}"
            )
        source = candidates[0]
        if source in expected_paths or source == expected:
            raise RuntimeError(f"Unsafe filename repair candidate: {source}")
        repairs.append((source, expected, expected_hash))

    if args.apply:
        for source, expected, expected_hash in repairs:
            if expected.exists():
                raise FileExistsError(expected)
            expected.parent.mkdir(parents=True, exist_ok=True)
            source.rename(expected)
            if sha256_file(expected) != expected_hash:
                raise RuntimeError(f"Post-rename hash mismatch: {expected}")
            verified += 1

    extras = [path for path in actual_files if path not in expected_paths]
    report = {
        "mode": "apply" if args.apply else "dry_run",
        "manifest_records": len(records),
        "actual_files": len(actual_files),
        "already_verified": verified - (len(repairs) if args.apply else 0),
        "repair_count": len(repairs),
        "post_apply_verified": verified,
        "pre_apply_extra_files": len(extras),
        "repairs": [
            {
                "source": str(source),
                "target": str(target),
                "sha256": digest,
            }
            for source, target, digest in repairs
        ],
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
