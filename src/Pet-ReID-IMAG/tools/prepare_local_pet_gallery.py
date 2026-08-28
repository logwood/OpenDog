#!/usr/bin/env python3
"""Deduplicate labeled local pet images and make a deterministic gallery split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_identity(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("identity must use NAME=PATH")
    identity, raw_path = value.split("=", 1)
    identity = identity.strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identity):
        raise argparse.ArgumentTypeError(
            "identity names may contain lowercase ASCII letters, numbers, _ and -"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"identity directory does not exist: {path}")
    return identity, path


def collect_images(path: Path) -> list[Path]:
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda item: item.as_posix().casefold(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identity",
        action="append",
        type=parse_identity,
        required=True,
        metavar="NAME=PATH",
        help="repeat once per identity",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if args.gallery_images_per_identity < 1:
        raise ValueError("--gallery-images-per-identity must be positive")
    identities = dict(args.identity)
    if len(identities) != len(args.identity):
        raise ValueError("identity names must be unique")

    all_hash_owners: dict[str, str] = {}
    manifest_records: list[dict] = []
    duplicate_records: list[dict] = []
    summaries = {}
    for identity, source_dir in identities.items():
        unique: list[dict] = []
        first_by_hash: dict[str, Path] = {}
        for source_path in collect_images(source_dir):
            digest = sha256_file(source_path)
            owner = all_hash_owners.get(digest)
            if owner is not None and owner != identity:
                raise ValueError(
                    f"identical file content is labeled as both {owner} and {identity}: "
                    f"{source_path}"
                )
            all_hash_owners[digest] = identity
            if digest in first_by_hash:
                duplicate_records.append(
                    {
                        "identity": identity,
                        "duplicate_path": str(source_path),
                        "kept_path": str(first_by_hash[digest]),
                        "sha256": digest,
                    }
                )
                continue
            first_by_hash[digest] = source_path
            with Image.open(source_path) as image:
                encoded_size = list(image.size)
                exif_orientation = int(image.getexif().get(274, 1))
            unique.append(
                {
                    "source_path": source_path,
                    "sha256": digest,
                    "encoded_size": encoded_size,
                    "exif_orientation": exif_orientation,
                }
            )

        if len(unique) <= args.gallery_images_per_identity:
            raise ValueError(
                f"{identity} needs more than {args.gallery_images_per_identity} unique images"
            )
        for index, item in enumerate(unique):
            split = (
                "gallery"
                if index < args.gallery_images_per_identity
                else "validation"
            )
            source_path = item.pop("source_path")
            filename = f"{index + 1:03d}_{source_path.name}"
            destination = output_dir / "images" / split / identity / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256_file(destination) != item["sha256"]:
                raise FileExistsError(f"refusing to overwrite different file: {destination}")
            if not destination.exists():
                shutil.copy2(source_path, destination)
            copied_hash = sha256_file(destination)
            if copied_hash != item["sha256"]:
                raise IOError(f"copy verification failed: {destination}")
            manifest_records.append(
                {
                    "identity": identity,
                    "split": split,
                    "source_path": str(source_path),
                    "library_path": str(destination.resolve()),
                    "canonical_filename": filename,
                    **item,
                }
            )
        summaries[identity] = {
            "input_files": len(collect_images(source_dir)),
            "unique_images": len(unique),
            "gallery_images": args.gallery_images_per_identity,
            "validation_images": len(unique) - args.gallery_images_per_identity,
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_policy": (
            "sort paths case-insensitively, keep first occurrence of each SHA-256, "
            "then assign the first N unique images per identity to gallery"
        ),
        "gallery_images_per_identity": args.gallery_images_per_identity,
        "identities": list(identities),
        "summaries": summaries,
        "records": manifest_records,
        "duplicates_excluded": duplicate_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
