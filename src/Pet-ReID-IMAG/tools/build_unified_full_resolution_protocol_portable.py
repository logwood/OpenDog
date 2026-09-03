#!/usr/bin/env python3
"""Build the full-resolution protocol while repairing legacy names by hash."""

from __future__ import annotations

from pathlib import Path

import build_unified_full_resolution_protocol as base


def portable_records(payload: dict, *, raw_root: Path) -> list[dict]:
    records = []
    image_root = raw_root / "images"
    for source in payload["records"]:
        filename = str(
            source.get("canonical_filename") or Path(source["source_path"]).name
        )
        expected = str(source.get("source_sha256", ""))
        image = image_root / filename
        if not image.is_file():
            suffix = filename.split(".", 2)[-1]
            candidates = list(image_root.glob(f"*{suffix}"))
            verified = [
                candidate
                for candidate in candidates
                if base.sha256_file(candidate) == expected
            ]
            if len(verified) != 1:
                raise FileNotFoundError(
                    f"Could not uniquely recover {filename!r}: "
                    f"{len(verified)} hash matches"
                )
            image = verified[0]
            filename = image.name
        actual = base.sha256_file(image)
        if expected and actual != expected:
            raise RuntimeError(f"Source hash mismatch: {image}")
        records.append(
            {
                "identity": str(source["identity"]).casefold(),
                "canonical_filename": filename,
                "source_path": image.relative_to(base.WORKSPACE).as_posix(),
                "source_sha256": actual,
                "original_size": source.get("original_size"),
            }
        )
    records.sort(key=lambda row: (row["identity"], row["canonical_filename"]))
    return records


if __name__ == "__main__":
    base.portable_records = portable_records
    base.main()
