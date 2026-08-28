#!/usr/bin/env python3
"""Package a validation-selected multimodal checkpoint before blind testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    config = args.config_file.resolve()
    selection = args.selection.resolve()
    protocol_dir = args.protocol_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model package: {output_dir}")
    for path in (checkpoint, config, selection):
        if not path.is_file():
            raise FileNotFoundError(path)
    split_paths = {
        name: protocol_dir / f"{name}_manifest.json"
        for name in ("train", "validation", "blind_test")
    }
    if not all(path.is_file() for path in split_paths.values()):
        raise FileNotFoundError("Protocol directory is missing one or more split manifests")

    selection_payload = json.loads(selection.read_text(encoding="utf-8"))
    selected_path = Path(selection_payload["selected"]["checkpoint"]).resolve()
    if selected_path != checkpoint:
        raise ValueError(
            f"Package checkpoint {checkpoint} differs from validation selection {selected_path}"
        )

    output_dir.mkdir(parents=True)
    model_path = output_dir / "model_final.pth"
    config_path = output_dir / "config.yaml"
    selection_path = output_dir / "validation_selection.json"
    shutil.copy2(checkpoint, model_path)
    shutil.copy2(config, config_path)
    shutil.copy2(selection, selection_path)
    source_hash = sha256_file(checkpoint)
    packaged_hash = sha256_file(model_path)
    if source_hash != packaged_hash:
        raise RuntimeError("Packaged checkpoint hash differs from selected source checkpoint")

    lock = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "validation identities only",
        "blind_test_status": "locked_before_first_blind_evaluation",
        "source_checkpoint": str(checkpoint),
        "packaged_checkpoint": str(model_path),
        "checkpoint_sha256": packaged_hash,
        "config_file": str(config_path),
        "validation_selection": str(selection_path),
        "protocol_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in split_paths.items()
        },
    }
    lock_path = output_dir / "lock_record.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
