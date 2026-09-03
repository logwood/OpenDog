#!/usr/bin/env python3
"""Lock a validation-selected low-rank body-fusion model before test use."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    selected_model = Path(evaluation["selected_model"]).resolve()
    model = args.model.resolve()
    if selected_model != model:
        raise ValueError(
            f"Evaluation selected {selected_model}, but lock requested {model}"
        )
    split_paths = {
        split: (args.protocol_dir / f"{split}_manifest.json").resolve()
        for split in ("train", "validation", "blind_test")
    }
    if not model.is_file() or not all(path.is_file() for path in split_paths.values()):
        raise FileNotFoundError("Model or protocol split is missing")
    payload = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "100 train identities and 20 validation identities only",
        "selection_rule": evaluation["selection_rule"],
        "selected": {
            "body_weight": evaluation["selected"]["body_weight"],
            "centered": evaluation["selected"]["centered"],
            "output_dim": evaluation["selected"]["output_dim"],
        },
        "model": {
            "path": str(model),
            "sha256": sha256_file(model),
        },
        "validation_evaluation": {
            "path": str(args.evaluation.resolve()),
            "sha256": sha256_file(args.evaluation.resolve()),
        },
        "protocol_files": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for split, path in split_paths.items()
        },
        "test_status": "locked_before_body_or_semantic feature extraction for this run",
        "historical_caveat": (
            "blind_test identities were used by earlier project experiments; "
            "the forthcoming result is a spent-test diagnostic, not a fresh blind claim"
        ),
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite lock: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

