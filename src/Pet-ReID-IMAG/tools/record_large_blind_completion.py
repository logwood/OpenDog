#!/usr/bin/env python3
"""Record completion of a fixed-budget large-gallery blind evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def concise(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {"evaluation": str(path.resolve())}
    for name, metrics in payload["branches"].items():
        result[name] = {
            key: metrics[key]
            for key in (
                "top1_correct",
                "top1_accuracy",
                "top5_correct",
                "top5_accuracy",
                "mean_reciprocal_rank",
                "auc",
            )
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock_record", type=Path)
    parser.add_argument("--frozen-evaluation", type=Path, required=True)
    parser.add_argument("--selected-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite blind completion: {args.output}")
    lock = json.loads(args.lock_record.read_text(encoding="utf-8"))
    actual_hash = sha256_file(Path(lock["packaged_checkpoint"]))
    if actual_hash != lock["checkpoint_sha256"]:
        raise RuntimeError("Locked checkpoint changed before or during blind evaluation")
    frozen = concise(args.frozen_evaluation)
    selected = concise(args.selected_evaluation)
    delta = {}
    for branch in selected.keys() - {"evaluation"}:
        delta[branch] = {
            key: selected[branch][key] - frozen[branch][key]
            for key in selected[branch]
        }
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "800 train identities -> 200 unseen identity-prototype gallery",
        "retrieval": "200-way; 2 gallery and 2 query images per identity",
        "queries": 400,
        "checkpoint_sha256_before_and_after_blind": actual_hash,
        "checkpoint_unchanged": True,
        "models_evaluated": [
            "predeclared frozen baseline",
            "a-priori fixed-budget locked model",
        ],
        "no_post_blind_model_selection": True,
        "frozen": frozen,
        "selected": selected,
        "delta": delta,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
