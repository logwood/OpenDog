#!/usr/bin/env python3
"""Record blind-test completion without mutating the locked model record."""

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


def concise_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fused = payload["branches"]["fused"]
    return {
        "evaluation": str(path.resolve()),
        "gallery_rank1": payload["gallery_query"]["rank1_accuracy"],
        "leave_one_out_rank1": payload["rank1_leave_one_out_accuracy"],
        "fused_auc": fused["auc"],
        "fused_balanced_accuracy": fused["pilot_best_threshold"]["balanced_accuracy"],
        "fused_mean_gap": fused["same"]["mean"] - fused["different"]["mean"],
        "fused_worst_gap": fused["same"]["min"] - fused["different"]["max"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock_record", type=Path)
    parser.add_argument("--frozen-evaluation", type=Path, required=True)
    parser.add_argument("--selected-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock_record.read_text(encoding="utf-8"))
    checkpoint = Path(lock["packaged_checkpoint"])
    actual_hash = sha256_file(checkpoint)
    expected_hash = lock["checkpoint_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError("Locked checkpoint changed before or during blind evaluation")
    frozen = concise_metrics(args.frozen_evaluation)
    selected = concise_metrics(args.selected_evaluation)
    metric_names = [name for name in selected if name != "evaluation"]
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256_before_and_after_blind": actual_hash,
        "checkpoint_unchanged": True,
        "models_evaluated": ["predeclared frozen baseline", "validation-locked selected model"],
        "no_post_blind_model_selection": True,
        "frozen": frozen,
        "selected": selected,
        "delta": {name: selected[name] - frozen[name] for name in metric_names},
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite blind completion: {args.output}")
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
