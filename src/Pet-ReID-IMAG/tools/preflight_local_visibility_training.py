"""Fail-closed preflight for an eventual local-visibility training run.

This command intentionally has no optimizer/training path. It validates that a
human-annotated pair is ready; unknown templates and proxy-only labels stop
before model loading. Remote jobs can run this exact gate safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.local_visibility_data import (  # noqa: E402
    load_visibility_manifest,
    require_independent_visibility_splits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--target-graph-sha256", required=True)
    parser.add_argument("--protected-records", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(report_path)
    result = {
        "status": "blocked",
        "training_started": False,
        "optimizer_created": False,
        "target_graph_sha256": args.target_graph_sha256,
        "train_manifest": str(args.train.resolve()),
        "validation_manifest": str(args.validation.resolve()),
        "protected_records": [str(path.resolve()) for path in args.protected_records],
    }
    try:
        train = load_visibility_manifest(
            args.train, expected_split="train", expected_target_graph_sha256=args.target_graph_sha256
        )
        validation = load_visibility_manifest(
            args.validation, expected_split="validation", expected_target_graph_sha256=args.target_graph_sha256
        )
        protected = []
        for path in args.protected_records:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            protected.extend(payload.get("records", []))
        require_independent_visibility_splits(train, validation, protected_records=protected)
        result.update(status="ready_for_training", training_ready=True, train_records=len(train["records"]), validation_records=len(validation["records"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(training_ready=False, blocker=f"{type(error).__name__}: {error}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "ready_for_training":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
