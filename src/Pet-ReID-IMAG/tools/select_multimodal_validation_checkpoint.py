#!/usr/bin/env python3
"""Select a multimodal checkpoint using validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ORDER = (
    "gallery_rank1",
    "leave_one_out_rank1",
    "fused_auc",
    "fused_balanced_accuracy",
    "fused_mean_gap",
    "fused_worst_gap",
)


def metrics_from_evaluation(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fused = payload["branches"]["fused"]
    return {
        "evaluation": str(path.resolve()),
        "checkpoint": payload.get("checkpoint"),
        "model_source": payload["model_source"],
        "gallery_rank1": float(payload["gallery_query"]["rank1_accuracy"]),
        "leave_one_out_rank1": float(payload["rank1_leave_one_out_accuracy"]),
        "fused_auc": float(fused["auc"]),
        "fused_balanced_accuracy": float(
            fused["pilot_best_threshold"]["balanced_accuracy"]
        ),
        "fused_mean_gap": float(fused["same"]["mean"] - fused["different"]["mean"]),
        "fused_worst_gap": float(fused["same"]["min"] - fused["different"]["max"]),
        "joint_mix": float(payload["joint_mix"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = metrics_from_evaluation(args.baseline)
    candidates = [metrics_from_evaluation(path) for path in args.evaluations]
    ranked = sorted(
        candidates,
        key=lambda item: tuple(item[name] for name in METRIC_ORDER),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank
        candidate["delta_vs_frozen"] = {
            name: candidate[name] - baseline[name] for name in METRIC_ORDER
        }
    selected = ranked[0]
    payload = {
        "schema_version": 1,
        "selection_data": "validation identities only; blind-test manifest was not read",
        "selection_rule": f"lexicographic descending: {', '.join(METRIC_ORDER)}",
        "baseline": baseline,
        "candidates_ranked": ranked,
        "selected": selected,
        "selected_beats_frozen_primary": (
            selected["gallery_rank1"] > baseline["gallery_rank1"]
            or (
                selected["gallery_rank1"] == baseline["gallery_rank1"]
                and selected["leave_one_out_rank1"] > baseline["leave_one_out_rank1"]
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite selection record: {args.output}")
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
