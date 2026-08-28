# encoding: utf-8
"""Judge whether the 100-step latent V2 microfit run actually learned."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--max-loss-ratio",
        type=float,
        default=0.85,
        help="Maximum allowed last-window / first-window median loss.",
    )
    return parser.parse_args()


def _read_records(path):
    if not path.is_file():
        raise FileNotFoundError(f"microfit metrics file does not exist: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on metrics line {line_number}") from error
        records.append(record)
    return records


def _finite_values(records, name):
    values = []
    for record in records:
        value = record.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _report(records, window, max_loss_ratio):
    losses = _finite_values(records, "total_loss")
    if len(losses) < 2 * window:
        raise ValueError(
            f"need at least {2 * window} total_loss records, found {len(losses)}"
        )
    first_loss = statistics.median(losses[:window])
    last_loss = statistics.median(losses[-window:])
    loss_ratio = last_loss / max(first_loss, 1e-12)

    slot_cosines = _finite_values(records, "latent/slot_cosine_max")
    effective_ranks = _finite_values(records, "latent/slot_effective_rank")
    gradient_norms = _finite_values(records, "latent/grad_norm")
    finite_fractions = _finite_values(records, "latent/grad_finite_fraction")
    fusion_ratios = _finite_values(records, "latent/slot_set_fusion_ratio")

    checks = {
        "loss_decreased": loss_ratio <= max_loss_ratio,
        "slot_diagnostics_present": bool(slot_cosines and effective_ranks),
        "slots_not_collapsed": bool(slot_cosines) and max(slot_cosines) < 0.995,
        "effective_rank_retained": bool(effective_ranks)
        and min(effective_ranks) >= 2.0,
        "workspace_received_gradients": bool(gradient_norms)
        and max(gradient_norms) > 0.0,
        "workspace_gradients_finite": bool(finite_fractions)
        and min(finite_fractions) == 1.0,
        "fusion_path_became_active": bool(fusion_ratios) and max(fusion_ratios) > 1e-6,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "records": len(records),
        "loss_records": len(losses),
        "first_loss_median": first_loss,
        "last_loss_median": last_loss,
        "last_to_first_loss_ratio": loss_ratio,
        "max_slot_cosine": max(slot_cosines) if slot_cosines else None,
        "min_slot_effective_rank": min(effective_ranks) if effective_ranks else None,
        "max_gradient_norm": max(gradient_norms) if gradient_norms else None,
        "max_fusion_ratio": max(fusion_ratios) if fusion_ratios else None,
        "checks": checks,
    }


def main():
    args = _parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")
    records = _read_records(args.metrics)
    report = _report(records, args.window, args.max_loss_ratio)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
