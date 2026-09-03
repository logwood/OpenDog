"""Judge whether the 100-step spatial-query latent microfit run is structurally healthy."""

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
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-loss-ratio", type=float, default=0.90)
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
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on metrics line {line_number}") from error
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
    slot_ranks = _finite_values(records, "latent/slot_effective_rank")
    query_cosines = _finite_values(records, "latent/identity_query_cosine_max")
    query_ranks = _finite_values(records, "latent/identity_query_effective_rank")
    attention_cosines = _finite_values(
        records, "latent/identity_query_attention_cosine"
    )
    gradient_norms = _finite_values(records, "latent/grad_norm")
    finite_fractions = _finite_values(records, "latent/grad_finite_fraction")
    fusion_weights = _finite_values(records, "latent/identity_fusion_weight")
    spatial_gates = {
        stage: _finite_values(records, f"latent/{stage}_gate")
        for stage in ("c2", "c3", "c4")
    }

    checks = {
        "loss_decreased": loss_ratio <= max_loss_ratio,
        "slot_diagnostics_present": bool(slot_cosines and slot_ranks),
        "slots_not_collapsed": bool(slot_cosines) and max(slot_cosines) < 0.995,
        "slot_rank_retained": bool(slot_ranks) and min(slot_ranks) >= 2.0,
        "identity_queries_diverse": bool(query_cosines)
        and max(query_cosines) < 0.995
        and bool(query_ranks)
        and min(query_ranks) >= 1.5,
        "identity_query_routes_diverse": bool(attention_cosines)
        and max(attention_cosines) < 0.999,
        "workspace_received_gradients": bool(gradient_norms)
        and max(gradient_norms) > 0.0,
        "workspace_gradients_finite": bool(finite_fractions)
        and min(finite_fractions) == 1.0,
        "fusion_gate_present": bool(fusion_weights)
        and min(fusion_weights) >= 0.0
        and max(fusion_weights) <= 0.5,
        "spatial_gates_present": all(spatial_gates.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "records": len(records),
        "first_loss_median": first_loss,
        "last_loss_median": last_loss,
        "last_to_first_loss_ratio": loss_ratio,
        "max_slot_cosine": max(slot_cosines) if slot_cosines else None,
        "min_slot_effective_rank": min(slot_ranks) if slot_ranks else None,
        "max_query_cosine": max(query_cosines) if query_cosines else None,
        "min_query_effective_rank": min(query_ranks) if query_ranks else None,
        "max_query_attention_cosine": (
            max(attention_cosines) if attention_cosines else None
        ),
        "fusion_weight_start": fusion_weights[0] if fusion_weights else None,
        "fusion_weight_end": fusion_weights[-1] if fusion_weights else None,
        "spatial_gate_ranges": {
            stage: [min(values), max(values)] if values else None
            for stage, values in spatial_gates.items()
        },
        "checks": checks,
    }


def main():
    args = _parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")
    report = _report(
        _read_records(args.metrics), args.window, args.max_loss_ratio
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

