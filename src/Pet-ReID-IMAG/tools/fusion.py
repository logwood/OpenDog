#!/usr/bin/env python3
"""Legacy rank-average CSV fusion with portable, explicit inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pet_id.workspace_paths import EVALUATIONS_ROOT, resolve_legacy_path


def rank_counts(values: np.ndarray) -> np.ndarray:
    """Return the historical 1-based count-of-values-less-or-equal rank."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("prediction columns must be finite one-dimensional arrays")
    return np.searchsorted(np.sort(values), values, side="right")


def fuse_rank_csvs(template: Path, inputs: list[Path], output: Path) -> Path:
    if len(inputs) < 2:
        raise ValueError("at least two --input CSV files are required")
    submit = pd.read_csv(template)
    ranked = []
    for path in inputs:
        frame = pd.read_csv(path)
        if "prediction" not in frame:
            raise ValueError(f"{path} has no prediction column")
        if len(frame) != len(submit):
            raise ValueError(
                f"row-count mismatch: {path} has {len(frame)}, template has {len(submit)}"
            )
        ranked.append(rank_counts(frame["prediction"].to_numpy()))
    submit["prediction"] = np.mean(np.stack(ranked, axis=0), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    submit.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank-average prediction CSVs. For the current four-branch Phase B "
            "workflow prefer workspace scripts/fuse_and_score.py."
        )
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATIONS_ROOT / "legacy-rank-fusion" / "submit.csv",
    )
    args = parser.parse_args()
    output = fuse_rank_csvs(
        resolve_legacy_path(args.template),
        [resolve_legacy_path(path) for path in args.input],
        resolve_legacy_path(args.output),
    )
    print(output)


if __name__ == "__main__":
    main()
