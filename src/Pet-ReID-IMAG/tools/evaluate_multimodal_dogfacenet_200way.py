#!/usr/bin/env python3
"""Run the scalable gallery evaluator with the tie-safe threshold sweep."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("evaluate_multimodal_dogfacenet_large.py")
SPEC = importlib.util.spec_from_file_location("dogfacenet_large_eval", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load evaluator: {SCRIPT}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def best_balanced_threshold(positive: list[float], negative: list[float]) -> dict:
    positives = [float(value) for value in positive]
    negatives = [float(value) for value in negative]
    rows = sorted(
        [(value, 1) for value in positives] + [(value, 0) for value in negatives],
        reverse=True,
    )
    positive_count = len(positives)
    negative_count = len(negatives)
    best = (0.5, 0.0, 1.0, rows[0][0] + 1e-6)
    true_positives = false_positives = 0
    cursor = 0
    while cursor < len(rows):
        score = rows[cursor][0]
        next_cursor = cursor
        while next_cursor < len(rows) and rows[next_cursor][0] == score:
            if rows[next_cursor][1]:
                true_positives += 1
            else:
                false_positives += 1
            next_cursor += 1
        next_score = rows[next_cursor][0] if next_cursor < len(rows) else score - 2e-6
        threshold = 0.5 * (score + next_score)
        true_positive_rate = true_positives / positive_count
        true_negative_rate = (negative_count - false_positives) / negative_count
        candidate = (
            0.5 * (true_positive_rate + true_negative_rate),
            true_positive_rate,
            true_negative_rate,
            threshold,
        )
        if candidate > best:
            best = candidate
        cursor = next_cursor
    return {
        "threshold": best[3],
        "balanced_accuracy": best[0],
        "same_recall": best[1],
        "different_recall": best[2],
    }


EVALUATOR.best_balanced_threshold = best_balanced_threshold


if __name__ == "__main__":
    EVALUATOR.main()
