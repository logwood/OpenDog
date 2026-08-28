#!/usr/bin/env python3
"""Build conservative pseudo-identity folders from a saved distance matrix."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pet_id.workspace_paths import LEGACY_RUNS_ROOT, PROCESSED_DATA_ROOT, resolve_legacy_path


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def topk_smallest(matrix: np.ndarray, count: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance matrix must be square, got {matrix.shape}")
    count = min(max(int(count), 1), matrix.shape[1])
    partition = np.argpartition(matrix, count - 1, axis=1)[:, :count]
    distances = np.take_along_axis(matrix, partition, axis=1)
    order = np.argsort(distances, axis=1)
    return np.take_along_axis(partition, order, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=LEGACY_RUNS_ROOT / "s101_submit",
        help="directory containing query_filename.txt and dist.npy",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_ROOT / "pseudo" / "score65",
    )
    parser.add_argument("--similarity-threshold", type=float, default=55.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--start-label", type=int, default=10000)
    args = parser.parse_args()

    feature_dir = resolve_legacy_path(args.feature_dir)
    source_dir = resolve_legacy_path(args.source_dir)
    output_dir = resolve_legacy_path(args.output_dir)
    query_files = read_lines(feature_dir / "query_filename.txt")
    distance = np.load(feature_dir / "dist.npy")
    if distance.shape != (len(query_files), len(query_files)):
        raise ValueError(
            f"distance/name mismatch: {distance.shape} versus {len(query_files)} names"
        )
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    similarity = (1.0 - distance) * 100.0
    topk_indices = topk_smallest(distance, args.top_k)
    consumed: set[str] = set()
    next_label = args.start_label
    groups = 0
    for index, query_file in enumerate(tqdm(query_files, colour="pink")):
        if query_file in consumed:
            continue
        members = [query_file]
        for candidate_index in topk_indices[index]:
            candidate = query_files[int(candidate_index)]
            if candidate != query_file and similarity[index, candidate_index] > args.similarity_threshold:
                consumed.add(candidate)
                members.append(candidate)
        if len(members) < 2:
            continue
        next_label += 1
        destination = output_dir / str(next_label)
        destination.mkdir(exist_ok=False)
        for name in members:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / f"{next_label}_{name}")
        groups += 1
    print(f"created {groups} pseudo-identity folders under {output_dir}")


if __name__ == "__main__":
    main()
