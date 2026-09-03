#!/usr/bin/env python3
"""Cluster saved descriptors into an explicitly selected pseudo-label dataset."""

from __future__ import annotations

import argparse
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from infomap_cluster import cluster_by_infomap, get_dist_nbr
from pet_id.workspace_paths import LEGACY_RUNS_ROOT, PROCESSED_DATA_ROOT, resolve_legacy_path


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def generate_cluster_features(labels: np.ndarray, features: torch.Tensor) -> torch.Tensor:
    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_value = int(label)
        if label_value >= 0:
            grouped[label_value].append(features[index])
    if not grouped:
        raise RuntimeError("Infomap produced no usable pseudo clusters")
    return torch.stack(
        [torch.stack(grouped[index]).mean(0) for index in sorted(grouped)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=LEGACY_RUNS_ROOT / "s101_submit",
        help="directory containing query_f.npy and query_filename.txt",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_ROOT / "pseudo" / "infomap-score50",
    )
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--minimum-similarity", type=float, default=0.5)
    parser.add_argument("--minimum-cluster-size", type=int, default=3)
    parser.add_argument("--knn-method", default="faiss-gpu")
    parser.add_argument("--label-offset", type=int, default=10000)
    args = parser.parse_args()

    feature_dir = resolve_legacy_path(args.feature_dir)
    source_dir = resolve_legacy_path(args.source_dir)
    output_dir = resolve_legacy_path(args.output_dir)
    names = read_lines(feature_dir / "query_filename.txt")
    features = torch.from_numpy(np.load(feature_dir / "query_f.npy"))
    if features.ndim != 2 or features.shape[0] != len(names):
        raise ValueError(
            f"feature/name mismatch: {tuple(features.shape)} versus {len(names)} names"
        )
    normalized = F.normalize(features, dim=1).cpu().numpy()
    distances, neighbors = get_dist_nbr(
        features=normalized,
        k=args.neighbors,
        knn_method=args.knn_method,
    )
    started = time.time()
    labels = cluster_by_infomap(
        neighbors,
        distances,
        min_sim=args.minimum_similarity,
        cluster_num=args.minimum_cluster_size,
    ).astype(np.intp)
    centers = generate_cluster_features(labels, features)
    if not torch.isfinite(centers).all():
        raise RuntimeError("cluster centers contain non-finite values")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name, label in zip(names, labels):
        label_value = int(label)
        if label_value < 0:
            continue
        destination_label = label_value + args.label_offset
        destination = output_dir / str(destination_label)
        destination.mkdir(exist_ok=True)
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / f"{destination_label}_{name}")
        copied += 1
    cluster_count = len({int(label) for label in labels if int(label) >= 0})
    print(
        f"created {cluster_count} clusters with {copied} images under {output_dir}; "
        f"clustering took {time.time() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
