#!/usr/bin/env python3
"""Rebase the selected full-resolution dependency chain to one workspace."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.release_compatibility import (  # noqa: E402
    historical_full_resolution_sources,
    with_parent_checkpoint_source,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--semantic-base",
        type=Path,
    )
    parser.add_argument(
        "--geometry",
        type=Path,
    )
    parser.add_argument(
        "--detail-checkpoint",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    historical_sources = historical_full_resolution_sources(workspace)
    semantic_base = (
        (workspace / args.semantic_base).resolve()
        if args.semantic_base is not None
        else historical_sources["semantic_base"].resolve()
    )
    geometry = (
        (workspace / args.geometry).resolve()
        if args.geometry is not None
        else historical_sources["geometry"].resolve()
    )
    semantic_profile = get_runtime_profile("legacy-semantic")
    production_profile = get_runtime_profile("production")
    candidate_profile = get_runtime_profile("candidate")
    semantic_package = workspace / "models/selected" / semantic_profile.model_package
    semantic_config = semantic_package / "config.yaml"
    semantic_checkpoint = semantic_package / "model_final.pth"
    arcface = workspace / "models/pretrained/dog.pt"
    parent_checkpoint = (
        workspace / production_profile.package_checkpoint_relative
    ).resolve()
    detail_checkpoint = (
        (workspace / args.detail_checkpoint).resolve()
        if args.detail_checkpoint is not None
        else (workspace / candidate_profile.package_checkpoint_relative).resolve()
    )

    semantic_payload = torch.load(semantic_base, map_location="cpu", weights_only=False)
    semantic_payload["sources"] = {
        "geometry_checkpoint": source(geometry),
        "semantic_config": source(semantic_config),
        "semantic_checkpoint": source(semantic_checkpoint),
        "arcface_checkpoint": source(arcface),
    }
    save(semantic_payload, semantic_base)

    parent_payload = torch.load(
        parent_checkpoint, map_location="cpu", weights_only=False
    )
    parent_payload["sources"]["base_checkpoint"] = source(semantic_base)
    save(parent_payload, parent_checkpoint)

    detail_payload = torch.load(
        detail_checkpoint, map_location="cpu", weights_only=False
    )
    detail_payload["sources"] = with_parent_checkpoint_source(
        detail_payload["sources"],
        source(parent_checkpoint),
    )
    save(detail_payload, detail_checkpoint)

    print(
        {
            "semantic_base": source(semantic_base),
            "production_parent": source(parent_checkpoint),
            "spatial_detail_candidate": source(detail_checkpoint),
        }
    )


if __name__ == "__main__":
    main()
