#!/usr/bin/env python3
"""Atomically re-encode the production Gallery for the candidate model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery_migration import migrate_pipeline_gallery  # noqa: E402
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.unified_highres_runtime import build_highres_onnx_pipeline  # noqa: E402


PRODUCTION_PROFILE = get_runtime_profile("production")
CANDIDATE_PROFILE = get_runtime_profile("candidate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-gallery",
        type=Path,
        default=PRODUCTION_PROFILE.persistent_gallery,
        help="existing Gallery whose original enrollment images are read only",
    )
    parser.add_argument(
        "--target-gallery",
        type=Path,
        default=CANDIDATE_PROFILE.persistent_gallery,
        help="new candidate Gallery published only after full verification",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=CANDIDATE_PROFILE.onnx,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=CANDIDATE_PROFILE.onnx.parent / "metadata.json",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=CANDIDATE_PROFILE.package_checkpoint,
        help="verified for provenance but not loaded for inference",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/gallery-migration/production-to-candidate/report.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = build_highres_onnx_pipeline(
        args.onnx_model.expanduser().resolve(),
        metadata_path=args.metadata.expanduser().resolve(),
        source_checkpoint=args.source_checkpoint.expanduser().resolve(),
        provider=args.onnx_provider,
        device=args.device,
        verify_hash=True,
        warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
    )
    report = migrate_pipeline_gallery(
        args.source_gallery,
        args.target_gallery,
        pipeline,
        require_single_pet=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
