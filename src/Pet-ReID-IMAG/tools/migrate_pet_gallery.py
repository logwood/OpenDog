#!/usr/bin/env python3
"""Atomically re-encode a persistent gallery with the BIFOR ONNX backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import build_pipeline  # noqa: E402
from pet_id.gallery_migration import migrate_pipeline_gallery  # noqa: E402
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.workspace_paths import (  # noqa: E402
    GALLERY_STORE_ROOT,
    SELECTED_MODELS_ROOT,
)


BIFOR_PACKAGE = SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_bifor_lowrank_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode every original image in a model-bound Gallery and publish "
            "a separate BIFOR Gallery only after full integrity verification"
        )
    )
    parser.add_argument(
        "--source-gallery",
        type=Path,
        default=GALLERY_STORE_ROOT / "pet_api_gallery_semantic_v3_v1",
    )
    parser.add_argument(
        "--target-gallery",
        type=Path,
        default=GALLERY_STORE_ROOT
        / "pet_api_gallery_semantic_v3_bifor_lowrank_v1",
    )
    parser.add_argument("--config-file", type=Path, default=BIFOR_PACKAGE / "config.yaml")
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "model_final.pth",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=BIFOR_PACKAGE / "onnx" / "pet_embedding.onnx",
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=SELECTED_MODELS_ROOT.parent
        / "pretrained"
        / "body_detection"
        / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument(
        "--require-single-pet",
        action="store_true",
        help="reject a trusted legacy reference unless exactly one pet is detected",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/bifor/gallery_migration_v1/report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = build_pipeline(
        args.config_file.expanduser().resolve(),
        args.identity_weights.expanduser().resolve(),
        args.device,
        backend="onnx-bifor",
        onnx_model=args.onnx_model.expanduser().resolve(),
        onnx_provider=args.onnx_provider,
        onnx_warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
        verify_onnx_source_checkpoint=True,
        body_detector=args.body_detector.expanduser().resolve(),
    )
    report = migrate_pipeline_gallery(
        args.source_gallery,
        args.target_gallery,
        pipeline,
        require_single_pet=args.require_single_pet,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
