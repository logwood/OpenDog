#!/usr/bin/env python3
"""Build an atomic BIFOR + MegaDescriptor Agent Gallery from original images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import build_pipeline  # noqa: E402
from pet_id.gallery_migration import migrate_gallery  # noqa: E402
from pet_id.gallery_service import MultimodalPipelineEncoder  # noqa: E402
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.recognition_agent import (  # noqa: E402
    AgentFeatureEncoder,
    MegaDescriptorEncoder,
)
from pet_id.model_profiles import get_runtime_profile  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode every reference with BIFOR and frozen MegaDescriptor, then "
            "publish an independent Agent Gallery after integrity verification"
        )
    )
    bifor = get_runtime_profile("research-bifor")
    agent = get_runtime_profile("research-agent")
    parser.add_argument(
        "--source-gallery",
        type=Path,
        default=bifor.persistent_gallery,
    )
    parser.add_argument(
        "--target-gallery",
        type=Path,
        default=agent.persistent_gallery,
    )
    parser.add_argument("--config-file", type=Path, default=bifor.config)
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=bifor.identity_weights,
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=bifor.onnx,
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=bifor.body_detector,
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        default=agent.expert_checkpoint,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megadescriptor-device")
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent.parent
        / "artifacts"
        / "runs"
        / "agent"
        / "gallery_migration"
        / "report.json",
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
    encoder = AgentFeatureEncoder(
        MultimodalPipelineEncoder(pipeline),
        [
            MegaDescriptorEncoder(
                args.megadescriptor_checkpoint,
                device=args.megadescriptor_device or args.device,
            )
        ],
    )
    report = migrate_gallery(
        args.source_gallery,
        args.target_gallery,
        encoder,
        require_single_pet=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
