#!/usr/bin/env python3
"""Run the legacy-semantic plus BIFOR research pipeline on dog images."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.bifor_onnx_runtime import build_bifor_onnx_multimodal_pipeline
from pet_id.config import add_retri_config
from pet_id.gallery import collect_images, load_exif_oriented_bgr
from pet_id.model_profiles import get_runtime_profile
from pet_id.onnx_runtime import parse_warmup_batches
from pet_id.workspace_paths import normalize_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    profile = get_runtime_profile("research-bifor")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=profile.config,
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=profile.onnx,
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=profile.body_detector,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup-batches", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (args.config_file, args.onnx_model, args.body_detector)
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing runtime inputs: {missing}")
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(args.config_file.resolve()))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    pipeline = build_bifor_onnx_multimodal_pipeline(
        cfg,
        model_path=args.onnx_model,
        body_detector_checkpoint=args.body_detector,
        provider=args.provider,
        device=args.device,
        warmup_batches=parse_warmup_batches(args.warmup_batches),
    )
    records = []
    arrays = {}
    for image_path in collect_images(args.images):
        descriptors = pipeline.encode_image(load_exif_oriented_bgr(image_path))
        descriptor_rows = []
        for index, descriptor in enumerate(descriptors):
            feature = descriptor.fused_feature.numpy().astype(np.float32, copy=False)
            array_name = f"embedding_{len(records)}_{index}"
            arrays[array_name] = feature
            descriptor_rows.append(
                {
                    "embedding": array_name,
                    "shape": list(feature.shape),
                    "l2_norm": float(np.linalg.norm(feature)),
                    "metadata": descriptor.metadata_dict(),
                }
            )
        records.append(
            {
                "source": str(image_path),
                "detections": len(descriptors),
                "descriptors": descriptor_rows,
            }
        )
    report = {
        "schema_version": 1,
        "backend": pipeline.identity_model.backend_info(),
        "records": records,
    }
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
