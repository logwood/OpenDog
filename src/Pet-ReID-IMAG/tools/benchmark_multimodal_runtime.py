#!/usr/bin/env python3
"""Benchmark warm end-to-end multimodal inference with stage attribution."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.config import add_retri_config
from pet_id.gallery import collect_images
from pet_id.multimodal import build_multimodal_pipeline
from pet_id.onnx_runtime import (
    build_onnx_multimodal_pipeline,
    parse_warmup_batches,
)
from pet_id.release_compatibility import historical_artifact_path
from pet_id.workspace_paths import normalize_runtime_config


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    rollback_package = historical_artifact_path(ROOT.parents[1], "joint-rollback-package")
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=rollback_package / "config.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("pytorch", "onnx"), default="onnx")
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=rollback_package / "model_final.pth",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=rollback_package / "onnx" / "pet_embedding.onnx",
    )
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1,4,8")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.iterations < 1:
        parser.error("--warmup-runs must be non-negative and --iterations positive")

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(args.config_file.resolve()))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.MULTIMODAL.IDENTITY_WEIGHTS = (
        str(args.identity_weights.resolve()) if args.backend == "pytorch" else ""
    )
    cfg.freeze()
    images = collect_images(args.images)
    build_start = time.perf_counter()
    if args.backend == "onnx":
        pipeline = build_onnx_multimodal_pipeline(
            cfg,
            model_path=args.onnx_model.resolve(),
            provider=args.onnx_provider,
            device=args.device,
            warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
        )
        backend_info = pipeline.identity_model.backend_info()
    else:
        pipeline = build_multimodal_pipeline(cfg, device=args.device)
        backend_info = {"backend": "pytorch", "device": str(pipeline.device)}
    build_ms = (time.perf_counter() - build_start) * 1000.0
    device = pipeline.device

    stage = {"detector_ms": 0.0, "segmenter_ms": 0.0, "identity_ms": 0.0}
    detector_call = pipeline.detector.detect
    segmenter_call = pipeline.segmenter.segment

    def timed_detector(*call_args, **call_kwargs):
        synchronize(device)
        start = time.perf_counter()
        result = detector_call(*call_args, **call_kwargs)
        synchronize(device)
        stage["detector_ms"] += (time.perf_counter() - start) * 1000.0
        return result

    def timed_segmenter(*call_args, **call_kwargs):
        synchronize(device)
        start = time.perf_counter()
        result = segmenter_call(*call_args, **call_kwargs)
        synchronize(device)
        stage["segmenter_ms"] += (time.perf_counter() - start) * 1000.0
        return result

    identity_start = {"value": 0.0}

    def identity_pre_hook(module, hook_args, hook_kwargs):
        synchronize(device)
        identity_start["value"] = time.perf_counter()

    def identity_post_hook(module, hook_args, hook_kwargs, output):
        synchronize(device)
        stage["identity_ms"] += (
            time.perf_counter() - identity_start["value"]
        ) * 1000.0

    pipeline.detector.detect = timed_detector
    pipeline.segmenter.segment = timed_segmenter
    pre_handle = pipeline.identity_model.register_forward_pre_hook(
        identity_pre_hook,
        with_kwargs=True,
    )
    post_handle = pipeline.identity_model.register_forward_hook(
        identity_post_hook,
        with_kwargs=True,
    )

    try:
        for _ in range(args.warmup_runs):
            for image in images:
                pipeline.encode_image(image)
        rows = []
        for iteration in range(args.iterations):
            for image in images:
                for key in stage:
                    stage[key] = 0.0
                synchronize(device)
                start = time.perf_counter()
                descriptors = pipeline.encode_image(image)
                synchronize(device)
                total_ms = (time.perf_counter() - start) * 1000.0
                attributed = sum(stage.values())
                rows.append(
                    {
                        "iteration": iteration + 1,
                        "image": str(image),
                        "pets": len(descriptors),
                        "total_ms": total_ms,
                        **stage,
                        "other_preprocessing_ms": max(total_ms - attributed, 0.0),
                    }
                )
    finally:
        pre_handle.remove()
        post_handle.remove()
        pipeline.detector.detect = detector_call
        pipeline.segmenter.segment = segmenter_call

    stage_names = (
        "total_ms",
        "detector_ms",
        "segmenter_ms",
        "identity_ms",
        "other_preprocessing_ms",
    )
    report = {
        "backend": backend_info,
        "pipeline_build_and_warmup_ms": build_ms,
        "images": len(images),
        "warmup_runs": args.warmup_runs,
        "timed_iterations": args.iterations,
        "stages": {
            name: summarize([row[name] for row in rows]) for name in stage_names
        },
        "throughput_images_per_second": 1000.0
        / statistics.fmean([row["total_ms"] for row in rows]),
        "runs": rows,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
