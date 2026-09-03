#!/usr/bin/env python3
"""Benchmark the dynamic one-graph spatial-detail ONNX runtime."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_highres_data import validate_highres_dimensions  # noqa: E402
from pet_id.unified_highres_runtime import (  # noqa: E402
    UnifiedHighResolutionONNXRuntimePipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider",
        action="append",
        choices=("cuda", "cpu"),
        default=[],
        help="May be repeated; defaults to CUDA and CPU",
    )
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def parse_shape(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        raise ValueError(f"Shape must be HxW, got {value!r}")
    height, width = (int(item) for item in text.split("x", 1))
    return validate_highres_dimensions(height, width)


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "runs": len(values),
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "maximum_ms": max(values),
        "images_per_second": 1000.0 / statistics.mean(values),
        "all_ms": ordered,
    }


def main() -> None:
    args = parse_args()
    model = args.model.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (model, metadata, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    providers = args.provider or ["cuda", "cpu"]
    shapes = [parse_shape(value) for value in args.shape]
    if not shapes:
        shapes = [(208, 126), (800, 600), (1824, 1368), (4032, 3024)]

    results = {}
    for provider in providers:
        start = time.perf_counter()
        pipeline = UnifiedHighResolutionONNXRuntimePipeline(
            model,
            metadata_path=metadata,
            source_checkpoint=checkpoint,
            provider=provider,
            device=provider,
            verify_hash=True,
        )
        session_initialization_ms = (time.perf_counter() - start) * 1000.0
        provider_rows = []
        for height, width in shapes:
            value = np.zeros((1, 3, height, width), dtype=np.float32)
            for _ in range(args.warmup):
                pipeline._run(value)
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                embedding = pipeline._run(value)
                elapsed = (time.perf_counter() - start) * 1000.0
                if tuple(embedding.shape) != (1, 512):
                    raise RuntimeError("Candidate benchmark output shape changed")
                timings.append(elapsed)
            provider_rows.append(
                {
                    "height": height,
                    "width": width,
                    "batch_size": 1,
                    "latency": summary(timings),
                }
            )
        backend = pipeline.backend_info()
        results[provider] = {
            "provider": backend["provider"],
            "provider_chain": backend["provider_chain"],
            "session_initialization_ms": session_initialization_ms,
            "shapes": provider_rows,
        }
        del pipeline
        gc.collect()

    expected = {
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    checks = {
        f"{provider}_no_fallback": results[provider]["provider"] == expected[provider]
        for provider in providers
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "spatial_detail_dynamic_onnx_runtime_benchmark",
        "model": str(model),
        "model_sha256": sha256_file(model),
        "metadata": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
        "checks": checks,
        "passed": all(checks.values()),
        "default_backend_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
