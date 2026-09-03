#!/usr/bin/env python3
"""Benchmark batch-1 PyTorch and ONNX runtimes for one unified candidate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_semantic import UnifiedSemanticPetReIDExport  # noqa: E402
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)
from pet_id.unified_training import sha256_file  # noqa: E402
from pet_id.release_compatibility import historical_run_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "fresh-baseline")
        / "prepared/development/manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    return parser.parse_args()


def summarize(values_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "iterations": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "images_per_second": float(1000.0 / values.mean()),
    }


def measure(
    call: Callable[[], torch.Tensor | np.ndarray],
    *,
    warmup: int,
    iterations: int,
    synchronize: Callable[[], None] | None = None,
) -> tuple[dict[str, float | int], np.ndarray]:
    durations = []
    last: torch.Tensor | np.ndarray | None = None
    for index in range(warmup + iterations):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter_ns()
        last = call()
        if synchronize is not None:
            synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if index >= warmup:
            durations.append(elapsed_ms)
    assert last is not None
    if torch.is_tensor(last):
        last = last.detach().float().cpu().numpy()
    output = np.asarray(last, dtype=np.float32)
    if output.shape != (1, 512) or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected benchmark output: {output.shape}")
    norm = float(np.linalg.norm(output[0]))
    if not np.isclose(norm, 1.0, atol=2e-4, rtol=2e-4):
        raise RuntimeError(f"Benchmark output is not normalized: {norm}")
    result = summarize(durations)
    result["output_l2_norm"] = norm
    return result, output


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    onnx_model = args.onnx_model.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (checkpoint, onnx_model, metadata, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)

    metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
    input_size = int(metadata_payload["inputs"]["rgb"]["shape"][2])
    dataset = UnifiedManifestDataset(
        manifest,
        input_size=input_size,
        training=False,
        allow_letterbox_upscale=False,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)
    sample = dataset[args.sample_index]["rgb"].unsqueeze(0).contiguous()
    sample_array = sample.numpy().astype(np.float32, copy=False)
    results: dict[str, dict] = {}
    outputs: dict[str, np.ndarray] = {}

    if not args.skip_pytorch:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA benchmark requested but CUDA is unavailable")
        started = time.perf_counter_ns()
        model, _ = build_unified_semantic_from_checkpoint(
            checkpoint,
            device="cuda",
            verify_sources=True,
        )
        wrapper = UnifiedSemanticPetReIDExport(model).cuda().eval()
        resident = sample.cuda().contiguous()
        torch.cuda.synchronize()
        build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            resident_result, outputs["pytorch_cuda_resident"] = measure(
                lambda: wrapper(resident),
                warmup=args.warmup,
                iterations=args.iterations,
                synchronize=torch.cuda.synchronize,
            )
            host_result, outputs["pytorch_cuda_host_io"] = measure(
                lambda: wrapper(
                    torch.from_numpy(sample_array).cuda().contiguous()
                ).cpu(),
                warmup=args.warmup,
                iterations=args.iterations,
                synchronize=torch.cuda.synchronize,
            )
        resident_result["scope"] = "model_forward_with_gpu_resident_input_output"
        host_result["scope"] = "cpu_input_to_cuda_model_to_cpu_output"
        resident_result["build_ms"] = build_ms
        host_result["build_ms"] = build_ms
        resident_result["torch_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        results["pytorch_cuda_resident"] = resident_result
        results["pytorch_cuda_host_io"] = host_result
        del resident, wrapper, model
        gc.collect()
        torch.cuda.empty_cache()

    for provider, skipped in (
        ("cuda", args.skip_cuda),
        ("cpu", args.skip_cpu),
    ):
        if skipped:
            continue
        started = time.perf_counter_ns()
        pipeline = UnifiedONNXRuntimePipeline(
            onnx_model,
            provider=provider,
            metadata_path=metadata,
            verify_hash=True,
            warmup_batches=(),
        )
        build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        result, outputs[f"onnx_{provider}"] = measure(
            lambda p=pipeline: p._run(sample_array),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result["scope"] = "cpu_input_through_deployment_pipeline_to_cpu_output"
        result["build_ms"] = build_ms
        result["provider"] = pipeline.session.get_providers()[0]
        results[f"onnx_{provider}"] = result
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parity = {}
    names = list(outputs)
    if names:
        reference = outputs[names[0]]
        for name in names[1:]:
            actual = outputs[name]
            parity[name] = {
                "reference": names[0],
                "cosine": float(
                    np.sum(reference * actual)
                    / max(
                        float(np.linalg.norm(reference) * np.linalg.norm(actual)),
                        1e-12,
                    )
                ),
                "max_abs_error": float(np.max(np.abs(reference - actual))),
            }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "unified_semantic_batch1_runtime_benchmark",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_model": str(onnx_model),
        "onnx_sha256": sha256_file(onnx_model),
        "metadata": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "sample_index": args.sample_index,
        "batch_size": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
        "parity": parity,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
