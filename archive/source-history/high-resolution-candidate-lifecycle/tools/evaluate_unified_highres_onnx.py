#!/usr/bin/env python3
"""Evaluate V4 ONNX on the locked high-resolution development split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_highres_data import load_raw_rgb  # noqa: E402
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402
from pet_id.unified_highres_runtime import (  # noqa: E402
    UnifiedHighResolutionONNXRuntimePipeline,
)
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pytorch-report", type=Path, required=True)
    parser.add_argument("--pytorch-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument("--max-abs-tolerance", type=float, default=3e-3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in (
            "model",
            "metadata",
            "checkpoint",
            "protocol_lock",
            "manifest",
            "pytorch_report",
            "pytorch_features",
        )
    }
    output_path = args.output.expanduser().resolve()
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)

    protocol = load_json(paths["protocol_lock"])
    if protocol.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 protocol lock")
    if protocol.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 protocol must remain locked and unscored")
    development = protocol["splits"]["development"]
    if paths["manifest"] != Path(development["path"]).expanduser().resolve():
        raise RuntimeError("Only the locked V4 development split may be evaluated")
    if sha256_file(paths["manifest"]) != development["sha256"]:
        raise RuntimeError("V4 development manifest hash mismatch")

    pytorch_report = load_json(paths["pytorch_report"])
    if pytorch_report.get("purpose") != "unified_v4_real_high_resolution_development_comparison":
        raise RuntimeError("Unexpected PyTorch development report")
    if pytorch_report.get("passed") is not True or pytorch_report.get("blind_data_used") is not False:
        raise RuntimeError("PyTorch V4 development evidence did not pass")
    candidate = pytorch_report["candidate"]
    if candidate.get("checkpoint_sha256") != sha256_file(paths["checkpoint"]):
        raise RuntimeError("PyTorch report checkpoint hash mismatch")
    feature_record = pytorch_report.get("feature_cache") or {}
    if feature_record.get("sha256") != sha256_file(paths["pytorch_features"]):
        raise RuntimeError("PyTorch feature cache hash mismatch")

    metadata = load_json(paths["metadata"])
    model_hash = sha256_file(paths["model"])
    checkpoint_hash = sha256_file(paths["checkpoint"])
    if metadata.get("model_type") != "unified_high_resolution_pet_reid":
        raise RuntimeError("Unexpected ONNX metadata model type")
    if metadata.get("onnx_sha256") != model_hash:
        raise RuntimeError("ONNX metadata hash mismatch")
    if metadata.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("ONNX metadata checkpoint hash mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("V4 ONNX metadata declares external models")

    manifest = load_json(paths["manifest"])
    if manifest.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 development manifest")
    records = list(manifest.get("records", []))
    reference = np.load(paths["pytorch_features"], allow_pickle=False)
    expected = np.asarray(reference["candidate"], dtype=np.float32)
    expected_identities = reference["identities"].astype(str).tolist()
    expected_sources = reference["source_paths"].astype(str).tolist()
    expected_hashes = reference["source_sha256"].astype(str).tolist()
    if expected.shape != (len(records), 512):
        raise RuntimeError("PyTorch feature cache has the wrong shape")

    pipeline = UnifiedHighResolutionONNXRuntimePipeline(
        paths["model"],
        metadata_path=paths["metadata"],
        source_checkpoint=paths["checkpoint"],
        provider=args.provider,
        device=args.provider,
        verify_hash=True,
    )
    backend = pipeline.backend_info()
    expected_provider = (
        "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    )
    if backend["provider"] != expected_provider:
        raise RuntimeError("ONNX Runtime provider fallback is forbidden")

    rows: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_hashes: list[str] = []
    fed_shapes: list[list[int]] = []
    maximum_side = int(backend["maximum_input_side"])
    for index, record in enumerate(records):
        source = Path(record["source_path"]).expanduser().resolve()
        digest = sha256_file(source)
        if digest != str(record["source_sha256"]).casefold():
            raise RuntimeError(f"Source hash differs from manifest: {source}")
        rgb, dimensions = load_raw_rgb(source, maximum_side=maximum_side)
        batch = rgb.numpy().astype(np.float32, copy=False)[None]
        rows.append(pipeline._run(batch)[0].cpu())
        identities.append(str(record["identity"]).casefold())
        source_paths.append(str(source))
        source_hashes.append(digest)
        fed_shapes.append([int(dimensions["fed_height"]), int(dimensions["fed_width"])])
        if index == 0 or (index + 1) % max(args.progress_every, 1) == 0:
            print(
                f"V4 ONNX {args.provider} development: {index + 1}/{len(records)}",
                flush=True,
            )

    actual = torch.stack(rows).numpy().astype(np.float32, copy=False)
    if identities != expected_identities:
        raise RuntimeError("ONNX/PyTorch development identity order differs")
    if source_paths != expected_sources or source_hashes != expected_hashes:
        raise RuntimeError("ONNX/PyTorch development source order differs")
    metrics = retrieval_metrics(torch.from_numpy(actual), identities, source_paths)
    expected_metrics = pytorch_report["candidate_metrics"]
    difference = np.abs(actual - expected)
    cosine = (actual * expected).sum(axis=1) / np.maximum(
        np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1),
        1e-12,
    )
    parity = {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "below_minimum_cosine": int((cosine < args.minimum_cosine).sum()),
    }
    checks = {
        "top1_not_below_pytorch": int(metrics["top1_correct"])
        >= int(expected_metrics["top1_correct"]),
        "top5_not_below_pytorch": int(metrics["top5_correct"])
        >= int(expected_metrics["top5_correct"]),
        "minimum_cosine": parity["minimum_cosine"] >= args.minimum_cosine,
        "max_abs_error": parity["max_abs_error"] <= args.max_abs_tolerance,
        "no_cosine_failures": parity["below_minimum_cosine"] == 0,
        "output_shape": list(actual.shape) == [len(records), 512],
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "unified_v4_high_resolution_onnx_development",
        "blind_data_used": False,
        "model": str(paths["model"]),
        "model_sha256": model_hash,
        "metadata": str(paths["metadata"]),
        "metadata_sha256": sha256_file(paths["metadata"]),
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "protocol_lock": str(paths["protocol_lock"]),
        "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
        "manifest": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "provider": backend,
        "records": len(records),
        "identities": len(set(identities)),
        "fed_shapes": fed_shapes,
        "retrieval": metrics,
        "pytorch_retrieval": expected_metrics,
        "parity_with_pytorch": parity,
        "thresholds": {
            "minimum_cosine": args.minimum_cosine,
            "max_abs_tolerance": args.max_abs_tolerance,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "feature_cache_persisted": False,
        "default_backend_changed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": sha256_file(output_path),
                "provider": backend["provider"],
                "retrieval": metrics,
                "parity_with_pytorch": parity,
                "checks": checks,
                "passed": report["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
