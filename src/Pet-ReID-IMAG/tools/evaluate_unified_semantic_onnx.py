#!/usr/bin/env python3
"""Run fixed-protocol retrieval with the exported one-graph ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_training import retrieval_metrics, sha256_file  # noqa: E402
from pet_id.release_compatibility import historical_run_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "shared-fusion-baseline")
        / "dev_validation_manifest.json",
    )
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--minimum-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-top5-correct", type=int, default=198)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    model_path = args.model.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    reference_path = args.reference_cache.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else model_path.with_name("metadata.json")
    )
    for path in (
        model_path,
        metadata_path,
        manifest_path,
        reference_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    pipeline = UnifiedONNXRuntimePipeline(
        model_path,
        provider=args.provider,
        metadata_path=metadata_path,
        verify_hash=True,
    )
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=pipeline.input_size,
        training=False,
        allow_letterbox_upscale=pipeline.letterbox_allow_upscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    outputs = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    processed = 0
    for batch in loader:
        rgb = batch["rgb"].numpy().astype(np.float32, copy=False)
        outputs.append(pipeline._run(rgb))
        identities.extend(batch["identity"])
        source_paths.extend(batch["source_path"])
        source_sha256.extend(batch["source_sha256"])
        processed += int(rgb.shape[0])
        if processed == args.batch_size or processed % 25 == 0:
            print(
                f"unified ONNX {args.provider}: {processed}/{len(dataset)}",
                flush=True,
            )
    embedding = torch.cat(outputs)
    metrics = retrieval_metrics(
        embedding,
        identities,
        source_paths,
        gallery_images_per_identity=2,
    )
    reference = np.load(reference_path, allow_pickle=False)
    if reference["source_sha256"].astype(str).tolist() != source_sha256:
        raise RuntimeError("Reference cache and manifest order differ")
    expected = torch.from_numpy(reference["embedding"]).float()
    actual = torch.nn.functional.normalize(embedding.float(), dim=1)
    expected = torch.nn.functional.normalize(expected, dim=1)
    difference = (expected - actual).abs()
    cosine = (expected * actual).sum(dim=1)
    worst_indices = cosine.argsort()[:20].tolist()
    worst_samples = [
        {
            "index": int(index),
            "identity": identities[index],
            "source_path": source_paths[index],
            "cosine": float(cosine[index]),
            "max_abs_error": float(difference[index].max()),
        }
        for index in worst_indices
    ]
    parity = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "below_minimum_cosine": int((cosine < args.minimum_cosine).sum()),
        "worst_samples": worst_samples,
    }
    feature_cache_record = None
    if args.feature_cache is not None:
        feature_cache_path = args.feature_cache.expanduser().resolve()
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_cache_path,
            embedding=actual.numpy(),
            cosine_to_pytorch=cosine.numpy(),
            source_sha256=np.asarray(source_sha256),
        )
        feature_cache_record = {"path": str(feature_cache_path)}
    checks = {
        "top1": metrics["top1_correct"] >= args.minimum_top1_correct,
        "top5": metrics["top5_correct"] >= args.minimum_top5_correct,
        "parity": parity["minimum_cosine"] >= args.minimum_cosine,
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_semantic_pet_reid_onnx",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reference_cache": str(reference_path),
        "reference_cache_sha256": sha256_file(reference_path),
        "records": len(dataset),
        "provider": pipeline.backend_info(),
        "retrieval": metrics,
        "parity_with_pytorch": parity,
        "thresholds": {
            "minimum_top1_correct": args.minimum_top1_correct,
            "minimum_top5_correct": args.minimum_top5_correct,
            "minimum_cosine": args.minimum_cosine,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "feature_cache": feature_cache_record,
        "default_backend_changed": False,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
