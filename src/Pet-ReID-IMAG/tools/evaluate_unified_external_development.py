#!/usr/bin/env python3
"""Evaluate one single-graph UnifiedPetReID ONNX on external development."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import load_exif_oriented_bgr  # noqa: E402
from pet_id.unified_external_protocol import (  # noqa: E402
    sha256_file,
    validate_raw_manifest,
)
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_training import retrieval_metrics  # noqa: E402
from pet_id.release_compatibility import acceptance_protocol_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--provider", choices=("cuda", "cpu", "auto"), default="cuda"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acceptance_path = args.acceptance.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (acceptance_path, model_path, metadata_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if (
        acceptance.get("protocol_name")
        != acceptance_protocol_name("external-development")
    ):
        raise RuntimeError("Unexpected external acceptance protocol")
    protocol_path = Path(acceptance["protocol_lock"]["path"]).resolve()
    if sha256_file(protocol_path) != acceptance["protocol_lock"]["sha256"]:
        raise RuntimeError("Protocol lock differs from acceptance")
    manifest_path = Path(acceptance["development"]["path"]).resolve()
    if sha256_file(manifest_path) != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest differs from acceptance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = validate_raw_manifest(manifest, expected_split="development")

    pipeline = UnifiedONNXRuntimePipeline(
        model_path,
        metadata_path=metadata_path,
        source_checkpoint=checkpoint_path,
        provider=args.provider,
        device=args.device,
        verify_hash=True,
        warmup_batches=(1, 2),
    )
    backend = pipeline.backend_info()
    if backend.get("single_graph") is not True or backend.get("external_models") != []:
        raise RuntimeError("Candidate is not a strict one-graph runtime")
    features: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    records = list(manifest["records"])
    for index, record in enumerate(records, start=1):
        source = Path(record["source_path"]).expanduser().resolve()
        descriptors = pipeline.encode_image(load_exif_oriented_bgr(source))
        if len(descriptors) != 1:
            raise RuntimeError("Unified runtime must emit exactly one descriptor")
        feature = torch.nn.functional.normalize(
            descriptors[0].fused_feature.float(), dim=0
        )
        if not torch.isfinite(feature).all():
            raise FloatingPointError(f"Non-finite descriptor: {source}")
        features.append(feature.cpu())
        identities.append(str(record["identity"]).casefold())
        source_paths.append(str(source))
        source_sha256.append(str(record["source_sha256"]))
        if index == 1 or index % max(args.progress_every, 1) == 0:
            print(f"unified development: {index}/{len(records)}", flush=True)

    feature_tensor = torch.stack(features)
    metrics = retrieval_metrics(feature_tensor, identities, source_paths)
    checks = {
        "top1": int(metrics["top1_correct"])
        >= int(acceptance["development"]["minimum_top1_correct"]),
        "top5": int(metrics["top5_correct"])
        >= int(acceptance["development"]["minimum_top5_correct"]),
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "external_development_model_selection",
        "blind_data_used": False,
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            **summary,
        },
        "candidate": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_model": str(model_path),
            "onnx_sha256": sha256_file(model_path),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "backend": backend,
        },
        "embedding": {
            "shape": list(feature_tensor.shape),
            "minimum_norm": float(feature_tensor.norm(dim=1).min()),
            "maximum_norm": float(feature_tensor.norm(dim=1).max()),
        },
        "metrics": metrics,
        "noninferiority": {
            "minimum_top1_correct": acceptance["development"][
                "minimum_top1_correct"
            ],
            "minimum_top5_correct": acceptance["development"][
                "minimum_top5_correct"
            ],
            "checks": checks,
            "passed": all(checks.values()),
        },
        "per_query_results_persisted": False,
        "feature_cache": None,
        "passed": all(checks.values()),
        "default_backend_changed": False,
    }
    if args.feature_cache is not None:
        cache_path = args.feature_cache.expanduser().resolve()
        if cache_path.exists():
            raise FileExistsError(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embedding=feature_tensor.numpy(),
            identities=np.asarray(identities),
            source_paths=np.asarray(source_paths),
            source_sha256=np.asarray(source_sha256),
        )
        report["feature_cache"] = {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
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
                "metrics": metrics,
                "noninferiority": report["noninferiority"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
