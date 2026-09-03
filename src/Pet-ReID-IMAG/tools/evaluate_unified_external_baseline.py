#!/usr/bin/env python3
"""Evaluate the locked legacy-semantic runtime on an external split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import build_pipeline, encode_primary  # noqa: E402
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.release_compatibility import acceptance_protocol_name  # noqa: E402
from pet_id.unified_external_protocol import (  # noqa: E402
    sha256_file,
    validate_raw_manifest,
)
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "blind_test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=get_runtime_profile("legacy-semantic").config,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_runtime_profile("legacy-semantic").identity_weights,
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=get_runtime_profile("legacy-semantic").onnx,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--provider", choices=("cuda", "cpu", "auto"), default="cuda"
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def reserve_attempt(
    output: Path, *, protocol_sha256: str, model_sha256: str
) -> Path:
    marker = output.with_name(output.name + ".attempt.json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "RESERVED",
        "purpose": "one_time_blind_baseline",
        "reserved_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "protocol_lock_sha256": protocol_sha256,
        "model_sha256": model_sha256,
    }
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return marker


def complete_attempt(marker: Path, report_sha256: str) -> None:
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "report_sha256": report_sha256,
        }
    )
    temporary = marker.with_name(marker.name + ".completing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_lock.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    config_path = args.config_file.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model_path = args.onnx_model.expanduser().resolve()
    required = (protocol_path, config_path, checkpoint_path, model_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if args.split == "blind_test" and args.feature_cache is not None:
        raise ValueError("Blind baseline evaluation must not persist features")

    protocol_sha256 = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_name") != acceptance_protocol_name("external-runtime"):
        raise RuntimeError("Unexpected external protocol")
    manifest_record = protocol["manifests"][args.split]
    manifest_path = Path(manifest_record["path"]).expanduser().resolve()
    if sha256_file(manifest_path) != manifest_record["sha256"]:
        raise RuntimeError("Protocol manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = validate_raw_manifest(manifest, expected_split=args.split)
    if summary != {
        "records": manifest_record["records"],
        "identities": manifest_record["identities"],
    }:
        raise RuntimeError("Protocol manifest summary mismatch")

    pipeline = build_pipeline(
        config_path,
        checkpoint_path,
        args.device,
        backend="onnx",
        onnx_model=model_path,
        onnx_provider=args.provider,
        onnx_warmup_batches=(1,),
        verify_onnx_source_checkpoint=True,
    )
    marker = None
    if args.split == "blind_test":
        marker = reserve_attempt(
            output_path,
            protocol_sha256=protocol_sha256,
            model_sha256=sha256_file(model_path),
        )

    features: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    detections_total = 0
    fallback_records = 0
    dual_branch_records = 0
    records = list(manifest["records"])
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            source = Path(record["source_path"]).expanduser().resolve()
            descriptor, diagnostic = encode_primary(pipeline, source)
            feature = torch.nn.functional.normalize(
                descriptor.fused_feature.float(), dim=0
            )
            if not torch.isfinite(feature).all():
                raise FloatingPointError(f"Non-finite descriptor: {source}")
            features.append(feature.cpu())
            identities.append(str(record["identity"]).casefold())
            source_paths.append(str(source))
            source_sha256.append(str(record["source_sha256"]))
            detections_total += int(diagnostic["detections"])
            fallback_records += int(descriptor.detection is None)
            dual_branch_records += int(all(descriptor.branch_available))
            if index == 1 or index % max(args.progress_every, 1) == 0:
                print(
                    f"legacy-semantic {args.split}: {index}/{len(records)}",
                    flush=True,
                )

    feature_tensor = torch.stack(features)
    metrics = retrieval_metrics(feature_tensor, identities, source_paths)
    compact_metrics = {
        key: value for key, value in metrics.items() if key != "queries"
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "locked_legacy_semantic_external_baseline",
        "split": args.split,
        "blind_candidate_used": False,
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            **summary,
        },
        "baseline": {
            "name": "legacy-semantic",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_model": str(model_path),
            "onnx_sha256": sha256_file(model_path),
            "backend": pipeline.identity_model.backend_info(),
        },
        "selection_policy": "largest detected dog, then detection confidence",
        "runtime_coverage": {
            "records": len(records),
            "detections_total": detections_total,
            "fallback_records": fallback_records,
            "dual_branch_records": dual_branch_records,
        },
        "embedding": {
            "shape": list(feature_tensor.shape),
            "minimum_norm": float(feature_tensor.norm(dim=1).min()),
            "maximum_norm": float(feature_tensor.norm(dim=1).max()),
        },
        "metrics": compact_metrics,
        "per_query_results_persisted": False,
        "feature_cache": None,
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
    report_sha256 = sha256_file(output_path)
    if marker is not None:
        complete_attempt(marker, report_sha256)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": report_sha256,
                "split": args.split,
                "runtime_coverage": report["runtime_coverage"],
                "metrics": compact_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
