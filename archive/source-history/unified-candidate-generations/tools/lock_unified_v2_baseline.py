#!/usr/bin/env python3
"""Begin or complete the one-time aggregate semantic-v3 v2 baseline lock."""

from __future__ import annotations

import argparse
import json
import os
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

from pet_id.unified_fresh_protocol import sha256_file  # noqa: E402
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("begin", "complete"))
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_unified_v2_protocol_20260831/protocol_lock.json",
    )
    parser.add_argument(
        "--development-cache",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/unified/v2/teacher_development_semantic_v3.npz",
    )
    parser.add_argument(
        "--blind-cache",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/unified/v2/teacher_blind_semantic_v3.npz",
    )
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=WORKSPACE
        / "models/selected/dogfacenet_semantic_v3_v1/model_final.pth",
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=WORKSPACE / "models/selected/dogfacenet_semantic_v3_v1/config.yaml",
    )
    parser.add_argument(
        "--acceptance-output",
        type=Path,
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v2.json",
    )
    return parser.parse_args()


def cache_metadata(path: Path) -> tuple[dict[str, Any], Path]:
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("archive_sha256") != sha256_file(path):
        raise RuntimeError(f"Teacher cache hash mismatch: {path}")
    return metadata, metadata_path


def aggregate_metrics(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as payload:
        features = torch.from_numpy(np.asarray(payload["embedding"], dtype=np.float32))
        identities = payload["identities"].astype(str).tolist()
        source_paths = payload["source_paths"].astype(str).tolist()
    metrics = retrieval_metrics(
        features,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=False,
    )
    result = {
        key: metrics[key]
        for key in (
            "gallery_images_per_identity",
            "gallery_records",
            "query_records",
            "top1_correct",
            "top1_accuracy",
            "top5_correct",
            "top5_accuracy",
            "mean_reciprocal_rank",
            "auc",
            "same_score_mean",
            "different_score_mean",
        )
    }
    result.update(
        {
            "records": len(identities),
            "identities": len({identity.casefold() for identity in identities}),
            "queries_count": metrics["query_records"],
        }
    )
    return result


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_lock.expanduser().resolve()
    semantic_checkpoint = args.semantic_checkpoint.expanduser().resolve()
    semantic_config = args.semantic_config.expanduser().resolve()
    development_cache = args.development_cache.expanduser().resolve()
    blind_cache = args.blind_cache.expanduser().resolve()
    acceptance_output = args.acceptance_output.expanduser().resolve()
    for path in (protocol_path, semantic_checkpoint, semantic_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "BASELINE_PENDING":
        raise RuntimeError("v2 protocol is not awaiting its baseline")
    marker_path = Path(protocol["baseline_attempt_marker"]).expanduser().resolve()
    baseline_dir = protocol_path.parent / "baseline"
    baseline_lock_path = baseline_dir / "baseline_lock.json"

    if args.mode == "begin":
        if marker_path.exists() or baseline_lock_path.exists():
            raise FileExistsError("The one-time v2 baseline attempt already exists")
        marker = {
            "schema_version": 1,
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "protocol_lock": str(protocol_path),
            "protocol_lock_sha256": sha256_file(protocol_path),
            "blind_manifest": protocol["splits"]["blind_test"],
            "semantic_checkpoint": {
                "path": str(semantic_checkpoint),
                "sha256": sha256_file(semantic_checkpoint),
            },
            "semantic_config": {
                "path": str(semantic_config),
                "sha256": sha256_file(semantic_config),
            },
            "expected_blind_cache": str(blind_cache),
            "aggregate_only": True,
        }
        atomic_json(marker_path, marker)
        print(json.dumps(marker, ensure_ascii=False, indent=2))
        return

    if not marker_path.is_file():
        raise RuntimeError("Baseline begin marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "RUNNING":
        raise RuntimeError("Baseline attempt is not running")
    if marker.get("protocol_lock_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Protocol lock changed after baseline begin")
    if acceptance_output.exists() or baseline_lock_path.exists():
        raise FileExistsError("Baseline acceptance has already been completed")
    development_metadata, development_metadata_path = cache_metadata(
        development_cache
    )
    blind_metadata, blind_metadata_path = cache_metadata(blind_cache)
    expected_checkpoint_hash = marker["semantic_checkpoint"]["sha256"]
    expected_config_hash = marker["semantic_config"]["sha256"]
    for name, metadata, split_name in (
        ("development", development_metadata, "development"),
        ("blind", blind_metadata, "blind_test"),
    ):
        if metadata.get("checkpoint_sha256") != expected_checkpoint_hash:
            raise RuntimeError(f"{name} cache uses the wrong semantic checkpoint")
        if metadata.get("config_sha256") != expected_config_hash:
            raise RuntimeError(f"{name} cache uses the wrong semantic config")
        if metadata.get("manifest_sha256") != protocol["splits"][split_name][
            "sha256"
        ]:
            raise RuntimeError(f"{name} cache uses the wrong locked manifest")

    development_metrics = aggregate_metrics(development_cache)
    blind_metrics = aggregate_metrics(blind_cache)
    if blind_metrics["queries_count"] != protocol["splits"]["blind_test"]["queries"]:
        raise RuntimeError("Blind query count differs from the protocol lock")
    reports = {}
    for name, metrics, cache, metadata_path in (
        (
            "development",
            development_metrics,
            development_cache,
            development_metadata_path,
        ),
        ("blind_test", blind_metrics, blind_cache, blind_metadata_path),
    ):
        report = {
            "schema_version": 1,
            "purpose": "semantic_v3_aggregate_noninferiority_baseline",
            "split": name,
            "aggregate_only": True,
            "per_query_results_stored": False,
            "teacher_cache": {
                "path": str(cache),
                "sha256": sha256_file(cache),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
            },
            "metrics": metrics,
        }
        report_path = baseline_dir / f"{name}.json"
        atomic_json(report_path, report)
        reports[name] = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "metrics": metrics,
        }

    baseline_lock = {
        "schema_version": 1,
        "status": "LOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_lock": str(protocol_path),
        "protocol_lock_sha256": sha256_file(protocol_path),
        "semantic_checkpoint": marker["semantic_checkpoint"],
        "semantic_config": marker["semantic_config"],
        "reports": reports,
        "blind_thresholds": {
            "minimum_top1_correct": blind_metrics["top1_correct"],
            "minimum_top5_correct": blind_metrics["top5_correct"],
            "queries": blind_metrics["queries_count"],
            "rule": "candidate counts must match or exceed semantic-v3",
        },
        "candidate_attempt_marker": protocol["candidate_attempt_marker"],
    }
    atomic_json(baseline_lock_path, baseline_lock)
    acceptance = {
        "schema_version": 2,
        "protocol_name": "unified_pet_reid_v2_strict_noninferiority",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "candidate_input": "float32 RGB [N,3,1280,1280] in 0..255",
            "candidate_output": "L2-normalized float32 [N,512]",
            "single_onnx_graph": True,
            "runtime_external_models": [],
            "blind_data_training_forbidden": True,
            "promotion_rule": (
                "Candidate must match or exceed semantic-v3 aggregate Top-1 "
                "and Top-5 counts on the locked fresh blind set."
            ),
            "failure_keeps_existing_default": True,
        },
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "baseline_lock": {
            "path": str(baseline_lock_path),
            "sha256": sha256_file(baseline_lock_path),
        },
        "development": protocol["splits"]["development"],
        "training": protocol["splits"]["training"],
        "blind": {
            **protocol["splits"]["blind_test"],
            **baseline_lock["blind_thresholds"],
        },
        "source_weight_locks": {
            "semantic_v3_checkpoint": marker["semantic_checkpoint"],
            "semantic_v3_config": marker["semantic_config"],
            "dog_arcface_checkpoint": {
                "path": str((WORKSPACE / "models/pretrained/dog.pt").resolve()),
                "sha256": sha256_file(WORKSPACE / "models/pretrained/dog.pt"),
            },
        },
        "candidate": {"status": "NOT_LOCKED", "promoted": False},
    }
    atomic_json(acceptance_output, acceptance)
    marker.update(
        {
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "baseline_lock": str(baseline_lock_path),
            "baseline_lock_sha256": sha256_file(baseline_lock_path),
            "acceptance": str(acceptance_output),
            "acceptance_sha256": sha256_file(acceptance_output),
        }
    )
    atomic_json(marker_path, marker)
    print(
        json.dumps(
            {
                "baseline_lock": str(baseline_lock_path),
                "baseline_lock_sha256": sha256_file(baseline_lock_path),
                "acceptance": str(acceptance_output),
                "acceptance_sha256": sha256_file(acceptance_output),
                "development_metrics": development_metrics,
                "blind_metrics": blind_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
