#!/usr/bin/env python3
"""Lock external semantic-v3 baselines and create the v3 acceptance contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_protocol import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--baseline-lock", type=Path, required=True)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v3.json",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def checked_report(path: Path, split: str, protocol_sha256: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("purpose") != "locked_semantic_v3_external_baseline":
        raise RuntimeError(f"Unexpected baseline report purpose: {path}")
    if payload.get("split") != split:
        raise RuntimeError(f"Unexpected baseline split: {path}")
    if payload["protocol_lock"]["sha256"] != protocol_sha256:
        raise RuntimeError(f"Baseline protocol hash mismatch: {path}")
    if payload.get("per_query_results_persisted") is not False:
        raise RuntimeError("Baseline report must not persist per-query results")
    metrics = payload["metrics"]
    if min(int(metrics["top1_correct"]), int(metrics["top5_correct"])) < 0:
        raise RuntimeError("Invalid baseline retrieval counts")
    if split == "blind_test" and payload.get("feature_cache") is not None:
        raise RuntimeError("Blind baseline must not persist a feature cache")
    return payload


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_lock.expanduser().resolve()
    development_path = args.development_report.expanduser().resolve()
    blind_path = args.blind_report.expanduser().resolve()
    baseline_lock_path = args.baseline_lock.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    for path in (protocol_path, development_path, blind_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_sha256 = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_AWAITING_BASELINE":
        raise RuntimeError("Unexpected protocol lock status")
    development = checked_report(
        development_path, "development", protocol_sha256
    )
    blind = checked_report(blind_path, "blind_test", protocol_sha256)
    baseline_keys = (
        "config_sha256",
        "checkpoint_sha256",
        "onnx_sha256",
    )
    for key in baseline_keys:
        if development["baseline"][key] != blind["baseline"][key]:
            raise RuntimeError(f"Development/blind baseline differs: {key}")
    blind_marker = blind_path.with_name(blind_path.name + ".attempt.json")
    if not blind_marker.is_file():
        raise FileNotFoundError(blind_marker)
    marker = json.loads(blind_marker.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETED":
        raise RuntimeError("Blind baseline attempt did not complete")
    if marker.get("report_sha256") != sha256_file(blind_path):
        raise RuntimeError("Blind baseline attempt/report mismatch")

    baseline_lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "name": "semantic_v3_external_bifor_baseline",
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "baseline": development["baseline"],
        "reports": {
            "development": {
                "path": str(development_path),
                "sha256": sha256_file(development_path),
                "metrics": development["metrics"],
                "runtime_coverage": development["runtime_coverage"],
            },
            "blind_test": {
                "path": str(blind_path),
                "sha256": sha256_file(blind_path),
                "attempt_marker": str(blind_marker),
                "attempt_marker_sha256": sha256_file(blind_marker),
                "metrics": blind["metrics"],
                "runtime_coverage": blind["runtime_coverage"],
            },
        },
        "blind_artifacts": {
            "feature_cache_persisted": False,
            "per_query_results_persisted": False,
        },
    }
    write_json(baseline_lock_path, baseline_lock)
    acceptance = {
        "schema_version": 3,
        "protocol_name": "unified_pet_reid_v3_external_strict_noninferiority",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "candidate_input": "float32 RGB [N,3,1280,1280] in 0..255",
            "candidate_output": "L2-normalized float32 [N,512]",
            "single_onnx_graph": True,
            "runtime_external_models": [],
            "blind_data_training_forbidden": True,
            "single_blind_candidate_attempt": True,
            "promotion_rule": (
                "Candidate must match or exceed semantic-v3 aggregate Top-1 "
                "and Top-5 counts on both external development and blind sets."
            ),
            "failure_keeps_existing_default": True,
        },
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "baseline_lock": {
            "path": str(baseline_lock_path),
            "sha256": sha256_file(baseline_lock_path),
        },
        "training": protocol["manifests"]["training_extension"],
        "development": {
            **protocol["manifests"]["development"],
            "minimum_top1_correct": development["metrics"]["top1_correct"],
            "minimum_top5_correct": development["metrics"]["top5_correct"],
        },
        "blind": {
            **protocol["manifests"]["blind_test"],
            "minimum_top1_correct": blind["metrics"]["top1_correct"],
            "minimum_top5_correct": blind["metrics"]["top5_correct"],
        },
        "source_weight_locks": {
            "semantic_v3_checkpoint": {
                "path": development["baseline"]["checkpoint"],
                "sha256": development["baseline"]["checkpoint_sha256"],
            },
            "semantic_v3_config": {
                "path": development["baseline"]["config"],
                "sha256": development["baseline"]["config_sha256"],
            },
            "semantic_v3_onnx": {
                "path": development["baseline"]["onnx_model"],
                "sha256": development["baseline"]["onnx_sha256"],
            },
        },
        "candidate": {"status": "NOT_LOCKED", "promoted": False},
    }
    write_json(acceptance_path, acceptance)
    print(
        json.dumps(
            {
                "baseline_lock": str(baseline_lock_path),
                "baseline_lock_sha256": sha256_file(baseline_lock_path),
                "acceptance": str(acceptance_path),
                "acceptance_sha256": sha256_file(acceptance_path),
                "development_thresholds": {
                    "top1_correct": development["metrics"]["top1_correct"],
                    "top5_correct": development["metrics"]["top5_correct"],
                },
                "blind_thresholds": {
                    "top1_correct": blind["metrics"]["top1_correct"],
                    "top5_correct": blind["metrics"]["top5_correct"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
