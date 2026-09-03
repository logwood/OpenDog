#!/usr/bin/env python3
"""Lock one fully validated external unified v3 candidate before blind use."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_training import load_acceptance, sha256_file  # noqa: E402
from pet_id.unified_v3_candidate import (  # noqa: E402
    EMBEDDING_OUTPUT_CONTRACT,
    PROTOCOL_NAME,
    RGB_INPUT_CONTRACT,
    validate_candidate_lock,
)


LOCKED_CODE_PATHS = (
    "src/Pet-ReID-IMAG/pet_id/unified_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_external_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_external_model.py",
    "src/Pet-ReID-IMAG/pet_id/unified_runtime.py",
    "src/Pet-ReID-IMAG/pet_id/unified_semantic.py",
    "src/Pet-ReID-IMAG/pet_id/unified_semantic_checkpoint.py",
    "src/Pet-ReID-IMAG/pet_id/unified_training.py",
    "src/Pet-ReID-IMAG/pet_id/unified_v3_candidate.py",
    "src/Pet-ReID-IMAG/tools/evaluate_unified_v3_blind.py",
    "src/Pet-ReID-IMAG/tools/lock_unified_v3_candidate.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--legacy-report", type=Path, required=True)
    parser.add_argument("--onnx-cpu-report", type=Path, required=True)
    parser.add_argument("--onnx-cuda-report", type=Path, required=True)
    parser.add_argument("--export-validation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v3.json",
    )
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--attempt-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_precision(payload: dict[str, Any], name: str) -> None:
    precision = payload.get("cuda_precision") or payload.get(
        "pytorch_cuda_precision"
    )
    if precision != {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}:
        raise RuntimeError(f"{name} was not evaluated with strict CUDA precision")


def main() -> None:
    args = parse_args()
    paths = {
        name: file(getattr(args, name))
        for name in (
            "checkpoint",
            "model",
            "metadata",
            "development_report",
            "legacy_report",
            "onnx_cpu_report",
            "onnx_cuda_report",
            "export_validation",
            "benchmark",
            "acceptance",
        )
    }
    output = args.output.expanduser().resolve()
    blind_output = args.blind_output.expanduser().resolve()
    attempt_marker = args.attempt_marker.expanduser().resolve()
    for target in (output, blind_output, attempt_marker):
        if target.exists():
            raise FileExistsError(target)
    if output.parent != paths["checkpoint"].parent:
        raise RuntimeError("Candidate lock must be beside its checkpoint")
    if blind_output.parent != output.parent or attempt_marker.parent != output.parent:
        raise RuntimeError("Blind report and marker must be beside the candidate lock")
    if len({output, blind_output, attempt_marker}) != 3:
        raise RuntimeError("Candidate lock, blind report, and marker must be distinct")

    acceptance = load_acceptance(
        paths["acceptance"],
        expected_protocol=PROTOCOL_NAME,
    )
    protocol_path = file(acceptance["protocol_lock"]["path"])
    baseline_path = file(acceptance["baseline_lock"]["path"])
    if sha256_file(protocol_path) != acceptance["protocol_lock"]["sha256"]:
        raise RuntimeError("Protocol lock changed")
    if sha256_file(baseline_path) != acceptance["baseline_lock"]["sha256"]:
        raise RuntimeError("Baseline lock changed")
    protocol = report(protocol_path)
    baseline = report(baseline_path)
    if protocol.get("policy", {}).get("blind_candidate_attempts") != 1:
        raise RuntimeError("External protocol is not one-shot")
    for split in ("development", "blind"):
        minimum = acceptance[split]
        baseline_key = "development" if split == "development" else "blind_test"
        metrics = baseline["reports"][baseline_key]["metrics"]
        if int(metrics["top1_correct"]) != int(minimum["minimum_top1_correct"]):
            raise RuntimeError(f"Locked {split} Top-1 baseline changed")
        if int(metrics["top5_correct"]) != int(minimum["minimum_top5_correct"]):
            raise RuntimeError(f"Locked {split} Top-5 baseline changed")

    checkpoint_hash = sha256_file(paths["checkpoint"])
    model_hash = sha256_file(paths["model"])
    metadata_hash = sha256_file(paths["metadata"])
    metadata = report(paths["metadata"])
    if metadata.get("model_type") != "unified_external_joint_pet_reid":
        raise RuntimeError("Unexpected candidate model type")
    if metadata.get("onnx_sha256") != model_hash:
        raise RuntimeError("Metadata ONNX hash mismatch")
    if metadata.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Metadata checkpoint hash mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("Unified v3 declares external runtime models")
    if metadata.get("inputs") != RGB_INPUT_CONTRACT:
        raise RuntimeError("Candidate RGB input contract changed")
    if metadata.get("outputs") != EMBEDDING_OUTPUT_CONTRACT:
        raise RuntimeError("Candidate embedding output contract changed")

    development = report(paths["development_report"])
    if development.get("purpose") != "unified_v3_external_joint_development":
        raise RuntimeError("Wrong PyTorch development purpose")
    if development.get("passed") is not True or development.get("blind_data_used") is not False:
        raise RuntimeError("PyTorch development evidence did not pass")
    if development.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Development checkpoint hash mismatch")
    if development.get("manifest", {}).get("sha256") != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest hash mismatch")
    strict_precision(development, "PyTorch development")
    candidate_metrics = development["candidate"]
    parent_metrics = development["parent_base"]
    for rank in ("top1", "top5"):
        key = f"{rank}_correct"
        if int(candidate_metrics[key]) < int(
            acceptance["development"][f"minimum_{key}"]
        ):
            raise RuntimeError(f"Development {rank} is below semantic-v3")
        if int(candidate_metrics[key]) < int(parent_metrics[key]):
            raise RuntimeError(f"Development {rank} is below the fixed parent")
    if not any(
        int(candidate_metrics[f"top{k}_correct"])
        > int(parent_metrics[f"top{k}_correct"])
        for k in (1, 5)
    ):
        raise RuntimeError("Candidate has no real development rank gain")

    legacy = report(paths["legacy_report"])
    if legacy.get("purpose") != "external_joint_legacy_v2_development_guard":
        raise RuntimeError("Wrong legacy guard purpose")
    if legacy.get("passed") is not True or legacy.get("blind_data_used") is not False:
        raise RuntimeError("Legacy clean/conflict evidence did not pass")
    if legacy.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Legacy checkpoint hash mismatch")
    strict_precision(legacy, "Legacy clean/conflict")
    if legacy.get("parent_noninferiority", {}).get("passed") is not True:
        raise RuntimeError("Legacy parent noninferiority failed")
    for split in ("clean", "conflict"):
        row = legacy["candidate"][split]
        if (int(row["top1_correct"]), int(row["top5_correct"])) != (70, 72):
            raise RuntimeError(f"Legacy {split} rank guard changed")

    onnx_reports = {}
    for name, provider in (
        ("onnx_cpu_report", "CPUExecutionProvider"),
        ("onnx_cuda_report", "CUDAExecutionProvider"),
    ):
        payload = report(paths[name])
        onnx_reports[name] = payload
        if payload.get("passed") is not True or payload.get("blind_data_used") is not False:
            raise RuntimeError(f"{name} did not pass")
        if payload.get("model_sha256") != model_hash:
            raise RuntimeError(f"{name} ONNX hash mismatch")
        if payload.get("manifest_sha256") != acceptance["development"]["sha256"]:
            raise RuntimeError(f"{name} manifest hash mismatch")
        if payload.get("provider", {}).get("provider") != provider:
            raise RuntimeError(f"{name} provider fallback detected")
        retrieval = payload["retrieval"]
        for key in ("top1_correct", "top5_correct"):
            if int(retrieval[key]) < int(candidate_metrics[key]):
                raise RuntimeError(f"{name} {key} is below PyTorch")
        parity = payload["parity_with_pytorch"]
        if float(parity["minimum_cosine"]) < 0.9999:
            raise RuntimeError(f"{name} full-set cosine failed")
        if int(parity["below_minimum_cosine"]) != 0:
            raise RuntimeError(f"{name} has full-set parity failures")

    export = report(paths["export_validation"])
    if export.get("passed") is not True or export.get("onnx_checker") != "passed":
        raise RuntimeError("Export validation did not pass")
    if export.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Export checkpoint hash mismatch")
    if export.get("onnx", {}).get("sha256") != model_hash:
        raise RuntimeError("Export ONNX hash mismatch")
    strict_precision(export, "ONNX export")
    providers = {row["provider"]: row for row in export.get("providers", [])}
    for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        runs = providers.get(provider, {}).get("runs", [])
        if sorted(int(row["batch_size"]) for row in runs) != [1, 2]:
            raise RuntimeError(f"Export lacks {provider} dynamic batch 1/2")
        if any(row["embedding"]["minimum_cosine"] < 0.9999 for row in runs):
            raise RuntimeError(f"Export {provider} parity failed")

    benchmark = report(paths["benchmark"])
    if benchmark.get("passed") is not True:
        raise RuntimeError("Runtime benchmark did not pass")
    if benchmark.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Benchmark checkpoint hash mismatch")
    if benchmark.get("onnx_sha256") != model_hash:
        raise RuntimeError("Benchmark ONNX hash mismatch")
    strict_precision(benchmark.get("environment", {}), "Benchmark")
    results = benchmark.get("results", {})
    if results.get("onnx_cuda", {}).get("provider") != "CUDAExecutionProvider":
        raise RuntimeError("CUDA benchmark provider mismatch")
    if results.get("onnx_cpu", {}).get("provider") != "CPUExecutionProvider":
        raise RuntimeError("CPU benchmark provider mismatch")

    evidence_paths = {
        "development_pytorch": paths["development_report"],
        "legacy_clean_conflict": paths["legacy_report"],
        "development_onnx_cpu": paths["onnx_cpu_report"],
        "development_onnx_cuda": paths["onnx_cuda_report"],
        "export_validation": paths["export_validation"],
        "benchmark": paths["benchmark"],
    }
    code_hashes = {}
    for relative in LOCKED_CODE_PATHS:
        code_path = file(WORKSPACE / relative)
        code_hashes[str(code_path)] = sha256_file(code_path)
    payload = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "status": "LOCKED_UNSCORED",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "single_blind_attempt": True,
            "aggregate_only_blind_report": True,
            "blind_features_must_not_be_persisted": True,
            "post_blind_tuning_forbidden": True,
            "failed_candidate_keeps_existing_default": True,
        },
        "candidate": {
            "checkpoint": str(paths["checkpoint"]),
            "checkpoint_sha256": checkpoint_hash,
            "onnx": str(paths["model"]),
            "onnx_sha256": model_hash,
            "metadata": str(paths["metadata"]),
            "metadata_sha256": metadata_hash,
            "single_onnx_graph": True,
            "external_models": [],
        },
        "acceptance": {
            "path": str(paths["acceptance"]),
            "sha256": sha256_file(paths["acceptance"]),
            "protocol_name": PROTOCOL_NAME,
        },
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "baseline_lock": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "development_evidence": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in evidence_paths.items()
        },
        "development_result": {
            "candidate_top1_correct": int(candidate_metrics["top1_correct"]),
            "candidate_top5_correct": int(candidate_metrics["top5_correct"]),
            "parent_top1_correct": int(parent_metrics["top1_correct"]),
            "parent_top5_correct": int(parent_metrics["top5_correct"]),
            "legacy_clean_top1_correct": int(
                legacy["candidate"]["clean"]["top1_correct"]
            ),
            "legacy_clean_top5_correct": int(
                legacy["candidate"]["clean"]["top5_correct"]
            ),
            "legacy_conflict_top1_correct": int(
                legacy["candidate"]["conflict"]["top1_correct"]
            ),
            "legacy_conflict_top5_correct": int(
                legacy["candidate"]["conflict"]["top5_correct"]
            ),
        },
        "blind_contract": {
            **{
                key: acceptance["blind"][key]
                for key in (
                    "path",
                    "sha256",
                    "records",
                    "identities",
                    "images_per_identity",
                    "minimum_top1_correct",
                    "minimum_top5_correct",
                )
            },
            "attempt_marker": str(attempt_marker),
            "report_path": str(blind_output),
        },
        "code_sha256": code_hashes,
        "default_backend_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validated = validate_candidate_lock(
        output,
        model_path=paths["model"],
        metadata_path=paths["metadata"],
        acceptance_path=paths["acceptance"],
    )
    print(
        json.dumps(
            {
                "candidate_lock": str(output),
                "candidate_lock_sha256": validated["sha256"],
                "status": payload["status"],
                "candidate": payload["candidate"],
                "development_result": payload["development_result"],
                "blind_contract": payload["blind_contract"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
