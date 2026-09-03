#!/usr/bin/env python3
"""Lock one fully validated unified v2 candidate before its only blind run."""

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
from pet_id.unified_v2_candidate import PROTOCOL_NAME  # noqa: E402


LOCKED_CODE_PATHS = (
    "src/Pet-ReID-IMAG/pet_id/unified.py",
    "src/Pet-ReID-IMAG/pet_id/unified_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_geometry_stability.py",
    "src/Pet-ReID-IMAG/pet_id/unified_runtime.py",
    "src/Pet-ReID-IMAG/pet_id/unified_semantic.py",
    "src/Pet-ReID-IMAG/pet_id/unified_semantic_checkpoint.py",
    "src/Pet-ReID-IMAG/pet_id/unified_training.py",
    "src/Pet-ReID-IMAG/pet_id/unified_v2_candidate.py",
    "src/Pet-ReID-IMAG/pet_id/arcface.py",
    "src/Pet-ReID-IMAG/pet_id/multimodal.py",
    "src/Pet-ReID-IMAG/tools/evaluate_unified_v2_blind.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--onnx-cpu-report", type=Path, required=True)
    parser.add_argument("--onnx-cuda-report", type=Path, required=True)
    parser.add_argument("--export-validation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--geometry-stability", type=Path, required=True)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    paths = {
        name: file(getattr(args, name))
        for name in (
            "checkpoint",
            "build_report",
            "model",
            "metadata",
            "development_report",
            "onnx_cpu_report",
            "onnx_cuda_report",
            "export_validation",
            "benchmark",
            "geometry_stability",
            "acceptance",
        )
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.parent != paths["checkpoint"].parent:
        raise RuntimeError("Candidate lock must be beside its checkpoint")

    acceptance = load_acceptance(
        paths["acceptance"],
        expected_protocol=PROTOCOL_NAME,
    )
    baseline_path = file(Path(acceptance["baseline_lock"]["path"]))
    protocol_path = file(Path(acceptance["protocol_lock"]["path"]))
    if sha256_file(baseline_path) != acceptance["baseline_lock"]["sha256"]:
        raise RuntimeError("Baseline lock changed")
    if sha256_file(protocol_path) != acceptance["protocol_lock"]["sha256"]:
        raise RuntimeError("Protocol lock changed")
    baseline = report(baseline_path)
    development_thresholds = baseline["reports"]["development"]["metrics"]

    checkpoint_hash = sha256_file(paths["checkpoint"])
    model_hash = sha256_file(paths["model"])
    metadata_hash = sha256_file(paths["metadata"])
    metadata = report(paths["metadata"])
    if metadata.get("model_type") != "unified_semantic_pet_reid":
        raise RuntimeError("Unexpected model type")
    if metadata.get("onnx_sha256") != model_hash:
        raise RuntimeError("Metadata ONNX hash mismatch")
    if metadata.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Metadata checkpoint hash mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("Unified candidate declares external models")
    model_config = metadata.get("model_config", {})
    if model_config.get("face_crop_scales") != [1.0, 1.6]:
        raise RuntimeError("Locked face crop scales changed")
    if model_config.get("face_crop_weights") != [0.05, 0.95]:
        raise RuntimeError("Locked face crop weights changed")

    development = report(paths["development_report"])
    if development.get("passed") is not True or development.get("blind_data_used") is not False:
        raise RuntimeError("PyTorch development evaluation did not pass")
    if development.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Development checkpoint hash mismatch")
    if development.get("manifest_sha256") != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest hash mismatch")
    clean = development.get("clean", {})
    if int(clean.get("top1_correct", -1)) < int(development_thresholds["top1_correct"]):
        raise RuntimeError("Development Top-1 is below semantic-v3")
    if int(clean.get("top5_correct", -1)) < int(development_thresholds["top5_correct"]):
        raise RuntimeError("Development Top-5 is below semantic-v3")
    metadata_development = metadata.get("development_report", {})
    if metadata_development.get("sha256") != sha256_file(paths["development_report"]):
        raise RuntimeError("Metadata development evidence mismatch")

    onnx_reports = {}
    for name, expected_provider in (
        ("onnx_cpu_report", "CPUExecutionProvider"),
        ("onnx_cuda_report", "CUDAExecutionProvider"),
    ):
        payload = report(paths[name])
        onnx_reports[name] = payload
        if payload.get("passed") is not True:
            raise RuntimeError(f"{name} did not pass")
        if payload.get("model_sha256") != model_hash:
            raise RuntimeError(f"{name} model hash mismatch")
        if payload.get("manifest_sha256") != acceptance["development"]["sha256"]:
            raise RuntimeError(f"{name} manifest hash mismatch")
        if payload.get("provider", {}).get("provider") != expected_provider:
            raise RuntimeError(f"{name} provider mismatch")
        retrieval = payload.get("retrieval", {})
        if int(retrieval.get("top1_correct", -1)) < int(
            development_thresholds["top1_correct"]
        ):
            raise RuntimeError(f"{name} Top-1 is below semantic-v3")
        if int(retrieval.get("top5_correct", -1)) < int(
            development_thresholds["top5_correct"]
        ):
            raise RuntimeError(f"{name} Top-5 is below semantic-v3")
        parity = payload.get("parity_with_pytorch", {})
        if float(parity.get("minimum_cosine", -1.0)) < 0.9999:
            raise RuntimeError(f"{name} full-development parity failed")
        if int(parity.get("below_minimum_cosine", -1)) != 0:
            raise RuntimeError(f"{name} has samples below parity")

    export = report(paths["export_validation"])
    if export.get("passed") is not True or export.get("onnx_checker") != "passed":
        raise RuntimeError("Export validation did not pass")
    if export.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Export checkpoint hash mismatch")
    if export.get("onnx", {}).get("sha256") != model_hash:
        raise RuntimeError("Export ONNX hash mismatch")
    providers = {row["provider"]: row for row in export.get("providers", [])}
    for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        runs = providers.get(provider, {}).get("runs", [])
        if sorted(int(row["batch_size"]) for row in runs) != [1, 2]:
            raise RuntimeError(f"Export lacks {provider} dynamic batch 1/2")
        if any(row["embedding"]["minimum_cosine"] < 0.9999 for row in runs):
            raise RuntimeError(f"Export {provider} parity failed")

    build = report(paths["build_report"])
    if build.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Build report checkpoint hash mismatch")
    if build.get("runtime_contract", {}).get("external_models") != []:
        raise RuntimeError("Build report declares external models")
    stability = report(paths["geometry_stability"])
    if stability.get("passed") is not True or stability.get("blind_data_used") is not False:
        raise RuntimeError("Geometry stability evidence did not pass")
    if stability.get("manifest_sha256") != acceptance["development"]["sha256"]:
        raise RuntimeError("Geometry stability used the wrong manifest")
    if build.get("geometry_stability_evidence", {}).get("sha256") != sha256_file(
        paths["geometry_stability"]
    ):
        raise RuntimeError("Build/stability evidence mismatch")

    benchmark = report(paths["benchmark"])
    if benchmark.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Benchmark checkpoint hash mismatch")
    if benchmark.get("onnx_sha256") != model_hash:
        raise RuntimeError("Benchmark ONNX hash mismatch")
    benchmark_results = benchmark.get("results", {})
    if benchmark_results.get("onnx_cuda", {}).get("provider") != "CUDAExecutionProvider":
        raise RuntimeError("CUDA benchmark provider mismatch")
    if benchmark_results.get("onnx_cpu", {}).get("provider") != "CPUExecutionProvider":
        raise RuntimeError("CPU benchmark provider mismatch")

    evidence_options = {
        "build_report": False,
        "development_pytorch": True,
        "development_onnx_cpu": True,
        "development_onnx_cuda": True,
        "export_validation": True,
        "benchmark": False,
        "geometry_stability": True,
    }
    evidence_paths = {
        "build_report": paths["build_report"],
        "development_pytorch": paths["development_report"],
        "development_onnx_cpu": paths["onnx_cpu_report"],
        "development_onnx_cuda": paths["onnx_cuda_report"],
        "export_validation": paths["export_validation"],
        "benchmark": paths["benchmark"],
        "geometry_stability": paths["geometry_stability"],
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
            "protocol_name": acceptance["protocol_name"],
        },
        "baseline_lock": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "development_evidence": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "requires_passed": evidence_options[name],
            }
            for name, path in evidence_paths.items()
        },
        "development_result": {
            "top1_correct": int(clean["top1_correct"]),
            "top5_correct": int(clean["top5_correct"]),
            "minimum_top1_correct": int(development_thresholds["top1_correct"]),
            "minimum_top5_correct": int(development_thresholds["top5_correct"]),
        },
        "blind_contract": {
            key: acceptance["blind"][key]
            for key in (
                "sha256",
                "records",
                "identities",
                "queries",
                "minimum_top1_correct",
                "minimum_top5_correct",
                "rule",
            )
        },
        "code_sha256": code_hashes,
        "default_backend_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_lock": str(output),
        "candidate_lock_sha256": sha256_file(output),
        "status": payload["status"],
        "candidate": payload["candidate"],
        "development_result": payload["development_result"],
        "blind_contract": payload["blind_contract"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()