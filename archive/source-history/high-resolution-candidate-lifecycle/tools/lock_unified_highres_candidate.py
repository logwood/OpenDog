#!/usr/bin/env python3
"""Freeze one fully validated UnifiedPetReID V4 candidate before blind use."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_highres import MODEL_TYPE  # noqa: E402
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402


LOCKED_CODE_PATHS = (
    "src/Pet-ReID-IMAG/pet_id/unified_highres.py",
    "src/Pet-ReID-IMAG/pet_id/unified_highres_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_highres_eval_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_highres_protocol.py",
    "src/Pet-ReID-IMAG/pet_id/unified_highres_runtime.py",
    "src/Pet-ReID-IMAG/pet_id/unified_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_runtime.py",
    "src/Pet-ReID-IMAG/pet_id/unified_training.py",
    "src/Pet-ReID-IMAG/tools/evaluate_unified_highres_blind.py",
    "src/Pet-ReID-IMAG/tools/lock_unified_highres_candidate.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--highres-development-report", type=Path, required=True)
    parser.add_argument("--v3-legacy-guard", type=Path, required=True)
    parser.add_argument("--onnx-cpu-report", type=Path, required=True)
    parser.add_argument("--onnx-cuda-report", type=Path, required=True)
    parser.add_argument("--export-validation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--parent-v3-model", type=Path, required=True)
    parser.add_argument("--parent-v3-metadata", type=Path, required=True)
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(path: Path, purpose: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("purpose") != purpose:
        raise RuntimeError(f"Unexpected report purpose: {path}")
    if payload.get("passed") is not True or payload.get("blind_data_used") is not False:
        raise RuntimeError(f"Development evidence did not pass: {path}")
    return payload


def main() -> None:
    args = parse_args()
    names = (
        "checkpoint",
        "model",
        "metadata",
        "protocol_lock",
        "highres_development_report",
        "v3_legacy_guard",
        "onnx_cpu_report",
        "onnx_cuda_report",
        "export_validation",
        "benchmark",
        "parent_v3_model",
        "parent_v3_metadata",
    )
    paths = {name: file(getattr(args, name)) for name in names}
    output = args.output.expanduser().resolve()
    blind_output = args.blind_output.expanduser().resolve()
    if output.exists() or blind_output.exists():
        raise FileExistsError("Candidate lock or blind output already exists")
    if output.parent != paths["checkpoint"].parent:
        raise RuntimeError("V4 candidate lock must be beside its checkpoint")
    if blind_output.parent != output.parent:
        raise RuntimeError("V4 blind report must be beside its candidate lock")

    protocol = load_json(paths["protocol_lock"])
    if protocol.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 protocol lock")
    if protocol.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 protocol must be locked and unscored")
    required_policy = (
        "v4_identity_disjoint",
        "exact_image_disjoint",
        "blind_single_candidate_attempt",
        "blind_training_forbidden",
        "blind_model_selection_forbidden",
        "blind_features_must_not_be_persisted",
        "failed_candidate_keeps_v3_default",
    )
    for key in required_policy:
        if protocol.get("policy", {}).get(key) is not True:
            raise RuntimeError(f"V4 protocol policy is missing {key}")
    marker = Path(protocol["blind_attempt_marker"]).expanduser().resolve()
    if marker.exists():
        raise FileExistsError("The V4 blind attempt is already spent")

    checkpoint_hash = sha256_file(paths["checkpoint"])
    model_hash = sha256_file(paths["model"])
    metadata_hash = sha256_file(paths["metadata"])
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Unexpected V4 checkpoint")
    training = checkpoint.get("training") or {}
    if training.get("blind_data_used") is not False:
        raise RuntimeError("V4 checkpoint provenance is not blind-safe")
    if training.get("protocol_lock_sha256") != sha256_file(paths["protocol_lock"]):
        raise RuntimeError("V4 checkpoint protocol lock mismatch")
    if training.get("manifest_sha256") != protocol["splits"]["training_extension"]["sha256"]:
        raise RuntimeError("V4 checkpoint training split mismatch")

    metadata = load_json(paths["metadata"])
    if metadata.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Unexpected V4 ONNX metadata type")
    if metadata.get("onnx_sha256") != model_hash:
        raise RuntimeError("V4 ONNX metadata hash mismatch")
    if metadata.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("V4 ONNX metadata checkpoint mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("V4 ONNX declares external runtime models")
    input_shape = metadata["runtime_contract"]["inputs"]["rgb"]["shape"]
    output_shape = metadata["runtime_contract"]["outputs"]["embedding"]["shape"]
    if input_shape != ["N", 3, "H", "W"] or output_shape != ["N", 512]:
        raise RuntimeError("V4 dynamic input or 512-D output contract changed")

    highres = evidence(
        paths["highres_development_report"],
        "unified_v4_real_high_resolution_development_comparison",
    )
    if highres["candidate"]["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("High-resolution development checkpoint mismatch")
    if highres["protocol"]["development_manifest_sha256"] != protocol["splits"]["development"]["sha256"]:
        raise RuntimeError("High-resolution development split mismatch")
    if highres.get("noninferiority", {}).get("passed") is not True:
        raise RuntimeError("High-resolution development noninferiority failed")

    guard = evidence(
        paths["v3_legacy_guard"],
        "unified_v4_locked_v3_and_legacy_development_guard",
    )
    if guard["checkpoint"]["sha256"] != checkpoint_hash:
        raise RuntimeError("V3/legacy guard checkpoint mismatch")
    if guard.get("noninferiority", {}).get("passed") is not True:
        raise RuntimeError("V3/legacy noninferiority failed")

    onnx_reports = {}
    for name, provider in (
        ("onnx_cpu_report", "CPUExecutionProvider"),
        ("onnx_cuda_report", "CUDAExecutionProvider"),
    ):
        payload = evidence(paths[name], "unified_v4_high_resolution_onnx_development")
        onnx_reports[name] = payload
        if payload.get("model_sha256") != model_hash:
            raise RuntimeError(f"{name} ONNX hash mismatch")
        if payload.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError(f"{name} checkpoint hash mismatch")
        if payload.get("manifest_sha256") != protocol["splits"]["development"]["sha256"]:
            raise RuntimeError(f"{name} manifest hash mismatch")
        if payload.get("provider", {}).get("provider") != provider:
            raise RuntimeError(f"{name} provider fallback detected")
        if float(payload["parity_with_pytorch"]["minimum_cosine"]) < 0.9999:
            raise RuntimeError(f"{name} full-set ONNX parity failed")

    export = load_json(paths["export_validation"])
    if export.get("purpose") != "unified_v4_dynamic_high_resolution_onnx_export_validation":
        raise RuntimeError("Unexpected V4 export validation")
    if export.get("passed") is not True or export.get("blind_data_used") is not False:
        raise RuntimeError("V4 export validation did not pass")
    if export.get("onnx", {}).get("sha256") != model_hash:
        raise RuntimeError("V4 export validation ONNX mismatch")
    contract = export.get("single_graph_contract", {})
    if contract.get("output_shape") != ["batch", 512]:
        raise RuntimeError("V4 export output is not statically 512-D")
    if contract.get("external_tensor_files") != []:
        raise RuntimeError("V4 export uses external tensor files")

    benchmark = load_json(paths["benchmark"])
    if benchmark.get("purpose") != "unified_v4_dynamic_onnx_runtime_benchmark":
        raise RuntimeError("Unexpected V4 benchmark")
    if benchmark.get("passed") is not True:
        raise RuntimeError("V4 benchmark failed")
    if benchmark.get("model_sha256") != model_hash or benchmark.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("V4 benchmark artifact mismatch")

    parent_model_hash = sha256_file(paths["parent_v3_model"])
    parent_metadata = load_json(paths["parent_v3_metadata"])
    if parent_metadata.get("onnx_sha256") != parent_model_hash:
        raise RuntimeError("Production V3 metadata/model mismatch")
    parent_checkpoint_hash = checkpoint["sources"]["parent_v3_checkpoint"]["sha256"]
    if parent_metadata.get("source_checkpoint_sha256") != parent_checkpoint_hash:
        raise RuntimeError("Production V3 is not the V4 parent")

    evidence_paths = {
        "highres_development_pytorch": paths["highres_development_report"],
        "v3_legacy_guard": paths["v3_legacy_guard"],
        "highres_development_onnx_cpu": paths["onnx_cpu_report"],
        "highres_development_onnx_cuda": paths["onnx_cuda_report"],
        "export_validation": paths["export_validation"],
        "benchmark": paths["benchmark"],
    }
    code_hashes = {}
    for relative in LOCKED_CODE_PATHS:
        code_path = file(WORKSPACE / relative)
        code_hashes[str(code_path)] = sha256_file(code_path)
    blind = protocol["splits"]["blind_test"]
    if (int(blind["records"]), int(blind["identities"])) != (32, 8):
        raise RuntimeError("V4 blind dimensions changed")
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
            "failed_candidate_keeps_v3_default": True,
        },
        "candidate": {
            "checkpoint": str(paths["checkpoint"]),
            "checkpoint_sha256": checkpoint_hash,
            "onnx": str(paths["model"]),
            "onnx_sha256": model_hash,
            "metadata": str(paths["metadata"]),
            "metadata_sha256": metadata_hash,
            "single_onnx_graph": True,
            "dynamic_raw_spatial_input": True,
            "output_dimension": 512,
            "external_models": [],
        },
        "parent_v3": {
            "onnx": str(paths["parent_v3_model"]),
            "onnx_sha256": parent_model_hash,
            "metadata": str(paths["parent_v3_metadata"]),
            "metadata_sha256": sha256_file(paths["parent_v3_metadata"]),
            "checkpoint_sha256": parent_checkpoint_hash,
        },
        "protocol_lock": {
            "path": str(paths["protocol_lock"]),
            "sha256": sha256_file(paths["protocol_lock"]),
        },
        "development_evidence": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in evidence_paths.items()
        },
        "development_result": {
            "highres_candidate_top1_correct": int(highres["candidate_metrics"]["top1_correct"]),
            "highres_candidate_top5_correct": int(highres["candidate_metrics"]["top5_correct"]),
            "highres_parent_top1_correct": int(highres["parent_production_metrics"]["top1_correct"]),
            "highres_parent_top5_correct": int(highres["parent_production_metrics"]["top5_correct"]),
            "v3_top1_correct": int(guard["v3_development"]["candidate_clean"]["top1_correct"]),
            "v3_top5_correct": int(guard["v3_development"]["candidate_clean"]["top5_correct"]),
            "legacy_clean_top1_correct": int(guard["legacy_clean_conflict"]["candidate_clean"]["top1_correct"]),
            "legacy_clean_top5_correct": int(guard["legacy_clean_conflict"]["candidate_clean"]["top5_correct"]),
            "legacy_conflict_top1_correct": int(guard["legacy_clean_conflict"]["candidate_conflict"]["top1_correct"]),
            "legacy_conflict_top5_correct": int(guard["legacy_clean_conflict"]["candidate_conflict"]["top5_correct"]),
        },
        "blind_contract": {
            "path": blind["path"],
            "sha256": blind["sha256"],
            "records": int(blind["records"]),
            "identities": int(blind["identities"]),
            "images_per_identity": int(protocol["criteria"]["images_per_identity"]),
            "acceptance": "candidate Top-1/Top-5 counts must be >= production V3 on identical images",
            "attempt_marker": str(marker),
            "report_path": str(blind_output),
        },
        "code_sha256": code_hashes,
        "default_backend_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "candidate_lock": str(output),
                "candidate_lock_sha256": sha256_file(output),
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
