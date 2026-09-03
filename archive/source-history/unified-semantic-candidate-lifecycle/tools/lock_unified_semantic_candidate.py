#!/usr/bin/env python3
"""Create an immutable, unscored lock for one UnifiedSemantic ONNX candidate."""

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

from pet_id.unified_blind_protocol import (  # noqa: E402
    JOINT800_BLIND_SHA256,
    JOINT800_TRAIN_SHA256,
    sha256_file,
    validate_candidate_lock,
    validate_disjoint_splits,
    validate_manifest,
)


LOCKED_CODE_PATHS = (
    "src/Pet-ReID-IMAG/tools/evaluate_unified_semantic_joint800.py",
    "src/Pet-ReID-IMAG/pet_id/unified_blind_protocol.py",
    "src/Pet-ReID-IMAG/pet_id/unified_runtime.py",
    "src/Pet-ReID-IMAG/pet_id/unified_data.py",
    "src/Pet-ReID-IMAG/pet_id/unified_training.py",
    "src/Pet-ReID-IMAG/pet_id/unified.py",
    "src/Pet-ReID-IMAG/pet_id/arcface.py",
    "src/Pet-ReID-IMAG/pet_id/multimodal.py",
    "src/Pet-ReID-IMAG/pet_id/dogfacenet_alignment.py",
    "src/Pet-ReID-IMAG/pet_id/workspace_paths.py",
)
DEV_MANIFEST_SHA256 = (
    "3ae881675a035a4e09f274097a839b0ad12dd30ccb2fd607b1d8257a3d0aa05d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pytorch-report", type=Path, required=True)
    parser.add_argument("--onnx-cpu-report", type=Path, required=True)
    parser.add_argument("--onnx-cuda-report", type=Path, required=True)
    parser.add_argument("--export-validation", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1"
        / "train_manifest.json",
    )
    parser.add_argument(
        "--blind-manifest",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1"
        / "blind_test_manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _workspace_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Locked artifact is outside the workspace: {path}") from error


def _load_report(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise RuntimeError(f"{name} did not pass")
    return payload


def _validate_pytorch_report(
    report: dict[str, Any],
    *,
    checkpoint_hash: str,
) -> None:
    if str(report.get("checkpoint_sha256", "")).casefold() != checkpoint_hash:
        raise RuntimeError("PyTorch report checkpoint hash mismatch")
    if str(report.get("manifest_sha256", "")).casefold() != DEV_MANIFEST_SHA256:
        raise RuntimeError("PyTorch report used a non-locked development manifest")
    required = {
        ("clean", "top1_correct"): 193,
        ("clean", "top5_correct"): 198,
        ("conflict", "top1_correct"): 193,
        ("conflict", "top5_correct"): 198,
    }
    for (section, field), minimum in required.items():
        if int(report.get(section, {}).get(field, -1)) < minimum:
            raise RuntimeError(f"PyTorch {section}.{field} is below {minimum}")


def _validate_onnx_report(
    report: dict[str, Any],
    *,
    model_hash: str,
    provider: str,
) -> None:
    if str(report.get("model_sha256", "")).casefold() != model_hash:
        raise RuntimeError(f"{provider} ONNX report model hash mismatch")
    if str(report.get("manifest_sha256", "")).casefold() != DEV_MANIFEST_SHA256:
        raise RuntimeError(f"{provider} ONNX report used a changed manifest")
    actual_provider = report.get("provider", {}).get("provider")
    if actual_provider != provider:
        raise RuntimeError(f"Expected {provider}, report used {actual_provider!r}")
    retrieval = report.get("retrieval", {})
    parity = report.get("parity_with_pytorch", {})
    if int(retrieval.get("top1_correct", -1)) < 193:
        raise RuntimeError(f"{provider} ONNX Top-1 is below 193")
    if int(retrieval.get("top5_correct", -1)) < 198:
        raise RuntimeError(f"{provider} ONNX Top-5 is below 198")
    if float(parity.get("minimum_cosine", -1.0)) < 0.9999:
        raise RuntimeError(f"{provider} ONNX parity is below 0.9999")
    if int(parity.get("below_minimum_cosine", -1)) != 0:
        raise RuntimeError(f"{provider} ONNX has samples below parity")


def _validate_export_report(
    report: dict[str, Any],
    *,
    checkpoint_hash: str,
    model_hash: str,
) -> None:
    if report.get("onnx_checker") != "passed":
        raise RuntimeError("ONNX checker did not pass")
    if str(report.get("checkpoint_sha256", "")).casefold() != checkpoint_hash:
        raise RuntimeError("Export validation checkpoint hash mismatch")
    if str(report.get("onnx", {}).get("sha256", "")).casefold() != model_hash:
        raise RuntimeError("Export validation ONNX hash mismatch")
    providers = {row.get("provider"): row for row in report.get("providers", [])}
    for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        runs = providers.get(provider, {}).get("runs", [])
        if sorted(int(row.get("batch_size", -1)) for row in runs) != [1, 2]:
            raise RuntimeError(f"Export validation lacks {provider} batch 1/2")
        if any(
            float(row.get("embedding", {}).get("minimum_cosine", -1.0))
            < 0.9999
            for row in runs
        ):
            raise RuntimeError(f"Export validation parity failed on {provider}")


def main() -> None:
    args = parse_args()
    checkpoint = _file(args.checkpoint)
    model = _file(args.model)
    metadata_path = _file(args.metadata)
    pytorch_path = _file(args.pytorch_report)
    cpu_path = _file(args.onnx_cpu_report)
    cuda_path = _file(args.onnx_cuda_report)
    export_path = _file(args.export_validation)
    acceptance_path = _file(args.acceptance)
    train_path = _file(args.train_manifest)
    blind_path = _file(args.blind_manifest)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(output.parents) < 6 or output.parents[5].resolve() != WORKSPACE:
        raise RuntimeError(
            "Candidate lock must be placed at "
            "artifacts/runs/unified/v1/<candidate>/candidate_lock.json"
        )

    checkpoint_hash = sha256_file(checkpoint)
    model_hash = sha256_file(model)
    metadata_hash = sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_type") != "unified_semantic_pet_reid":
        raise RuntimeError("Unexpected deployment model type")
    if metadata.get("external_models") != []:
        raise RuntimeError("Unified deployment must not load external models")
    if str(metadata.get("onnx_sha256", "")).casefold() != model_hash:
        raise RuntimeError("Metadata ONNX hash mismatch")
    if (
        str(metadata.get("source_checkpoint_sha256", "")).casefold()
        != checkpoint_hash
    ):
        raise RuntimeError("Metadata checkpoint hash mismatch")

    pytorch_report = _load_report(pytorch_path, "PyTorch development report")
    cpu_report = _load_report(cpu_path, "CPU ONNX development report")
    cuda_report = _load_report(cuda_path, "CUDA ONNX development report")
    export_report = _load_report(export_path, "ONNX export validation")
    _validate_pytorch_report(pytorch_report, checkpoint_hash=checkpoint_hash)
    _validate_onnx_report(
        cpu_report,
        model_hash=model_hash,
        provider="CPUExecutionProvider",
    )
    _validate_onnx_report(
        cuda_report,
        model_hash=model_hash,
        provider="CUDAExecutionProvider",
    )
    _validate_export_report(
        export_report,
        checkpoint_hash=checkpoint_hash,
        model_hash=model_hash,
    )

    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    protocol = acceptance["required_evaluations"]["joint800_unseen200"]
    if str(protocol.get("manifest_sha256", "")).casefold() != (
        JOINT800_BLIND_SHA256
    ):
        raise RuntimeError("Acceptance Joint800 blind hash changed")
    if (
        int(protocol.get("records", -1)),
        int(protocol.get("identities", -1)),
        int(protocol.get("queries", -1)),
        int(protocol.get("minimum_top1_correct", -1)),
        int(protocol.get("minimum_top5_correct", -1)),
    ) != (800, 200, 400, 380, 397):
        raise RuntimeError("Acceptance Joint800 dimensions or thresholds changed")

    train = validate_manifest(
        train_path,
        expected_sha256=JOINT800_TRAIN_SHA256,
        expected_split="train",
        expected_records=3200,
        expected_identities=800,
        expected_images_per_identity=4,
    )
    blind = validate_manifest(
        blind_path,
        expected_sha256=JOINT800_BLIND_SHA256,
        expected_split="blind_test",
        expected_records=800,
        expected_identities=200,
        expected_images_per_identity=4,
    )
    validate_disjoint_splits(train, blind)

    evidence_paths = {
        "pytorch": pytorch_path,
        "onnx_cpu": cpu_path,
        "onnx_cuda": cuda_path,
        "export_validation": export_path,
    }
    code_hashes = {}
    for relative in LOCKED_CODE_PATHS:
        code_hashes[relative] = sha256_file(_file(WORKSPACE / relative))

    payload = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_UNSCORED",
        "policy": {
            "single_blind_attempt": True,
            "blind_results_must_not_change_candidate": True,
            "failed_candidate_keeps_existing_default": True,
        },
        "candidate": {
            "checkpoint": _workspace_relative(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "onnx": _workspace_relative(model),
            "onnx_sha256": model_hash,
            "metadata": _workspace_relative(metadata_path),
            "metadata_sha256": metadata_hash,
            "model_type": "unified_semantic_pet_reid",
            "input": "float32 RGB [N,3,1280,1280], 0..255",
            "output": "L2-normalized float32 [N,512]",
            "external_models": [],
        },
        "protocol": {
            "train_manifest": _workspace_relative(train_path),
            "train_manifest_sha256": JOINT800_TRAIN_SHA256,
            "blind_manifest": _workspace_relative(blind_path),
            "blind_manifest_sha256": JOINT800_BLIND_SHA256,
            "train_records": 3200,
            "train_identities": 800,
            "blind_records": 800,
            "blind_identities": 200,
            "queries": 400,
            "split_overlap": {"identities": 0, "source_sha256": 0},
        },
        "development_evidence": {
            name: {
                "path": _workspace_relative(path),
                "sha256": sha256_file(path),
            }
            for name, path in evidence_paths.items()
        },
        "acceptance": {
            "path": _workspace_relative(acceptance_path),
            "sha256": sha256_file(acceptance_path),
            "minimum_top1_correct": 380,
            "minimum_top5_correct": 397,
        },
        "code_sha256": code_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validated = validate_candidate_lock(
        output,
        model_path=model,
        metadata_path=metadata_path,
    )
    print(
        json.dumps(
            {
                "candidate_lock": str(output),
                "candidate_lock_sha256": validated["sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "onnx_sha256": model_hash,
                "metadata_sha256": metadata_hash,
                "development_evidence": payload["development_evidence"],
                "code_files_locked": len(code_hashes),
                "status": payload["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
