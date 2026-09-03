"""Immutable candidate and one-shot controls for external unified v3."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .unified_training import load_acceptance, sha256_file


PROTOCOL_NAME = "unified_pet_reid_v3_external_strict_noninferiority"
BLIND_CONFIRMATION = "RUN_UNIFIED_V3_BLIND_ONCE"

RGB_INPUT_CONTRACT = {
    "rgb": {"dtype": "float32", "shape": ["N", 3, 1280, 1280]}
}
EMBEDDING_OUTPUT_CONTRACT = {
    "embedding": {
        "dtype": "float32",
        "shape": ["N", 512],
        "l2_normalized": True,
    }
}
REQUIRED_EVIDENCE = (
    "development_pytorch",
    "legacy_clean_conflict",
    "development_onnx_cpu",
    "development_onnx_cuda",
    "export_validation",
    "benchmark",
)


def _file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_candidate_lock(
    lock_path: str | Path,
    *,
    model_path: str | Path,
    metadata_path: str | Path,
    acceptance_path: str | Path,
) -> dict[str, Any]:
    """Validate every immutable input before the blind manifest is parsed."""

    lock_path = _file(lock_path)
    model_path = _file(model_path)
    metadata_path = _file(metadata_path)
    acceptance_path = _file(acceptance_path)
    lock = _json(lock_path)
    if lock.get("schema_version") != 1 or lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Candidate lock is not unified v3")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("Candidate must be locked and unscored")
    policy = lock.get("policy", {})
    for key in (
        "single_blind_attempt",
        "aggregate_only_blind_report",
        "blind_features_must_not_be_persisted",
        "post_blind_tuning_forbidden",
        "failed_candidate_keeps_existing_default",
    ):
        if policy.get(key) is not True:
            raise RuntimeError(f"Candidate lock policy is missing {key}")

    acceptance = load_acceptance(
        acceptance_path,
        expected_protocol=PROTOCOL_NAME,
    )
    acceptance_record = lock.get("acceptance", {})
    if acceptance_record.get("sha256") != sha256_file(acceptance_path):
        raise RuntimeError("Candidate lock acceptance hash mismatch")
    if acceptance_record.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Candidate lock acceptance protocol mismatch")

    for name in ("protocol_lock", "baseline_lock"):
        accepted = acceptance[name]
        accepted_path = _file(accepted["path"])
        if sha256_file(accepted_path) != accepted["sha256"]:
            raise RuntimeError(f"Acceptance {name} changed")
        locked = lock.get(name, {})
        if Path(str(locked.get("path", ""))).expanduser().resolve() != accepted_path:
            raise RuntimeError(f"Candidate {name} path mismatch")
        if locked.get("sha256") != accepted["sha256"]:
            raise RuntimeError(f"Candidate {name} hash mismatch")

    candidate = lock.get("candidate", {})
    model_hash = sha256_file(model_path)
    metadata_hash = sha256_file(metadata_path)
    if candidate.get("onnx_sha256") != model_hash:
        raise RuntimeError("Candidate lock ONNX hash mismatch")
    if candidate.get("metadata_sha256") != metadata_hash:
        raise RuntimeError("Candidate lock metadata hash mismatch")
    checkpoint_path = _file(candidate.get("checkpoint", ""))
    checkpoint_hash = sha256_file(checkpoint_path)
    if candidate.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Candidate lock checkpoint hash mismatch")
    if candidate.get("single_onnx_graph") is not True:
        raise RuntimeError("Candidate is not declared as one ONNX graph")
    if candidate.get("external_models") != []:
        raise RuntimeError("Candidate lock declares external runtime models")

    metadata = _json(metadata_path)
    if metadata.get("model_type") != "unified_external_joint_pet_reid":
        raise RuntimeError("Unexpected unified v3 model type")
    if metadata.get("onnx_sha256") != model_hash:
        raise RuntimeError("Deployment metadata ONNX hash mismatch")
    if metadata.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Deployment metadata checkpoint hash mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("Deployment metadata declares external runtime models")
    if metadata.get("inputs") != RGB_INPUT_CONTRACT:
        raise RuntimeError("Unified v3 RGB input contract changed")
    if metadata.get("outputs") != EMBEDDING_OUTPUT_CONTRACT:
        raise RuntimeError("Unified v3 embedding output contract changed")
    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("input_range") != [0, 255]:
        raise RuntimeError("Unified v3 input range changed")
    if preprocessing.get("color_order") != "RGB":
        raise RuntimeError("Unified v3 color order changed")
    if preprocessing.get("letterbox_allow_upscale") is not False:
        raise RuntimeError("Unified v3 must disable letterbox upscaling")

    evidence = lock.get("development_evidence", {})
    if set(evidence) != set(REQUIRED_EVIDENCE):
        raise RuntimeError("Candidate lock development evidence set changed")
    for name in REQUIRED_EVIDENCE:
        record = evidence[name]
        evidence_path = _file(record.get("path", ""))
        if sha256_file(evidence_path) != record.get("sha256"):
            raise RuntimeError(f"Development evidence hash mismatch: {name}")
        report = _json(evidence_path)
        if report.get("passed") is not True:
            raise RuntimeError(f"Development evidence did not pass: {name}")

    for path_text, expected_hash in lock.get("code_sha256", {}).items():
        code_path = _file(path_text)
        if sha256_file(code_path) != expected_hash:
            raise RuntimeError(f"Locked code changed: {code_path}")

    blind = acceptance["blind"]
    contract = lock.get("blind_contract", {})
    for key in (
        "path",
        "sha256",
        "records",
        "identities",
        "images_per_identity",
        "minimum_top1_correct",
        "minimum_top5_correct",
    ):
        if contract.get(key) != blind.get(key):
            raise RuntimeError(f"Candidate blind contract changed: {key}")
    blind_manifest = Path(blind["path"]).expanduser().resolve()
    if not blind_manifest.is_file():
        raise FileNotFoundError(blind_manifest)
    if sha256_file(blind_manifest) != blind["sha256"]:
        raise RuntimeError("Locked blind manifest hash changed")
    marker_path = Path(str(contract.get("attempt_marker", ""))).expanduser().resolve()
    report_path = Path(str(contract.get("report_path", ""))).expanduser().resolve()
    if marker_path == report_path or not marker_path.name or not report_path.name:
        raise RuntimeError("Candidate blind artifact paths are invalid")
    return {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "payload": lock,
        "acceptance": acceptance,
        "checkpoint": checkpoint_path,
        "blind_manifest": blind_manifest,
        "attempt_marker": marker_path,
        "report_path": report_path,
    }


def reserve_blind_attempt(
    marker_path: str | Path,
    *,
    output_path: str | Path,
    candidate_lock_sha256: str,
) -> Path:
    """Atomically and permanently spend the candidate's only blind attempt."""

    marker = Path(marker_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "RUNNING",
        "reserved_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "candidate_lock_sha256": candidate_lock_sha256,
        "single_attempt": True,
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise FileExistsError(
            "The unified v3 candidate blind attempt is already spent"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except BaseException:
        # Reservation remains permanent even if later setup or inference fails.
        raise
    return marker


def complete_blind_attempt(marker_path: str | Path, report_sha256: str) -> None:
    marker = Path(marker_path).expanduser().resolve()
    payload = _json(marker)
    if payload.get("status") != "RUNNING":
        raise RuntimeError("Blind attempt marker is not running")
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
