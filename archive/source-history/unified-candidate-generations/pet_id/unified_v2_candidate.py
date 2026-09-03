"""Immutable candidate and single-attempt controls for unified v2."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .unified_training import load_acceptance, sha256_file


PROTOCOL_NAME = "unified_pet_reid_v2_strict_noninferiority"
BLIND_CONFIRMATION = "RUN_UNIFIED_V2_BLIND_ONCE"


def _file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_candidate_lock(
    lock_path: str | Path,
    *,
    model_path: str | Path,
    metadata_path: str | Path,
    acceptance_path: str | Path,
) -> dict[str, Any]:
    lock_path = _file(lock_path)
    model_path = _file(model_path)
    metadata_path = _file(metadata_path)
    acceptance_path = _file(acceptance_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Candidate lock is not unified v2")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("Candidate must be locked and unscored")
    policy = lock.get("policy", {})
    for key in (
        "single_blind_attempt",
        "aggregate_only_blind_report",
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

    candidate = lock.get("candidate", {})
    model_hash = sha256_file(model_path)
    metadata_hash = sha256_file(metadata_path)
    if candidate.get("onnx_sha256") != model_hash:
        raise RuntimeError("Candidate lock ONNX hash mismatch")
    if candidate.get("metadata_sha256") != metadata_hash:
        raise RuntimeError("Candidate lock metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_type") != "unified_semantic_pet_reid":
        raise RuntimeError("Unexpected candidate model type")
    if metadata.get("onnx_sha256") != model_hash:
        raise RuntimeError("Metadata ONNX hash mismatch")
    if metadata.get("source_checkpoint_sha256") != candidate.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Metadata checkpoint hash mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("Unified candidate must have no external runtime models")
    if metadata.get("inputs") != {
        "rgb": {"dtype": "float32", "shape": ["N", 3, 1280, 1280]}
    }:
        raise RuntimeError("Unified v2 input contract changed")
    if metadata.get("outputs") != {
        "embedding": {
            "dtype": "float32",
            "shape": ["N", 512],
            "l2_normalized": True,
        }
    }:
        raise RuntimeError("Unified v2 output contract changed")

    for name, record in lock.get("development_evidence", {}).items():
        evidence_path = _file(record["path"])
        if sha256_file(evidence_path) != record.get("sha256"):
            raise RuntimeError(f"Development evidence hash mismatch: {name}")
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if record.get("requires_passed", True) and payload.get("passed") is not True:
            raise RuntimeError(f"Development evidence did not pass: {name}")

    for path_text, expected_hash in lock.get("code_sha256", {}).items():
        code_path = _file(path_text)
        if sha256_file(code_path) != expected_hash:
            raise RuntimeError(f"Locked code changed: {code_path}")

    baseline_path = _file(acceptance["baseline_lock"]["path"])
    if sha256_file(baseline_path) != acceptance["baseline_lock"]["sha256"]:
        raise RuntimeError("v2 baseline lock changed")
    protocol_path = _file(acceptance["protocol_lock"]["path"])
    if sha256_file(protocol_path) != acceptance["protocol_lock"]["sha256"]:
        raise RuntimeError("v2 protocol lock changed")
    return {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "payload": lock,
        "acceptance": acceptance,
    }


def reserve_blind_attempt(
    marker_path: str | Path,
    *,
    output_path: str | Path,
    candidate_lock_sha256: str,
) -> Path:
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
        raise FileExistsError("The unified v2 candidate blind attempt is already spent") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except BaseException:
        # The reservation remains permanent by protocol, even on failure.
        raise
    return marker


def complete_blind_attempt(marker_path: str | Path, report_sha256: str) -> None:
    marker = Path(marker_path).expanduser().resolve()
    payload = json.loads(marker.read_text(encoding="utf-8"))
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