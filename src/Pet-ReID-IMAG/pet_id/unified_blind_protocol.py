"""Locked protocol helpers for one-shot UnifiedPetReID evaluation.

The helpers deliberately keep protocol validation separate from model inference.
The blind evaluator can therefore reject a changed manifest, an overlapping
split, an unlocked candidate, or a repeated attempt before starting ONNX
Runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOINT800_TRAIN_SHA256 = (
    "c1dec0eaa73efc71885e9617ebd80d13a2d039c3fbf1a6675205744c81f188a4"
)
JOINT800_BLIND_SHA256 = (
    "6f7ec3164e7f1535cc9eaf26c0cfa57abb0f62e40a6307eecb3ab034a9137d5a"
)
JOINT800_CONFIRMATION = "RUN_JOINT800_UNSEEN200_ONCE"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestSummary:
    path: Path
    sha256: str
    protocol_split: str
    records: int
    identities: int
    images_per_identity: int
    identity_names: frozenset[str]
    source_hashes: frozenset[str]

    def report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "protocol_split": self.protocol_split,
            "records": self.records,
            "identities": self.identities,
            "images_per_identity": self.images_per_identity,
        }


def validate_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_split: str,
    expected_records: int,
    expected_identities: int,
    expected_images_per_identity: int,
) -> ManifestSummary:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_hash = sha256_file(resolved)
    if actual_hash.casefold() != expected_sha256.casefold():
        raise RuntimeError(
            f"Manifest hash mismatch for {resolved}: {actual_hash}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    split = str(payload.get("protocol_split", ""))
    if split != expected_split:
        raise RuntimeError(
            f"Expected protocol_split={expected_split!r}, got {split!r}"
        )
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} records, got "
            f"{len(records) if isinstance(records, list) else 'non-list'}"
        )
    identities = [str(row.get("identity", "")).casefold() for row in records]
    if any(not identity for identity in identities):
        raise RuntimeError("Every manifest record must have a non-empty identity")
    counts = Counter(identities)
    if len(counts) != expected_identities:
        raise RuntimeError(
            f"Expected {expected_identities} identities, got {len(counts)}"
        )
    invalid_counts = {
        identity: count
        for identity, count in counts.items()
        if count != expected_images_per_identity
    }
    if invalid_counts:
        raise RuntimeError(
            "Every identity must have exactly "
            f"{expected_images_per_identity} images; violations={invalid_counts}"
        )
    source_hashes = [str(row.get("source_sha256", "")).casefold() for row in records]
    if any(not value for value in source_hashes):
        raise RuntimeError("Every manifest record must have source_sha256")
    if len(set(source_hashes)) != len(source_hashes):
        raise RuntimeError("Manifest source_sha256 values must be unique")
    return ManifestSummary(
        path=resolved,
        sha256=actual_hash,
        protocol_split=split,
        records=len(records),
        identities=len(counts),
        images_per_identity=expected_images_per_identity,
        identity_names=frozenset(counts),
        source_hashes=frozenset(source_hashes),
    )


def validate_disjoint_splits(
    train: ManifestSummary,
    blind: ManifestSummary,
) -> None:
    shared_identities = train.identity_names.intersection(blind.identity_names)
    if shared_identities:
        raise RuntimeError(
            f"Train/blind identity overlap: {len(shared_identities)} identities"
        )
    shared_sources = train.source_hashes.intersection(blind.source_hashes)
    if shared_sources:
        raise RuntimeError(
            f"Train/blind source overlap: {len(shared_sources)} images"
        )


def validate_candidate_lock(
    lock_path: str | Path,
    *,
    model_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    resolved_lock = Path(lock_path).expanduser().resolve()
    resolved_model = Path(model_path).expanduser().resolve()
    resolved_metadata = Path(metadata_path).expanduser().resolve()
    for path in (resolved_lock, resolved_model, resolved_metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = json.loads(resolved_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("Candidate lock must have status LOCKED_UNSCORED")
    policy = lock.get("policy", {})
    required_policy = (
        "single_blind_attempt",
        "blind_results_must_not_change_candidate",
        "failed_candidate_keeps_existing_default",
    )
    if any(policy.get(name) is not True for name in required_policy):
        raise RuntimeError("Candidate lock does not enforce the one-shot policy")
    candidate = lock.get("candidate", {})
    actual_model_hash = sha256_file(resolved_model)
    actual_metadata_hash = sha256_file(resolved_metadata)
    if candidate.get("onnx_sha256", "").casefold() != actual_model_hash.casefold():
        raise RuntimeError("Candidate lock ONNX hash mismatch")
    if (
        candidate.get("metadata_sha256", "").casefold()
        != actual_metadata_hash.casefold()
    ):
        raise RuntimeError("Candidate lock metadata hash mismatch")
    protocol = lock.get("protocol", {})
    if protocol.get("train_manifest_sha256", "").casefold() != (
        JOINT800_TRAIN_SHA256
    ):
        raise RuntimeError("Candidate lock has the wrong Joint800 train hash")
    if protocol.get("blind_manifest_sha256", "").casefold() != (
        JOINT800_BLIND_SHA256
    ):
        raise RuntimeError("Candidate lock has the wrong Joint800 blind hash")
    metadata = json.loads(resolved_metadata.read_text(encoding="utf-8"))
    if metadata.get("onnx_sha256", "").casefold() != actual_model_hash.casefold():
        raise RuntimeError("Deployment metadata ONNX hash mismatch")
    if metadata.get("model_type") != "unified_semantic_pet_reid":
        raise RuntimeError("Unexpected unified model type")
    if metadata.get("external_models") != []:
        raise RuntimeError("Unified deployment metadata must have no external models")
    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("letterbox_allow_upscale") is not False:
        raise RuntimeError("Locked preprocessing must disable letterbox upscaling")
    inputs = metadata.get("inputs", {})
    outputs = metadata.get("outputs", {})
    if inputs != {
        "rgb": {"dtype": "float32", "shape": ["N", 3, 1280, 1280]}
    }:
        raise RuntimeError("Locked model must have one RGB [N,3,1280,1280] input")
    if outputs != {
        "embedding": {
            "dtype": "float32",
            "shape": ["N", 512],
            "l2_normalized": True,
        }
    }:
        raise RuntimeError("Locked model must have one normalized [N,512] output")
    source_hash = str(metadata.get("source_checkpoint_sha256", ""))
    if candidate.get("checkpoint_sha256", "").casefold() != source_hash.casefold():
        raise RuntimeError("Candidate lock checkpoint hash mismatch")
    evidence = lock.get("development_evidence", {})
    required_evidence = ("pytorch", "onnx_cpu", "onnx_cuda", "export_validation")
    for name in required_evidence:
        record = evidence.get(name, {})
        evidence_path = Path(str(record.get("path", ""))).expanduser()
        if not evidence_path.is_absolute():
            evidence_path = resolved_lock.parents[5] / evidence_path
        evidence_path = evidence_path.resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence_hash = sha256_file(evidence_path)
        if record.get("sha256", "").casefold() != evidence_hash.casefold():
            raise RuntimeError(f"Development evidence hash mismatch: {name}")
        report = json.loads(evidence_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise RuntimeError(f"Development evidence did not pass: {name}")
    for relative, expected_hash in lock.get("code_sha256", {}).items():
        code_path = Path(relative)
        if not code_path.is_absolute():
            code_path = resolved_lock.parents[5] / code_path
        code_path = code_path.resolve()
        if not code_path.is_file() or sha256_file(code_path) != expected_hash:
            raise RuntimeError(f"Locked code changed: {relative}")
    return {
        "path": str(resolved_lock),
        "sha256": sha256_file(resolved_lock),
        "payload": lock,
    }


def reserve_single_attempt(
    output_path: str | Path,
    *,
    candidate_lock_sha256: str,
) -> Path:
    """Atomically reserve the only permitted blind attempt.

    The marker is intentionally permanent, including after a failed process.
    Removing it would be an explicit protocol violation rather than an implicit
    retry hidden by a transient runtime failure.
    """

    output = Path(output_path).expanduser().resolve()
    marker = output.with_name(output.name + ".attempt.json")
    if output.exists():
        raise FileExistsError(output)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "RESERVED",
        "reserved_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "candidate_lock_sha256": candidate_lock_sha256,
        "single_attempt": True,
    }
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except BaseException:
        # The marker remains by design if reservation started.
        raise
    return marker


def complete_attempt_marker(marker_path: str | Path, report_sha256: str) -> None:
    marker = Path(marker_path).expanduser().resolve()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["status"] = "COMPLETED"
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    payload["report_sha256"] = report_sha256
    temporary = marker.with_name(marker.name + ".completing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)
