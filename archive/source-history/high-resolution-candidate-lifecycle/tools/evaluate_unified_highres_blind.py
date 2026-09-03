#!/usr/bin/env python3
"""Run the single aggregate blind comparison for locked UnifiedPetReID V4.

The candidate, both ONNX runtimes, all non-blind evidence, and the output
contract are checked before the blind attempt is irreversibly reserved.  The
blind manifest is not read or hashed until after that reservation.  Only
aggregate retrieval metrics are persisted; embeddings and per-query results
remain in memory and are discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_highres import MODEL_TYPE  # noqa: E402
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402
from pet_id.unified_highres_runtime import (  # noqa: E402
    UnifiedHighResolutionONNXRuntimePipeline,
)
from pet_id.unified_runtime import (  # noqa: E402
    UnifiedONNXRuntimePipeline,
    sha256_file,
)
from pet_id.unified_training import retrieval_metrics  # noqa: E402


BLIND_CONFIRMATION = "RUN_UNIFIED_V4_BLIND_ONCE"
EXPECTED_RECORDS = 32
EXPECTED_IDENTITIES = 8
EXPECTED_IMAGES_PER_IDENTITY = 4
GALLERY_IMAGES_PER_IDENTITY = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--parent-metadata", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"Must equal {BLIND_CONFIRMATION!r}",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def existing_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def locked_path(record: dict[str, Any], key: str) -> Path:
    value = str(record.get(key, ""))
    if not value:
        raise RuntimeError(f"Candidate lock is missing {key}")
    return Path(value).expanduser().resolve()


def require_same_path(actual: Path, expected: Path, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} path differs from the candidate lock")


def require_hash(path: Path, expected: str, label: str) -> str:
    digest = sha256_file(path)
    if digest.casefold() != str(expected).casefold():
        raise RuntimeError(f"{label} SHA-256 differs from the candidate lock")
    return digest


def manifest_sets(
    payload: dict[str, Any],
    *,
    expected_split: str,
) -> tuple[set[str], set[str]]:
    if payload.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Manifest protocol name changed")
    if payload.get("protocol_split") != expected_split:
        raise RuntimeError(f"Expected the {expected_split} manifest")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Locked manifest has no records")
    identities = [str(row.get("identity", "")).casefold() for row in records]
    sources = [str(row.get("source_sha256", "")).casefold() for row in records]
    if any(not value for value in identities + sources):
        raise RuntimeError("Locked manifest has incomplete records")
    if len(sources) != len(set(sources)):
        raise RuntimeError("Locked manifest contains duplicate source images")
    declared = int(payload.get("images_per_identity", 0))
    counts = Counter(identities)
    if declared < 1 or any(count != declared for count in counts.values()):
        raise RuntimeError("Locked manifest identity counts changed")
    return set(identities), set(sources)


def validate_candidate_lock(
    *,
    lock_path: Path,
    protocol_path: Path,
    candidate_model: Path,
    candidate_metadata: Path,
    candidate_checkpoint: Path,
    parent_model: Path,
    parent_metadata: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate immutable inputs without reading the blind manifest."""

    lock = read_json(lock_path)
    if lock.get("schema_version") != 1 or lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Candidate lock is not UnifiedPetReID V4")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 candidate must be locked and unscored")
    for key in (
        "single_blind_attempt",
        "aggregate_only_blind_report",
        "blind_features_must_not_be_persisted",
        "post_blind_tuning_forbidden",
        "failed_candidate_keeps_v3_default",
    ):
        if lock.get("policy", {}).get(key) is not True:
            raise RuntimeError(f"Candidate lock policy is missing {key}")

    candidate = lock.get("candidate", {})
    require_same_path(candidate_model, locked_path(candidate, "onnx"), "Candidate ONNX")
    require_same_path(
        candidate_metadata,
        locked_path(candidate, "metadata"),
        "Candidate metadata",
    )
    require_same_path(
        candidate_checkpoint,
        locked_path(candidate, "checkpoint"),
        "Candidate checkpoint",
    )
    require_hash(candidate_model, candidate.get("onnx_sha256", ""), "Candidate ONNX")
    require_hash(
        candidate_metadata,
        candidate.get("metadata_sha256", ""),
        "Candidate metadata",
    )
    require_hash(
        candidate_checkpoint,
        candidate.get("checkpoint_sha256", ""),
        "Candidate checkpoint",
    )
    if candidate.get("single_onnx_graph") is not True:
        raise RuntimeError("V4 candidate is not one ONNX graph")
    if candidate.get("dynamic_raw_spatial_input") is not True:
        raise RuntimeError("V4 candidate is not dynamic raw-spatial input")
    if int(candidate.get("output_dimension", 0)) != 512:
        raise RuntimeError("V4 candidate output dimension changed")
    if candidate.get("external_models") != []:
        raise RuntimeError("V4 candidate declares external runtime models")

    metadata_payload = read_json(candidate_metadata)
    if metadata_payload.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Unexpected V4 ONNX metadata model type")
    if metadata_payload.get("external_models") != []:
        raise RuntimeError("V4 ONNX metadata declares external runtime models")
    runtime_contract = metadata_payload.get("runtime_contract", {})
    if runtime_contract.get("inputs", {}).get("rgb", {}).get("shape") != [
        "N",
        3,
        "H",
        "W",
    ]:
        raise RuntimeError("V4 dynamic RGB contract changed")
    if runtime_contract.get("outputs", {}).get("embedding", {}).get("shape") != [
        "N",
        512,
    ]:
        raise RuntimeError("V4 embedding contract changed")

    parent = lock.get("parent_v3", {})
    require_same_path(parent_model, locked_path(parent, "onnx"), "Parent V3 ONNX")
    require_same_path(
        parent_metadata,
        locked_path(parent, "metadata"),
        "Parent V3 metadata",
    )
    require_hash(parent_model, parent.get("onnx_sha256", ""), "Parent V3 ONNX")
    require_hash(
        parent_metadata,
        parent.get("metadata_sha256", ""),
        "Parent V3 metadata",
    )
    parent_payload = read_json(parent_metadata)
    if parent_payload.get("onnx_sha256") != parent.get("onnx_sha256"):
        raise RuntimeError("Parent V3 metadata/model hash mismatch")
    if parent_payload.get("source_checkpoint_sha256") != parent.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Parent V3 checkpoint provenance changed")
    if parent_payload.get("external_models") != []:
        raise RuntimeError("Parent V3 declares external runtime models")

    protocol_record = lock.get("protocol_lock", {})
    require_same_path(
        protocol_path,
        locked_path(protocol_record, "path"),
        "V4 protocol lock",
    )
    require_hash(
        protocol_path,
        protocol_record.get("sha256", ""),
        "V4 protocol lock",
    )
    protocol = read_json(protocol_path)
    if protocol.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 protocol")
    if protocol.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 protocol is no longer locked and unscored")
    for key in (
        "v4_identity_disjoint",
        "exact_image_disjoint",
        "blind_single_candidate_attempt",
        "blind_training_forbidden",
        "blind_model_selection_forbidden",
        "blind_features_must_not_be_persisted",
        "failed_candidate_keeps_v3_default",
    ):
        if protocol.get("policy", {}).get(key) is not True:
            raise RuntimeError(f"V4 protocol policy is missing {key}")

    for name, record in lock.get("development_evidence", {}).items():
        evidence_path = existing_file(record.get("path", ""))
        require_hash(
            evidence_path,
            record.get("sha256", ""),
            f"Development evidence {name}",
        )
        evidence = read_json(evidence_path)
        if evidence.get("passed") is not True:
            raise RuntimeError(f"Development evidence did not pass: {name}")
        if name != "benchmark" and evidence.get("blind_data_used") is not False:
            raise RuntimeError(f"Development evidence is not blind-safe: {name}")

    code_hashes = lock.get("code_sha256", {})
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise RuntimeError("Candidate lock has no code hashes")
    for path_text, digest in code_hashes.items():
        code_path = existing_file(path_text)
        require_hash(code_path, digest, f"Locked code {code_path.name}")

    blind = protocol.get("splits", {}).get("blind_test", {})
    contract = lock.get("blind_contract", {})
    for key in ("path", "sha256", "records", "identities"):
        if contract.get(key) != blind.get(key):
            raise RuntimeError(f"V4 blind contract changed: {key}")
    if int(contract.get("images_per_identity", 0)) != int(
        protocol.get("criteria", {}).get("images_per_identity", 0)
    ):
        raise RuntimeError("V4 blind images-per-identity changed")
    dimensions = (
        int(contract.get("records", 0)),
        int(contract.get("identities", 0)),
        int(contract.get("images_per_identity", 0)),
    )
    if dimensions != (
        EXPECTED_RECORDS,
        EXPECTED_IDENTITIES,
        EXPECTED_IMAGES_PER_IDENTITY,
    ):
        raise RuntimeError("V4 blind dimensions changed")
    if contract.get("acceptance") != (
        "candidate Top-1/Top-5 counts must be >= production V3 on identical images"
    ):
        raise RuntimeError("V4 blind acceptance rule changed")

    blind_manifest = Path(str(contract.get("path", ""))).expanduser().resolve()
    marker = Path(str(contract.get("attempt_marker", ""))).expanduser().resolve()
    report_path = Path(str(contract.get("report_path", ""))).expanduser().resolve()
    require_same_path(output, report_path, "V4 blind report")
    require_same_path(
        marker,
        Path(str(protocol.get("blind_attempt_marker", ""))).expanduser().resolve(),
        "V4 blind attempt marker",
    )
    # File existence may be checked during preflight, but the protected bytes
    # are deliberately not read or hashed until reserve_blind_attempt returns.
    if not blind_manifest.is_file():
        raise FileNotFoundError(blind_manifest)
    if marker == output or marker == blind_manifest or output == blind_manifest:
        raise RuntimeError("V4 blind artifact paths collide")
    if marker.exists() or output.exists():
        raise FileExistsError("The V4 blind attempt is already spent")
    return {
        "lock": lock,
        "lock_sha256": sha256_file(lock_path),
        "protocol": protocol,
        "blind_manifest": blind_manifest,
        "blind_sha256": str(contract["sha256"]),
        "marker": marker,
    }


def load_nonblind_sets(protocol: dict[str, Any]) -> dict[str, tuple[set[str], set[str]]]:
    result: dict[str, tuple[set[str], set[str]]] = {}
    for split in ("training_extension", "development"):
        record = protocol["splits"][split]
        path = existing_file(record["path"])
        require_hash(path, record["sha256"], f"V4 {split} manifest")
        result[split] = manifest_sets(read_json(path), expected_split=split)
    return result


def reserve_blind_attempt(
    marker: Path,
    *,
    output: Path,
    candidate_lock_sha256: str,
    provider: str,
) -> None:
    """Atomically and permanently spend the only V4 blind attempt."""

    if output.exists():
        raise FileExistsError(output)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "status": "RUNNING",
        "reserved_at": utc_now(),
        "output": str(output),
        "candidate_lock_sha256": candidate_lock_sha256,
        "provider": provider,
        "single_attempt": True,
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise FileExistsError("The V4 blind attempt is already spent") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def complete_blind_attempt(marker: Path, report_sha256: str, passed: bool) -> None:
    payload = read_json(marker)
    if payload.get("status") != "RUNNING":
        raise RuntimeError("V4 blind attempt marker is not running")
    payload.update(
        {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "report_sha256": report_sha256,
            "passed": bool(passed),
            "post_blind_tuning_permitted": False,
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
    if args.confirm != BLIND_CONFIRMATION:
        raise RuntimeError("Explicit V4 one-shot confirmation is incorrect")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")

    candidate_model = existing_file(args.candidate_model)
    candidate_metadata = existing_file(args.candidate_metadata)
    candidate_checkpoint = existing_file(args.candidate_checkpoint)
    parent_model = existing_file(args.parent_model)
    parent_metadata = existing_file(args.parent_metadata)
    candidate_lock_path = existing_file(args.candidate_lock)
    protocol_path = existing_file(args.protocol_lock)
    output = args.output.expanduser().resolve()

    validated = validate_candidate_lock(
        lock_path=candidate_lock_path,
        protocol_path=protocol_path,
        candidate_model=candidate_model,
        candidate_metadata=candidate_metadata,
        candidate_checkpoint=candidate_checkpoint,
        parent_model=parent_model,
        parent_metadata=parent_metadata,
        output=output,
    )
    nonblind_sets = load_nonblind_sets(validated["protocol"])

    # Both sessions and provider selection are proven usable before the blind
    # attempt is spent.  Warmup uses synthetic zeros only.
    candidate_pipeline = UnifiedHighResolutionONNXRuntimePipeline(
        candidate_model,
        metadata_path=candidate_metadata,
        source_checkpoint=candidate_checkpoint,
        provider=args.provider,
        device=args.provider,
        verify_hash=True,
        warmup_batches=(1,),
        warmup_shapes=((1280, 1280),),
    )
    parent_pipeline = UnifiedONNXRuntimePipeline(
        parent_model,
        metadata_path=parent_metadata,
        provider=args.provider,
        device=args.provider,
        verify_hash=True,
        warmup_batches=(1,),
    )
    expected_provider = (
        "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    )
    if candidate_pipeline.session.get_providers()[0] != expected_provider:
        raise RuntimeError("V4 blind runtime provider fallback is forbidden")
    if parent_pipeline.session.get_providers()[0] != expected_provider:
        raise RuntimeError("V3 blind runtime provider fallback is forbidden")

    marker = validated["marker"]
    reserve_blind_attempt(
        marker,
        output=output,
        candidate_lock_sha256=validated["lock_sha256"],
        provider=expected_provider,
    )

    # Protected content is first consumed after the exclusive marker exists.
    blind_manifest = validated["blind_manifest"]
    require_hash(blind_manifest, validated["blind_sha256"], "V4 blind manifest")
    manifest = read_json(blind_manifest)
    blind_sets = manifest_sets(manifest, expected_split="blind_test")
    records = manifest["records"]
    if len(records) != EXPECTED_RECORDS or len(blind_sets[0]) != EXPECTED_IDENTITIES:
        raise RuntimeError("V4 blind manifest dimensions changed")
    for split, sets in nonblind_sets.items():
        if sets[0].intersection(blind_sets[0]):
            raise RuntimeError(f"{split}/blind identity overlap")
        if sets[1].intersection(blind_sets[1]):
            raise RuntimeError(f"{split}/blind source overlap")

    candidate_rows: list[torch.Tensor] = []
    parent_rows: list[torch.Tensor] = []
    identities: list[str] = []
    source_keys: list[str] = []
    for index, record in enumerate(records):
        source = existing_file(record.get("source_path", ""))
        source_digest = require_hash(
            source,
            record.get("source_sha256", ""),
            "V4 blind source image",
        )
        image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError("Failed to decode a V4 blind source image")
        # The exact same decoded BGR array is given to both preprocessing paths.
        candidate_descriptor = candidate_pipeline.encode_image(image_bgr)[0]
        parent_descriptor = parent_pipeline.encode_image(image_bgr)[0]
        candidate_rows.append(candidate_descriptor.fused_feature.detach().float().cpu())
        parent_rows.append(parent_descriptor.fused_feature.detach().float().cpu())
        identities.append(str(record["identity"]).casefold())
        source_keys.append(source_digest)
        processed = index + 1
        if processed == 1 or processed % args.progress_every == 0:
            print(f"V4/V3 blind ONNX comparison: {processed}/{len(records)}", flush=True)

    candidate_features = torch.stack(candidate_rows)
    parent_features = torch.stack(parent_rows)
    candidate_metrics = retrieval_metrics(
        candidate_features,
        identities,
        source_keys,
        gallery_images_per_identity=GALLERY_IMAGES_PER_IDENTITY,
        include_queries=False,
    )
    parent_metrics = retrieval_metrics(
        parent_features,
        identities,
        source_keys,
        gallery_images_per_identity=GALLERY_IMAGES_PER_IDENTITY,
        include_queries=False,
    )
    expected_retrieval_dimensions = (
        EXPECTED_IDENTITIES,
        EXPECTED_IDENTITIES * GALLERY_IMAGES_PER_IDENTITY,
        EXPECTED_IDENTITIES
        * (EXPECTED_IMAGES_PER_IDENTITY - GALLERY_IMAGES_PER_IDENTITY),
    )
    actual_retrieval_dimensions = (
        int(candidate_metrics["gallery_identities"]),
        int(candidate_metrics["gallery_records"]),
        int(candidate_metrics["query_records"]),
    )
    if actual_retrieval_dimensions != expected_retrieval_dimensions:
        raise RuntimeError("V4 blind gallery/query dimensions changed")
    if (
        int(parent_metrics["gallery_identities"]),
        int(parent_metrics["gallery_records"]),
        int(parent_metrics["query_records"]),
    ) != expected_retrieval_dimensions:
        raise RuntimeError("V3 blind gallery/query dimensions changed")

    checks = {
        "same_decoded_images_for_both_models": True,
        "candidate_output_shape": list(candidate_features.shape)
        == [EXPECTED_RECORDS, 512],
        "parent_output_shape": list(parent_features.shape)
        == [EXPECTED_RECORDS, 512],
        "candidate_top1_not_below_parent": int(candidate_metrics["top1_correct"])
        >= int(parent_metrics["top1_correct"]),
        "candidate_top5_not_below_parent": int(candidate_metrics["top5_correct"])
        >= int(parent_metrics["top5_correct"]),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol_name": PROTOCOL_NAME,
        "purpose": "single_aggregate_unified_v4_vs_production_v3_blind_comparison",
        "single_attempt": True,
        "aggregate_only": True,
        "blind_data_used": True,
        "per_query_results_stored": False,
        "feature_cache_persisted": False,
        "candidate_lock": {
            "path": str(candidate_lock_path),
            "sha256": validated["lock_sha256"],
        },
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "blind_manifest": {
            "path": str(blind_manifest),
            "sha256": validated["blind_sha256"],
            "records": EXPECTED_RECORDS,
            "identities": EXPECTED_IDENTITIES,
            "images_per_identity": EXPECTED_IMAGES_PER_IDENTITY,
            "gallery_records": expected_retrieval_dimensions[1],
            "query_records": expected_retrieval_dimensions[2],
        },
        "attempt_marker": str(marker),
        "provider": expected_provider,
        "candidate": {
            "model": str(candidate_model),
            "model_sha256": sha256_file(candidate_model),
            "metadata": str(candidate_metadata),
            "metadata_sha256": sha256_file(candidate_metadata),
            "backend": candidate_pipeline.backend_info(),
            "retrieval": candidate_metrics,
        },
        "parent_production_v3": {
            "model": str(parent_model),
            "model_sha256": sha256_file(parent_model),
            "metadata": str(parent_metadata),
            "metadata_sha256": sha256_file(parent_metadata),
            "backend": parent_pipeline.backend_info(),
            "retrieval": parent_metrics,
        },
        "metric_deltas_candidate_minus_parent": {
            "top1_correct": int(candidate_metrics["top1_correct"])
            - int(parent_metrics["top1_correct"]),
            "top5_correct": int(candidate_metrics["top5_correct"])
            - int(parent_metrics["top5_correct"]),
        },
        "acceptance": (
            "candidate Top-1/Top-5 counts must be >= production V3 on identical images"
        ),
        "checks": checks,
        "passed": passed,
        "promotion_eligible": passed,
        "default_backend_changed": False,
        "post_blind_tuning_permitted": False,
    }
    del candidate_features, parent_features, candidate_rows, parent_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    report_hash = sha256_file(output)
    complete_blind_attempt(marker, report_hash, passed)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": report_hash,
                "candidate_retrieval": candidate_metrics,
                "parent_retrieval": parent_metrics,
                "metric_deltas_candidate_minus_parent": report[
                    "metric_deltas_candidate_minus_parent"
                ],
                "checks": checks,
                "passed": passed,
                "post_blind_tuning_permitted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
