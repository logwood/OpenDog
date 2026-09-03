#!/usr/bin/env python3
"""Run the only permitted aggregate blind evaluation for unified v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_training import retrieval_metrics, sha256_file  # noqa: E402
from pet_id.unified_v2_candidate import (  # noqa: E402
    BLIND_CONFIRMATION,
    complete_blind_attempt,
    reserve_blind_attempt,
    validate_candidate_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"Must equal {BLIND_CONFIRMATION!r}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm != BLIND_CONFIRMATION:
        raise RuntimeError("Explicit unified v2 one-shot confirmation is incorrect")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    model = args.model.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    candidate_lock_path = args.candidate_lock.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    output = args.output.expanduser().resolve()
    candidate = validate_candidate_lock(
        candidate_lock_path,
        model_path=model,
        metadata_path=metadata,
        acceptance_path=acceptance_path,
    )
    acceptance = candidate["acceptance"]
    blind_contract = acceptance["blind"]
    blind_manifest = Path(blind_contract["path"]).expanduser().resolve()
    if not blind_manifest.is_file():
        raise FileNotFoundError(blind_manifest)
    if sha256_file(blind_manifest) != blind_contract["sha256"]:
        raise RuntimeError("Locked v2 blind manifest hash changed")
    expected_dimensions = (
        int(blind_contract["records"]),
        int(blind_contract["identities"]),
        int(blind_contract["queries"]),
    )
    if expected_dimensions != (144, 36, 72):
        raise RuntimeError("Locked v2 blind dimensions changed")
    thresholds = {
        "minimum_top1_correct": int(blind_contract["minimum_top1_correct"]),
        "minimum_top5_correct": int(blind_contract["minimum_top5_correct"]),
    }
    if thresholds != {
        "minimum_top1_correct": 65,
        "minimum_top5_correct": 71,
    }:
        raise RuntimeError("Locked v2 blind thresholds changed")

    baseline_lock_path = Path(acceptance["baseline_lock"]["path"]).resolve()
    baseline_lock = json.loads(baseline_lock_path.read_text(encoding="utf-8"))
    marker_path = Path(baseline_lock["candidate_attempt_marker"]).resolve()
    protocol_lock_path = Path(acceptance["protocol_lock"]["path"]).resolve()
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    if Path(protocol_lock["candidate_attempt_marker"]).resolve() != marker_path:
        raise RuntimeError("Protocol and baseline candidate markers differ")
    if marker_path.exists() or output.exists():
        raise FileExistsError("The unified v2 blind candidate attempt is already spent")

    # Finish all model/provider preflight before reserving the irreversible attempt.
    pipeline = UnifiedONNXRuntimePipeline(
        model,
        provider=args.provider,
        metadata_path=metadata,
        verify_hash=True,
        warmup_batches=(1,),
    )
    expected_provider = (
        "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    )
    if pipeline.session.get_providers()[0] != expected_provider:
        raise RuntimeError("Blind runtime provider fallback is forbidden")
    marker = reserve_blind_attempt(
        marker_path,
        output_path=output,
        candidate_lock_sha256=candidate["sha256"],
    )

    # Protected data is first parsed only after the permanent attempt reservation.
    manifest_payload = json.loads(blind_manifest.read_text(encoding="utf-8"))
    records = manifest_payload.get("records", [])
    if len(records) != 144:
        raise RuntimeError("Blind manifest record count changed")
    counts = Counter(str(record["identity"]).casefold() for record in records)
    if len(counts) != 36 or set(counts.values()) != {4}:
        raise RuntimeError("Blind manifest must contain 36 identities with four images each")
    source_hashes = [str(record.get("source_sha256", "")) for record in records]
    if any(not value for value in source_hashes) or len(set(source_hashes)) != 144:
        raise RuntimeError("Blind manifest source hashes are missing or duplicated")

    dataset = UnifiedManifestDataset(
        blind_manifest,
        input_size=pipeline.input_size,
        training=False,
        allow_letterbox_upscale=pipeline.letterbox_allow_upscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    embeddings: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    processed = 0
    for batch in loader:
        rgb = batch["rgb"].numpy().astype(np.float32, copy=False)
        embeddings.append(pipeline._run(rgb))
        identities.extend(batch["identity"])
        source_paths.extend(batch["source_path"])
        processed += int(rgb.shape[0])
        if processed == args.batch_size or processed % 25 == 0:
            print(f"unified v2 blind ONNX: {processed}/{len(dataset)}", flush=True)
    embedding = torch.cat(embeddings)
    metrics = retrieval_metrics(
        embedding,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=False,
    )
    if (
        metrics["gallery_identities"],
        metrics["gallery_records"],
        metrics["query_records"],
    ) != (36, 72, 72):
        raise RuntimeError("Blind gallery/query dimensions changed")
    checks = {
        "top1": metrics["top1_correct"] >= thresholds["minimum_top1_correct"],
        "top5": metrics["top5_correct"] >= thresholds["minimum_top5_correct"],
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_name": acceptance["protocol_name"],
        "purpose": "single_aggregate_candidate_blind_evaluation",
        "single_attempt": True,
        "aggregate_only": True,
        "per_query_results_stored": False,
        "candidate_lock": {
            "path": candidate["path"],
            "sha256": candidate["sha256"],
        },
        "model": str(model),
        "model_sha256": sha256_file(model),
        "metadata": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
        },
        "blind_manifest": {
            "path": str(blind_manifest),
            "sha256": sha256_file(blind_manifest),
            "records": 144,
            "identities": 36,
            "queries": 72,
        },
        "attempt_marker": str(marker),
        "provider": pipeline.backend_info(),
        "retrieval": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "promotion_eligible": all(checks.values()),
        "default_backend_changed": False,
        "post_blind_tuning_permitted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    complete_blind_attempt(marker, sha256_file(output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()