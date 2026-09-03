#!/usr/bin/env python3
"""Run the only aggregate blind evaluation allowed for unified external v3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_data import UnifiedRawManifestDataset  # noqa: E402
from pet_id.unified_external_protocol import validate_raw_manifest  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_training import retrieval_metrics, sha256_file  # noqa: E402
from pet_id.unified_v3_candidate import (  # noqa: E402
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
        default=WORKSPACE / "models/acceptance/unified_pet_reid_v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"Must equal {BLIND_CONFIRMATION!r}",
    )
    return parser.parse_args()


def identity_and_source_sets(payload: dict) -> tuple[set[str], set[str]]:
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Locked manifest has no records")
    identities = [str(row.get("identity", "")).casefold() for row in rows]
    sources = [str(row.get("source_sha256", "")).casefold() for row in rows]
    if any(not value for value in identities + sources):
        raise RuntimeError("Locked manifest has incomplete records")
    if len(sources) != len(set(sources)):
        raise RuntimeError("Locked manifest contains duplicate source images")
    declared = int(payload.get("images_per_identity", 0))
    counts = Counter(identities)
    if declared < 1 or any(count != declared for count in counts.values()):
        raise RuntimeError("Locked manifest identity counts changed")
    return set(identities), set(sources)


def main() -> None:
    args = parse_args()
    if args.confirm != BLIND_CONFIRMATION:
        raise RuntimeError("Explicit unified v3 one-shot confirmation is incorrect")
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
    if output != candidate["report_path"]:
        raise RuntimeError("Blind output differs from the locked report path")
    marker_path = candidate["attempt_marker"]
    if marker_path.exists() or output.exists():
        raise FileExistsError("The unified v3 blind candidate attempt is already spent")

    blind = acceptance["blind"]
    expected_dimensions = (
        int(blind["records"]),
        int(blind["identities"]),
        int(blind["images_per_identity"]),
    )
    if expected_dimensions != (512, 128, 4):
        raise RuntimeError("Locked v3 blind dimensions changed")
    thresholds = {
        "minimum_top1_correct": int(blind["minimum_top1_correct"]),
        "minimum_top5_correct": int(blind["minimum_top5_correct"]),
    }
    if thresholds != {
        "minimum_top1_correct": 69,
        "minimum_top5_correct": 104,
    }:
        raise RuntimeError("Locked v3 blind thresholds changed")

    # Complete model/provider preflight before permanently reserving the attempt.
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

    # Protected manifest contents are first parsed after the irreversible marker.
    blind_manifest = candidate["blind_manifest"]
    manifest_payload = json.loads(blind_manifest.read_text(encoding="utf-8"))
    summary = validate_raw_manifest(
        manifest_payload,
        expected_split="blind_test",
    )
    if (summary["records"], summary["identities"]) != (512, 128):
        raise RuntimeError("Blind manifest dimensions changed")

    split_sets = {}
    for name in ("training", "development"):
        record = acceptance[name]
        path = Path(record["path"]).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Locked {name} manifest changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_split = "training_extension" if name == "training" else name
        if payload.get("protocol_split") != expected_split:
            raise RuntimeError(f"Locked {name} protocol split changed")
        split_sets[name] = identity_and_source_sets(payload)
    split_sets["blind"] = identity_and_source_sets(manifest_payload)
    for left, right in (
        ("training", "development"),
        ("training", "blind"),
        ("development", "blind"),
    ):
        if split_sets[left][0].intersection(split_sets[right][0]):
            raise RuntimeError(f"{left}/{right} identity overlap")
        if split_sets[left][1].intersection(split_sets[right][1]):
            raise RuntimeError(f"{left}/{right} source overlap")

    dataset = UnifiedRawManifestDataset(
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
    embedding_rows: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    processed = 0
    for batch in loader:
        rgb = batch["rgb"].numpy().astype(np.float32, copy=False)
        embedding_rows.append(pipeline._run(rgb))
        identities.extend(batch["identity"])
        source_paths.extend(batch["source_path"])
        processed += int(rgb.shape[0])
        if processed == args.batch_size or processed % 64 == 0:
            print(f"unified v3 blind ONNX: {processed}/{len(dataset)}", flush=True)
    embeddings = torch.cat(embedding_rows)
    metrics = retrieval_metrics(
        embeddings,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=False,
    )
    del embeddings, embedding_rows, identities, source_paths
    if (
        metrics["gallery_identities"],
        metrics["gallery_records"],
        metrics["query_records"],
    ) != (128, 256, 256):
        raise RuntimeError("Blind gallery/query dimensions changed")
    checks = {
        "top1": int(metrics["top1_correct"]) >= thresholds["minimum_top1_correct"],
        "top5": int(metrics["top5_correct"]) >= thresholds["minimum_top5_correct"],
    }
    report = {
        "schema_version": 1,
        "protocol_name": acceptance["protocol_name"],
        "purpose": "single_aggregate_external_v3_candidate_blind_evaluation",
        "single_attempt": True,
        "aggregate_only": True,
        "per_query_results_stored": False,
        "feature_cache_persisted": False,
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
            "records": 512,
            "identities": 128,
            "queries": 256,
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
