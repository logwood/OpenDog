#!/usr/bin/env python3
"""Run the locked, single-attempt Joint800 unseen-200 ONNX evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.release_compatibility import historical_run_path  # noqa: E402
from pet_id.unified_blind_protocol import (  # noqa: E402
    JOINT800_BLIND_SHA256,
    JOINT800_CONFIRMATION,
    JOINT800_TRAIN_SHA256,
    complete_attempt_marker,
    reserve_single_attempt,
    sha256_file,
    validate_candidate_lock,
    validate_disjoint_splits,
    validate_manifest,
)
from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "joint-rollback-protocol")
        / "train_manifest.json",
    )
    parser.add_argument(
        "--blind-manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "joint-rollback-protocol")
        / "blind_test_manifest.json",
    )
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"Must equal {JOINT800_CONFIRMATION!r}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm != JOINT800_CONFIRMATION:
        raise RuntimeError("Explicit one-shot confirmation text is incorrect")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    model_path = args.model.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    if output_path.exists() or output_path.with_name(
        output_path.name + ".attempt.json"
    ).exists():
        raise FileExistsError("Joint800 blind evaluation was already attempted")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    protocol = acceptance["required_evaluations"]["joint800_unseen200"]
    if protocol["manifest_sha256"] != JOINT800_BLIND_SHA256:
        raise RuntimeError("Acceptance file has a changed Joint800 blind hash")
    if int(protocol["records"]) != 800 or int(protocol["identities"]) != 200:
        raise RuntimeError("Acceptance file has changed Joint800 dimensions")
    if int(protocol["queries"]) != 400:
        raise RuntimeError("Acceptance file has changed Joint800 query count")
    candidate_lock = validate_candidate_lock(
        args.candidate_lock,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    train = validate_manifest(
        args.train_manifest,
        expected_sha256=JOINT800_TRAIN_SHA256,
        expected_split="train",
        expected_records=3200,
        expected_identities=800,
        expected_images_per_identity=4,
    )
    blind = validate_manifest(
        args.blind_manifest,
        expected_sha256=JOINT800_BLIND_SHA256,
        expected_split="blind_test",
        expected_records=800,
        expected_identities=200,
        expected_images_per_identity=4,
    )
    validate_disjoint_splits(train, blind)
    marker = reserve_single_attempt(
        output_path,
        candidate_lock_sha256=candidate_lock["sha256"],
    )

    pipeline = UnifiedONNXRuntimePipeline(
        model_path,
        provider=args.provider,
        metadata_path=metadata_path,
        verify_hash=True,
    )
    dataset = UnifiedManifestDataset(
        blind.path,
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
            print(f"Joint800 unified ONNX: {processed}/{len(dataset)}", flush=True)
    embedding = torch.cat(embeddings)
    metrics = retrieval_metrics(
        embedding,
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=False,
    )
    if metrics["gallery_identities"] != 200:
        raise RuntimeError("Joint800 gallery identity count changed")
    if metrics["gallery_records"] != 400 or metrics["query_records"] != 400:
        raise RuntimeError("Joint800 gallery/query split changed")
    thresholds = {
        "minimum_top1_correct": int(protocol["minimum_top1_correct"]),
        "minimum_top5_correct": int(protocol["minimum_top5_correct"]),
    }
    checks = {
        "top1": metrics["top1_correct"] >= thresholds["minimum_top1_correct"],
        "top5": metrics["top5_correct"] >= thresholds["minimum_top5_correct"],
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "joint800_unseen200",
        "single_attempt": True,
        "candidate_lock": {
            "path": candidate_lock["path"],
            "sha256": candidate_lock["sha256"],
        },
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
        },
        "train_manifest": train.report(),
        "blind_manifest": blind.report(),
        "split_overlap": {"identities": 0, "source_sha256": 0},
        "provider": pipeline.backend_info(),
        "retrieval": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "default_backend_changed": False,
        "post_blind_tuning_permitted": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    report_hash = sha256_file(output_path)
    complete_attempt_marker(marker, report_hash)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
