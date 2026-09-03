#!/usr/bin/env python3
"""Evaluate deterministic cross-identity nose injection inside UnifiedPetReID."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset
from pet_id.unified_training import (
    build_model_from_checkpoint,
    load_acceptance,
    retrieval_metrics,
    sha256_file,
)
from pet_id.release_compatibility import (  # noqa: E402
    acceptance_path,
    historical_run_path,
)


@dataclass(frozen=True)
class ConflictPair:
    query_index: int
    donor_index: int
    query_identity: str
    donor_identity: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "shared-fusion-baseline")
        / "dev_validation_manifest.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=acceptance_path(WORKSPACE, "legacy-training"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_conflict_pairs(
    identities: list[str], gallery_per_identity: int
) -> list[ConflictPair]:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    ordered = sorted(grouped)
    if len(ordered) < 2:
        raise ValueError("At least two identities are required")
    pairs = []
    for identity_position, identity in enumerate(ordered):
        queries = grouped[identity][gallery_per_identity:]
        if not queries:
            raise ValueError(f"Identity {identity!r} has no held-out query")
        donor_identity = ordered[(identity_position + 1) % len(ordered)]
        donors = grouped[donor_identity][gallery_per_identity:]
        if not donors:
            raise ValueError(
                f"Donor identity {donor_identity!r} has no query"
            )
        for offset, query_index in enumerate(queries):
            pairs.append(
                ConflictPair(
                    query_index=query_index,
                    donor_index=donors[offset % len(donors)],
                    query_identity=identity,
                    donor_identity=donor_identity,
                )
            )
    return pairs


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def query_transition(clean: dict, corrupted: dict) -> dict:
    clean_by_path = {
        row["query_source_path"]: row for row in clean["queries"]
    }
    rows = []
    for row in corrupted["queries"]:
        before = clean_by_path[row["query_source_path"]]
        rows.append(
            {
                "source_path": row["query_source_path"],
                "identity": row["query_identity"],
                "clean_rank": before["true_identity_rank"],
                "corrupted_rank": row["true_identity_rank"],
            }
        )
    return {
        "correct_to_correct": sum(
            row["clean_rank"] == 1 and row["corrupted_rank"] == 1
            for row in rows
        ),
        "correct_to_wrong": sum(
            row["clean_rank"] == 1 and row["corrupted_rank"] != 1
            for row in rows
        ),
        "wrong_to_correct": sum(
            row["clean_rank"] != 1 and row["corrupted_rank"] == 1
            for row in rows
        ),
        "wrong_to_wrong": sum(
            row["clean_rank"] != 1 and row["corrupted_rank"] != 1
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.gallery_images_per_identity < 1:
        raise ValueError("batch sizes and gallery count must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    arcface_path = args.arcface_checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    acceptance = load_acceptance(acceptance_path)
    expected_manifest_hash = acceptance["development"][
        "validation_manifest"
    ]["sha256"]
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            "Conflict evaluation only accepts the locked development "
            "validation manifest"
        )
    expected_arcface_hash = acceptance["source_weight_locks"][
        "dog_arcface_checkpoint"
    ]["sha256"]
    if sha256_file(arcface_path) != expected_arcface_hash:
        raise RuntimeError("ArcFace checkpoint differs from acceptance lock")

    device = torch.device(args.device)
    model, checkpoint = build_model_from_checkpoint(
        checkpoint_path, arcface_path, device=device
    )
    model.eval()
    allow_upscale = bool(
        checkpoint.get("preprocessing", {}).get(
            "letterbox_allow_upscale", True
        )
    )
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=allow_upscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    clean_rows = []
    clean_scale_rows = []
    identities: list[str] = []
    source_paths: list[str] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(batch["rgb"], return_aux=True)
            clean_rows.append(output["embedding"].float().cpu())
            clean_scale_rows.append(output["residual_scale"].float().cpu())
            identities.extend(
                identity.casefold() for identity in raw_batch["identity"]
            )
            source_paths.extend(raw_batch["source_path"])
    clean_features = torch.cat(clean_rows)
    clean_scales = torch.cat(clean_scale_rows).flatten()
    pairs = build_conflict_pairs(
        identities, args.gallery_images_per_identity
    )
    corrupted_rows = []
    corrupted_scale_rows = []
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            part = pairs[start : start + args.batch_size]
            query_rgb = torch.stack(
                [dataset[pair.query_index]["rgb"] for pair in part]
            ).to(device)
            donor_rgb = torch.stack(
                [dataset[pair.donor_index]["rgb"] for pair in part]
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                query = model(query_rgb, return_aux=True)
                donor = model(donor_rgb, return_aux=True)
                corrupted, _, scale = model.semantic_fusion(
                    query["face_descriptor"],
                    donor["nose_descriptor"],
                    query["semantic_queries"],
                    query["geometry_confidence"],
                )
            corrupted_rows.append(corrupted.float().cpu())
            corrupted_scale_rows.append(scale.float().cpu())
    query_indices = torch.tensor(
        [pair.query_index for pair in pairs], dtype=torch.long
    )
    corrupted_query_features = torch.cat(corrupted_rows)
    corrupted_features = clean_features.clone()
    corrupted_features.index_copy_(
        0, query_indices, corrupted_query_features
    )
    clean_metrics = retrieval_metrics(
        clean_features,
        identities,
        source_paths,
        gallery_images_per_identity=args.gallery_images_per_identity,
        include_queries=True,
    )
    corrupted_metrics = retrieval_metrics(
        corrupted_features,
        identities,
        source_paths,
        gallery_images_per_identity=args.gallery_images_per_identity,
        include_queries=True,
    )
    protocol = acceptance["required_evaluations"][
        "nose_face_conflict_dev"
    ]
    passed = (
        corrupted_metrics["top1_correct"]
        >= protocol["minimum_corrupted_top1_correct"]
        and corrupted_metrics["top1_accuracy"]
        >= protocol["minimum_corrupted_top1_accuracy"]
    )
    corrupted_scales = torch.cat(corrupted_scale_rows).flatten()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "cross_identity_nose_injection",
        "protocol_guard": "locked_development_validation_only",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "runtime_contract_unchanged": True,
        "deployment_external_models": [],
        "pairs": [asdict(pair) for pair in pairs],
        "clean": clean_metrics,
        "corrupted": corrupted_metrics,
        "transition": query_transition(clean_metrics, corrupted_metrics),
        "gate": {
            "clean_mean_residual_scale": float(
                clean_scales.index_select(0, query_indices).mean()
            ),
            "corrupted_mean_residual_scale": float(
                corrupted_scales.mean()
            ),
        },
        "acceptance": {
            "passed": passed,
            "minimum_corrupted_top1_correct": protocol[
                "minimum_corrupted_top1_correct"
            ],
            "minimum_corrupted_top1_accuracy": protocol[
                "minimum_corrupted_top1_accuracy"
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "clean_top1": clean_metrics["top1_correct"],
                "corrupted_top1": corrupted_metrics["top1_correct"],
                "clean_mean_scale": report["gate"][
                    "clean_mean_residual_scale"
                ],
                "corrupted_mean_scale": report["gate"][
                    "corrupted_mean_residual_scale"
                ],
                "acceptance": report["acceptance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
