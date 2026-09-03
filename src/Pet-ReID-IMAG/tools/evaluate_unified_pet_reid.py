#!/usr/bin/env python3
"""Evaluate a one-input UnifiedPetReID checkpoint on a locked manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified import UnifiedPetReID
from pet_id.unified_data import UnifiedManifestDataset
from pet_id.unified_training import (
    build_model_from_checkpoint,
    geometry_losses,
    load_acceptance,
    retrieval_metrics,
    sha256_file,
)
from pet_id.release_compatibility import acceptance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
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
    parser.add_argument(
        "--geometry-source",
        choices=("predicted", "teacher"),
        default="predicted",
        help="teacher is an oracle diagnostic and can never pass deployment acceptance.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--input-size",
        type=int,
        help=(
            "Fixed square RGB size for an untrained ArcFace-anchor oracle. "
            "For a checkpoint this is only an optional compatibility assertion."
        ),
    )
    letterbox = parser.add_mutually_exclusive_group()
    letterbox.add_argument(
        "--letterbox-upscale",
        dest="letterbox_upscale",
        action="store_true",
        help="Allow smaller images to be enlarged to fill the fixed canvas.",
    )
    letterbox.add_argument(
        "--no-letterbox-upscale",
        dest="letterbox_upscale",
        action="store_false",
        help="Preserve smaller manifest images at native resolution and only pad them.",
    )
    parser.set_defaults(letterbox_upscale=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--include-queries", action="store_true")
    return parser.parse_args()


def identify_protocol(
    acceptance: dict[str, Any], manifest_hash: str
) -> tuple[str | None, dict[str, Any] | None]:
    for name, protocol in acceptance["required_evaluations"].items():
        if protocol.get("manifest_sha256") == manifest_hash:
            return name, protocol
    return None, None


def acceptance_result(
    metrics: dict[str, Any],
    protocol: dict[str, Any] | None,
    *,
    oracle: bool,
) -> dict[str, Any]:
    if protocol is None:
        return {
            "eligible": False,
            "passed": False,
            "reason": "manifest is not a locked deployment evaluation",
        }
    checks = {
        "top1_correct": metrics["top1_correct"]
        >= protocol["minimum_top1_correct"],
        "top1_accuracy": metrics["top1_accuracy"]
        >= protocol["minimum_top1_accuracy"],
        "top5_correct": metrics["top5_correct"]
        >= protocol["minimum_top5_correct"],
        "top5_accuracy": metrics["top5_accuracy"]
        >= protocol["minimum_top5_accuracy"],
    }
    eligible = not oracle
    return {
        "eligible": eligible,
        "passed": eligible and all(checks.values()),
        "checks": checks,
        "thresholds": {
            key: value
            for key, value in protocol.items()
            if key.startswith("minimum_")
        },
        "reason": (
            "teacher geometry is an oracle diagnostic"
            if oracle
            else "all locked thresholds passed"
            if all(checks.values())
            else "one or more locked noninferiority thresholds failed"
        ),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    arcface_path = args.arcface_checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    acceptance = load_acceptance(acceptance_path)
    expected_arcface_hash = acceptance["source_weight_locks"][
        "dog_arcface_checkpoint"
    ]["sha256"]
    if sha256_file(arcface_path) != expected_arcface_hash:
        raise RuntimeError("ArcFace checkpoint differs from the acceptance lock")
    device = torch.device(args.device)
    if args.checkpoint:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        model, checkpoint = build_model_from_checkpoint(
            checkpoint_path,
            arcface_path,
            device=device,
        )
        if args.input_size is not None and args.input_size != model.input_size:
            raise ValueError(
                f"Checkpoint input_size is {model.input_size}, not {args.input_size}"
            )
        letterbox_upscale = bool(
            checkpoint.get("preprocessing", {}).get(
                "letterbox_allow_upscale", True
            )
        )
        if (
            args.letterbox_upscale is not None
            and args.letterbox_upscale != letterbox_upscale
        ):
            raise ValueError("CLI letterbox policy differs from checkpoint")
    else:
        checkpoint_path = None
        checkpoint = None
        model = UnifiedPetReID.from_arcface_checkpoint(
            arcface_path,
            input_size=args.input_size or 1280,
        ).to(device)
        letterbox_upscale = bool(args.letterbox_upscale)
    model.eval()
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=letterbox_upscale,
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
    embeddings = []
    face_embeddings = []
    identities: list[str] = []
    source_paths: list[str] = []
    geometry_sums = {
        "geometry_center": 0.0,
        "geometry_size": 0.0,
        "geometry_angle": 0.0,
        "geometry_containment": 0.0,
        "geometry_total": 0.0,
    }
    records = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in raw_batch.items()
            }
            override = (
                {
                    "boxes_cxcywh": batch["boxes_cxcywh"],
                    "angle_radians": batch["angle_radians"],
                }
                if args.geometry_source == "teacher"
                else None
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(
                    batch["rgb"],
                    return_aux=True,
                    geometry_override=override,
                )
            losses = geometry_losses(
                output["boxes_cxcywh"].float(),
                output["angle_radians"].float(),
                batch["boxes_cxcywh"].float(),
                batch["angle_radians"].float(),
            )
            count = int(batch["rgb"].shape[0])
            records += count
            for name in geometry_sums:
                geometry_sums[name] += float(losses[name]) * count
            embeddings.append(output["embedding"].float().cpu())
            face_embeddings.append(
                output["face_descriptor"].float().cpu()
            )
            identities.extend(
                identity.casefold() for identity in raw_batch["identity"]
            )
            source_paths.extend(raw_batch["source_path"])
            print(f"unified evaluation: {records}/{len(dataset)}", flush=True)

    embedding_metrics = retrieval_metrics(
        torch.cat(embeddings),
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=args.include_queries,
    )
    face_metrics = retrieval_metrics(
        torch.cat(face_embeddings),
        identities,
        source_paths,
        gallery_images_per_identity=2,
        include_queries=False,
    )
    manifest_hash = sha256_file(manifest_path)
    protocol_name, protocol = identify_protocol(acceptance, manifest_hash)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_pet_reid",
        "runtime_contract": {
            "inputs": [
                {
                    "name": "rgb",
                    "shape": ["N", 3, model.input_size, model.input_size],
                }
            ],
            "outputs": [{"name": "embedding", "shape": ["N", 512]}],
            "external_models": [],
            "letterbox_allow_upscale": letterbox_upscale,
        },
        "geometry_source": args.geometry_source,
        "oracle_diagnostic": args.geometry_source == "teacher",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "protocol_name": protocol_name,
        "records": records,
        "identities": len(set(identities)),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path)
        if checkpoint_path
        else None,
        "checkpoint_stage": checkpoint.get("stage")
        if checkpoint
        else "untrained_arcface_anchor",
        "arcface_checkpoint_sha256": expected_arcface_hash,
        "amp_dtype": str(amp_dtype).removeprefix("torch.")
        if use_amp
        else "float32",
        "geometry": {
            name: value / records
            for name, value in geometry_sums.items()
        },
        "embedding": embedding_metrics,
        "face_descriptor": face_metrics,
        "acceptance": acceptance_result(
            embedding_metrics,
            protocol,
            oracle=args.geometry_source == "teacher",
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    concise = {
        "output": str(output_path),
        "protocol_name": protocol_name,
        "geometry_source": args.geometry_source,
        "embedding": {
            key: embedding_metrics[key]
            for key in (
                "top1_correct",
                "top1_accuracy",
                "top5_correct",
                "top5_accuracy",
                "mean_reciprocal_rank",
                "auc",
            )
        },
        "face_descriptor": {
            key: face_metrics[key]
            for key in (
                "top1_correct",
                "top1_accuracy",
                "top5_correct",
                "top5_accuracy",
            )
        },
        "geometry_total": report["geometry"]["geometry_total"],
        "acceptance": report["acceptance"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

