#!/usr/bin/env python3
"""Evaluate one spatial-detail checkpoint on the locked validation set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_highres import build_highres_from_checkpoint  # noqa: E402
from pet_id.release_compatibility import locked_protocol_paths  # noqa: E402
from train_unified_nose_detail import (  # noqa: E402
    LockedDetailDataset,
    evaluate,
    read_json,
    sha256_file,
    workspace_path,
)


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = workspace_path(args.checkpoint)
    output = workspace_path(args.output)
    lock_path, _, manifest_path = locked_protocol_paths(
        WORKSPACE, config["protocol"]
    )
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("The validation protocol is not LOCKED_UNSCORED")
    expected_manifest_hash = lock["splits"]["validation"]["sha256"]
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_hash:
        raise RuntimeError("Locked validation manifest hash mismatch")

    manifest = read_json(manifest_path)
    verified_bytes = 0
    for row in manifest["records"]:
        source = workspace_path(row["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256_file(source) != str(row["source_sha256"]):
            raise RuntimeError(f"Locked validation source hash mismatch: {source}")
        verified_bytes += source.stat().st_size

    device = torch.device(args.device)
    model, model_payload = build_highres_from_checkpoint(
        checkpoint,
        device=device,
        verify_sources=True,
    )
    model.eval()
    training_size = int(config["model"]["training_size"])
    dataset = LockedDetailDataset(
        manifest_path,
        training_size=training_size,
        degraded_size=int(config["model"]["degraded_detail_size"]),
        training=False,
        horizontal_flip=0.0,
        color_jitter=0.0,
    )
    expected_identities = int(config["protocol"]["validation_identities"])
    if dataset.num_classes != expected_identities:
        raise RuntimeError(
            f"Validation identity mismatch: {dataset.num_classes} != {expected_identities}"
        )
    amp_name = str(config["training"]["amp"]).casefold()
    use_amp = device.type == "cuda" and amp_name != "float32"
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    started = time.monotonic()
    metrics = evaluate(
        model,
        dataset,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        gallery_images=int(config["protocol"]["gallery_images_per_identity"]),
    )
    payload = {
        "schema_version": 1,
        "evaluation": "spatial_detail_locked_validation",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "model_type": model_payload.get("model_type"),
        },
        "protocol": {
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "validation_manifest": str(manifest_path),
            "validation_manifest_sha256": manifest_hash,
            "verified_images": len(manifest["records"]),
            "verified_bytes": verified_bytes,
            "gallery_images_per_identity": int(
                config["protocol"]["gallery_images_per_identity"]
            ),
        },
        "runtime": {
            "device": str(device),
            "amp": amp_name,
            "training_size": training_size,
            "elapsed_seconds": time.monotonic() - started,
            "cuda_max_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
        "metrics": metrics,
    }
    atomic_json_dump(payload, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
