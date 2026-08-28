#!/usr/bin/env python3
"""Cache frozen semantic-v3 nose/face features for body-fusion experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model
from pet_id.workspace_paths import SELECTED_MODELS_ROOT, normalize_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "model_final.pth",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "config.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PreparedDogFaceNetDataset(args.manifest, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prepared_dogfacenet,
    )

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(args.config_file))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=str(args.checkpoint),
    )
    model.eval()
    if not getattr(model, "joint_enabled", False):
        raise RuntimeError("The source model must expose semantic adapters")

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    tensor_rows: dict[str, list[torch.Tensor]] = {
        "baseline_features": [],
        "raw_nose_features": [],
        "adapted_nose_features": [],
        "raw_face_features": [],
        "adapted_face_features": [],
        "gate_quality_signals": [],
        "quality_signals": [],
        "viewpoint_signals": [],
        "branch_available": [],
        "fusion_weights": [],
    }
    identities: list[str] = []
    source_paths: list[str] = []

    for batch_index, batch in enumerate(loader, start=1):
        inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"targets", "identities", "source_paths"}
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = model(**inputs)
            raw_nose = output["nose_features"]
            raw_face = output["face_features"]
            adapted_nose = model.nose_adapter(raw_nose)
            adapted_face = model.face_adapter(raw_face)

        quality = inputs["quality_signals"].float()
        viewpoint = inputs["viewpoint_signals"].float()
        joint_quality = quality.clone()
        joint_quality[:, 0] *= output["viewpoint_frontality"].float()
        gate_quality = torch.cat((joint_quality, viewpoint), dim=1)
        effective_available = output["effective_branch_available"].bool()

        values = {
            "baseline_features": output["features"],
            "raw_nose_features": raw_nose,
            "adapted_nose_features": adapted_nose,
            "raw_face_features": raw_face,
            "adapted_face_features": adapted_face,
            "gate_quality_signals": gate_quality,
            "quality_signals": quality,
            "viewpoint_signals": viewpoint,
            "branch_available": effective_available,
            "fusion_weights": output["fusion_weights"],
        }
        for name, value in values.items():
            tensor_rows[name].append(value.detach().float().cpu())
        identities.extend(identity.casefold() for identity in batch["identities"])
        source_paths.extend(str(Path(path).resolve()) for path in batch["source_paths"])
        processed = min(batch_index * args.batch_size, len(dataset))
        print(f"semantic features: {processed}/{len(dataset)}", flush=True)

    arrays = {
        name: torch.cat(rows).numpy()
        for name, rows in tensor_rows.items()
    }
    arrays["branch_available"] = arrays["branch_available"].astype(np.bool_)
    arrays["identities"] = np.asarray(identities)
    arrays["source_paths"] = np.asarray(source_paths)
    if arrays["baseline_features"].shape[1] != 512:
        raise RuntimeError(
            "Expected the semantic-v3 public embedding to be 512-D, got "
            f"{arrays['baseline_features'].shape}"
        )
    for name in ("adapted_nose_features", "adapted_face_features"):
        arrays[name] = F.normalize(
            torch.from_numpy(arrays[name]).float(), dim=1
        ).numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "multimodal_branch_features.npz"
    np.savez_compressed(archive_path, **arrays)
    metadata = {
        "schema_version": 1,
        "purpose": "frozen_semantic_v3_features_for_body_primary_fusion",
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "config_file": str(args.config_file.resolve()),
        "records": len(dataset),
        "identities": len(set(identities)),
        "public_input": "one prepared dog image plus existing internal geometry",
        "public_output": "512-D L2-normalized identity embedding",
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in arrays.items()
        },
        "archive": str(archive_path.resolve()),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
