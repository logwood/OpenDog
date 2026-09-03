#!/usr/bin/env python3
"""Transplant a FastReID nose checkpoint into a UnifiedPetReID V4 model.

This tool intentionally changes only the specialist nose encoder.  Geometry,
ArcFace, the nose-to-face adapter, the V3 bounded fusion module, and the V4
high-resolution refiner remain byte-for-byte inherited from the base V4
checkpoint.  The resulting checkpoint is a development candidate and must be
evaluated before it can be considered for deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_highres import (  # noqa: E402
    MODEL_TYPE,
    build_highres_from_checkpoint,
    save_highres_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--nose-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-compatible-ratio", type=float, default=0.80)
    return parser.parse_args()


def source_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Nose checkpoint does not contain a non-empty model state")
    return state


def compatible_nose_state(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    unexpected: list[str] = []
    for key, value in source.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key not in target:
            unexpected.append(clean_key)
            continue
        if tuple(value.shape) != tuple(target[clean_key].shape):
            shape_mismatches.append(clean_key)
            continue
        compatible[clean_key] = value

    target_numel = sum(value.numel() for value in target.values())
    compatible_numel = sum(value.numel() for value in compatible.values())
    changed_tensors = sum(
        not torch.equal(value, target[key]) for key, value in compatible.items()
    )
    report = {
        "source_tensor_count": len(source),
        "target_tensor_count": len(target),
        "compatible_tensor_count": len(compatible),
        "compatible_tensor_ratio": len(compatible) / max(len(target), 1),
        "compatible_parameter_ratio": compatible_numel / max(target_numel, 1),
        "changed_tensor_count": changed_tensors,
        "missing_target_keys": sorted(set(target) - set(compatible)),
        "shape_mismatch_keys": sorted(shape_mismatches),
        "ignored_source_keys": sorted(unexpected),
    }
    return compatible, report


def main() -> None:
    args = parse_args()
    base_path = args.base_checkpoint.expanduser().resolve()
    nose_path = args.nose_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (base_path, nose_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if not 0.0 < args.minimum_compatible_ratio <= 1.0:
        raise ValueError("minimum-compatible-ratio must be in (0, 1]")

    model, payload = build_highres_from_checkpoint(
        base_path,
        device="cpu",
        verify_sources=True,
    )
    if payload.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Base checkpoint is not UnifiedPetReID V4")

    nose_model = model.parent_model.base_model.nose_encoder.model
    target_state = nose_model.state_dict()
    compatible, compatibility = compatible_nose_state(
        checkpoint_state(nose_path),
        target_state,
    )
    if compatibility["compatible_tensor_ratio"] < args.minimum_compatible_ratio:
        raise RuntimeError(
            "Nose checkpoint tensor compatibility is below the required ratio: "
            f"{compatibility['compatible_tensor_count']}/"
            f"{compatibility['target_tensor_count']}"
        )
    incompatible = nose_model.load_state_dict(compatible, strict=False)
    if sorted(incompatible.missing_keys) != compatibility["missing_target_keys"]:
        raise RuntimeError("Unexpected missing keys while loading the nose checkpoint")
    model.configure_trainable(refiner=False)
    model.eval()

    base_source = source_record(base_path)
    nose_source = source_record(nose_path)
    payload["model"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload.setdefault("sources", {})["nose_checkpoint_override"] = nose_source
    inherited_training = dict(payload.get("training") or {})
    inherited_training.update(
        {
            "blind_data_used": False,
            "integration_mode": "direct_nose_encoder_transplant",
            "adapter_retrained": False,
            "fusion_retrained": False,
            "nose_checkpoint_override": nose_source,
        }
    )
    payload["training"] = inherited_training
    payload["selection"] = {
        "status": "development_candidate_unscored",
        "rule": "nose encoder transplant only; no blind data; development evaluation required",
        "base_v4_checkpoint": base_source,
        "base_v4_selection": payload.get("selection"),
    }
    payload["integration"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "changed_component": "parent_model.base_model.nose_encoder.model",
        "unchanged_components": [
            "geometry_frontend",
            "arcface_encoder",
            "nose_adapter",
            "v3_bounded_fusion",
            "v4_high_resolution_refiner",
        ],
        "compatibility": compatibility,
    }
    payload["promotion_status"] = "development_validation_required"
    payload["default_backend_changed"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_highres_checkpoint(payload, output_path)
    print(
        json.dumps(
            {
                "output": source_record(output_path),
                "base_checkpoint": base_source,
                "nose_checkpoint": nose_source,
                "compatibility": compatibility,
                "promotion_status": payload["promotion_status"],
                "default_backend_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
