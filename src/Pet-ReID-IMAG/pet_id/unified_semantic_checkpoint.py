"""Checkpoint construction for the single-graph semantic pet ReID model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from fastreid.config import get_cfg

from .config import add_retri_config
from .multimodal import build_local_identity_model
from .unified_geometry_stability import (
    DEFAULT_GEOMETRY_ANGLE_OFFSET,
    DEFAULT_GEOMETRY_BOX_OFFSETS,
)
from .unified_semantic import UnifiedSemanticPetReID
from .unified_training import (
    atomic_torch_save,
    build_model_from_checkpoint,
    sha256_file,
)
from .workspace_paths import normalize_runtime_config, resolve_legacy_path


MODEL_TYPE = "unified_semantic_pet_reid"
SCHEMA_VERSION = 1


def _source_record(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verified_source(record: dict[str, Any]) -> Path:
    path = resolve_legacy_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise RuntimeError(
            f"Locked source hash mismatch for {path}: "
            f"expected {record['sha256']}, got {actual}"
        )
    return path


def build_semantic_identity_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
):
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(torch.device(device))
    cfg.freeze()
    return build_local_identity_model(
        cfg,
        device=torch.device(device),
        for_training=False,
        identity_weights=checkpoint_path,
    )


def semantic_model_configuration(
    model: UnifiedSemanticPetReID,
) -> dict[str, Any]:
    geometry = model.geometry_frontend
    return {
        "input_size": model.input_size,
        "nose_crop_size": model.nose_crop_size,
        "face_crop_scales": list(model.face_crop_scales),
        "face_crop_weights": list(model.face_crop_weights),
        "geometry": {
            "input_size": geometry.input_size,
            "localization_size": geometry.localization_size,
            "crop_size": geometry.crop_size,
            "geometry_hidden_channels": geometry.geometry.hidden_channels,
            "fusion_hidden_dim": int(
                geometry.semantic_fusion.context_projection[1].out_features
            ),
            "geometry_feature_mode": geometry.geometry_feature_mode,
            "maximum_residual_scale": (geometry.semantic_fusion.maximum_residual_scale),
            "geometry_minimum_sizes": list(geometry.geometry.minimum_sizes),
            "geometry_maximum_sizes": list(geometry.geometry.maximum_sizes),
        },
        "geometry_discretization": model.geometry_discretizer.configuration(),
        "fusion": model.fusion.configuration(),
    }


def build_unified_semantic_from_sources(
    geometry_checkpoint: str | Path,
    semantic_config: str | Path,
    semantic_checkpoint: str | Path,
    arcface_checkpoint: str | Path,
    *,
    maximum_nose_weight: float = 0.225,
    face_confidence_threshold: float = 0.44,
    temperature: float = 0.02,
    face_crop_scales: Sequence[float] = (1.0,),
    face_crop_weights: Sequence[float] | None = None,
    geometry_box_step: float | list[list[float]] | None = None,
    geometry_box_offsets: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] | list[list[float]] = DEFAULT_GEOMETRY_BOX_OFFSETS,
    geometry_angle_step: float | None = None,
    geometry_angle_offset: float = DEFAULT_GEOMETRY_ANGLE_OFFSET,
    geometry_box_piecewise: list[dict[str, Any]] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[UnifiedSemanticPetReID, dict[str, Any]]:
    device = torch.device(device)
    geometry_model, geometry_payload = build_model_from_checkpoint(
        geometry_checkpoint,
        arcface_checkpoint,
        device=device,
    )
    identity_model = build_semantic_identity_model(
        semantic_config,
        semantic_checkpoint,
        device=device,
    )
    model = UnifiedSemanticPetReID.from_semantic_residual(
        geometry_model,
        identity_model,
        maximum_nose_weight=maximum_nose_weight,
        face_confidence_threshold=face_confidence_threshold,
        temperature=temperature,
        face_crop_scales=face_crop_scales,
        face_crop_weights=face_crop_weights,
        gate_trainable=False,
        geometry_box_step=geometry_box_step,
        geometry_box_offsets=geometry_box_offsets,
        geometry_angle_step=geometry_angle_step,
        geometry_angle_offset=geometry_angle_offset,
        geometry_box_piecewise=geometry_box_piecewise,
    )
    model.configure_trainable(
        geometry=False,
        nose_encoder_parts=(),
        nose_adapter=False,
        fusion=False,
    )
    model.to(device).eval()
    return model, geometry_payload


def create_unified_semantic_checkpoint(
    model: UnifiedSemanticPetReID,
    *,
    geometry_checkpoint: str | Path,
    semantic_config: str | Path,
    semantic_checkpoint: str | Path,
    arcface_checkpoint: str | Path,
    policy_evidence: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources = {
        "geometry_checkpoint": _source_record(geometry_checkpoint),
        "semantic_config": _source_record(semantic_config),
        "semantic_checkpoint": _source_record(semantic_checkpoint),
        "arcface_checkpoint": _source_record(arcface_checkpoint),
    }
    evidence = _source_record(policy_evidence) if policy_evidence is not None else None
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_config": semantic_model_configuration(model),
        "model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "sources": sources,
        "policy_evidence": evidence,
        "preprocessing": {
            "input_range": [0, 255],
            "color_order": "RGB",
            "letterbox": "centered_black",
            "letterbox_allow_upscale": False,
        },
        "runtime_contract": {
            "inputs": {
                "rgb": {
                    "dtype": "float32",
                    "shape": ["N", 3, model.input_size, model.input_size],
                }
            },
            "outputs": {
                "embedding": {
                    "dtype": "float32",
                    "shape": ["N", 512],
                    "l2_normalized": True,
                }
            },
            "external_models": [],
        },
        "selection": {
            "epoch": 0,
            "rule": "locked safe development policy; later epochs may only replace it after all noninferiority gates pass",
        },
        **(extra or {}),
    }


def save_unified_semantic_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    atomic_torch_save(payload, path)


def build_unified_semantic_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
) -> tuple[UnifiedSemanticPetReID, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported unified semantic checkpoint schema")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint is not UnifiedSemanticPetReID")
    sources = payload["sources"]
    if verify_sources:
        resolved = {name: _verified_source(record) for name, record in sources.items()}
    else:
        resolved = {
            name: resolve_legacy_path(record["path"])
            for name, record in sources.items()
        }
    configuration = payload["model_config"]
    fusion = configuration["fusion"]
    discretization = configuration.get("geometry_discretization", {})
    box_step_value = discretization.get("box_step")
    angle_step_value = discretization.get("angle_step")
    model, _ = build_unified_semantic_from_sources(
        resolved["geometry_checkpoint"],
        resolved["semantic_config"],
        resolved["semantic_checkpoint"],
        resolved["arcface_checkpoint"],
        maximum_nose_weight=float(fusion["maximum_nose_weight"]),
        face_confidence_threshold=float(fusion["face_confidence_threshold"]),
        temperature=float(fusion["temperature"]),
        face_crop_scales=configuration.get("face_crop_scales", (1.0,)),
        face_crop_weights=configuration.get("face_crop_weights"),
        geometry_box_step=box_step_value,
        geometry_box_offsets=discretization.get(
            "box_offsets", DEFAULT_GEOMETRY_BOX_OFFSETS
        ),
        geometry_angle_step=(
            float(angle_step_value) if angle_step_value is not None else None
        ),
        geometry_angle_offset=float(
            discretization.get(
                "angle_offset",
                DEFAULT_GEOMETRY_ANGLE_OFFSET,
            )
        ),
        geometry_box_piecewise=discretization.get("box_piecewise"),
        device=device,
    )
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unified semantic checkpoint mismatch: {incompatible}")
    model.configure_trainable(
        geometry=False,
        nose_encoder_parts=(),
        nose_adapter=False,
        fusion=False,
    )
    return model.to(device).eval(), payload
