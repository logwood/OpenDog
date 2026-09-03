"""Portable loader for structural end-to-end high-resolution checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .unified_highres import sha256_file
from .release_compatibility import (
    detail_checkpoint_source,
    migrate_structural_state_dict,
)
from .unified_highres_structural import (
    SCHEMA_VERSION,
    MODEL_TYPE,
    UnifiedHighResolutionStructuralPetReID,
    build_structural_from_detail_checkpoint,
)


def build_structural_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    detail_checkpoint_override: str | Path | None = None,
    verify_sources: bool = True,
) -> tuple[UnifiedHighResolutionStructuralPetReID, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported structural checkpoint schema")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint is not a structural high-resolution model")
    source = detail_checkpoint_source(payload["sources"])
    base = (
        Path(detail_checkpoint_override).expanduser().resolve()
        if detail_checkpoint_override is not None
        else Path(source["path"]).expanduser().resolve()
    )
    if not base.is_file():
        raise FileNotFoundError(base)
    if verify_sources and sha256_file(base) != source["sha256"]:
        raise RuntimeError("Spatial-detail source checkpoint hash mismatch")
    configuration = payload["model_config"]
    bridge = configuration["bridge"]
    residual = configuration["structural_residual"]
    model, _ = build_structural_from_detail_checkpoint(
        base,
        device=device,
        verify_sources=verify_sources,
        bridge_variant=str(bridge["variant"]),
        bridge_token_dim=int(bridge["token_dim"]),
        bridge_bottleneck_dim=int(bridge["bottleneck_dim"]),
        bridge_hidden_dim=int(bridge["hidden_dim"]),
        bridge_attention_heads=int(bridge["attention_heads"]),
        bridge_dropout=float(bridge["dropout"]),
        residual_hidden_dim=int(residual["hidden_dim"]),
        maximum_structural_residual=float(residual["maximum_residual_scale"]),
    )
    incompatible = model.load_state_dict(
        migrate_structural_state_dict(payload["model"]), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Structural checkpoint mismatch: {incompatible}")
    model.configure_trainable(nose_encoder_parts=(), structural=False)
    return model.to(device).eval(), payload
