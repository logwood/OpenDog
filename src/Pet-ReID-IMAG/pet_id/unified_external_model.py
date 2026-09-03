"""Gradient-controlled external fusion stage for one-graph UnifiedPetReID."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .unified_semantic import UnifiedSemanticPetReID
from .unified_semantic_checkpoint import build_unified_semantic_from_checkpoint
from .unified_training import atomic_torch_save
from .workspace_paths import resolve_legacy_path


MODEL_TYPE = "unified_external_joint_pet_reid"
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, SCHEMA_VERSION}


def configure_strict_cuda_precision() -> dict[str, bool]:
    """Match PyTorch convolution precision to the strict ONNX CUDA provider."""

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GradientControlledFusionRefiner(nn.Module):
    """Add a bounded learned nose residual without changing initialization.

    ``direction_gain_logit`` starts at zero, so the exact initial result is the
    locked base embedding.  Its derivative is non-zero, allowing the global
    gate to move on the first optimizer step.  Only after that move can the
    sample reliability network and upstream nose adapter receive gradients.
    """

    def __init__(
        self,
        descriptor_dim: int = 512,
        *,
        hidden_dim: int = 32,
        maximum_residual_weight: float = 0.10,
        maximum_interaction_norm: float = 0.05,
        interaction_scale_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_residual_weight = float(maximum_residual_weight)
        self.maximum_interaction_norm = float(maximum_interaction_norm)
        self.interaction_scale_mode = str(interaction_scale_mode)
        if min(self.descriptor_dim, self.hidden_dim) <= 0:
            raise ValueError("Refiner dimensions must be positive")
        if not 0.0 < self.maximum_residual_weight <= 0.25:
            raise ValueError("maximum_residual_weight must be in (0,0.25]")
        if not 0.0 < self.maximum_interaction_norm <= 0.10:
            raise ValueError("maximum_interaction_norm must be in (0,0.10]")
        if self.interaction_scale_mode not in {"constant", "reliability"}:
            raise ValueError(
                "interaction_scale_mode must be 'constant' or 'reliability'"
            )
        self.signal_norm = nn.LayerNorm(5)
        self.reliability = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.interaction_norm = nn.LayerNorm(4 * self.descriptor_dim)
        self.interaction = nn.Sequential(
            nn.Linear(4 * self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.descriptor_dim),
        )
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)
        self.direction_gain_logit = nn.Parameter(torch.zeros(()))

    def configuration(self) -> dict[str, Any]:
        return {
            "descriptor_dim": self.descriptor_dim,
            "hidden_dim": self.hidden_dim,
            "maximum_residual_weight": self.maximum_residual_weight,
            "maximum_interaction_norm": self.maximum_interaction_norm,
            "interaction_scale_mode": self.interaction_scale_mode,
            "zero_initialized_exact_base_anchor": True,
        }

    def forward(
        self,
        base_embedding: torch.Tensor,
        face_descriptor: torch.Tensor,
        adapted_nose_descriptor: torch.Tensor,
        face_geometry_confidence: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        if base_embedding.ndim != 2 or base_embedding.shape[1] != self.descriptor_dim:
            raise ValueError("base_embedding has the wrong shape")
        if face_descriptor.shape != base_embedding.shape:
            raise ValueError("face_descriptor must match base_embedding")
        if adapted_nose_descriptor.shape != base_embedding.shape:
            raise ValueError("adapted_nose_descriptor must match base_embedding")
        if face_geometry_confidence.numel() != base_embedding.shape[0]:
            raise ValueError("One face confidence is required per descriptor")
        base = base_embedding
        unit_base = F.normalize(base_embedding, dim=1)
        face = F.normalize(face_descriptor, dim=1)
        nose = F.normalize(adapted_nose_descriptor, dim=1)
        confidence = face_geometry_confidence.reshape(-1).clamp(0.0, 1.0)
        signals = torch.stack(
            (
                confidence,
                (unit_base * nose).sum(dim=1),
                (face * nose).sum(dim=1),
                (unit_base - nose).abs().mean(dim=1),
                (face - nose).abs().mean(dim=1),
            ),
            dim=1,
        )
        reliability = torch.sigmoid(
            self.reliability(self.signal_norm(signals.float()))
        ).reshape(-1)
        global_gain = (
            self.maximum_residual_weight * self.direction_gain_logit.tanh()
        )
        residual_weight = reliability.to(base.dtype) * global_gain.to(base.dtype)
        relation = torch.cat(
            (unit_base, nose, unit_base * nose, (unit_base - nose).abs()), dim=1
        )
        interaction = torch.tanh(
            self.interaction(self.interaction_norm(relation.float()))
        ) / float(self.descriptor_dim) ** 0.5
        if self.interaction_scale_mode == "reliability":
            interaction_scale = (
                self.maximum_interaction_norm * reliability
            ).to(base.dtype)
        else:
            interaction_scale = torch.full_like(
                reliability,
                self.maximum_interaction_norm,
                dtype=base.dtype,
            )
        candidate = (
            base
            + residual_weight[:, None] * (nose - unit_base)
            + interaction_scale[:, None] * interaction.to(base.dtype)
        )
        candidate_norm = candidate.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
        # sign(abs(gain)) is exactly zero only at initialization and one after
        # the first non-zero update.  It keeps the zero-gain output bitwise
        # equal to the locked base while normalizing every trained candidate.
        activity = global_gain.abs() + interaction.abs().sum(dim=1)
        active = torch.sign(activity).to(candidate.dtype)[:, None]
        normalization = 1.0 + active * (
            candidate_norm.reciprocal().to(candidate.dtype) - 1.0
        )
        embedding = candidate * normalization
        if not return_aux:
            return embedding
        return {
            "embedding": embedding,
            "base_embedding": base,
            "refiner_reliability": reliability,
            "refiner_global_gain": global_gain,
            "refiner_residual_weight": residual_weight,
            "refiner_signals": signals,
            "refiner_interaction": interaction,
            "refiner_interaction_scale": interaction_scale,
        }


class UnifiedExternalJointPetReID(nn.Module):
    """One RGB graph: locked unified base plus gradient-controlled refiner."""

    descriptor_dim = 512

    def __init__(
        self,
        base_model: UnifiedSemanticPetReID,
        *,
        hidden_dim: int = 32,
        maximum_residual_weight: float = 0.10,
        maximum_interaction_norm: float = 0.05,
        interaction_scale_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.refiner = GradientControlledFusionRefiner(
            self.descriptor_dim,
            hidden_dim=hidden_dim,
            maximum_residual_weight=maximum_residual_weight,
            maximum_interaction_norm=maximum_interaction_norm,
            interaction_scale_mode=interaction_scale_mode,
        )

    @property
    def input_size(self) -> int:
        return self.base_model.input_size

    def configure_trainable(
        self, *, nose_adapter: bool = False, refiner: bool = False
    ) -> None:
        self.base_model.configure_trainable(
            geometry=False,
            nose_encoder_parts=(),
            nose_adapter=nose_adapter,
            fusion=False,
        )
        self.refiner.requires_grad_(bool(refiner))
        self.base_model.eval()

    def forward(self, rgb_0_255: torch.Tensor, *, return_aux: bool = False):
        base = self.base_model(rgb_0_255, return_aux=True)
        refined = self.refiner(
            base["embedding"],
            base["face_descriptor"],
            base["adapted_nose_descriptor"],
            base["geometry_confidence"][:, 0],
            return_aux=return_aux,
        )
        if not return_aux:
            return refined
        return {**base, **refined}


class UnifiedExternalJointPetReIDExport(nn.Module):
    def __init__(self, model: UnifiedExternalJointPetReID) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)


def build_external_joint_from_base_checkpoint(
    base_checkpoint: str | Path,
    *,
    hidden_dim: int = 32,
    maximum_residual_weight: float = 0.10,
    maximum_interaction_norm: float = 0.05,
    interaction_scale_mode: str = "constant",
    device: str | torch.device = "cpu",
) -> UnifiedExternalJointPetReID:
    base, _ = build_unified_semantic_from_checkpoint(
        base_checkpoint, device=device, verify_sources=True
    )
    model = UnifiedExternalJointPetReID(
        base,
        hidden_dim=hidden_dim,
        maximum_residual_weight=maximum_residual_weight,
        maximum_interaction_norm=maximum_interaction_norm,
        interaction_scale_mode=interaction_scale_mode,
    )
    model.configure_trainable(nose_adapter=False, refiner=False)
    return model.to(device).eval()


def create_external_joint_checkpoint(
    model: UnifiedExternalJointPetReID,
    *,
    base_checkpoint: str | Path,
    training: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_path = Path(base_checkpoint).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_config": {
            "input_size": model.input_size,
            "refiner": model.refiner.configuration(),
        },
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "sources": {
            "base_checkpoint": {
                "path": str(base_path),
                "sha256": sha256_file(base_path),
                "bytes": base_path.stat().st_size,
            }
        },
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
        "training": training,
        "selection": selection,
        "promotion_status": "development_validation_required",
        "default_backend_changed": False,
    }


def save_external_joint_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    atomic_torch_save(payload, path)


def build_external_joint_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
) -> tuple[UnifiedExternalJointPetReID, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported external joint checkpoint schema")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint is not UnifiedExternalJointPetReID")
    source = payload["sources"]["base_checkpoint"]
    base_path = resolve_legacy_path(source["path"])
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if verify_sources and sha256_file(base_path) != source["sha256"]:
        raise RuntimeError("Base unified checkpoint hash mismatch")
    config = payload["model_config"]["refiner"]
    model = build_external_joint_from_base_checkpoint(
        base_path,
        hidden_dim=int(config["hidden_dim"]),
        maximum_residual_weight=float(config["maximum_residual_weight"]),
        maximum_interaction_norm=float(
            config.get("maximum_interaction_norm", 0.05)
        ),
        interaction_scale_mode=str(
            config.get("interaction_scale_mode", "reliability")
        ),
        device=device,
    )
    strict = schema_version == SCHEMA_VERSION
    incompatible = model.load_state_dict(payload["model"], strict=strict)
    allowed_legacy_missing = {
        "refiner.interaction_norm.weight",
        "refiner.interaction_norm.bias",
        "refiner.interaction.0.weight",
        "refiner.interaction.0.bias",
        "refiner.interaction.2.weight",
        "refiner.interaction.2.bias",
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    if unexpected or (missing and (strict or not missing <= allowed_legacy_missing)):
        raise RuntimeError(f"External joint checkpoint mismatch: {incompatible}")
    model.configure_trainable(nose_adapter=False, refiner=False)
    return model.to(device).eval(), payload
