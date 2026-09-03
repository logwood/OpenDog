"""High-resolution, one-graph extension of the locked production parent.

The production parent intentionally receives a fixed 1280 x 1280 letterbox.
That contract remains untouched. This spatial-detail candidate accepts one raw
RGB tensor with dynamic spatial dimensions. A graph-internal sampler creates
the parent's exact 1280 coordinate system, while the same predicted face and
nose boxes sample larger crops directly from the source pixels before
irreversible global downsampling.

The added fusion is bounded and zero initialized. At initialization its output
is exactly the protected parent embedding. After training it is still forced
to return the parent embedding when the source maximum side is at most 1280,
so low-resolution behaviour cannot drift through the detail branch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import operators as onnx_operators

from .unified_external_model import (
    UnifiedExternalJointPetReID,
    build_external_joint_from_checkpoint,
    sha256_file,
)
from .unified_training import atomic_torch_save
from .release_compatibility import parent_checkpoint_source


MODEL_TYPE = "unified_high_resolution_pet_reid"
SCHEMA_VERSION = 1


def _validate_rgb(rgb_0_255: torch.Tensor) -> torch.Tensor:
    if rgb_0_255.ndim != 4 or rgb_0_255.shape[1] != 3:
        raise ValueError("rgb_0_255 must have shape [batch, 3, height, width]")
    if rgb_0_255.shape[-2] < 2 or rgb_0_255.shape[-1] < 2:
        raise ValueError("RGB height and width must both be at least two")
    return rgb_0_255.float()


def _source_geometry(
    rgb: torch.Tensor,
    *,
    minimum_side: int,
) -> tuple[torch.Tensor, ...]:
    """Return tensor-valued source-to-centered-square geometry.

    ``torch.onnx.operators.shape_as_tensor`` is deliberate.  The current
    dynamo exporter specializes SymInt-to-float arithmetic for GridSample,
    whereas the legacy exporter preserves this operation as ONNX Shape.  The
    spatial-detail exporter therefore uses the legacy exporter path and
    validates several H/W pairs with ONNX Runtime before publishing an artifact.
    """

    shape = onnx_operators.shape_as_tensor(rgb).to(
        device=rgb.device,
        dtype=rgb.dtype,
    )
    height = shape[-2]
    width = shape[-1]
    side = torch.maximum(
        torch.maximum(height, width),
        torch.full_like(height, float(minimum_side)),
    )
    pad_left = torch.floor((side - width) * 0.5)
    pad_top = torch.floor((side - height) * 0.5)
    x_scale = side / width
    y_scale = side / height
    # Account for the one-pixel asymmetry when (side - extent) is odd.  With
    # symmetric padding these translations are exactly zero.
    x_translation = (side - 2.0 * pad_left) / width - 1.0
    y_translation = (side - 2.0 * pad_top) / height - 1.0
    return (
        height,
        width,
        side,
        x_scale,
        y_scale,
        x_translation,
        y_translation,
    )


class ShapeDrivenGlobalLetterbox(nn.Module):
    """Sample a centered black square directly into a fixed global view."""

    def __init__(self, output_size: int = 1280) -> None:
        super().__init__()
        self.output_size = int(output_size)
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

    def forward(
        self,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = _validate_rgb(rgb)
        (
            _,
            _,
            side,
            x_scale,
            y_scale,
            x_translation,
            y_translation,
        ) = _source_geometry(rgb, minimum_side=self.output_size)
        zero = torch.zeros_like(side)
        theta = torch.stack(
            (
                x_scale,
                zero,
                x_translation,
                zero,
                y_scale,
                y_translation,
            )
        ).reshape(1, 2, 3)
        theta = theta.expand(rgb.shape[0], -1, -1)
        grid = F.affine_grid(
            theta,
            (rgb.shape[0], rgb.shape[1], self.output_size, self.output_size),
            align_corners=False,
        )
        global_rgb = F.grid_sample(
            rgb,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        detail_scale = side / float(self.output_size)
        detail_availability = (detail_scale - 1.0).clamp(0.0, 1.0)
        return global_rgb, detail_scale, detail_availability


class ShapeDrivenRotatedCropper(nn.Module):
    """Crop a normalized square-coordinate ROI from dynamic raw H/W pixels."""

    def __init__(
        self,
        output_size: Sequence[int],
        *,
        minimum_side: int = 1280,
    ) -> None:
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        self.minimum_side = int(minimum_side)
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError("output_size must contain two positive integers")
        if self.minimum_side <= 0:
            raise ValueError("minimum_side must be positive")

    def forward(
        self,
        rgb: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        angles_radians: torch.Tensor,
    ) -> torch.Tensor:
        rgb = _validate_rgb(rgb)
        if boxes_cxcywh.shape != (rgb.shape[0], 4):
            raise ValueError("boxes_cxcywh must have shape [batch, 4]")
        if angles_radians.shape != (rgb.shape[0],):
            raise ValueError("angles_radians must have shape [batch]")
        (
            _,
            _,
            side,
            x_scale,
            y_scale,
            x_translation,
            y_translation,
        ) = _source_geometry(rgb, minimum_side=self.minimum_side)
        center_x, center_y, width, height = boxes_cxcywh.unbind(dim=1)
        minimum_size = side.reciprocal()
        width = torch.minimum(
            torch.maximum(width, minimum_size),
            torch.ones_like(width),
        )
        height = torch.minimum(
            torch.maximum(height, minimum_size),
            torch.ones_like(height),
        )
        cosine = torch.cos(angles_radians)
        sine = torch.sin(angles_radians)
        theta = torch.stack(
            (
                cosine * width * x_scale,
                -sine * height * x_scale,
                (2.0 * center_x - 1.0) * x_scale + x_translation,
                sine * width * y_scale,
                cosine * height * y_scale,
                (2.0 * center_y - 1.0) * y_scale + y_translation,
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        grid = F.affine_grid(
            theta,
            (
                rgb.shape[0],
                rgb.shape[1],
                self.output_size[0],
                self.output_size[1],
            ),
            align_corners=False,
        )
        return F.grid_sample(
            rgb,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


def _detail_energy(crop_0_255: torch.Tensor) -> torch.Tensor:
    """Return a bounded, differentiable local high-frequency signal."""

    gray = (
        0.2989 * crop_0_255[:, 0]
        + 0.5870 * crop_0_255[:, 1]
        + 0.1140 * crop_0_255[:, 2]
    ) / 255.0
    horizontal = (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean(dim=(1, 2))
    vertical = (gray[:, 1:, :] - gray[:, :-1, :]).abs().mean(dim=(1, 2))
    return (0.5 * (horizontal + vertical)).clamp(0.0, 1.0)


class HighResolutionDetailRefiner(nn.Module):
    """Bounded face/nose detail residual with an exact parent anchor."""

    signal_dim = 10

    def __init__(
        self,
        descriptor_dim: int = 512,
        *,
        hidden_dim: int = 64,
        maximum_detail_weight: float = 0.08,
        maximum_interaction_norm: float = 0.03,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_detail_weight = float(maximum_detail_weight)
        self.maximum_interaction_norm = float(maximum_interaction_norm)
        if min(self.descriptor_dim, self.hidden_dim) <= 0:
            raise ValueError("refiner dimensions must be positive")
        if not 0.0 < self.maximum_detail_weight <= 0.20:
            raise ValueError("maximum_detail_weight must be in (0, 0.20]")
        if not 0.0 < self.maximum_interaction_norm <= 0.10:
            raise ValueError("maximum_interaction_norm must be in (0, 0.10]")

        self.signal_norm = nn.LayerNorm(self.signal_dim)
        self.reliability = nn.Sequential(
            nn.Linear(self.signal_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

        relation_dim = 6 * self.descriptor_dim
        self.interaction_norm = nn.LayerNorm(relation_dim)
        self.interaction = nn.Sequential(
            nn.Linear(relation_dim, self.hidden_dim),
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
            "maximum_detail_weight": self.maximum_detail_weight,
            "maximum_interaction_norm": self.maximum_interaction_norm,
            "zero_initialized_exact_parent_anchor": True,
            "low_resolution_exact_parent_anchor": True,
        }

    def forward(
        self,
        base_embedding: torch.Tensor,
        detail_face_descriptor: torch.Tensor,
        detail_nose_descriptor: torch.Tensor,
        face_geometry_confidence: torch.Tensor,
        detail_scale: torch.Tensor,
        detail_availability: torch.Tensor,
        face_detail_energy: torch.Tensor,
        nose_detail_energy: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        expected = (base_embedding.shape[0], self.descriptor_dim)
        if tuple(base_embedding.shape) != expected:
            raise ValueError("base_embedding has the wrong shape")
        if tuple(detail_face_descriptor.shape) != expected:
            raise ValueError("detail_face_descriptor must match base_embedding")
        if tuple(detail_nose_descriptor.shape) != expected:
            raise ValueError("detail_nose_descriptor must match base_embedding")

        base = base_embedding
        unit_base = F.normalize(base_embedding, dim=1)
        face = F.normalize(detail_face_descriptor, dim=1)
        nose = F.normalize(detail_nose_descriptor, dim=1)
        batch = base.shape[0]
        confidence = face_geometry_confidence.reshape(batch).clamp(0.0, 1.0)
        availability = detail_availability.reshape(-1).expand(batch).clamp(0.0, 1.0)
        scale_signal = (
            torch.log2(detail_scale.reshape(-1).expand(batch).clamp_min(1.0)) / 2.0
        ).clamp(0.0, 1.0)
        face_cosine = (unit_base * face).sum(dim=1)
        nose_cosine = (unit_base * nose).sum(dim=1)
        cross_cosine = (face * nose).sum(dim=1)
        face_difference = (unit_base - face).abs().mean(dim=1)
        nose_difference = (unit_base - nose).abs().mean(dim=1)
        signals = torch.stack(
            (
                availability,
                scale_signal,
                confidence,
                face_cosine,
                nose_cosine,
                cross_cosine,
                face_difference,
                nose_difference,
                face_detail_energy.reshape(batch).clamp(0.0, 1.0),
                nose_detail_energy.reshape(batch).clamp(0.0, 1.0),
            ),
            dim=1,
        )
        detail_weights = self.reliability(
            self.signal_norm(signals.float())
        ).softmax(dim=1).to(base.dtype)
        direction = (
            detail_weights[:, 0:1] * (face - unit_base)
            + detail_weights[:, 1:2] * (nose - unit_base)
        )
        global_gain = self.maximum_detail_weight * self.direction_gain_logit.tanh()

        relation = torch.cat(
            (
                unit_base,
                face,
                nose,
                (unit_base - face).abs(),
                (unit_base - nose).abs(),
                face * nose,
            ),
            dim=1,
        )
        interaction = torch.tanh(
            self.interaction(self.interaction_norm(relation.float()))
        ) / math.sqrt(self.descriptor_dim)
        active_scale = availability.to(base.dtype)[:, None]
        candidate = base + active_scale * (
            global_gain.to(base.dtype) * direction
            + self.maximum_interaction_norm * interaction.to(base.dtype)
        )
        candidate_norm = candidate.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
        activity = availability * (
            global_gain.abs().to(availability.dtype)
            + interaction.abs().sum(dim=1).to(availability.dtype)
        )
        active = torch.sign(activity).to(candidate.dtype)[:, None]
        normalization = 1.0 + active * (
            candidate_norm.reciprocal().to(candidate.dtype) - 1.0
        )
        embedding = candidate * normalization
        if not return_aux:
            return embedding
        return {
            "embedding": embedding,
            "highres_parent_embedding": base,
            "detail_face_descriptor": face,
            "detail_nose_descriptor": nose,
            "detail_weights": detail_weights,
            "detail_signals": signals,
            "detail_scale": detail_scale.reshape(-1).expand(batch),
            "detail_availability": availability,
            "detail_global_gain": global_gain,
            "detail_interaction": interaction,
        }


class UnifiedHighResolutionPetReID(nn.Module):
    """One raw RGB graph: protected global anchor plus source-resolution details."""

    descriptor_dim = 512

    def __init__(
        self,
        parent_model: UnifiedExternalJointPetReID,
        *,
        face_detail_size: int = 384,
        nose_detail_size: int = 320,
        refiner_hidden_dim: int = 64,
        maximum_detail_weight: float = 0.08,
        maximum_interaction_norm: float = 0.03,
        maximum_input_side: int = 4096,
    ) -> None:
        super().__init__()
        self.parent_model = parent_model
        self.global_input_size = int(parent_model.input_size)
        self.face_detail_size = int(face_detail_size)
        self.nose_detail_size = int(nose_detail_size)
        self.maximum_input_side = int(maximum_input_side)
        if self.global_input_size != 1280:
            raise ValueError(
                "The spatial-detail model requires the locked 1280 parent"
            )
        if min(
            self.face_detail_size,
            self.nose_detail_size,
            self.maximum_input_side,
        ) <= 0:
            raise ValueError("detail and input sizes must be positive")
        if self.maximum_input_side < self.global_input_size:
            raise ValueError("maximum_input_side must be at least 1280")

        self.global_sampler = ShapeDrivenGlobalLetterbox(self.global_input_size)
        self.face_detail_cropper = ShapeDrivenRotatedCropper(
            (self.face_detail_size, self.face_detail_size),
            minimum_side=self.global_input_size,
        )
        self.nose_detail_cropper = ShapeDrivenRotatedCropper(
            (self.nose_detail_size, self.nose_detail_size),
            minimum_side=self.global_input_size,
        )
        self.refiner = HighResolutionDetailRefiner(
            self.descriptor_dim,
            hidden_dim=refiner_hidden_dim,
            maximum_detail_weight=maximum_detail_weight,
            maximum_interaction_norm=maximum_interaction_norm,
        )
        full_mask = torch.ones(
            1,
            1,
            self.nose_detail_size,
            self.nose_detail_size,
        )
        self.register_buffer(
            "nose_detail_feather_mask",
            F.avg_pool2d(full_mask, kernel_size=5, stride=1, padding=2),
            persistent=False,
        )
        self.configure_trainable(refiner=False)

    @property
    def input_size(self) -> int:
        """Return the fixed global canvas used by fixed-size data paths.

        The deployment graph also accepts dynamic raw H/W, but exposing the
        parent canvas keeps generic descriptor wrappers (including the
        reference-aware trainer) from guessing which size to materialize.
        """

        return self.global_input_size

    def configuration(self) -> dict[str, Any]:
        return {
            "global_input_size": self.global_input_size,
            "face_detail_size": self.face_detail_size,
            "nose_detail_size": self.nose_detail_size,
            "maximum_input_side": self.maximum_input_side,
            "refiner": self.refiner.configuration(),
        }

    def configure_trainable(self, *, refiner: bool = False) -> None:
        self.parent_model.requires_grad_(False)
        self.parent_model.configure_trainable(nose_adapter=False, refiner=False)
        self.parent_model.eval()
        self.refiner.requires_grad_(bool(refiner))

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm statistics in both inherited identity encoders remain the
        # locked parent values. Only the explicitly enabled detail refiner trains.
        self.parent_model.eval()
        return self

    def forward(self, rgb_0_255: torch.Tensor, *, return_aux: bool = False):
        rgb_0_255 = _validate_rgb(rgb_0_255)
        global_rgb, detail_scale, detail_availability = self.global_sampler(
            rgb_0_255
        )
        parent = self.parent_model(global_rgb, return_aux=True)
        boxes = parent["boxes_cxcywh"]
        angles = parent["angle_radians"]
        face_crop = self.face_detail_cropper(
            rgb_0_255,
            boxes[:, 0],
            angles,
        )
        nose_crop = self.nose_detail_cropper(
            rgb_0_255,
            boxes[:, 1],
            angles,
        )

        semantic = self.parent_model.base_model
        geometry_frontend = semantic.geometry_frontend
        detail_face_descriptor = geometry_frontend._backbone_descriptor(
            geometry_frontend._normalize(face_crop)
        )
        feather = self.nose_detail_feather_mask.to(dtype=nose_crop.dtype)
        background = semantic.nose_encoder.model.pixel_mean.to(dtype=nose_crop.dtype)
        feathered_nose = nose_crop * feather + background * (1.0 - feather)
        raw_detail_nose = F.normalize(
            semantic.nose_encoder(nose_crop)
            + semantic.nose_encoder(feathered_nose),
            dim=1,
        )
        detail_nose_descriptor = semantic.nose_adapter(raw_detail_nose)
        refined = self.refiner(
            parent["embedding"],
            detail_face_descriptor,
            detail_nose_descriptor,
            parent["geometry_confidence"][:, 0],
            detail_scale,
            detail_availability,
            _detail_energy(face_crop),
            _detail_energy(nose_crop),
            return_aux=return_aux,
        )
        if not return_aux:
            return refined
        return {
            **parent,
            **refined,
            "global_rgb": global_rgb,
            "detail_face_crop": face_crop,
            "detail_nose_crop": nose_crop,
            "raw_detail_nose_descriptor": raw_detail_nose,
        }


class UnifiedHighResolutionPetReIDExport(nn.Module):
    """Strict one-input/one-output spatial-detail deployment boundary."""

    def __init__(self, model: UnifiedHighResolutionPetReID) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)


def build_highres_from_parent_checkpoint(
    parent_checkpoint: str | Path,
    *,
    face_detail_size: int = 384,
    nose_detail_size: int = 320,
    refiner_hidden_dim: int = 64,
    maximum_detail_weight: float = 0.08,
    maximum_interaction_norm: float = 0.03,
    maximum_input_side: int = 4096,
    device: str | torch.device = "cpu",
) -> UnifiedHighResolutionPetReID:
    parent, _ = build_external_joint_from_checkpoint(
        parent_checkpoint,
        device=device,
        verify_sources=True,
    )
    model = UnifiedHighResolutionPetReID(
        parent,
        face_detail_size=face_detail_size,
        nose_detail_size=nose_detail_size,
        refiner_hidden_dim=refiner_hidden_dim,
        maximum_detail_weight=maximum_detail_weight,
        maximum_interaction_norm=maximum_interaction_norm,
        maximum_input_side=maximum_input_side,
    )
    model.configure_trainable(refiner=False)
    return model.to(device).eval()


def create_highres_checkpoint(
    model: UnifiedHighResolutionPetReID,
    *,
    parent_checkpoint: str | Path,
    training: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).expanduser().resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_config": model.configuration(),
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "sources": {
            "parent_checkpoint": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
                "bytes": parent_path.stat().st_size,
            }
        },
        "preprocessing": {
            "input_range": [0, 255],
            "color_order": "RGB",
            "external_letterbox": False,
            "raw_spatial_input": True,
            "recommended_maximum_side": model.maximum_input_side,
            "oversize_policy": "resize_long_side_before_onnx",
            "internal_global_view": "centered_black_square_to_1280",
        },
        "runtime_contract": {
            "inputs": {
                "rgb": {
                    "dtype": "float32",
                    "shape": ["N", 3, "H", "W"],
                    "height_width_minimum": 64,
                    "height_width_maximum": model.maximum_input_side,
                    "same_shape_within_batch": True,
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


def save_highres_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    atomic_torch_save(payload, path)


def build_highres_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
) -> tuple[UnifiedHighResolutionPetReID, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported high-resolution checkpoint schema")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint is not UnifiedHighResolutionPetReID")
    source = parent_checkpoint_source(payload["sources"])
    parent_path = Path(source["path"]).expanduser().resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    if verify_sources and sha256_file(parent_path) != source["sha256"]:
        raise RuntimeError("Parent checkpoint hash mismatch")
    config = payload["model_config"]
    refiner = config["refiner"]
    model = build_highres_from_parent_checkpoint(
        parent_path,
        face_detail_size=int(config["face_detail_size"]),
        nose_detail_size=int(config["nose_detail_size"]),
        refiner_hidden_dim=int(refiner["hidden_dim"]),
        maximum_detail_weight=float(refiner["maximum_detail_weight"]),
        maximum_interaction_norm=float(refiner["maximum_interaction_norm"]),
        maximum_input_side=int(config["maximum_input_side"]),
        device=device,
    )
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"High-resolution checkpoint mismatch: {incompatible}")
    model.configure_trainable(refiner=False)
    return model.to(device).eval(), payload
