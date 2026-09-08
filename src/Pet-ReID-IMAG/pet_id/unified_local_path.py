"""End-to-end local-path candidate: source-aware localization and bounded ROIs.

This is not a global-identity fallback. Face/nose encoders, their fusion, and
the final refiner remain the released V3 modules. Only the graph-internal
localization view and sampling geometry change. Quality outputs are geometric
diagnostics, not calibrated face-presence probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import operators as onnx_operators

from .arcface import DogArcFaceEncoder
from .unified import UnifiedPetReID
from .unified_e2e import GraphInternalLetterbox
from .unified_external_model import UnifiedExternalJointPetReID
from .unified_semantic import UnifiedSemanticPetReID
from .unified_semantic_checkpoint import build_semantic_identity_model
from .unified_training import sha256_file
from .workspace_paths import resolve_legacy_path

MODEL_TYPE = "unified_content_bounded_local_path"

# Only the original pre-reliability schema is migrated. Partially specified or
# unknown configurations are errors, not an invitation to guess graph semantics.
LEGACY_RELIABILITY_DEFAULTS = {
    "reliability_mode": "legacy",
    "minimum_nose_valid_fraction": 0.50,
    "minimum_nose_face_cosine": 0.0,
    "maximum_nose_weight_override": None,
    "final_refiner_enabled": True,
}


def normalize_local_path_configuration(config, expected):
    if not isinstance(config, dict):
        raise ValueError("Candidate configuration must be a dictionary")
    keys, current = set(config), set(expected)
    if keys == current - set(LEGACY_RELIABILITY_DEFAULTS):
        return {**config, **LEGACY_RELIABILITY_DEFAULTS}
    if keys != current:
        raise ValueError("Unknown or incomplete candidate configuration fields")
    return dict(config)


def project_rotated_box(boxes, angles, image_width, image_height):
    """Project physical-pixel rotated boxes inside source pixel-center bounds.

    Shrink a rotated rectangle only when necessary, then move its center to the
    feasible interval. All four corners (not only unrotated corners) remain
    inside the image. This guarantees valid sampling, not semantic face quality.
    """

    center = boxes[:, :2]
    sizes = boxes[:, 2:].clamp_min(1.0)
    cosine, sine = angles.cos().abs(), angles.sin().abs()
    half_x = 0.5 * (cosine * sizes[:, 0] + sine * sizes[:, 1])
    half_y = 0.5 * (sine * sizes[:, 0] + cosine * sizes[:, 1])
    span_x, span_y = image_width - 1.0, image_height - 1.0
    factor = torch.minimum(
        torch.ones_like(half_x),
        torch.minimum(
            span_x / (2 * half_x).clamp_min(1e-8), span_y / (2 * half_y).clamp_min(1e-8)
        ),
    )
    sizes = sizes * factor[:, None]
    half_x, half_y = half_x * factor, half_y * factor
    x = torch.minimum(torch.maximum(center[:, 0], half_x), span_x - half_x)
    y = torch.minimum(torch.maximum(center[:, 1], half_y), span_y - half_y)
    fitted_center = torch.stack((x, y), dim=1)
    shift = (fitted_center - center).norm(dim=1)
    return torch.cat((fitted_center, sizes), dim=1), factor, shift


class ContentBoundedRawCropper(nn.Module):
    """Convert learned canonical-canvas coordinates to original source pixels."""

    def __init__(self, output_size: int, canonical_size: int = 1280):
        super().__init__()
        if output_size < 2 or canonical_size < 2:
            raise ValueError("crop and canonical sizes must be at least two")
        self.output_size = int(output_size)
        self.canonical_size = int(canonical_size)

    def forward(self, rgb, boxes, angles, *, return_aux=False):
        shape = onnx_operators.shape_as_tensor(rgb).to(
            device=rgb.device, dtype=rgb.dtype
        )
        height, width = shape[-2], shape[-1]
        side = torch.maximum(height, width)
        pad_x, pad_y = (
            torch.floor((side - width) * 0.5),
            torch.floor((side - height) * 0.5),
        )
        # GraphInternalLetterbox(allow_upscale=True) samples at half-pixel
        # centers. The existing canonical cropper defines its center at c*S.
        canonical = float(self.canonical_size)
        center = (boxes[:, :2] * canonical + 0.5) * (side / canonical)
        center = center - torch.stack((pad_x, pad_y))[None] - 0.5
        sizes = boxes[:, 2:].clamp(1.0 / canonical, 1.0) * side
        raw_boxes = torch.cat((center, sizes), dim=1)
        fitted, factor, shift = project_rotated_box(raw_boxes, angles, width, height)
        cx, cy, box_width, box_height = fitted.unbind(1)
        cosine, sine = angles.cos(), angles.sin()
        theta = torch.stack(
            (
                cosine * box_width / (width - 1.0),
                -sine * box_height / (width - 1.0),
                2.0 * cx / (width - 1.0) - 1.0,
                sine * box_width / (height - 1.0),
                cosine * box_height / (height - 1.0),
                2.0 * cy / (height - 1.0) - 1.0,
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        grid = F.affine_grid(
            theta,
            (rgb.shape[0], 3, self.output_size, self.output_size),
            align_corners=True,
        )
        crop = F.grid_sample(
            rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        if not return_aux:
            return crop
        return crop, {
            "source_boxes_cxcywh": fitted,
            "size_retention": factor,
            "center_shift_source_pixels": shift,
        }


class LocalPathUnifiedPetReID(nn.Module):
    """One raw RGB input -> learned geometry + local identity fusion -> 512D."""

    descriptor_dim = 512

    def __init__(
        self,
        model: UnifiedExternalJointPetReID,
        *,
        reliability_mode: str = "legacy",
        minimum_nose_valid_fraction: float = 0.50,
        minimum_nose_face_cosine: float = 0.0,
        maximum_nose_weight: float | None = None,
        final_refiner_enabled: bool = True,
    ):
        super().__init__()
        self.model = model
        semantic = model.base_model
        self.localization_view = GraphInternalLetterbox(
            semantic.input_size, allow_upscale=True
        )
        self.face_cropper = ContentBoundedRawCropper(
            semantic.geometry_frontend.crop_size, semantic.input_size
        )
        self.nose_cropper = ContentBoundedRawCropper(
            semantic.nose_crop_size, semantic.input_size
        )
        self.reliability_mode = str(reliability_mode)
        if self.reliability_mode not in {"legacy", "self_consistency", "off"}:
            raise ValueError(
                "reliability_mode must be legacy, self_consistency, or off"
            )
        self.minimum_nose_valid_fraction = float(minimum_nose_valid_fraction)
        self.minimum_nose_face_cosine = float(minimum_nose_face_cosine)
        self.maximum_nose_weight_override = (
            None if maximum_nose_weight is None else float(maximum_nose_weight)
        )
        self.final_refiner_enabled = bool(final_refiner_enabled)
        if not 0.0 <= self.minimum_nose_valid_fraction <= 1.0:
            raise ValueError("minimum_nose_valid_fraction must be in [0,1]")
        if not -1.0 <= self.minimum_nose_face_cosine <= 1.0:
            raise ValueError("minimum_nose_face_cosine must be in [-1,1]")
        if (
            self.maximum_nose_weight_override is not None
            and not 0.0 <= self.maximum_nose_weight_override <= 0.49
        ):
            raise ValueError("maximum_nose_weight must be in [0,0.49]")

    def configuration(self):
        return {
            "model_type": MODEL_TYPE,
            "descriptor_dim": 512,
            "input": "float32 raw RGB 0..255 [N,3,H,W]",
            "minimum_side": 2,
            "maximum_side": 4096,
            "localization_canvas": self.model.input_size,
            "localization_allow_upscale": True,
            "sampling": "direct_raw_pixels_rotated_roi_projected_inside_source",
            "global_identity_branch": False,
            "external_models": [],
            "fusion_and_refiner": "released_weights_unchanged",
            "face_presence_calibrated": False,
            "attention_concentration_is_probability": False,
            "reliability_mode": self.reliability_mode,
            "minimum_nose_valid_fraction": self.minimum_nose_valid_fraction,
            "minimum_nose_face_cosine": self.minimum_nose_face_cosine,
            "maximum_nose_weight_override": self.maximum_nose_weight_override,
            "final_refiner_enabled": self.final_refiner_enabled,
        }

    def forward(self, rgb, *, return_aux=False):
        if not torch.onnx.is_in_onnx_export():
            if (
                rgb.ndim != 4
                or rgb.shape[0] < 1
                or rgb.shape[1] != 3
                or min(rgb.shape[-2:]) < 2
            ):
                raise ValueError("rgb must have shape [N,3,H,W] with N>=1 and H/W>=2")
            if (
                not torch.is_floating_point(rgb)
                or not torch.isfinite(rgb).all()
                or rgb.min() < 0
                or rgb.max() > 255
            ):
                raise ValueError("rgb must contain finite floating-point 0..255 pixels")
        rgb = rgb.float()
        semantic = self.model.base_model
        frontend = semantic.geometry_frontend
        prediction, geometry_features = frontend._localize(self.localization_view(rgb))
        boxes, angles = semantic.geometry_discretizer(
            prediction.boxes_cxcywh, prediction.angle_radians
        )
        face_crops, face_diagnostics = [], []
        for scale in semantic.face_crop_scales:
            face_box = torch.cat((boxes[:, 0, :2], boxes[:, 0, 2:] * scale), dim=1)
            crop, diagnostic = self.face_cropper(rgb, face_box, angles, return_aux=True)
            face_crops.append(crop)
            face_diagnostics.append(diagnostic)
        packed = frontend._backbone_descriptor(
            frontend._normalize(torch.cat(face_crops, dim=0))
        )
        packed = packed.reshape(len(face_crops), -1, self.descriptor_dim)
        weights = semantic.face_crop_weights_tensor.to(packed).view(-1, 1, 1)
        face = F.normalize((packed * weights).sum(0), dim=1)
        nose_crop, nose_diagnostic = self.nose_cropper(
            rgb, boxes[:, 1], angles, return_aux=True
        )
        valid_mask = torch.ones_like(rgb)
        valid_nose = self.nose_cropper(valid_mask, boxes[:, 1], angles)
        nose_valid_fraction = valid_nose.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
        feather = semantic.nose_feather_mask.to(nose_crop)
        background = semantic.nose_encoder.model.pixel_mean.to(nose_crop)
        feathered_nose = nose_crop * feather + background * (1.0 - feather)
        raw_nose = F.normalize(
            semantic.nose_encoder(nose_crop) + semantic.nose_encoder(feathered_nose),
            dim=1,
        )
        nose = semantic.nose_adapter(raw_nose)
        fusion = self._fuse_local_descriptors(
            face, nose, prediction, nose_valid_fraction, geometry_features
        )
        if not return_aux:
            return fusion["embedding"]
        return {
            **fusion,
            "face_descriptor": face,
            "adapted_nose_descriptor": nose,
            "attention_concentration": prediction.confidence[:, 0],
            "nose_valid_fraction": nose_valid_fraction,
            "face_nose_cosine": F.cosine_similarity(face, nose, dim=1),
            "face_crop_diagnostics": face_diagnostics,
            "nose_crop_diagnostics": nose_diagnostic,
            "face_crops": torch.stack(face_crops, dim=1),
            "nose_crop": nose_crop,
        }

    def _fuse_local_descriptors(
        self, face, nose, prediction, nose_valid_fraction, geometry_features
    ):
        """Keep legacy arithmetic reproducible, including the non-exact `off`.

        New exact-zero visibility semantics live in a separate opt-in model.
        """
        semantic = self.model.base_model
        concentration = prediction.confidence[:, 0]
        face_nose_cosine = F.cosine_similarity(face, nose, dim=1)
        if self.reliability_mode == "legacy":
            nose_reliability = torch.ones_like(concentration)
        elif self.reliability_mode == "off":
            nose_reliability = torch.zeros_like(concentration)
        else:
            nose_reliability = (
                (nose_valid_fraction >= self.minimum_nose_valid_fraction)
                & (face_nose_cosine >= self.minimum_nose_face_cosine)
            ).to(dtype=concentration.dtype)
        # Keep the original differentiable fusion in the graph but suppress it
        # by passing an unambiguous high concentration when evidence is absent.
        gate_concentration = torch.where(
            nose_reliability > 0,
            concentration,
            torch.ones_like(concentration),
        )
        fused = semantic.fusion(face, nose, gate_concentration)
        if self.maximum_nose_weight_override is not None:
            original_weight = semantic.fusion.nose_weight(gate_concentration)
            bounded_weight = (
                torch.minimum(
                    original_weight,
                    original_weight.new_full(
                        original_weight.shape, self.maximum_nose_weight_override
                    ),
                )
                * nose_reliability
            )
            fused = F.normalize(
                F.normalize(face, dim=1) * (1.0 - bounded_weight[:, None])
                + F.normalize(nose, dim=1) * bounded_weight[:, None],
                dim=1,
            )
        if self.final_refiner_enabled:
            embedding = self.model.refiner(fused, face, nose, gate_concentration)
        else:
            embedding = fused
        return {
            "embedding": embedding,
            "base_embedding": fused,
            "gate_concentration": gate_concentration,
            "nose_reliability": nose_reliability,
        }


class LocalPathUnifiedPetReIDExport(nn.Module):
    """Expose exactly one tensor argument to the legacy ONNX tracer."""

    def __init__(self, model: LocalPathUnifiedPetReID):
        super().__init__()
        self.model = model

    def forward(self, rgb):
        return self.model(rgb)


def load_local_path_checkpoint(path: str | Path, *, device="cpu"):
    """Restore an explicitly saved research candidate without unsafe unpickling."""
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != 1 or payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Unsupported local-path checkpoint envelope")
    source = payload["source_v3_checkpoint"]
    source_path = resolve_legacy_path(source["path"])
    if sha256_file(source_path) != source["sha256"]:
        raise ValueError("Source V3 hash differs from candidate record")
    model, _ = build_local_path_candidate(source_path, device=device)
    config = normalize_local_path_configuration(
        payload["model_config"], model.configuration()
    )
    model = LocalPathUnifiedPetReID(
        model.model,
        reliability_mode=config["reliability_mode"],
        minimum_nose_valid_fraction=config["minimum_nose_valid_fraction"],
        minimum_nose_face_cosine=config["minimum_nose_face_cosine"],
        maximum_nose_weight=config["maximum_nose_weight_override"],
        final_refiner_enabled=config["final_refiner_enabled"],
    ).to(device)
    if config != model.configuration():
        raise ValueError("Candidate configuration differs from its graph definition")
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False), payload


def build_released_v3_safely(checkpoint_path: str | Path, *, device="cpu"):
    """Strict reconstruction of the full released model using safe source loads."""
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("model_type") != "unified_external_joint_pet_reid":
        raise ValueError("Expected the released external-joint V3 model")
    base_record = payload["sources"]["base_checkpoint"]
    base_path = resolve_legacy_path(base_record["path"])
    if sha256_file(base_path) != base_record["sha256"]:
        raise ValueError("V3 source checkpoint hash mismatch")
    base = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    sources = {}
    for name, record in base["sources"].items():
        source = resolve_legacy_path(record["path"])
        if sha256_file(source) != record["sha256"]:
            raise ValueError(f"V3 source hash mismatch: {name}")
        sources[name] = source
    config = base["model_config"]
    geometry = UnifiedPetReID(
        DogArcFaceEncoder(sources["arcface_checkpoint"], freeze=True),
        **config["geometry"],
    )
    identity = build_semantic_identity_model(
        sources["semantic_config"], sources["semantic_checkpoint"], device="cpu"
    )
    discretization = config["geometry_discretization"]
    semantic = UnifiedSemanticPetReID.from_semantic_residual(
        geometry,
        identity,
        **config["fusion"],
        face_crop_scales=config["face_crop_scales"],
        face_crop_weights=config["face_crop_weights"],
        geometry_box_step=discretization["box_step"],
        geometry_box_offsets=discretization["box_offsets"],
        geometry_angle_step=discretization["angle_step"],
        geometry_angle_offset=discretization["angle_offset"],
        geometry_box_piecewise=discretization["box_piecewise"],
    )
    refiner_config = {
        k: v
        for k, v in payload["model_config"]["refiner"].items()
        if k
        in {
            "hidden_dim",
            "maximum_residual_weight",
            "maximum_interaction_norm",
            "interaction_scale_mode",
        }
    }
    model = UnifiedExternalJointPetReID(semantic, **refiner_config)
    model.load_state_dict(payload["model"], strict=True)
    model.configure_trainable(nose_adapter=False, refiner=False)
    return model.eval().requires_grad_(False).to(device), payload


def build_local_path_candidate(
    checkpoint_path: str | Path, *, device="cpu"
) -> tuple[LocalPathUnifiedPetReID, dict[str, Any]]:
    model, payload = build_released_v3_safely(checkpoint_path, device=device)
    return LocalPathUnifiedPetReID(model).eval().to(device), payload
