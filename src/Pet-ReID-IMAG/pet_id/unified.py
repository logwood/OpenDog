# encoding: utf-8
"""Single-graph RGB-to-descriptor model for pet re-identification.

The deployment contract intentionally has one tensor input and one 512-D
output. Geometry supervision is training-only: a soft spatial query predicts
face/nose locations, a differentiable crop samples the face, and the same
identity backbone is reused for localization features and the final face
descriptor. A bounded, zero-initialized residual lets semantic context help
without destroying the pretrained face identity space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .arcface import DogArcFaceEncoder


PART_NAMES = ("face", "nose")
GEOMETRY_FEATURE_MODES = ("layer3", "fpn")


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized corner boxes to center/size form."""

    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1),
        dim=-1,
    )


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized center/size boxes to corner form."""

    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ),
        dim=-1,
    )


class NormalizedRotatedCropper(nn.Module):
    """ONNX-friendly differentiable crop from normalized center/size boxes."""

    def __init__(self, output_size: Sequence[int] = (224, 224)):
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError("output_size must contain two positive integers")

    def forward(
        self,
        images: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        angles_radians: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if boxes_cxcywh.shape != (images.shape[0], 4):
            raise ValueError("boxes_cxcywh must have shape [batch, 4]")
        if angles_radians.shape != (images.shape[0],):
            raise ValueError("angles_radians must have shape [batch]")

        center_x, center_y, width, height = boxes_cxcywh.unbind(dim=1)
        image_height = images.shape[-2]
        image_width = images.shape[-1]
        width = width.clamp(1.0 / image_width, 1.0)
        height = height.clamp(1.0 / image_height, 1.0)
        x_per_width = float(image_width) / max(float(image_width - 1), 1.0)
        y_per_height = float(image_height) / max(float(image_height - 1), 1.0)
        y_per_width = float(image_height) / max(float(image_width - 1), 1.0)
        x_per_height = float(image_width) / max(float(image_height - 1), 1.0)
        cosine = torch.cos(angles_radians)
        sine = torch.sin(angles_radians)
        theta = torch.stack(
            (
                cosine * width * x_per_width,
                -sine * height * y_per_width,
                center_x * (2.0 * x_per_width) - 1.0,
                sine * width * x_per_height,
                cosine * height * y_per_height,
                center_y * (2.0 * y_per_height) - 1.0,
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        grid = F.affine_grid(
            theta,
            (
                images.shape[0],
                images.shape[1],
                self.output_size[0],
                self.output_size[1],
            ),
            align_corners=True,
        )
        return F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )


@dataclass(frozen=True)
class GeometryPrediction:
    boxes_cxcywh: torch.Tensor
    angle_radians: torch.Tensor
    attention: torch.Tensor
    pooled_queries: torch.Tensor
    confidence: torch.Tensor


class PartGeometryCalibration(nn.Module):
    """Bounded graph-internal calibration for predicted part boxes."""

    maximum_center_offset = 0.50
    minimum_size_scale = 0.50
    maximum_size_scale = 2.00

    def __init__(self):
        super().__init__()
        self.center_offset_logits = nn.Parameter(
            torch.zeros(len(PART_NAMES), 2)
        )
        self.log_size_scales = nn.Parameter(
            torch.zeros(len(PART_NAMES), 2)
        )

    def forward(self, prediction: GeometryPrediction) -> GeometryPrediction:
        boxes = prediction.boxes_cxcywh
        relative_offsets = self.maximum_center_offset * torch.tanh(
            self.center_offset_logits
        )
        log_minimum = math.log(self.minimum_size_scale)
        log_maximum = math.log(self.maximum_size_scale)
        size_scales = self.log_size_scales.clamp(
            log_minimum, log_maximum
        ).exp()
        centers = boxes[..., :2] + (
            relative_offsets.unsqueeze(0) * boxes[..., 2:]
        )
        sizes = boxes[..., 2:] * size_scales.unsqueeze(0)
        calibrated = torch.cat(
            (centers.clamp(0.0, 1.0), sizes.clamp(1e-4, 1.0)), dim=-1
        )
        return GeometryPrediction(
            boxes_cxcywh=calibrated,
            angle_radians=prediction.angle_radians,
            attention=prediction.attention,
            pooled_queries=prediction.pooled_queries,
            confidence=prediction.confidence,
        )

    def set_part(
        self,
        part: str,
        *,
        center_offset: Sequence[float] = (0.0, 0.0),
        size_scale: Sequence[float] = (1.0, 1.0),
    ) -> None:
        if part not in PART_NAMES:
            raise ValueError(f"Unknown semantic part: {part!r}")
        if len(center_offset) != 2 or len(size_scale) != 2:
            raise ValueError(
                "center_offset and size_scale must each contain 2 values"
            )
        offsets = torch.as_tensor(center_offset, dtype=torch.float32)
        scales = torch.as_tensor(size_scale, dtype=torch.float32)
        if not torch.isfinite(offsets).all() or not torch.isfinite(scales).all():
            raise ValueError("Calibration values must be finite")
        if (offsets.abs() >= self.maximum_center_offset).any():
            raise ValueError("Center calibration exceeds its bounded interval")
        if (scales < self.minimum_size_scale).any() or (
            scales > self.maximum_size_scale
        ).any():
            raise ValueError("Size calibration exceeds its bounded interval")
        offset_logits = torch.atanh(offsets / self.maximum_center_offset)
        with torch.no_grad():
            index = PART_NAMES.index(part)
            self.center_offset_logits[index].copy_(offset_logits)
            self.log_size_scales[index].copy_(scales.log())


class Layer2Layer3GeometryAdapter(nn.Module):
    """Fuse stride-8 detail with stride-16 semantics for precise part geometry."""

    def __init__(self, output_channels: int = 128):
        super().__init__()
        output_channels = int(output_channels)
        if output_channels <= 0 or output_channels % 16:
            raise ValueError("output_channels must be a positive multiple of 16")
        self.layer2_projection = nn.Sequential(
            nn.Conv2d(512, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, output_channels),
            nn.GELU(),
        )
        self.layer3_projection = nn.Sequential(
            nn.Conv2d(1024, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, output_channels),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                2 * output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, output_channels),
            nn.GELU(),
        )

    def forward(
        self, layer2: torch.Tensor, layer3: torch.Tensor
    ) -> torch.Tensor:
        high_resolution = self.layer2_projection(layer2)
        semantic = F.interpolate(
            self.layer3_projection(layer3),
            size=high_resolution.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fusion(torch.cat((high_resolution, semantic), dim=1))


class SoftPartGeometryHead(nn.Module):
    """Predict face/nose geometry with differentiable spatial expectations."""

    def __init__(
        self,
        input_channels: int,
        feature_height: int,
        feature_width: int,
        *,
        hidden_channels: int = 128,
        center_priors: Sequence[Sequence[float]] = ((0.50, 0.45), (0.50, 0.52)),
        size_priors: Sequence[Sequence[float]] = ((0.36, 0.48), (0.15, 0.16)),
        minimum_size: float = 0.04,
        maximum_size: float = 0.98,
        minimum_sizes: Sequence[float] | None = None,
        maximum_sizes: Sequence[float] | None = None,
        prior_sigma: float = 0.22,
        maximum_angle_radians: float = math.pi * 0.60,
    ):
        super().__init__()
        self.feature_height = int(feature_height)
        self.feature_width = int(feature_width)
        if minimum_sizes is None:
            minimum_sizes = (float(minimum_size),) * len(PART_NAMES)
        if maximum_sizes is None:
            maximum_sizes = (float(maximum_size),) * len(PART_NAMES)
        if len(minimum_sizes) != len(PART_NAMES) or len(maximum_sizes) != len(PART_NAMES):
            raise ValueError("one minimum and maximum size is required per semantic part")
        self.minimum_sizes = tuple(float(value) for value in minimum_sizes)
        self.maximum_sizes = tuple(float(value) for value in maximum_sizes)
        if any(
            not math.isfinite(value) or value <= 0.0 or value >= 1.0
            for value in self.minimum_sizes
        ):
            raise ValueError("minimum sizes must be finite values in (0, 1)")
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in self.maximum_sizes
        ):
            raise ValueError("maximum sizes must be finite values in (0, 1]")
        if any(
            minimum >= maximum
            for minimum, maximum in zip(self.minimum_sizes, self.maximum_sizes)
        ):
            raise ValueError("each minimum size must be below its maximum size")
        # Legacy scalar attributes remain available for callers and diagnostics.
        self.minimum_size = float(min(self.minimum_sizes))
        self.maximum_size = float(max(self.maximum_sizes))
        self.register_buffer(
            "minimum_sizes_tensor",
            torch.tensor(self.minimum_sizes, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "maximum_sizes_tensor",
            torch.tensor(self.maximum_sizes, dtype=torch.float32),
            persistent=False,
        )
        self.maximum_angle_radians = float(maximum_angle_radians)
        if self.feature_height <= 0 or self.feature_width <= 0:
            raise ValueError("feature map dimensions must be positive")
        if len(center_priors) != len(PART_NAMES) or len(size_priors) != len(PART_NAMES):
            raise ValueError("one center and size prior is required per semantic part")

        self.reduction = nn.Sequential(
            nn.Conv2d(int(input_channels), int(hidden_channels), kernel_size=1, bias=False),
            nn.GroupNorm(16, int(hidden_channels)),
            nn.GELU(),
        )
        self.query_logits = nn.Conv2d(
            int(hidden_channels), len(PART_NAMES), kernel_size=1, bias=True
        )
        nn.init.zeros_(self.query_logits.weight)
        nn.init.zeros_(self.query_logits.bias)

        y = torch.linspace(0.0, 1.0, self.feature_height)
        x = torch.linspace(0.0, 1.0, self.feature_width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        prior_logits = []
        sigma2 = max(float(prior_sigma), 1e-4) ** 2
        for center_x, center_y in center_priors:
            prior_logits.append(
                -(
                    (xx - float(center_x)).square()
                    + (yy - float(center_y)).square()
                )
                / (2.0 * sigma2)
            )
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer(
            "prior_logits", torch.stack(prior_logits), persistent=True
        )

        self.size_heads = nn.ModuleList(
            nn.Linear(int(hidden_channels), 2) for _ in PART_NAMES
        )
        for head, prior, minimum, maximum in zip(
            self.size_heads,
            size_priors,
            self.minimum_sizes,
            self.maximum_sizes,
        ):
            nn.init.zeros_(head.weight)
            normalized = [
                (float(value) - minimum) / (maximum - minimum)
                for value in prior
            ]
            if any(value <= 0.0 or value >= 1.0 for value in normalized):
                raise ValueError("size priors must lie strictly inside each size bound")
            with torch.no_grad():
                head.bias.copy_(
                    torch.tensor([_logit(value) for value in normalized])
                )
        self.angle_head = nn.Linear(int(hidden_channels), 1)
        nn.init.zeros_(self.angle_head.weight)
        nn.init.zeros_(self.angle_head.bias)

    @property
    def hidden_channels(self) -> int:
        return int(self.reduction[0].out_channels)

    def forward(self, spatial_features: torch.Tensor) -> GeometryPrediction:
        if spatial_features.ndim != 4:
            raise ValueError("spatial_features must have shape [batch, channels, h, w]")
        if tuple(spatial_features.shape[-2:]) != (
            self.feature_height,
            self.feature_width,
        ):
            raise ValueError(
                "Unexpected geometry feature map size: "
                f"{tuple(spatial_features.shape[-2:])}"
            )
        reduced = self.reduction(spatial_features)
        logits = self.query_logits(reduced) + self.prior_logits.unsqueeze(0)
        attention = logits.flatten(2).softmax(dim=-1)
        centers = attention @ self.coordinates.to(dtype=attention.dtype)
        flattened = reduced.flatten(2).transpose(1, 2)
        pooled = attention @ flattened

        sizes = []
        for index, head in enumerate(self.size_heads):
            raw = head(pooled[:, index])
            minimum = self.minimum_sizes_tensor[index].to(
                dtype=raw.dtype, device=raw.device
            )
            maximum = self.maximum_sizes_tensor[index].to(
                dtype=raw.dtype, device=raw.device
            )
            sizes.append(
                minimum + (maximum - minimum) * raw.sigmoid()
            )
        sizes = torch.stack(sizes, dim=1)
        boxes = torch.cat((centers, sizes), dim=-1)
        angle = self.maximum_angle_radians * torch.tanh(
            self.angle_head(pooled[:, 0]).squeeze(1)
        )
        entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(
            float(self.feature_height * self.feature_width)
        )
        return GeometryPrediction(
            boxes_cxcywh=boxes,
            angle_radians=angle,
            attention=attention.reshape(
                spatial_features.shape[0],
                len(PART_NAMES),
                self.feature_height,
                self.feature_width,
            ),
            pooled_queries=pooled,
            confidence=confidence,
        )


class BoundedSemanticResidual(nn.Module):
    """Face-anchored fusion whose initial output is exactly the face feature."""

    def __init__(
        self,
        query_dim: int,
        descriptor_dim: int = 512,
        *,
        hidden_dim: int = 256,
        maximum_residual_scale: float = 0.35,
    ):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.maximum_residual_scale = float(maximum_residual_scale)
        if not 0.0 < self.maximum_residual_scale <= 1.0:
            raise ValueError(
                "maximum_residual_scale must be in the interval (0, 1]"
            )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(3 * int(query_dim)),
            nn.Linear(3 * int(query_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.descriptor_dim),
        )
        self.reliability = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(8 * self.descriptor_dim),
            nn.Linear(8 * self.descriptor_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.descriptor_dim, bias=False),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.reliability[-1].weight)
        initial_scale = min(0.10, 0.5 * self.maximum_residual_scale)
        nn.init.constant_(
            self.reliability[-1].bias,
            _logit(initial_scale / self.maximum_residual_scale),
        )

    def forward(
        self,
        face_descriptor: torch.Tensor,
        nose_descriptor: torch.Tensor,
        query_features: torch.Tensor,
        geometry_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_features.ndim != 3 or query_features.shape[1] != 3:
            raise ValueError("query_features must have shape [batch, 3, query_dim]")
        if face_descriptor.shape != nose_descriptor.shape:
            raise ValueError("face and nose descriptors must have identical shapes")
        context = F.normalize(
            self.context_projection(query_features.flatten(1)), dim=1
        )
        face_context_cosine = F.cosine_similarity(
            face_descriptor.float(), context.float(), dim=1
        )
        face_context_difference = (
            face_descriptor.float() - context.float()
        ).abs().mean(dim=1)
        face_nose_cosine = F.cosine_similarity(
            face_descriptor.float(), nose_descriptor.float(), dim=1
        )
        face_nose_difference = (
            face_descriptor.float() - nose_descriptor.float()
        ).abs().mean(dim=1)
        evidence = torch.stack(
            (
                face_context_cosine,
                face_context_difference,
                face_nose_cosine,
                face_nose_difference,
                geometry_confidence[:, 0],
                geometry_confidence[:, 1],
            ),
            dim=1,
        ).to(dtype=face_descriptor.dtype)
        residual_scale = self.maximum_residual_scale * self.reliability(evidence).sigmoid()
        relation = torch.cat(
            (
                face_descriptor,
                nose_descriptor,
                context,
                (face_descriptor - nose_descriptor).abs(),
                (face_descriptor - context).abs(),
                face_descriptor * nose_descriptor,
                face_descriptor * context,
                nose_descriptor * context,
            ),
            dim=1,
        )
        delta = torch.tanh(self.delta(relation)) / math.sqrt(self.descriptor_dim)
        embedding = F.normalize(face_descriptor + residual_scale * delta, dim=1)
        return embedding, context, residual_scale


class UnifiedPetReID(nn.Module):
    """One-input pet descriptor model with learned internal geometry."""

    descriptor_dim = 512

    def __init__(
        self,
        identity_encoder: DogArcFaceEncoder,
        *,
        input_size: int = 1280,
        localization_size: int = 320,
        crop_size: int = 224,
        geometry_hidden_channels: int = 128,
        fusion_hidden_dim: int = 256,
        geometry_feature_mode: str = "layer3",
        maximum_residual_scale: float = 0.35,
        geometry_minimum_sizes: Sequence[float] | None = None,
        geometry_maximum_sizes: Sequence[float] | None = None,
    ):
        super().__init__()
        self.identity_encoder = identity_encoder
        self.input_size = int(input_size)
        self.localization_size = int(localization_size)
        self.crop_size = int(crop_size)
        if min(self.input_size, self.localization_size, self.crop_size) <= 0:
            raise ValueError("all spatial sizes must be positive")
        if self.localization_size % 16:
            raise ValueError("localization_size must be divisible by 16")
        self.geometry_feature_mode = str(geometry_feature_mode)
        if self.geometry_feature_mode not in GEOMETRY_FEATURE_MODES:
            raise ValueError(
                f"geometry_feature_mode must be one of {GEOMETRY_FEATURE_MODES}"
            )
        if self.geometry_feature_mode == "fpn":
            self.geometry_adapter = Layer2Layer3GeometryAdapter(
                geometry_hidden_channels
            )
            geometry_input_channels = int(geometry_hidden_channels)
            geometry_feature_size = self.localization_size // 8
        else:
            self.geometry_adapter = nn.Identity()
            geometry_input_channels = 1024
            geometry_feature_size = self.localization_size // 16
        self.geometry = SoftPartGeometryHead(
            geometry_input_channels,
            geometry_feature_size,
            geometry_feature_size,
            hidden_channels=geometry_hidden_channels,
            minimum_sizes=geometry_minimum_sizes,
            maximum_sizes=geometry_maximum_sizes,
        )
        self.geometry_calibration = PartGeometryCalibration()
        self.cropper = NormalizedRotatedCropper((self.crop_size, self.crop_size))
        self.semantic_fusion = BoundedSemanticResidual(
            geometry_hidden_channels,
            self.descriptor_dim,
            hidden_dim=fusion_hidden_dim,
            maximum_residual_scale=maximum_residual_scale,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=True,
        )

    @classmethod
    def from_arcface_checkpoint(
        cls,
        checkpoint_path,
        **kwargs,
    ) -> "UnifiedPetReID":
        encoder = DogArcFaceEncoder(
            checkpoint_path,
            freeze=True,
            normalize=True,
        )
        return cls(encoder, **kwargs)

    def configure_identity_trainable(self, parts: Sequence[str] = ()) -> None:
        self.identity_encoder.configure_trainable_parts(parts)

    def as_reference_aware(
        self,
        matcher=None,
        *,
        max_references: int | None = None,
    ):
        """Attach the optional query-conditioned reference-set head.

        The returned module keeps this single-image model as its shared
        encoder. Importing lazily avoids coupling the ordinary RGB-to-
        descriptor path to the experimental set-scoring component.
        """

        from .reference_aware_model import ReferenceAwarePetReID
        from .reference_set_model import QueryConditionedReferenceMatcher

        if matcher is None:
            matcher = QueryConditionedReferenceMatcher(
                descriptor_dim=self.descriptor_dim,
                max_references=(
                    4 if max_references is None else int(max_references)
                ),
            )
        return ReferenceAwarePetReID(
            self,
            matcher,
            max_references=max_references,
        )

    def _normalize(self, images_0_255: torch.Tensor) -> torch.Tensor:
        return (images_0_255.div(255.0) - self.pixel_mean) / self.pixel_std

    def _backbone_spatial(
        self, normalized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.identity_encoder.backbone
        x = backbone.conv1(normalized)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        layer2 = backbone.layer2(x)
        return layer2, backbone.layer3(layer2)

    def _backbone_layer3(self, normalized: torch.Tensor) -> torch.Tensor:
        return self._backbone_spatial(normalized)[1]

    def _backbone_descriptor(self, normalized: torch.Tensor) -> torch.Tensor:
        backbone = self.identity_encoder.backbone
        x = self._backbone_layer3(normalized)
        x = backbone.layer4(x)
        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = backbone.fc(x)
        return F.normalize(x, dim=1)

    def _validate_rgb(self, rgb_0_255: torch.Tensor) -> torch.Tensor:
        if rgb_0_255.ndim != 4 or rgb_0_255.shape[1] != 3:
            raise ValueError("rgb_0_255 must have shape [batch, 3, height, width]")
        if tuple(rgb_0_255.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} RGB input, "
                f"got {tuple(rgb_0_255.shape[-2:])}"
            )
        return rgb_0_255.float()

    def _localize(
        self, rgb_0_255: torch.Tensor
    ) -> tuple[GeometryPrediction, torch.Tensor]:
        localization_input = F.interpolate(
            rgb_0_255,
            size=(self.localization_size, self.localization_size),
            mode="bilinear",
            align_corners=False,
        )
        layer2, layer3 = self._backbone_spatial(
            self._normalize(localization_input)
        )
        spatial_features = (
            self.geometry_adapter(layer2, layer3)
            if self.geometry_feature_mode == "fpn"
            else layer3
        )
        prediction = self.geometry(spatial_features)
        return (
            self.geometry_calibration(prediction),
            spatial_features,
        )

    def predict_geometry(self, rgb_0_255: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training/diagnostic geometry output; deployment uses forward()."""

        rgb_0_255 = self._validate_rgb(rgb_0_255)
        geometry, _ = self._localize(rgb_0_255)
        return {
            "boxes_cxcywh": geometry.boxes_cxcywh,
            "angle_radians": geometry.angle_radians,
            "attention": geometry.attention,
            "confidence": geometry.confidence,
        }


    @staticmethod
    def _override_geometry(
        prediction: GeometryPrediction,
        override: Mapping[str, torch.Tensor] | None,
    ) -> GeometryPrediction:
        if override is None:
            return prediction
        boxes = override.get("boxes_cxcywh", prediction.boxes_cxcywh)
        angle = override.get("angle_radians", prediction.angle_radians)
        return GeometryPrediction(
            boxes_cxcywh=boxes,
            angle_radians=angle,
            attention=prediction.attention,
            pooled_queries=prediction.pooled_queries,
            confidence=prediction.confidence,
        )

    def forward(
        self,
        rgb_0_255: torch.Tensor,
        *,
        return_aux: bool = False,
        geometry_override: Mapping[str, torch.Tensor] | None = None,
    ):
        rgb_0_255 = self._validate_rgb(rgb_0_255)
        geometry, geometry_features = self._localize(rgb_0_255)
        geometry = self._override_geometry(geometry, geometry_override)
        face_crops = self.cropper(
            rgb_0_255,
            geometry.boxes_cxcywh[:, 0],
            geometry.angle_radians,
        )
        nose_crops = self.cropper(
            rgb_0_255,
            geometry.boxes_cxcywh[:, 1],
            geometry.angle_radians,
        )
        branch_crops = torch.cat((face_crops, nose_crops), dim=0)
        branch_descriptors = self._backbone_descriptor(
            self._normalize(branch_crops)
        )
        face_descriptor, nose_descriptor = branch_descriptors.chunk(2, dim=0)
        reduced_geometry = self.geometry.reduction(geometry_features)
        global_query = F.adaptive_avg_pool2d(
            reduced_geometry, output_size=1
        ).flatten(1)
        semantic_queries = torch.cat(
            (geometry.pooled_queries, global_query.unsqueeze(1)), dim=1
        )
        embedding, semantic_context, residual_scale = self.semantic_fusion(
            face_descriptor,
            nose_descriptor,
            semantic_queries,
            geometry.confidence,
        )
        if not return_aux:
            return embedding
        return {
            "embedding": embedding,
            "face_descriptor": face_descriptor,
            "nose_descriptor": nose_descriptor,
            "semantic_context": semantic_context,
            "residual_scale": residual_scale,
            "boxes_cxcywh": geometry.boxes_cxcywh,
            "angle_radians": geometry.angle_radians,
            "attention": geometry.attention,
            "geometry_confidence": geometry.confidence,
            "semantic_queries": semantic_queries,
            "face_crops": face_crops,
            "nose_crops": nose_crops,
        }


class UnifiedPetReIDExport(nn.Module):
    """Strict one-input/one-output wrapper used for ONNX export."""

    def __init__(self, model: UnifiedPetReID):
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)

