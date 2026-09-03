"""Graph-internal preprocessing for a genuinely end-to-end UnifiedPetReID.

The earlier unified export stopped at a fixed, already-letterboxed tensor.
That is useful for a training parity check, but it leaves an important part
of the image contract in Python.  The classes in this module move the
centered black letterbox into the exported graph.  The graph therefore accepts
raw RGB pixel tensors (``0..255``) with dynamic spatial dimensions and emits
the model's one normalized embedding.  JPEG/PNG decoding and EXIF handling
remain transport concerns of the HTTP boundary; no learned or geometric
inference is performed outside the graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import operators as onnx_operators


def _validate_raw_rgb(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [batch, 3, height, width]")
    if rgb.shape[-2] < 2 or rgb.shape[-1] < 2:
        raise ValueError("RGB height and width must both be at least two")
    return rgb.float()


def _source_geometry(
    rgb: torch.Tensor,
    *,
    output_size: int,
    allow_upscale: bool,
) -> tuple[torch.Tensor, ...]:
    """Return tensor-valued source-to-centered-square geometry.

    ``shape_as_tensor`` keeps height and width symbolic in the legacy ONNX
    exporter.  When upscaling is disabled, the virtual square is at least the
    requested output size, which exactly represents a native-resolution image
    surrounded by black padding.  When enabled, the source's own long side is
    used and a smaller image may fill the output square.
    """

    shape = onnx_operators.shape_as_tensor(rgb).to(
        device=rgb.device,
        dtype=rgb.dtype,
    )
    height = shape[-2]
    width = shape[-1]
    extent = torch.maximum(height, width)
    if allow_upscale:
        side = extent
    else:
        side = torch.maximum(
            extent,
            torch.full_like(extent, float(output_size)),
        )
    pad_left = torch.floor((side - width) * 0.5)
    pad_top = torch.floor((side - height) * 0.5)
    x_scale = side / width
    y_scale = side / height
    # ``align_corners=False`` maps a pixel centre to a half-pixel coordinate.
    # These translations retain the one-pixel asymmetry of integer padding.
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


class GraphInternalLetterbox(nn.Module):
    """Center-letterbox raw RGB pixels into a fixed square inside the graph."""

    def __init__(self, output_size: int = 1280, *, allow_upscale: bool = False):
        super().__init__()
        self.output_size = int(output_size)
        self.allow_upscale = bool(allow_upscale)
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = _validate_raw_rgb(rgb)
        (
            _,
            _,
            side,
            x_scale,
            y_scale,
            x_translation,
            y_translation,
        ) = _source_geometry(
            rgb,
            output_size=self.output_size,
            allow_upscale=self.allow_upscale,
        )
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
        return F.grid_sample(
            rgb,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class UnifiedEndToEndPetReID(nn.Module):
    """Wrap a fixed-square unified model with graph-internal raw preprocessing."""

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: int | None = None,
        allow_upscale: bool = False,
    ) -> None:
        super().__init__()
        inferred = input_size if input_size is not None else getattr(model, "input_size", None)
        if inferred is None:
            raise ValueError("input_size is required when the model has no input_size")
        self.input_size = int(inferred)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        self.model = model
        self.preprocessing = GraphInternalLetterbox(
            self.input_size,
            allow_upscale=allow_upscale,
        )
        self.allow_upscale = bool(allow_upscale)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocessing(rgb))


class UnifiedEndToEndPetReIDExport(nn.Module):
    """Strict one-input/one-output deployment boundary for raw RGB pixels."""

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: int | None = None,
        allow_upscale: bool = False,
    ) -> None:
        super().__init__()
        self.model = UnifiedEndToEndPetReID(
            model,
            input_size=input_size,
            allow_upscale=allow_upscale,
        )

    @property
    def input_size(self) -> int:
        return self.model.input_size

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)


# Descriptive aliases used by downstream exporters and integrations.
RawRGBUnifiedPetReID = UnifiedEndToEndPetReID
RawRGBUnifiedPetReIDExport = UnifiedEndToEndPetReIDExport


__all__ = [
    "GraphInternalLetterbox",
    "UnifiedEndToEndPetReID",
    "UnifiedEndToEndPetReIDExport",
    "RawRGBUnifiedPetReID",
    "RawRGBUnifiedPetReIDExport",
]
