"""Numerically stable graph-internal geometry discretization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_GEOMETRY_BOX_OFFSETS = (
    (
        0.149932861328125,
        0.2910614013671875,
        0.3283538818359375,
        0.0924224853515625,
    ),
    (
        0.9654998779296875,
        0.1184234619140625,
        0.377593994140625,
        0.17376708984375,
    ),
)
DEFAULT_GEOMETRY_ANGLE_OFFSET = 0.269317626953125
LOCKED_GEOMETRY_BOX_STEP = 1.0 / 300.0
# Keep the angle continuous; angle quantization harmed development retrieval
# while the raw backend discrepancy remains below the cosine tolerance.
LOCKED_GEOMETRY_ANGLE_STEP: float | None = None


def choose_backend_stable_offset(
    reference: np.ndarray,
    backends: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    step: float,
    require_all: bool = True,
    preferred_step: float | None = None,
    preferred_offset: float | None = None,
    minimum_boundary_margin: float = 0.0,
) -> dict[str, Any]:
    """Choose one quantization phase with identical bins on every backend.

    StableGeometryDiscretizer quantizes round(value / step + offset).
    Candidate boundaries are evaluated between every observed fractional
    coordinate. Among phases with complete PyTorch/backend agreement, the
    phase furthest from all observed values is selected.
    """

    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    backend_arrays = [
        np.asarray(value, dtype=np.float32).reshape(-1) for value in backends
    ]
    if not backend_arrays:
        raise ValueError("at least one backend array is required")
    if reference.size == 0:
        raise ValueError("reference values must not be empty")
    if any(value.shape != reference.shape for value in backend_arrays):
        raise ValueError("reference and backend arrays must share one shape")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0:
        raise ValueError("step must be finite and positive")
    all_values = np.concatenate((reference, *backend_arrays))
    if not np.isfinite(all_values).all():
        raise ValueError("geometry values must be finite")
    residues = np.mod(
        all_values.astype(np.float64) / float(step32),
        1.0,
    )
    points = np.unique(residues)
    following = np.roll(points, -1)
    gaps = np.mod(following - points, 1.0)
    boundaries = np.mod(points + 0.5 * gaps, 1.0)

    best: dict[str, Any] | None = None
    total_pairs = int(reference.size * len(backend_arrays))
    preserve_values = (
        preferred_step is not None and preferred_offset is not None
    )
    preferred_quantized = None
    if preserve_values:
        preferred_step32 = np.float32(preferred_step)
        preferred_offset32 = np.float32(preferred_offset)
        preferred_quantized = (
            np.rint(reference / preferred_step32 + preferred_offset32)
            - preferred_offset32
        ) * preferred_step32
    for boundary in boundaries:
        offset = np.float32(np.mod(0.5 - boundary, 1.0))
        reference_bins = np.rint(reference / step32 + offset).astype(np.int64)
        backend_bins = [
            np.rint(value / step32 + offset).astype(np.int64)
            for value in backend_arrays
        ]
        matching_pairs = sum(
            int(np.count_nonzero(value == reference_bins))
            for value in backend_bins
        )
        circular_distance = np.abs(
            np.mod(residues - boundary + 0.5, 1.0) - 0.5
        )
        margin = float(circular_distance.min(initial=0.5) * float(step32))
        candidate = {
            "offset": float(offset),
            "boundary_phase": float(boundary),
            "matching_pairs": matching_pairs,
            "total_pairs": total_pairs,
            "all_match": matching_pairs == total_pairs,
            "minimum_boundary_margin": margin,
        }
        if preferred_quantized is not None:
            quantized = (reference_bins - offset) * step32
            change = np.abs(quantized - preferred_quantized)
            candidate["mean_abs_change_from_preferred"] = float(change.mean())
            candidate["max_abs_change_from_preferred"] = float(change.max())
            key = (
                int(
                    candidate["all_match"]
                    and margin >= minimum_boundary_margin
                ),
                int(candidate["all_match"]),
                matching_pairs,
                -candidate["mean_abs_change_from_preferred"],
                -candidate["max_abs_change_from_preferred"],
                margin,
                -float(offset),
            )
        else:
            key = (
                int(
                    candidate["all_match"]
                    and margin >= minimum_boundary_margin
                ),
                int(candidate["all_match"]),
                matching_pairs,
                margin,
                -float(offset),
            )
        if best is None or key > best["_key"]:
            best = {**candidate, "_key": key}
    assert best is not None
    best.pop("_key")
    if require_all and (
        not best["all_match"]
        or best["minimum_boundary_margin"] < minimum_boundary_margin
    ):
        raise RuntimeError(
            "No quantization phase makes every backend share one bin; "
            f"best={best['matching_pairs']}/{best['total_pairs']}"
        )
    return best


class StableGeometryDiscretizer(nn.Module):
    """Make downstream crops robust to backend numeric differences.

    Quantization uses a straight-through estimator in training and exact
    rounding in evaluation/export. Offsets select the centers of development
    intervals where PyTorch and ONNX geometry predictions share one bin.
    """

    def __init__(
        self,
        *,
        box_step: (
            float
            | tuple[
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
            | list[list[float]]
            | None
        ) = None,
        box_offsets: tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ] = DEFAULT_GEOMETRY_BOX_OFFSETS,
        angle_step: float | None = None,
        angle_offset: float = DEFAULT_GEOMETRY_ANGLE_OFFSET,
        box_piecewise: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if box_step is None:
            self.box_step: float | list[list[float]] | None = None
            box_steps = torch.ones(2, 4, dtype=torch.float32)
        elif np.isscalar(box_step):
            self.box_step = float(box_step)
            box_steps = torch.full((2, 4), self.box_step, dtype=torch.float32)
        else:
            box_steps = torch.as_tensor(box_step, dtype=torch.float32)
            if tuple(box_steps.shape) != (2, 4):
                raise ValueError("per-coordinate box steps must have shape [2, 4]")
            self.box_step = box_steps.tolist()
        self.angle_step = (
            float(angle_step) if angle_step is not None else None
        )
        self.angle_offset = float(angle_offset)
        self.box_piecewise_configuration: list[dict[str, Any]] | None = None
        if self.box_step is not None and (
            not torch.isfinite(box_steps).all() or (box_steps <= 0).any()
        ):
            raise ValueError("box quantization steps must be finite and positive")
        if self.angle_step is not None and self.angle_step <= 0:
            raise ValueError("angle quantization step must be positive")
        offsets = torch.as_tensor(box_offsets, dtype=torch.float32)
        if tuple(offsets.shape) != (2, 4):
            raise ValueError("box_offsets must have shape [2, 4]")
        if not torch.isfinite(offsets).all():
            raise ValueError("box_offsets must be finite")
        self.register_buffer(
            "box_offsets", offsets.unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "box_steps", box_steps.unsqueeze(0), persistent=False
        )
        if box_piecewise is not None:
            if len(box_piecewise) != 8:
                raise ValueError("box_piecewise must define all 8 box coordinates")
            normalized_piecewise = []
            for index, record in enumerate(box_piecewise):
                thresholds = torch.as_tensor(
                    record["thresholds"], dtype=torch.float32
                )
                levels = torch.as_tensor(record["levels"], dtype=torch.float32)
                if thresholds.ndim != 1 or levels.ndim != 1:
                    raise ValueError("piecewise thresholds/levels must be 1-D")
                if levels.numel() != thresholds.numel() + 1:
                    raise ValueError(
                        "piecewise levels must have one more item than thresholds"
                    )
                if (
                    not torch.isfinite(thresholds).all()
                    or not torch.isfinite(levels).all()
                ):
                    raise ValueError("piecewise geometry values must be finite")
                if thresholds.numel() > 1 and not torch.all(
                    thresholds[1:] > thresholds[:-1]
                ):
                    raise ValueError("piecewise thresholds must be strictly sorted")
                self.register_buffer(
                    f"box_thresholds_{index}", thresholds, persistent=False
                )
                self.register_buffer(
                    f"box_levels_{index}", levels, persistent=False
                )
                normalized_piecewise.append(
                    {
                        "part": index // 4,
                        "coordinate": index % 4,
                        "thresholds": thresholds.tolist(),
                        "levels": levels.tolist(),
                    }
                )
            self.box_piecewise_configuration = normalized_piecewise

    @property
    def enabled(self) -> bool:
        return (
            self.box_step is not None
            or self.box_piecewise_configuration is not None
            or self.angle_step is not None
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "box_step": self.box_step,
            "box_offsets": self.box_offsets.squeeze(0).tolist(),
            "box_piecewise": self.box_piecewise_configuration,
            "angle_step": self.angle_step,
            "angle_offset": self.angle_offset,
            "training_gradient": "straight_through_estimator",
        }

    def _quantize(
        self,
        value: torch.Tensor,
        step: float,
        offset: torch.Tensor | float,
    ) -> torch.Tensor:
        quantized = (
            torch.round(value / step + offset) - offset
        ) * step
        if self.training:
            return value + (quantized - value).detach()
        return quantized

    def forward(
        self, boxes: torch.Tensor, angles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return boxes, angles
        stable_boxes = boxes
        if self.box_piecewise_configuration is not None:
            parts = []
            for part in range(2):
                coordinates = []
                for coordinate in range(4):
                    index = part * 4 + coordinate
                    value = boxes[:, part, coordinate]
                    thresholds = getattr(
                        self, f"box_thresholds_{index}"
                    ).to(dtype=boxes.dtype)
                    levels = getattr(self, f"box_levels_{index}").to(
                        dtype=boxes.dtype
                    )
                    bucket = (
                        value[:, None] >= thresholds[None, :]
                    ).to(torch.int64).sum(dim=1)
                    coordinates.append(levels[bucket])
                parts.append(torch.stack(coordinates, dim=1))
            quantized_boxes = torch.stack(parts, dim=1)
            stable_boxes = (
                boxes + (quantized_boxes - boxes).detach()
                if self.training
                else quantized_boxes
            )
        elif self.box_step is not None:
            offsets = self.box_offsets.to(dtype=boxes.dtype)
            steps = self.box_steps.to(dtype=boxes.dtype)
            stable_boxes = self._quantize(boxes, steps, offsets)
            centers = stable_boxes[..., :2].clamp(0.0, 1.0)
            sizes = stable_boxes[..., 2:].clamp(1e-4, 1.0)
            stable_boxes = torch.cat((centers, sizes), dim=-1)
        stable_angles = angles
        if self.angle_step is not None:
            stable_angles = self._quantize(
                angles, self.angle_step, self.angle_offset
            )
        return stable_boxes, stable_angles
