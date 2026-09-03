from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pet_id.unified_data import letterbox_rgb
from pet_id.unified_highres import (
    HighResolutionDetailRefiner,
    ShapeDrivenGlobalLetterbox,
    ShapeDrivenRotatedCropper,
)


@pytest.mark.parametrize(
    ("height", "width"),
    ((208, 126), (375, 223), (640, 1280), (1279, 801)),
)
def test_shape_driven_global_sampler_matches_protected_parent_letterbox(
    height: int,
    width: int,
) -> None:
    image = np.random.default_rng(height + width).integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )
    expected, _, _ = letterbox_rgb(
        image,
        size=1280,
        fill_value=0,
        allow_upscale=False,
    )
    value = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None]
    actual, detail_scale, availability = ShapeDrivenGlobalLetterbox(1280)(value)
    actual_hwc = actual[0].permute(1, 2, 0).numpy()

    # GridSample uses floating coordinates even when the virtual square is an
    # identity-size 1280 canvas.  The error stays far below one uint8 level.
    assert np.max(np.abs(actual_hwc - expected.astype(np.float32))) < 0.06
    assert float(detail_scale) == pytest.approx(1.0)
    assert float(availability) == 0.0


def test_shape_driven_rotated_cropper_has_source_pixel_gradients() -> None:
    image = torch.rand(2, 3, 1400, 1000, requires_grad=True)
    boxes = torch.tensor(
        ((0.50, 0.50, 0.30, 0.25), (0.45, 0.55, 0.20, 0.20)),
        requires_grad=True,
    )
    angles = torch.tensor((0.1, -0.2), requires_grad=True)
    cropper = ShapeDrivenRotatedCropper((32, 40), minimum_side=1280)

    crops = cropper(image, boxes, angles)
    crops.square().mean().backward()

    assert crops.shape == (2, 3, 32, 40)
    assert image.grad is not None and float(image.grad.abs().sum()) > 0.0
    assert boxes.grad is not None and float(boxes.grad.abs().sum()) > 0.0
    assert angles.grad is not None and float(angles.grad.abs().sum()) > 0.0


def _normalized(batch: int, width: int = 512) -> torch.Tensor:
    return F.normalize(torch.randn(batch, width), dim=1)


def test_highres_refiner_is_exact_parent_at_zero_initialization() -> None:
    refiner = HighResolutionDetailRefiner()
    base = _normalized(3)
    output = refiner(
        base,
        _normalized(3),
        _normalized(3),
        torch.rand(3),
        torch.tensor(2.0),
        torch.tensor(1.0),
        torch.rand(3),
        torch.rand(3),
    )

    assert torch.equal(output, base)


def test_highres_refiner_cannot_change_low_resolution_after_training() -> None:
    refiner = HighResolutionDetailRefiner()
    with torch.no_grad():
        refiner.direction_gain_logit.fill_(0.8)
        for parameter in refiner.interaction.parameters():
            parameter.normal_(0.0, 0.05)
        for parameter in refiner.reliability.parameters():
            parameter.normal_(0.0, 0.05)
    base = _normalized(4)
    output = refiner(
        base,
        _normalized(4),
        _normalized(4),
        torch.rand(4),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.rand(4),
        torch.rand(4),
    )

    assert torch.equal(output, base)


def test_first_highres_backward_reaches_bounded_residual_controls() -> None:
    refiner = HighResolutionDetailRefiner()
    base = _normalized(4)
    face = _normalized(4)
    output = refiner(
        base,
        face,
        _normalized(4),
        torch.rand(4),
        torch.tensor(2.0),
        torch.tensor(1.0),
        torch.rand(4),
        torch.rand(4),
    )
    loss = 1.0 - F.cosine_similarity(output, face, dim=1).mean()
    loss.backward()

    assert refiner.direction_gain_logit.grad is not None
    assert float(refiner.direction_gain_logit.grad.abs()) > 0.0
    assert refiner.interaction[-1].weight.grad is not None
    assert float(refiner.interaction[-1].weight.grad.norm()) > 0.0
    # The reliability path intentionally starts receiving gradients after the
    # global direction gate's first non-zero optimizer update.
    assert refiner.reliability[-1].weight.grad is not None
    assert float(refiner.reliability[-1].weight.grad.norm()) == 0.0
