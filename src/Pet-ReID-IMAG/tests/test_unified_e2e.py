"""Contract tests for graph-internal UnifiedPetReID preprocessing."""

from __future__ import annotations

import torch

from pet_id.unified_e2e import GraphInternalLetterbox


def test_graph_letterbox_preserves_native_pixels_and_black_padding() -> None:
    module = GraphInternalLetterbox(output_size=8, allow_upscale=False)
    source = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)

    actual = module(source)

    assert tuple(actual.shape) == (2, 3, 8, 8)
    torch.testing.assert_close(actual[:, :, 3:5, 2:6], source)
    expected = torch.zeros_like(actual)
    expected[:, :, 3:5, 2:6] = source
    torch.testing.assert_close(actual, expected)


def test_graph_letterbox_accepts_dynamic_spatial_shapes() -> None:
    module = GraphInternalLetterbox(output_size=16, allow_upscale=False)

    portrait = module(torch.full((1, 3, 9, 5), 17.0))
    landscape = module(torch.full((2, 3, 6, 12), 29.0))

    assert tuple(portrait.shape) == (1, 3, 16, 16)
    assert tuple(landscape.shape) == (2, 3, 16, 16)
    assert torch.isfinite(portrait).all()
    assert torch.isfinite(landscape).all()
