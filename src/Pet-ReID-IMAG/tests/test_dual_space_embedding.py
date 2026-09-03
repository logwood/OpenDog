from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pet_id.dual_space_embedding import (
    STRUCTURAL_VARIANTS,
    DualSpaceNoseEmbeddingBridge,
)


@pytest.mark.parametrize("variant", STRUCTURAL_VARIANTS)
def test_bridge_is_face_anchored_and_normalized_at_initialization(variant: str):
    torch.manual_seed(7)
    model = DualSpaceNoseEmbeddingBridge(
        variant=variant,
        token_dim=32,
        bottleneck_dim=16,
        hidden_dim=32,
        attention_heads=4,
        dropout=0.0,
    ).eval()
    face = torch.randn(5, 512)
    nose_pre = torch.randn(5, 2048)
    nose_post = torch.randn(5, 2048)

    output = model(face, nose_pre, nose_post, return_aux=True)

    expected = F.normalize(face, dim=1)
    assert output["embedding"].shape == (5, 512)
    assert torch.allclose(output["embedding"], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        output["embedding"].norm(dim=1),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )
    assert output["nose_post_token"].shape == (5, 32)
    if variant == "post_residual":
        assert output["nose_pre_token"] is None
        assert output["attention_weights"] is None
    else:
        assert output["nose_pre_token"].shape == (5, 32)
    if variant == "dual_cross_attention":
        assert output["attention_weights"].shape == (5, 2)
        assert torch.allclose(
            output["attention_weights"].sum(dim=1),
            torch.ones(5),
            atol=1e-6,
            rtol=1e-6,
        )


def test_bridge_receives_gradient_without_identity_classifier():
    torch.manual_seed(11)
    model = DualSpaceNoseEmbeddingBridge(
        variant="dual_cross_attention",
        token_dim=32,
        bottleneck_dim=16,
        hidden_dim=32,
        attention_heads=4,
        dropout=0.0,
    )
    face = torch.randn(8, 512)
    nose_pre = torch.randn(8, 2048)
    nose_post = torch.randn(8, 2048)
    embedding = model(face, nose_pre, nose_post)
    target = F.normalize(torch.randn_like(embedding), dim=1)

    (1.0 - F.cosine_similarity(embedding, target, dim=1).mean()).backward()

    gradient = model.residual_head[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0
    assert not any("classifier" in name for name, _ in model.named_parameters())


def test_bridge_rejects_mismatched_native_descriptor_shape():
    model = DualSpaceNoseEmbeddingBridge(
        variant="dual_consensus",
        token_dim=32,
        bottleneck_dim=16,
        hidden_dim=32,
        attention_heads=4,
    )
    with pytest.raises(ValueError, match="nose_pre_descriptor"):
        model(
            torch.randn(2, 512),
            torch.randn(2, 1024),
            torch.randn(2, 2048),
        )

