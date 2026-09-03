from __future__ import annotations

import torch
import torch.nn.functional as F

from pet_id.unified_highres_structural_onepass import (
    EndToEndStructuralIdentityCore,
)


def _descriptor(batch: int, width: int) -> torch.Tensor:
    return F.normalize(torch.randn(batch, width), dim=1)


def test_integrated_core_is_unbounded_token_backbone_with_nose_gradients() -> None:
    torch.manual_seed(91)
    batch = 3
    core = EndToEndStructuralIdentityCore(
        depth=2,
        attention_heads=8,
        dropout=0.0,
    )
    global_pre = _descriptor(batch, 256).requires_grad_()
    global_post = _descriptor(batch, 256).requires_grad_()
    detail_pre = _descriptor(batch, 256).requires_grad_()
    detail_post = _descriptor(batch, 256).requires_grad_()
    inputs = {
        "semantic_embedding": _descriptor(batch, 512),
        "parent_embedding": _descriptor(batch, 512),
        "detail_embedding": _descriptor(batch, 512),
        "global_face": _descriptor(batch, 512),
        "global_nose_pre": global_pre,
        "global_nose_post": global_post,
        "global_structural": _descriptor(batch, 512),
        "detail_face": _descriptor(batch, 512),
        "detail_nose_pre": detail_pre,
        "detail_nose_post": detail_post,
        "detail_structural": _descriptor(batch, 512),
        "continuous_context": torch.randn(batch, 15),
    }

    output = core(**inputs, return_aux=True)
    assert output["embedding"].shape == (batch, 512)
    torch.testing.assert_close(
        output["embedding"].norm(dim=1),
        torch.ones(batch),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert output["token_attention"].shape == (batch, 2, 8, 12, 12)
    assert not hasattr(core, "gain_logit")
    assert not hasattr(core, "maximum_residual_scale")
    assert core.configuration()["manual_branch_thresholds"] is False
    assert core.configuration()["bounded_output_residual"] is False

    target = _descriptor(batch, 512)
    (1.0 - F.cosine_similarity(output["embedding"], target, dim=1)).mean().backward()
    for value in (global_pre, global_post, detail_pre, detail_post):
        assert value.grad is not None
        assert float(value.grad.abs().sum()) > 0.0


def test_changing_native_nose_tokens_changes_final_identity_vector() -> None:
    torch.manual_seed(19)
    core = EndToEndStructuralIdentityCore(dropout=0.0).eval()
    batch = 2
    inputs = {
        "semantic_embedding": _descriptor(batch, 512),
        "parent_embedding": _descriptor(batch, 512),
        "detail_embedding": _descriptor(batch, 512),
        "global_face": _descriptor(batch, 512),
        "global_nose_pre": _descriptor(batch, 256),
        "global_nose_post": _descriptor(batch, 256),
        "global_structural": _descriptor(batch, 512),
        "detail_face": _descriptor(batch, 512),
        "detail_nose_pre": _descriptor(batch, 256),
        "detail_nose_post": _descriptor(batch, 256),
        "detail_structural": _descriptor(batch, 512),
        "continuous_context": torch.randn(batch, 15),
    }
    baseline = core(**inputs)
    changed = dict(inputs)
    changed["global_nose_pre"] = -inputs["global_nose_pre"]
    changed["detail_nose_post"] = -inputs["detail_nose_post"]
    perturbed = core(**changed)
    difference = (baseline - perturbed).norm(dim=1).mean().detach()
    assert float(difference) > 1.0e-6
