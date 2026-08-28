from __future__ import annotations

import torch
from torch.nn import functional as F

from pet_id.body_fusion import BodyPrimaryFusionNeck


def _inputs(batch: int = 6) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "nose_features": F.normalize(torch.randn(batch, 512), dim=1),
        "face_features": F.normalize(torch.randn(batch, 512), dim=1),
        "body_features": F.normalize(torch.randn(batch, 1024), dim=1),
        "nose_quality_signals": torch.rand(batch, 10),
        "body_quality_signals": torch.rand(batch, 4),
        "branch_available": torch.ones(batch, 3, dtype=torch.bool),
    }


def test_body_primary_fusion_preserves_512d_cosine_interface():
    model = BodyPrimaryFusionNeck()
    output = model(**_inputs())

    assert output["features"].shape == (6, 512)
    assert output["primary_features"].shape == (6, 512)
    assert output["adapted_body_features"].shape == (6, 512)
    assert torch.allclose(output["features"].norm(dim=1), torch.ones(6), atol=1e-5)
    assert torch.allclose(
        output["primary_features"].norm(dim=1), torch.ones(6), atol=1e-5
    )


def test_body_and_nose_weights_are_bounded_when_all_branches_exist():
    model = BodyPrimaryFusionNeck(max_body_weight=0.55, max_nose_weight=0.35)
    output = model(**_inputs())

    assert torch.all(output["body_weights"][:, 0] <= 0.55 + 1e-7)
    assert torch.all(output["body_weights"][:, 0] >= 0.0)
    assert torch.all(output["nose_weights"][:, 0] <= 0.35 + 1e-7)
    assert torch.all(output["nose_weights"][:, 0] >= 0.0)
    assert torch.allclose(output["body_weights"][:, 0], torch.full((6,), 0.30))
    assert torch.allclose(output["nose_weights"][:, 0], torch.full((6,), 0.10))


def test_missing_body_falls_back_to_nose_face_path():
    model = BodyPrimaryFusionNeck()
    inputs = _inputs()
    inputs["branch_available"][:, 2] = False
    output = model(**inputs)

    assert torch.allclose(output["body_weights"][:, 0], torch.zeros(6))
    assert torch.allclose(
        output["primary_features"], inputs["face_features"], atol=1e-6
    )
    assert torch.allclose(output["nose_weights"][:, 0], torch.full((6,), 0.10))


def test_missing_nose_returns_face_body_primary_descriptor():
    model = BodyPrimaryFusionNeck()
    inputs = _inputs()
    inputs["branch_available"][:, 0] = False
    output = model(**inputs)

    assert torch.allclose(output["nose_weights"][:, 0], torch.zeros(6))
    assert torch.allclose(
        output["features"], output["primary_features"], atol=1e-6
    )


def test_single_primary_branch_has_exact_weight_one_fallback():
    model = BodyPrimaryFusionNeck()
    inputs = _inputs(batch=2)
    inputs["branch_available"] = torch.tensor(
        [
            [False, False, True],
            [False, True, False],
        ]
    )
    output = model(**inputs)

    assert torch.allclose(output["body_weights"][:, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(
        output["features"][0], output["adapted_body_features"][0], atol=1e-6
    )
    assert torch.allclose(output["features"][1], inputs["face_features"][1], atol=1e-6)

