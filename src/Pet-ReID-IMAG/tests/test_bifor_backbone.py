from __future__ import annotations

import torch

from pet_id.bifor_backbone import FrozenBIFORBodyBackbone
from pet_id.workspace_paths import WORKSPACE_ROOT


def test_bifor_checkpoint_loads_strictly_and_preserves_feature_contract():
    model = FrozenBIFORBodyBackbone(
        WORKSPACE_ROOT / "models/pretrained/BIFOR/f2/bifor.pth",
        frozen=True,
    )
    output = model(torch.zeros(1, 3, 224, 224))

    assert model.feature_dim == 768
    assert output["global_features"].shape == (1, 768)
    assert output["feature_map"].shape[1] == 768
    assert output["tokens"].shape[2] == 768
    assert torch.allclose(
        output["global_features"].norm(dim=1), torch.ones(1), atol=1e-5
    )
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
