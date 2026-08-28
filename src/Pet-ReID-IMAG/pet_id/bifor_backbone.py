"""Frozen feature-only BIFOR body backbone.

BIFOR f(2) is an official ConvNeXt-Small whose classifier is removed.  This
wrapper preserves the upstream checkpoint key layout while exposing the same
feature dictionary used by the project's other body backbone.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import convnext_small


class FrozenBIFORBodyBackbone(nn.Module):
    """Load BIFOR f(2) as a frozen, headless 768-D body encoder."""

    feature_dim = 768
    input_size = (224, 224)

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        frozen: bool = True,
        strict: bool = True,
    ) -> None:
        super().__init__()
        model = convnext_small(weights=None)
        # This name and nesting exactly match upstream models/Bifor.py.
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path).resolve()
        )
        if self.checkpoint_path is not None:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.load_state_dict(state_dict, strict=strict)
        self.frozen = bool(frozen)
        if self.frozen:
            self.requires_grad_(False)
            self.eval()

    @staticmethod
    def preprocessing() -> transforms.Compose:
        """Return the preprocessing used by the official BIFOR repository."""
        return transforms.Compose(
            (
                transforms.Resize((224, 224), antialias=True),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        )

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        feature_map = self.feature_extractor[0](images)
        global_features = self.feature_extractor[1](feature_map).flatten(1)
        return {
            "feature_map": feature_map,
            "tokens": feature_map.flatten(2).transpose(1, 2),
            "global_features": F.normalize(global_features, dim=1),
        }
