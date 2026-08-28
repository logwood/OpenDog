"""Feature-only whole-dog backbone built from official Torchvision weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import Swin_V2_B_Weights, swin_v2_b


class FrozenSwinV2BodyBackbone(nn.Module):
    """Expose Swin V2-B spatial and global features without a classifier head.

    Inputs must already use the preprocessing associated with
    :class:`~torchvision.models.Swin_V2_B_Weights`: 256 x 256 RGB tensors with
    ImageNet normalization. The returned spatial tokens make it possible to
    fuse the body branch before global pooling, while ``global_features`` is a
    compact descriptor for later fusion.
    """

    feature_dim = 1024
    input_size = (256, 256)

    def __init__(self, *, pretrained: bool = True, frozen: bool = True):
        super().__init__()
        weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
        model = swin_v2_b(weights=weights)
        model.head = nn.Identity()
        self.features = model.features
        self.norm = model.norm
        self.permute = model.permute
        self.avgpool = model.avgpool
        self.flatten = model.flatten
        self.weights = weights
        self.frozen = bool(frozen)
        if self.frozen:
            self.requires_grad_(False)
            self.eval()

    @staticmethod
    def preprocessing():
        """Return the official pretrained inference preprocessing transform."""
        return Swin_V2_B_Weights.DEFAULT.transforms()

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        channels_last = self.norm(self.features(images))
        feature_map = self.permute(channels_last)
        global_features = self.flatten(self.avgpool(feature_map))
        return {
            "feature_map": feature_map,
            "tokens": feature_map.flatten(2).transpose(1, 2),
            "global_features": F.normalize(global_features, dim=1),
        }

