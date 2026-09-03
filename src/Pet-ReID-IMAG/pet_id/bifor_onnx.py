"""ONNX deployment boundary for semantic identity plus locked BIFOR fusion."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .bifor_backbone import FrozenBIFORBodyBackbone
from .multimodal import LocalEndToEndPetIDModel
from .onnx_export import ONNX_OUTPUT_NAMES, PreCroppedPetEmbeddingModel
from .release_compatibility import is_lowrank_body_fusion_architecture


BIFOR_ONNX_INPUT_NAMES = (
    "nose_crop",
    "face_crop",
    "body_crop",
    "nose_mask",
    "quality_signals",
    "viewpoint_signals",
    "branch_available",
)


class PreCroppedBIFORPetEmbeddingModel(nn.Module):
    """Frozen semantic and BIFOR encoders with the selected body projection.

    The established diagnostic outputs are preserved. Only the main embedding
    becomes the 512-D low-rank body-fused descriptor. BIFOR's classifier is not
    present, and the locked experiment's horizontal-flip TTA remains in-graph.
    """

    input_names = BIFOR_ONNX_INPUT_NAMES
    output_names = ONNX_OUTPUT_NAMES

    def __init__(
        self,
        identity_model: LocalEndToEndPetIDModel,
        body_encoder: FrozenBIFORBodyBackbone,
        fusion_checkpoint: str | Path,
    ) -> None:
        super().__init__()
        self.semantic_model = PreCroppedPetEmbeddingModel(identity_model)
        self.body_encoder = body_encoder
        checkpoint_path = Path(fusion_checkpoint).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        architecture = checkpoint.get("architecture", {})
        if not is_lowrank_body_fusion_architecture(architecture.get("name")):
            raise ValueError(
                "Expected the low-rank semantic body-fusion architecture, got "
                f"{architecture.get('name')!r}"
            )
        body_weight = float(checkpoint["body_weight"])
        semantic_weight = float(checkpoint["semantic_weight"])
        if not math.isclose(
            body_weight + semantic_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("Body and semantic weights must sum to one")

        mean = checkpoint["mean"].detach().float()
        projection = checkpoint["projection"].detach().float()
        expected_input_dim = int(identity_model.fused_dim) + int(
            body_encoder.feature_dim
        )
        if tuple(mean.shape) != (1, expected_input_dim):
            raise ValueError(
                f"Unexpected low-rank mean shape {tuple(mean.shape)}; "
                f"expected (1, {expected_input_dim})"
            )
        if projection.ndim != 2 or projection.shape[1] != expected_input_dim:
            raise ValueError(
                f"Unexpected low-rank projection shape: {tuple(projection.shape)}"
            )
        self.output_dim = int(architecture["output_dim"])
        if projection.shape[0] > self.output_dim:
            raise ValueError("Projection rank exceeds output dimension")

        self.projection_rank = int(projection.shape[0])
        self.body_weight = body_weight
        self.semantic_weight = semantic_weight
        self.fusion_checkpoint_path = checkpoint_path
        self.register_buffer("fusion_mean", mean)
        self.register_buffer("fusion_projection", projection)
        self.register_buffer(
            "body_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "body_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()

    def _body_features(self, body_crop: torch.Tensor) -> torch.Tensor:
        normalized = (
            body_crop.div(255.0) - self.body_mean.to(dtype=body_crop.dtype)
        ) / self.body_std.to(dtype=body_crop.dtype)
        original = self.body_encoder(normalized)["global_features"]
        flipped = self.body_encoder(normalized.flip(-1))["global_features"]
        return F.normalize(original + flipped, dim=1)

    def forward(
        self,
        nose_crop: torch.Tensor,
        face_crop: torch.Tensor,
        body_crop: torch.Tensor,
        nose_mask: torch.Tensor,
        quality_signals: torch.Tensor,
        viewpoint_signals: torch.Tensor,
        branch_available: torch.Tensor,
    ):
        semantic_outputs = self.semantic_model(
            nose_crop,
            face_crop,
            nose_mask,
            quality_signals,
            viewpoint_signals,
            branch_available,
        )
        semantic_embedding = F.normalize(semantic_outputs[0].float(), dim=1)
        body_embedding = self._body_features(body_crop).float()
        joint = torch.cat(
            (
                semantic_embedding * math.sqrt(self.semantic_weight),
                body_embedding * math.sqrt(self.body_weight),
            ),
            dim=1,
        )
        projected = F.linear(
            joint - self.fusion_mean,
            self.fusion_projection,
        )
        projected = F.pad(
            projected,
            (0, self.output_dim - self.projection_rank),
        )
        embedding = F.normalize(projected, dim=1)
        return (embedding, *semantic_outputs[1:])
