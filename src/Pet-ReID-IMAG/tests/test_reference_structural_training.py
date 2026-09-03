from __future__ import annotations

import torch
from torch import nn

from pet_id.reference_aware_training import (
    ReferenceImageBatch,
    ReferenceImageEpisode,
    hard_negative_margin_loss,
    materialize_reference_image_episode,
    reference_episode_loss,
)
from pet_id.reference_token_model import (
    TokenConditionedReferenceMatcher,
    TokenReferenceAwarePetReID,
)


class _SpatialEncoder(nn.Module):
    descriptor_dim = 8
    input_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer4 = nn.Conv2d(3, 6, kernel_size=1)
        self.projection = nn.Linear(6, self.descriptor_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.layer4(images.float())
        return self.projection(features.mean(dim=(2, 3)))


def _token_model() -> TokenReferenceAwarePetReID:
    torch.manual_seed(23)
    matcher = TokenConditionedReferenceMatcher(
        descriptor_dim=8,
        token_dim=6,
        hidden_dim=5,
        max_references=2,
        reference_top_k=2,
    )
    return TokenReferenceAwarePetReID(
        _SpatialEncoder(),
        matcher,
        token_dim=6,
        token_grid=2,
    )


def _metadata_batch() -> ReferenceImageBatch:
    return ReferenceImageBatch(
        query_images=torch.rand(2, 3, 4, 4),
        reference_images=torch.rand(2, 2, 3, 4, 4),
        reference_mask=torch.ones(2, 2, dtype=torch.bool),
        targets=torch.tensor([0, 1]),
        identity_names=("a", "b"),
        query_view_features=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        reference_view_features=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ]
        ),
        query_view_valid=torch.ones(2, dtype=torch.bool),
        reference_view_valid=torch.ones(2, 2, dtype=torch.bool),
        query_quality_features=torch.ones(2, 6),
        reference_quality_features=torch.ones(2, 2, 6),
        query_quality_valid=torch.ones(2, dtype=torch.bool),
        reference_quality_valid=torch.ones(2, 2, dtype=torch.bool),
    )


def test_hard_negative_margin_uses_strongest_impostor() -> None:
    loss, observed = hard_negative_margin_loss(
        torch.tensor([[0.8, 0.7, 0.2], [0.1, 0.4, 0.39]]),
        torch.tensor([0, 1]),
        margin=0.2,
    )
    torch.testing.assert_close(loss, torch.tensor(0.145))
    torch.testing.assert_close(observed, torch.tensor(0.055))


def test_view_coverage_supervision_reaches_token_gate() -> None:
    model = _token_model()
    loss, details = reference_episode_loss(
        model,
        _metadata_batch(),
        hard_negative_weight=0.25,
        view_coverage_weight=0.2,
    )
    assert torch.isfinite(loss)
    assert float(details["coverage_valid_fraction"]) == 1.0
    assert float(details["view_coverage_loss"]) > 0.0
    loss.backward()
    gradient = model.matcher.coverage_head[-1].weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_metadata_free_batch_keeps_coverage_objective_as_zero() -> None:
    model = _token_model()
    batch = _metadata_batch()
    batch = ReferenceImageBatch(
        query_images=batch.query_images,
        reference_images=batch.reference_images,
        reference_mask=batch.reference_mask,
        targets=batch.targets,
        identity_names=batch.identity_names,
    )
    _, details = reference_episode_loss(model, batch, view_coverage_weight=1.0)
    torch.testing.assert_close(details["view_coverage_loss"], torch.tensor(0.0))
    torch.testing.assert_close(details["coverage_valid_fraction"], torch.tensor(0.0))


class _MetadataDataset:
    def __init__(self) -> None:
        self.records = [
            {
                "identity": "a",
                "viewpoint_signals": [1, 0, 0, 0],
                "quality_signals": [1] * 6,
            },
            {
                "identity": "a",
                "viewpoint_signals": [0, 1, 0, 0],
                "quality_signals": [0.8] * 6,
            },
            {
                "identity": "b",
                "viewpoint_signals": [0, 0, 1, 0],
                "quality_signals": [0.9] * 6,
            },
            {
                "identity": "b",
                "viewpoint_signals": [0, 0, 0, 1],
                "quality_signals": [0.7] * 6,
            },
        ]

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            "rgb": torch.full((3, 4, 4), float(index)),
            "record": record,
        }


def test_episode_materialization_propagates_view_metadata() -> None:
    dataset = _MetadataDataset()
    episode = ReferenceImageEpisode(
        identity_names=("a", "b"),
        query_indices=(1, 3),
        reference_indices=((0,), (2,)),
        targets=torch.tensor([0, 1]),
    )
    batch = materialize_reference_image_episode(dataset, episode)
    assert batch.query_view_features is not None
    assert batch.reference_view_features is not None
    assert tuple(batch.query_view_features.shape) == (2, 4)
    assert tuple(batch.reference_view_features.shape) == (2, 1, 4)
    assert bool(batch.query_view_valid.all())
    assert bool(batch.reference_view_valid.all())
