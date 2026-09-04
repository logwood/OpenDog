from __future__ import annotations

import torch
from torch import nn

from pet_id.reference_aware_training import (
    ReferenceImageBatch,
    ReferenceImageEpisode,
    baseline_no_harm_loss,
    hard_negative_margin_loss,
    materialize_reference_image_episode,
    reference_episode_loss,
    score_reference_image_episode,
    view_coverage_loss,
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


def _three_reference_token_model() -> TokenReferenceAwarePetReID:
    torch.manual_seed(29)
    matcher = TokenConditionedReferenceMatcher(
        descriptor_dim=8,
        token_dim=6,
        hidden_dim=5,
        max_references=3,
        reference_top_k=3,
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
    assert float(details["view_coverage_loss"].detach()) > 0.0
    loss.backward()
    coverage_gradient = model.matcher.coverage_head[-1].weight.grad
    novelty_gradient = model.matcher.view_projection[1].weight.grad
    assert coverage_gradient is not None
    assert novelty_gradient is not None
    assert float(coverage_gradient.abs().sum()) > 0.0
    assert float(novelty_gradient.abs().sum()) > 0.0
    assert float(details["novelty_alignment_loss"].detach()) > 0.0
    assert float(details["reliability_alignment_loss"].detach()) > 0.0


def test_baseline_no_harm_penalizes_every_reduced_pairwise_margin() -> None:
    baseline = torch.tensor([[0.8, 0.4, 0.2], [0.1, 0.6, 0.5]])
    score = torch.tensor(
        [[0.7, 0.5, 0.2], [0.0, 0.8, 0.5]],
        requires_grad=True,
    )
    loss = baseline_no_harm_loss(
        {"score": score, "baseline_score": baseline},
        torch.tensor([0, 1]),
    )
    catalog_spread = baseline[0].std(unbiased=False)
    neighbor_weights = torch.softmax(
        torch.tensor([0.4, 0.2]) / catalog_spread,
        dim=0,
    )
    expected_loss = 0.5 * (0.2 * neighbor_weights[0] + 0.1 * neighbor_weights[1])
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    torch.testing.assert_close(
        score.grad,
        torch.tensor(
            [
                [
                    -0.5,
                    0.5 * neighbor_weights[0],
                    0.5 * neighbor_weights[1],
                ],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert float(score.grad[0, 1]) > float(score.grad[0, 2]) > 0.0


def test_baseline_no_harm_does_not_move_the_protected_baseline() -> None:
    baseline = torch.tensor([[0.8, 0.4]], requires_grad=True)
    residual = torch.tensor([[-0.2, 0.1]], requires_grad=True)
    loss = baseline_no_harm_loss(
        {
            "score": baseline + residual,
            "baseline_score": baseline,
        },
        torch.tensor([0]),
    )
    loss.backward()
    torch.testing.assert_close(baseline.grad, torch.zeros_like(baseline))
    torch.testing.assert_close(residual.grad, torch.tensor([[-1.0, 1.0]]))


def test_direct_alignment_losses_reach_their_dedicated_heads() -> None:
    model = _token_model()
    batch = _metadata_batch()
    output = score_reference_image_episode(model, batch, return_aux=True)
    assert isinstance(output, dict)
    _, details = view_coverage_loss(output, batch)
    novelty_gradient = torch.autograd.grad(
        details["novelty_alignment_loss"],
        model.matcher.view_projection[1].weight,
        retain_graph=True,
    )[0]
    reliability_gradient = torch.autograd.grad(
        details["reliability_alignment_loss"],
        model.matcher.coverage_head[-1].weight,
    )[0]
    assert float(novelty_gradient.abs().sum()) > 0.0
    assert float(reliability_gradient.abs().sum()) > 0.0


def test_descriptor_only_attention_supervision_skips_token_only_targets() -> None:
    scores = torch.zeros(2, 2, requires_grad=True)
    attention_logits = torch.tensor(
        [
            [[0.2, 0.8], [0.7, 0.3]],
            [[0.6, 0.4], [0.1, 0.9]],
        ],
        requires_grad=True,
    )
    coverage, details = view_coverage_loss(
        {
            "score": scores,
            "attention": attention_logits.softmax(dim=-1),
        },
        _metadata_batch(),
    )
    torch.testing.assert_close(coverage, details["attention_alignment_loss"])
    for name in (
        "token_alignment_loss",
        "novelty_alignment_loss",
        "reliability_alignment_loss",
    ):
        torch.testing.assert_close(details[name], torch.tensor(0.0))


def test_nested_training_evaluates_one_two_and_three_reference_prefixes() -> None:
    model = _three_reference_token_model()
    batch = ReferenceImageBatch(
        query_images=torch.rand(2, 3, 4, 4),
        reference_images=torch.rand(2, 3, 3, 4, 4),
        reference_mask=torch.ones(2, 3, dtype=torch.bool),
        targets=torch.tensor([0, 1]),
        identity_names=("a", "b"),
    )
    seen_counts: list[int] = []
    seen_catalog_gates: list[torch.Tensor] = []
    original_forward = model.forward_encoded

    def recording_forward(
        query_descriptor,
        reference_descriptors,
        reference_mask=None,
        **kwargs,
    ):
        assert reference_mask is not None
        counts = reference_mask.sum(dim=1).unique()
        assert counts.numel() == 1
        seen_counts.append(int(counts.item()))
        catalog_gate = kwargs["catalog_confidence_gate"]
        assert not catalog_gate.requires_grad
        gate_matrix = catalog_gate.reshape(2, 2)
        torch.testing.assert_close(
            gate_matrix,
            gate_matrix[:, :1].expand_as(gate_matrix),
        )
        seen_catalog_gates.append(catalog_gate)
        return original_forward(
            query_descriptor,
            reference_descriptors,
            reference_mask,
            **kwargs,
        )

    model.forward_encoded = recording_forward
    loss, details = reference_episode_loss(
        model,
        batch,
        nested_reference_counts=True,
    )
    assert torch.isfinite(loss)
    assert seen_counts == [1, 2, 3]
    assert len(seen_catalog_gates) == 3
    torch.testing.assert_close(
        details["nested_reference_counts"],
        torch.tensor([1, 2, 3]),
    )


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
