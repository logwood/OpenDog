"""Tests for token-level query/reference interaction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from pet_id.reference_aware_training import (
    ReferenceImageBatch,
    score_reference_image_episode,
)
from pet_id.reference_token_model import (
    TokenConditionedReferenceMatcher,
    TokenReferenceAwarePetReID,
    TokenReferenceAwarePetReIDExport,
    build_token_reference_aware_model_from_checkpoint,
    save_token_reference_aware_model,
)


class TinySpatialEncoder(nn.Module):
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

    def configuration(self):
        return {"type": "tiny-spatial", "descriptor_dim": self.descriptor_dim}


class TinyDescriptorEncoder(nn.Module):
    descriptor_dim = 8
    input_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.descriptor_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.float().mean(dim=(2, 3)))


def make_model(encoder: nn.Module | None = None) -> TokenReferenceAwarePetReID:
    torch.manual_seed(19)
    matcher = TokenConditionedReferenceMatcher(
        descriptor_dim=8,
        token_dim=6,
        hidden_dim=5,
        max_references=3,
        reference_top_k=2,
        coverage_weight=0.4,
    )
    return TokenReferenceAwarePetReID(
        encoder or TinySpatialEncoder(), matcher, token_dim=6, token_grid=2
    )


class ReferenceTokenModelTest(unittest.TestCase):
    @staticmethod
    def _matcher_inputs():
        generator = torch.Generator().manual_seed(211)
        query = torch.rand(2, 8, generator=generator)
        references = torch.rand(2, 3, 8, generator=generator)
        query_tokens = torch.rand(2, 4, 6, generator=generator)
        reference_tokens = torch.rand(2, 3, 4, 6, generator=generator)
        return query, references, query_tokens, reference_tokens

    def test_spatial_tokens_and_coverage_diagnostics(self):
        model = make_model()
        query = torch.rand(2, 3, 4, 4)
        references = torch.rand(2, 3, 3, 4, 4)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        output = model(query, references, mask, return_aux=True)
        self.assertEqual(tuple(output["query_tokens"].shape), (2, 4, 6))
        self.assertEqual(tuple(output["reference_tokens"].shape), (2, 3, 4, 6))
        self.assertEqual(tuple(output["token_similarity"].shape), (2, 3, 4, 4))
        self.assertEqual(tuple(output["coverage_gate"].shape), (2, 3))
        self.assertTrue(torch.isfinite(output["score"]).all())
        self.assertTrue(torch.isfinite(output["novelty"]).all())
        self.assertTrue(torch.allclose(output["attention"][0, 2], torch.zeros(())))

    def test_descriptor_only_encoder_has_safe_token_fallback(self):
        model = make_model(TinyDescriptorEncoder())
        output = model(
            torch.rand(1, 3, 4, 4),
            torch.rand(1, 2, 3, 4, 4),
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (1,))
        self.assertIsNone(model.image_encoder._hook_target_name)

    def test_episode_helper_expands_tokens_before_matching(self):
        model = make_model()
        batch = ReferenceImageBatch(
            query_images=torch.rand(2, 3, 4, 4),
            reference_images=torch.rand(2, 2, 3, 4, 4),
            reference_mask=torch.ones(2, 2, dtype=torch.bool),
            targets=torch.tensor([0, 1]),
            identity_names=("a", "b"),
        )
        scores = score_reference_image_episode(model, batch)
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores).all())

    def test_gradients_and_checkpoint_round_trip(self):
        model = make_model()
        query = torch.rand(1, 3, 4, 4)
        references = torch.rand(1, 2, 3, 4, 4)
        expected = model(query, references)
        expected.square().mean().backward()
        self.assertIsNotNone(model.matcher.pair_head[-1].weight.grad)
        self.assertIsNotNone(model.image_encoder.encoder.projection.weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token-model.pth"
            save_token_reference_aware_model(model, path, encoder_fingerprint="test")
            restored, payload = build_token_reference_aware_model_from_checkpoint(
                path, TinySpatialEncoder()
            )
            torch.testing.assert_close(restored(query, references), expected)
            self.assertEqual(payload["format"], "reference-token-aware-pet-reid")
            self.assertEqual(
                payload["model_config"]["matcher"]["strategy"],
                "evidence_gated_spatial_delta",
            )

    def test_single_reference_is_exact_centroid_for_arbitrary_score_head(self):
        matcher = make_model().matcher
        with torch.no_grad():
            for parameter in matcher.score_head.parameters():
                parameter.uniform_(-3.0, 3.0)
        query, references, query_tokens, reference_tokens = self._matcher_inputs()
        mask = torch.tensor([[True, False, False], [True, False, False]])
        output = matcher(
            query,
            references,
            query_tokens,
            reference_tokens,
            mask,
            return_aux=True,
        )
        torch.testing.assert_close(
            output["score"],
            output["centroid_score"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            output["residual"],
            torch.zeros_like(output["residual"]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            output["residual_gate"],
            torch.zeros_like(output["residual_gate"]),
            rtol=0.0,
            atol=0.0,
        )

    def test_duplicate_reference_tokens_disable_the_residual(self):
        matcher = make_model().matcher
        with torch.no_grad():
            for parameter in matcher.score_head.parameters():
                parameter.uniform_(-2.0, 2.0)
        query, references, query_tokens, reference_tokens = self._matcher_inputs()
        reference_tokens[:, 1] = reference_tokens[:, 0]
        reference_tokens[:, 2] = reference_tokens[:, 0]
        output = matcher(
            query,
            references,
            query_tokens,
            reference_tokens,
            torch.ones(2, 3, dtype=torch.bool),
            return_aux=True,
        )
        torch.testing.assert_close(
            output["novelty"],
            torch.zeros_like(output["novelty"]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            output["score"],
            output["centroid_score"],
            rtol=0.0,
            atol=0.0,
        )

    def test_descriptor_changes_do_not_route_reference_attention(self):
        matcher = make_model().matcher.eval()
        query, references, query_tokens, reference_tokens = self._matcher_inputs()
        mask = torch.ones(2, 3, dtype=torch.bool)
        first = matcher(
            query,
            references,
            query_tokens,
            reference_tokens,
            mask,
            return_aux=True,
        )
        second = matcher(
            torch.flip(query, dims=(1,)),
            references,
            query_tokens,
            reference_tokens,
            mask,
            return_aux=True,
        )
        torch.testing.assert_close(
            first["attention"],
            second["attention"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            first["raw_residual"],
            second["raw_residual"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            first["residual_gate"],
            second["residual_gate"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            torch.allclose(first["centroid_score"], second["centroid_score"])
        )

    def test_old_matcher_checkpoint_is_rejected_explicitly(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-token-model.pth"
            save_token_reference_aware_model(model, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["model_config"]["matcher"].pop("strategy")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "retired matcher strategy"):
                build_token_reference_aware_model_from_checkpoint(
                    path,
                    TinySpatialEncoder(),
                )

    def test_export_boundary(self):
        model = make_model().eval()
        wrapper = TokenReferenceAwarePetReIDExport(model)
        score = wrapper(
            torch.rand(2, 3, 4, 4),
            torch.rand(2, 2, 3, 4, 4),
            torch.ones(2, 2, dtype=torch.bool),
        )
        self.assertEqual(tuple(score.shape), (2,))

    def test_onnx_export_uses_fixed_token_grid(self):
        model = make_model().eval()
        wrapper = TokenReferenceAwarePetReIDExport(model)
        query = torch.rand(1, 3, 4, 4)
        references = torch.rand(1, 2, 3, 4, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        # Materialize LazyLinear before tracing.
        _ = wrapper(query, references, mask)
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "token.onnx"
            torch.onnx.export(
                wrapper,
                (query, references, mask),
                graph,
                input_names=["query_rgb", "reference_rgb", "reference_mask"],
                output_names=["score"],
                dynamic_axes={
                    "query_rgb": {0: "batch"},
                    "reference_rgb": {0: "batch"},
                    "reference_mask": {0: "batch"},
                    "score": {0: "batch"},
                },
                opset_version=20,
                dynamo=False,
            )
            self.assertGreater(graph.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
