"""Tests for the learnable query-conditioned reference-set matcher."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pet_id.reference_set_model import (
    QueryConditionedReferenceMatcher,
    ReferenceSetMatcherRuntime,
    build_reference_set_matcher_from_checkpoint,
    save_reference_set_matcher,
)


class ReferenceSetModelTest(unittest.TestCase):
    def make_model(self) -> QueryConditionedReferenceMatcher:
        torch.manual_seed(7)
        return QueryConditionedReferenceMatcher(
            descriptor_dim=8,
            hidden_dim=6,
            max_references=4,
            reference_top_k=2,
            reference_score_weight=0.4,
        )

    def make_inputs(self):
        torch.manual_seed(11)
        query = F.normalize(torch.randn(2, 8), dim=1)
        references = F.normalize(torch.randn(2, 4, 8), dim=2)
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        return query, references, mask

    def test_query_conditioned_attention_respects_mask(self):
        model = self.make_model().eval()
        query, references, mask = self.make_inputs()
        output = model(query, references, mask, return_aux=True)
        self.assertEqual(tuple(output["score"].shape), (2,))
        self.assertEqual(tuple(output["attention"].shape), (2, 4))
        torch.testing.assert_close(
            output["attention"].sum(dim=1), torch.ones(2), atol=1e-6, rtol=0
        )
        self.assertTrue(torch.equal(output["attention"][0, 2:] == 0, torch.ones(2, dtype=torch.bool)))
        self.assertTrue(torch.equal(output["attention"][1, 3:] == 0, torch.ones(1, dtype=torch.bool)))

    def test_zero_initialized_residual_matches_centroid_top_k_baseline(self):
        model = self.make_model().eval()
        query, references, mask = self.make_inputs()
        output = model(query, references, mask, return_aux=True)
        self.assertTrue(torch.equal(output["residual"], torch.zeros_like(output["residual"])))
        query = F.normalize(query, dim=1)
        references = F.normalize(references, dim=2)
        expected = []
        for row in range(query.shape[0]):
            rows = references[row][mask[row]]
            centroid = F.normalize(rows.mean(dim=0), dim=0)
            centroid_score = query[row] @ centroid
            similarities = rows @ query[row]
            top = similarities.topk(min(2, similarities.numel())).values.mean()
            expected.append(0.6 * centroid_score + 0.4 * top)
        torch.testing.assert_close(output["score"], torch.stack(expected), atol=1e-6, rtol=0)
        torch.testing.assert_close(output["score"], output["baseline_score"], atol=1e-6, rtol=0)

    def test_head_and_attention_are_differentiable(self):
        model = self.make_model()
        query, references, mask = self.make_inputs()
        loss = model(query, references, mask).square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_empty_set_is_rejected(self):
        model = self.make_model()
        query = torch.ones(1, 8)
        references = torch.ones(1, 4, 8)
        with self.assertRaisesRegex(ValueError, "at least one reference"):
            model(query, references, torch.zeros(1, 4, dtype=torch.bool))

    def test_runtime_and_checkpoint_round_trip(self):
        model = self.make_model().eval()
        query, references, mask = self.make_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matcher.pth"
            save_reference_set_matcher(
                model, path, encoder_fingerprint="encoder-hash", training={"step": 3}
            )
            loaded, payload = build_reference_set_matcher_from_checkpoint(path)
            self.assertEqual(payload["format"], "reference-set-matcher")
            torch.testing.assert_close(
                loaded(query, references, mask), model(query, references, mask)
            )
            runtime = ReferenceSetMatcherRuntime.from_checkpoint(path)
            scores, details = runtime.score_many(
                query[0].numpy(),
                [references[0, :2].numpy(), references[1, :3].numpy()],
            )
            self.assertEqual(scores.shape, (2,))
            self.assertEqual(len(details), 2)
            self.assertEqual(details[0]["reference_count"], 2)
            self.assertTrue(np.isfinite(scores).all())

    def test_runtime_chunks_identity_with_more_references_than_model_width(self):
        model = QueryConditionedReferenceMatcher(
            descriptor_dim=8,
            hidden_dim=6,
            max_references=2,
            reference_top_k=2,
        ).eval()
        query = F.normalize(torch.randn(8), dim=0).numpy()
        references = F.normalize(torch.randn(5, 8), dim=1).numpy()
        runtime = ReferenceSetMatcherRuntime(model)
        scores, details = runtime.score_many(query, [references])
        self.assertEqual(scores.shape, (1,))
        self.assertEqual(details[0]["reference_count"], 5)
        self.assertEqual(details[0]["chunk_count"], 3)
        self.assertEqual(len(details[0]["chunk_scores"]), 3)
        self.assertEqual(details[0]["chunk_aggregation"], "top2_mean")


if __name__ == "__main__":
    unittest.main()
