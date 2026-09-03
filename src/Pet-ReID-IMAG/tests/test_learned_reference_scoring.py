"""Integration tests for the learned reference-set scoring mode."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pet_id.reference_scoring import (
    LEARNED_REFERENCE_SET_SCORING,
    score_gallery,
)
from pet_id.reference_set_model import (
    QueryConditionedReferenceMatcher,
    ReferenceSetMatcherRuntime,
    save_reference_set_matcher,
)


class LearnedReferenceScoringTest(unittest.TestCase):
    def test_runtime_scores_gallery_in_one_call(self):
        torch.manual_seed(4)
        model = QueryConditionedReferenceMatcher(
            descriptor_dim=6,
            hidden_dim=8,
            max_references=3,
            reference_top_k=2,
        )
        query = F.normalize(torch.randn(6), dim=0).numpy()
        refs_a = F.normalize(torch.randn(2, 6), dim=1).numpy()
        refs_b = F.normalize(torch.randn(3, 6), dim=1).numpy()
        prototypes = [
            {"pet_id": "a", "prototype": refs_a.mean(axis=0), "reference_features": refs_a},
            {"pet_id": "b", "prototype": refs_b.mean(axis=0), "reference_features": refs_b},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matcher.pth"
            save_reference_set_matcher(model, path)
            runtime = ReferenceSetMatcherRuntime.from_checkpoint(path)
            scores, details = score_gallery(
                query,
                prototypes,
                scoring_mode=LEARNED_REFERENCE_SET_SCORING,
                learned_scorer=runtime,
            )
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(set(details), {"a", "b"})
        self.assertEqual(details["a"]["mode"], LEARNED_REFERENCE_SET_SCORING)
        self.assertTrue(np.isfinite(scores).all())

    def test_mode_requires_a_trained_scorer(self):
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        prototypes = [
            {
                "pet_id": "a",
                "prototype": query,
                "reference_features": np.stack((query, query)),
            }
        ]
        with self.assertRaisesRegex(ValueError, "trained reference scorer"):
            score_gallery(
                query,
                prototypes,
                scoring_mode=LEARNED_REFERENCE_SET_SCORING,
            )


if __name__ == "__main__":
    unittest.main()
