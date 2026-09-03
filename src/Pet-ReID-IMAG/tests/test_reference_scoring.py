"""Unit tests for robust scoring against an identity's reference set."""

from __future__ import annotations

import unittest

import numpy as np

from pet_id.reference_scoring import (
    CENTROID_SCORING,
    DEFAULT_REFERENCE_SCORE_WEIGHT,
    DEFAULT_REFERENCE_TOP_K,
    MAX_REFERENCE_TOP_K,
    REFERENCE_SET_SCORING,
    score_gallery,
    score_identity,
    validate_reference_score_weight,
    validate_reference_top_k,
    validate_scoring_mode,
)


class ReferenceScoringTest(unittest.TestCase):
    def test_centroid_mode_is_backward_compatible(self):
        score, detail = score_identity(
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            references=np.asarray([[1.0, 0.0]]),
            scoring_mode=CENTROID_SCORING,
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(detail["mode"], CENTROID_SCORING)
        self.assertEqual(detail["score"], 0.0)
        self.assertIsNone(detail["reference_score"])

    def test_reference_set_blends_centroid_and_top_k_mean(self):
        # The strongest reference is a perfect match, but the second strongest
        # is orthogonal.  A top-2 mean therefore contributes 0.5 rather than
        # letting the raw maximum of 1.0 decide the identity.
        score, detail = score_identity(
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            references=np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            scoring_mode=REFERENCE_SET_SCORING,
            reference_top_k=2,
            reference_score_weight=0.5,
        )
        self.assertAlmostEqual(score, 0.25)
        self.assertAlmostEqual(detail["centroid_score"], 0.0)
        self.assertAlmostEqual(detail["reference_best"], 1.0)
        self.assertAlmostEqual(detail["reference_score"], 0.5)
        self.assertEqual(detail["reference_top_k"], 2)
        self.assertEqual(detail["reference_count"], 3)

    def test_top_k_is_clamped_to_available_references(self):
        score, detail = score_identity(
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            references=np.asarray([[1.0, 0.0]]),
            scoring_mode=REFERENCE_SET_SCORING,
            reference_top_k=MAX_REFERENCE_TOP_K,
            reference_score_weight=1.0,
        )
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(detail["reference_top_k"], 1)

    def test_gallery_returns_scores_in_identity_order_with_diagnostics(self):
        scores, details = score_gallery(
            np.asarray([1.0, 0.0]),
            [
                {
                    "pet_id": "dog-a",
                    "prototype": np.asarray([0.0, 1.0]),
                    "reference_features": np.asarray([[1.0, 0.0]]),
                },
                {
                    "pet_id": "dog-b",
                    "prototype": np.asarray([1.0, 0.0]),
                    "reference_features": np.asarray([[-1.0, 0.0]]),
                },
            ],
            scoring_mode=REFERENCE_SET_SCORING,
            reference_top_k=1,
            reference_score_weight=0.5,
        )
        self.assertEqual(scores.shape, (2,))
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertEqual(set(details), {"dog-a", "dog-b"})

    def test_reference_set_can_recover_a_view_specific_match(self):
        # Under centroid scoring dog-b has the stronger average direction. A
        # query from dog-a's first viewpoint is recovered when the individual
        # reference evidence is enabled.
        prototypes = [
            {
                "pet_id": "dog-a",
                "prototype": np.asarray([0.8, 0.98]),
                "reference_features": np.asarray([[1.0, 0.0], [-0.2, 0.98]]),
            },
            {
                "pet_id": "dog-b",
                "prototype": np.asarray([0.7, 0.714]),
                "reference_features": np.asarray([[0.7, 0.714], [0.7, 0.714]]),
            },
        ]
        centroid_scores, _ = score_gallery(
            np.asarray([1.0, 0.0]), prototypes, scoring_mode=CENTROID_SCORING
        )
        reference_scores, _ = score_gallery(
            np.asarray([1.0, 0.0]),
            prototypes,
            scoring_mode=REFERENCE_SET_SCORING,
            reference_top_k=1,
            reference_score_weight=1.0,
        )
        self.assertEqual(int(np.argmax(centroid_scores)), 1)
        self.assertEqual(int(np.argmax(reference_scores)), 0)

    def test_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_scoring_mode("unknown")
        with self.assertRaises(ValueError):
            validate_reference_top_k(0)
        with self.assertRaises(ValueError):
            validate_reference_top_k(1.5)
        with self.assertRaises(ValueError):
            validate_reference_top_k(MAX_REFERENCE_TOP_K + 1)
        with self.assertRaises(ValueError):
            validate_reference_top_k(float("inf"))
        with self.assertRaises(ValueError):
            validate_reference_score_weight(float("nan"))
        with self.assertRaises(ValueError):
            validate_reference_score_weight(1.01)

    def test_defaults_are_stable_and_semantic(self):
        self.assertEqual(DEFAULT_REFERENCE_TOP_K, 3)
        self.assertAlmostEqual(DEFAULT_REFERENCE_SCORE_WEIGHT, 0.4)


if __name__ == "__main__":
    unittest.main()
