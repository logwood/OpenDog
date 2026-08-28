# encoding: utf-8
"""Unit tests for deterministic cross-identity conflict evaluation."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_multimodal_conflict_robustness.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_multimodal_conflict_evaluation_tool",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
CONFLICT_EVALUATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONFLICT_EVALUATION
SPEC.loader.exec_module(CONFLICT_EVALUATION)

build_conflict_pairs = CONFLICT_EVALUATION.build_conflict_pairs
replace_nose_quality = CONFLICT_EVALUATION.replace_nose_quality
transition_summary = CONFLICT_EVALUATION.transition_summary
validate_development_manifest = CONFLICT_EVALUATION.validate_development_manifest


class MultimodalConflictEvaluationTest(unittest.TestCase):
    def test_pairs_use_next_different_identity_and_only_queries(self):
        identities = [
            "b",
            "b",
            "b",
            "b",
            "a",
            "a",
            "a",
            "a",
            "c",
            "c",
            "c",
            "c",
        ]
        pairs = build_conflict_pairs(identities, gallery_per_identity=2)
        self.assertEqual(
            [
                (
                    pair.query_index,
                    pair.donor_index,
                    pair.query_identity,
                    pair.donor_identity,
                )
                for pair in pairs
            ],
            [
                (6, 2, "a", "b"),
                (7, 3, "a", "b"),
                (2, 10, "b", "c"),
                (3, 11, "b", "c"),
                (10, 6, "c", "a"),
                (11, 7, "c", "a"),
            ],
        )
        self.assertTrue(
            all(pair.query_identity != pair.donor_identity for pair in pairs)
        )

    def test_only_nose_quality_columns_are_replaced(self):
        query = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        donor = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
        mixed = replace_nose_quality(query, donor)
        torch.testing.assert_close(
            mixed,
            torch.tensor([[10.0, 2.0, 3.0, 40.0, 50.0, 6.0]]),
        )
        torch.testing.assert_close(
            query,
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        )

    def test_transition_counts_regression_and_donor_hijack(self):
        clean = {
            "top1_accuracy": 0.5,
            "top1_correct": 1,
            "queries": [
                {
                    "query_index": 2,
                    "query_source_path": "q0.jpg",
                    "correct": True,
                    "true_identity_rank": 1,
                    "top5": [{"identity": "a", "score": 0.9}],
                },
                {
                    "query_index": 3,
                    "query_source_path": "q1.jpg",
                    "correct": False,
                    "true_identity_rank": 2,
                    "top5": [{"identity": "c", "score": 0.8}],
                },
            ],
        }
        corrupted = {
            "top1_accuracy": 0.0,
            "top1_correct": 0,
            "queries": [
                {
                    "query_index": 2,
                    "query_source_path": "q0.jpg",
                    "correct": False,
                    "true_identity_rank": 3,
                    "top5": [{"identity": "b", "score": 0.95}],
                },
                {
                    "query_index": 3,
                    "query_source_path": "q1.jpg",
                    "correct": False,
                    "true_identity_rank": 2,
                    "top5": [{"identity": "b", "score": 0.82}],
                },
            ],
        }
        pairs = build_conflict_pairs(
            ["a", "a", "a", "a", "b", "b", "b", "b"],
            gallery_per_identity=2,
        )[:2]
        summary = transition_summary(clean, corrupted, pairs)
        self.assertEqual(summary["regressed"], 1)
        self.assertEqual(summary["both_wrong"], 1)
        self.assertEqual(summary["donor_hijacked"], 2)
        self.assertEqual(summary["donor_hijack_rate"], 1.0)

    def test_locked_fresh_blind_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "fresh-blind"):
            validate_development_manifest(
                {
                    "protocol_split": "fresh_blind",
                    "usage_policy": ("single_final_evaluation_after_model_lock"),
                }
            )
        validate_development_manifest(
            {
                "protocol_split": "dev_validation",
                "usage_policy": "model_selection_only_no_gradient_updates",
            }
        )


if __name__ == "__main__":
    unittest.main()
