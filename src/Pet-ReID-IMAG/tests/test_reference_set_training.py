"""Tests for descriptor-episode sampling and held-out evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pet_id.reference_set_model import QueryConditionedReferenceMatcher
from pet_id.reference_set_training import (
    DescriptorTable,
    ReferenceEpisodeSampler,
    episode_retrieval_loss,
    evaluate_reference_matcher,
)


class ReferenceSetTrainingTest(unittest.TestCase):
    def make_cache(self, path: Path) -> None:
        rng = np.random.default_rng(3)
        # Four records per identity gives two references and two held-out queries.
        centers = rng.normal(size=(5, 8)).astype(np.float32)
        rows = []
        identities = []
        for index, center in enumerate(centers):
            for _ in range(4):
                rows.append(center + 0.04 * rng.normal(size=8))
                identities.append(f"pet-{index}")
        rows = np.asarray(rows, dtype=np.float32)
        np.savez_compressed(
            path,
            embedding=rows,
            identities=np.asarray(identities),
            source_paths=np.asarray([f"image-{i}.jpg" for i in range(len(rows))]),
        )

    def test_table_and_variable_size_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            self.make_cache(path)
            table = DescriptorTable.from_npz(path)
            self.assertEqual(table.num_identities, 5)
            self.assertEqual(table.descriptor_dim, 8)
            sampler = ReferenceEpisodeSampler(
                table,
                identities_per_batch=3,
                reference_count=2,
                queries_per_identity=1,
                max_references=2,
                variable_reference_count=True,
            )
            batch = sampler.sample(epoch=1, step=2)
            self.assertEqual(tuple(batch.queries.shape), (9, 8))
            self.assertEqual(tuple(batch.references.shape), (9, 2, 8))
            self.assertEqual(tuple(batch.reference_mask.shape), (9, 2))
            self.assertEqual(tuple(batch.targets.shape), (3,))
            self.assertTrue(torch.all(batch.reference_mask.any(dim=1)))
            self.assertTrue(torch.all(batch.targets < 3))

    def test_episode_loss_and_evaluation(self):
        scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        loss = episode_retrieval_loss(scores, torch.tensor([0, 1]))
        self.assertLess(float(loss), 0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            self.make_cache(path)
            table = DescriptorTable.from_npz(path)
            torch.manual_seed(9)
            model = QueryConditionedReferenceMatcher(
                descriptor_dim=8,
                hidden_dim=8,
                max_references=2,
                reference_top_k=2,
            )
            metrics = evaluate_reference_matcher(
                model, table, reference_count=2, max_references=2, batch_size=4
            )
            self.assertEqual(metrics["learned"]["query_records"], 10)
            self.assertEqual(metrics["baseline"]["gallery_identities"], 5)
            self.assertTrue(0.0 <= metrics["learned"]["top1_accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
