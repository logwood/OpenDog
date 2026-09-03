"""Tests for the end-to-end query/reference image-set model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from pet_id.reference_aware_model import (
    ReferenceAwareDescriptorScorer,
    ReferenceAwarePetReID,
    ReferenceAwarePetReIDExport,
    build_reference_aware_model_from_checkpoint,
    save_reference_aware_model,
)
from pet_id.reference_aware_training import (
    ReferenceImageEpisodeSampler,
    materialize_reference_image_episode,
    reference_episode_loss,
    score_reference_image_episode,
)
from pet_id.reference_set_model import QueryConditionedReferenceMatcher


class TinyImageEncoder(nn.Module):
    descriptor_dim = 8
    input_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.descriptor_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.float().mean(dim=(2, 3))
        return self.projection(pooled)

    def configuration(self):
        return {"type": "tiny-test-encoder", "descriptor_dim": self.descriptor_dim}


class TinyDataset:
    def __init__(self) -> None:
        self.records = [
            {"identity": identity}
            for identity in ("a", "a", "a", "b", "b", "b", "c", "c", "c")
        ]

    def __getitem__(self, index: int):
        image = torch.full((3, 4, 4), float(index + 1))
        return {"rgb": image}


class ReferenceAwareModelTest(unittest.TestCase):
    def make_model(self) -> ReferenceAwarePetReID:
        torch.manual_seed(31)
        matcher = QueryConditionedReferenceMatcher(
            descriptor_dim=8,
            hidden_dim=6,
            max_references=3,
            reference_top_k=2,
        )
        return ReferenceAwarePetReID(TinyImageEncoder(), matcher)

    def test_image_set_forward_and_mask(self):
        model = self.make_model()
        query = torch.rand(2, 3, 4, 4)
        references = torch.rand(2, 3, 3, 4, 4)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        output = model(query, references, mask, return_aux=True)
        self.assertEqual(tuple(output["score"].shape), (2,))
        self.assertEqual(tuple(output["query_descriptor"].shape), (2, 8))
        self.assertEqual(tuple(output["reference_descriptors"].shape), (2, 3, 8))

        changed = references.clone()
        changed[0, 2] = torch.randn_like(changed[0, 2]) * 100.0
        changed_output = model(query, changed, mask)
        torch.testing.assert_close(changed_output, output["score"])

    def test_gradients_reach_encoder_and_matcher(self):
        model = self.make_model()
        query = torch.rand(2, 3, 4, 4)
        references = torch.rand(2, 2, 3, 4, 4)
        loss = model(query, references).square().mean()
        loss.backward()
        self.assertIsNotNone(model.image_encoder.projection.weight.grad)
        self.assertTrue(
            torch.isfinite(model.image_encoder.projection.weight.grad).all()
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.matcher.parameters())
        )

    def test_descriptor_path_matches_image_path(self):
        model = self.make_model().eval()
        query = torch.rand(2, 3, 4, 4)
        references = torch.rand(2, 2, 3, 4, 4)
        mask = torch.tensor([[True, False], [True, True]])
        image_score = model(query, references, mask)
        descriptor_score = model.score_descriptors(
            model.encode_images(query), model.encode_reference_images(references), mask
        )
        torch.testing.assert_close(image_score, descriptor_score)

    def test_checkpoint_round_trip(self):
        model = self.make_model().eval()
        query = torch.rand(1, 3, 4, 4)
        references = torch.rand(1, 2, 3, 4, 4)
        expected = model(query, references)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference-aware.pth"
            save_reference_aware_model(
                model,
                path,
                base_encoder_checkpoint="base.pth",
                encoder_fingerprint="encoder-test",
                optimizer_state={"step": 3},
            )
            restored, payload = build_reference_aware_model_from_checkpoint(
                path, TinyImageEncoder()
            )
            torch.testing.assert_close(restored(query, references), expected)
            self.assertEqual(payload["format"], "reference-aware-pet-reid")
            self.assertEqual(payload["encoder_fingerprint"], "encoder-test")
            self.assertEqual(payload["optimizer"]["step"], 3)

    def test_export_boundary_is_tensor_only(self):
        model = self.make_model().eval()
        wrapper = ReferenceAwarePetReIDExport(model)
        output = wrapper(
            torch.rand(2, 3, 4, 4),
            torch.rand(2, 2, 3, 4, 4),
            torch.ones(2, 2, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (2,))

    def test_export_path_matches_eager_path(self):
        model = self.make_model().eval()
        query = torch.rand(2, 3, 4, 4)
        references = torch.rand(2, 2, 3, 4, 4)
        mask = torch.tensor([[True, False], [True, True]])
        eager = model(query, references, mask)
        exported = model.forward_export(query, references, mask)
        torch.testing.assert_close(exported, eager)

    def test_joint_model_exposes_descriptor_gallery_scorer(self):
        model = self.make_model().eval()
        scorer = model.descriptor_scorer()
        self.assertIsInstance(scorer, ReferenceAwareDescriptorScorer)
        query = torch.rand(3, 4, 4)
        references = torch.rand(2, 3, 4, 4)
        query_descriptor = model.encode_images(query[None])[0].detach().numpy()
        reference_descriptors = model.encode_images(references).detach().numpy()
        score, detail = scorer.score(query_descriptor, reference_descriptors)
        self.assertTrue(torch.isfinite(torch.tensor(score)))
        self.assertEqual(detail["reference_count"], 2)

    def test_image_episode_sampler_and_end_to_end_loss(self):
        dataset = TinyDataset()
        sampler = ReferenceImageEpisodeSampler(
            dataset,
            identities_per_batch=2,
            reference_count=2,
            queries_per_identity=1,
            max_references=2,
            variable_reference_count=False,
            seed=4,
        )
        episode = sampler.sample(epoch=1, step=2)
        batch = materialize_reference_image_episode(dataset, episode)
        self.assertEqual(tuple(batch.query_images.shape), (2, 3, 4, 4))
        self.assertEqual(tuple(batch.reference_images.shape), (2, 2, 3, 4, 4))
        model = self.make_model()
        scores = score_reference_image_episode(model, batch)
        self.assertEqual(tuple(scores.shape), (2, 2))
        loss, details = reference_episode_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("retrieval_loss", details)


if __name__ == "__main__":
    unittest.main()
