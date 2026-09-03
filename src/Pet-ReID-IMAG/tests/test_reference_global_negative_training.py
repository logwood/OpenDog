"""Focused regression tests for all-identity cached reference training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from pet_id.reference_aware_training import (
    AllIdentityReferenceEpisodeSampler,
    ReferenceImageEpisodeSampler,
    build_reference_spatial_feature_cache,
    cached_reference_episode_loss,
    hard_negative_margin_loss,
    load_reference_spatial_feature_cache,
    materialize_cached_reference_episode,
    materialize_reference_image_episode,
    save_reference_spatial_feature_cache,
    score_cached_reference_episode,
    score_reference_image_episode,
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


class _DescriptorEncoder(nn.Module):
    descriptor_dim = 8
    input_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.descriptor_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.float().mean(dim=(2, 3)))


class _ManifestDataset:
    def __init__(self, identities: tuple[str, ...] = ("a", "b", "c", "d")) -> None:
        self.training = False
        self.records: list[dict[str, object]] = []
        self.images: list[torch.Tensor] = []
        generator = torch.Generator().manual_seed(71)
        for identity_index, identity in enumerate(identities):
            for image_index in range(3):
                index = len(self.records)
                self.records.append(
                    {
                        "identity": identity,
                        "source_sha256": f"{index + 1:064x}",
                        "viewpoint_signals": [
                            float(identity_index == 0),
                            float(identity_index == 1),
                            float(image_index - 1),
                            1.0,
                        ],
                        "quality_signals": [0.7 + 0.1 * image_index] * 6,
                    }
                )
                self.images.append(torch.rand(3, 4, 4, generator=generator))

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        return {
            "rgb": self.images[int(index)].clone(),
            "record": record,
        }


def _model(encoder: nn.Module | None = None) -> TokenReferenceAwarePetReID:
    torch.manual_seed(73)
    matcher = TokenConditionedReferenceMatcher(
        descriptor_dim=8,
        token_dim=6,
        hidden_dim=5,
        max_references=2,
        reference_top_k=2,
    )
    return TokenReferenceAwarePetReID(
        encoder or _SpatialEncoder(),
        matcher,
        token_dim=6,
        token_grid=2,
    )


def _write_manifest(path: Path, dataset: _ManifestDataset) -> None:
    path.write_text(
        json.dumps({"records": dataset.records}, sort_keys=True),
        encoding="utf-8",
    )


class GlobalNegativeTrainingTest(unittest.TestCase):
    def test_all_identity_sampler_keeps_every_candidate_and_disjoint_positive(self):
        dataset = _ManifestDataset()
        sampler = AllIdentityReferenceEpisodeSampler(
            dataset,
            identities_per_batch=2,
            reference_count=2,
            queries_per_identity=1,
            max_references=2,
            variable_reference_count=False,
            seed=79,
        )
        episode = sampler.sample(epoch=2, step=3)
        self.assertEqual(episode.identity_names, ("a", "b", "c", "d"))
        self.assertEqual(len(episode.reference_indices), 4)
        self.assertEqual(len(episode.query_indices), 2)
        for query_index, target in zip(
            episode.query_indices, episode.targets.tolist()
        ):
            identity = str(dataset.records[query_index]["identity"])
            self.assertEqual(episode.identity_names[target], identity)
            self.assertNotIn(query_index, episode.reference_indices[target])

    def test_live_and_cached_spatial_scores_match(self):
        dataset = _ManifestDataset()
        model = _model()
        model.freeze_encoder()
        sampler = AllIdentityReferenceEpisodeSampler(
            dataset,
            identities_per_batch=2,
            reference_count=2,
            queries_per_identity=1,
            max_references=2,
            variable_reference_count=False,
            seed=83,
        )
        episode = sampler.sample(epoch=1, step=1)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _write_manifest(manifest, dataset)
            cache = build_reference_spatial_feature_cache(
                model,
                dataset,
                manifest_path=manifest,
                base_checkpoint_sha256="b" * 64,
                device="cpu",
                batch_size=4,
            )
            live_batch = materialize_reference_image_episode(dataset, episode)
            cached_batch = materialize_cached_reference_episode(
                cache, dataset, episode
            )
            model.eval()
            live_scores = score_reference_image_episode(model, live_batch)
            cached_scores = score_cached_reference_episode(model, cached_batch)
        self.assertEqual(tuple(cached_scores.shape), (2, 4))
        torch.testing.assert_close(cached_scores, live_scores)

    def test_cached_path_trains_projection_and_matcher_but_not_encoder(self):
        dataset = _ManifestDataset()
        model = _model()
        model.freeze_encoder()
        sampler = AllIdentityReferenceEpisodeSampler(
            dataset,
            identities_per_batch=2,
            reference_count=2,
            queries_per_identity=1,
            max_references=2,
            variable_reference_count=False,
            seed=89,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _write_manifest(manifest, dataset)
            cache = build_reference_spatial_feature_cache(
                model,
                dataset,
                manifest_path=manifest,
                base_checkpoint_sha256="c" * 64,
                device="cpu",
                batch_size=3,
            )
            batch = materialize_cached_reference_episode(
                cache,
                dataset,
                sampler.sample(epoch=0, step=1),
            )
            loss, _details = cached_reference_episode_loss(
                model,
                batch,
                hard_negative_weight=0.25,
                view_coverage_weight=0.2,
            )
            loss.backward()
        projection_gradient = model.image_encoder.feature_projection.weight.grad
        self.assertIsNotNone(projection_gradient)
        self.assertGreater(float(projection_gradient.abs().sum()), 0.0)
        matcher_gradients = [
            parameter.grad
            for parameter in model.matcher.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(matcher_gradients)
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in matcher_gradients),
            0.0,
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.image_encoder.encoder.parameters())
        )

    def test_hard_negative_sees_candidates_outside_small_query_episode(self):
        limited_loss, limited_margin = hard_negative_margin_loss(
            torch.tensor([[0.8, 0.1]]),
            torch.tensor([0]),
            margin=0.15,
        )
        global_loss, global_margin = hard_negative_margin_loss(
            torch.tensor([[0.8, 0.1, 0.79, 0.2]]),
            torch.tensor([0]),
            margin=0.15,
        )
        torch.testing.assert_close(limited_loss, torch.tensor(0.0))
        torch.testing.assert_close(limited_margin, torch.tensor(0.7))
        torch.testing.assert_close(global_loss, torch.tensor(0.14))
        torch.testing.assert_close(global_margin, torch.tensor(0.01))

    def test_cache_rejects_manifest_base_or_record_order_mismatch(self):
        dataset = _ManifestDataset()
        model = _model()
        model.freeze_encoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            cache_path = root / "features.pth"
            _write_manifest(manifest, dataset)
            cache = build_reference_spatial_feature_cache(
                model,
                dataset,
                manifest_path=manifest,
                base_checkpoint_sha256="d" * 64,
                device="cpu",
                batch_size=4,
            )
            save_reference_spatial_feature_cache(cache, cache_path)
            hook = model.image_encoder.feature_hook_name
            self.assertIsInstance(hook, str)
            loaded = load_reference_spatial_feature_cache(
                cache_path,
                dataset=dataset,
                manifest_path=manifest,
                base_checkpoint_sha256="d" * 64,
                feature_hook=str(hook),
                token_grid=2,
                descriptor_dim=8,
            )
            self.assertEqual(loaded.source_sha256s, cache.source_sha256s)
            with self.assertRaisesRegex(ValueError, "base checkpoint SHA-256"):
                load_reference_spatial_feature_cache(
                    cache_path,
                    dataset=dataset,
                    manifest_path=manifest,
                    base_checkpoint_sha256="e" * 64,
                    feature_hook=str(hook),
                    token_grid=2,
                    descriptor_dim=8,
                )
            manifest.write_text(
                json.dumps({"records": dataset.records, "changed": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                load_reference_spatial_feature_cache(
                    cache_path,
                    dataset=dataset,
                    manifest_path=manifest,
                    base_checkpoint_sha256="d" * 64,
                    feature_hook=str(hook),
                    token_grid=2,
                    descriptor_dim=8,
                )
            _write_manifest(manifest, dataset)
            reordered = _ManifestDataset()
            reordered.records = list(reversed(reordered.records))
            reordered.images = list(reversed(reordered.images))
            with self.assertRaisesRegex(ValueError, "record source_sha256 order"):
                load_reference_spatial_feature_cache(
                    cache_path,
                    dataset=reordered,
                    manifest_path=manifest,
                    base_checkpoint_sha256="d" * 64,
                    feature_hook=str(hook),
                    token_grid=2,
                    descriptor_dim=8,
                )

    def test_cache_rejects_descriptor_fallback(self):
        dataset = _ManifestDataset()
        model = _model(_DescriptorEncoder())
        model.freeze_encoder()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _write_manifest(manifest, dataset)
            with self.assertRaisesRegex(RuntimeError, "layer4 spatial feature hook"):
                build_reference_spatial_feature_cache(
                    model,
                    dataset,
                    manifest_path=manifest,
                    base_checkpoint_sha256="f" * 64,
                    device="cpu",
                )

    def test_original_small_image_episode_path_is_unchanged(self):
        dataset = _ManifestDataset()
        model = _model().eval()
        sampler = ReferenceImageEpisodeSampler(
            dataset,
            identities_per_batch=2,
            reference_count=2,
            queries_per_identity=1,
            max_references=2,
            variable_reference_count=False,
            seed=97,
        )
        episode = sampler.sample(epoch=0, step=1)
        batch = materialize_reference_image_episode(dataset, episode)
        scores = score_reference_image_episode(model, batch)
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(bool(torch.isfinite(scores).all()))


if __name__ == "__main__":
    unittest.main()
