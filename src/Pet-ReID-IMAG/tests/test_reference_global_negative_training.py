"""Focused regression tests for all-identity cached reference training."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from pet_id.reference_aware_training import (
    AllIdentityReferenceEpisodeSampler,
    ReferenceImageEpisodeSampler,
    build_full_catalog_validation_episodes,
    build_reference_spatial_feature_cache,
    cached_reference_episode_loss,
    evaluate_cached_reference_catalog,
    hard_negative_margin_loss,
    load_reference_spatial_feature_cache,
    materialize_cached_reference_episode,
    materialize_reference_image_episode,
    paired_retrieval_error_summary,
    reference_validation_checkpoint_eligible,
    reference_validation_is_better,
    reference_validation_selection_summary,
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


class _MultiScaleSpatialEncoder(nn.Module):
    """Mimic unified face crops concatenated as [scale * batch, ...]."""

    descriptor_dim = 8
    input_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer4 = nn.Identity()
        self.projection = nn.Linear(6, self.descriptor_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        features = self.backbone.layer4(torch.cat((images, images + 1.0), dim=0))
        first_scale, second_scale = features.chunk(2, dim=0)
        combined = torch.cat(
            (
                first_scale.mean(dim=(2, 3)),
                second_scale.mean(dim=(2, 3)),
            ),
            dim=1,
        )
        return self.projection(combined)


class _ManifestDataset:
    def __init__(
        self,
        identities: tuple[str, ...] = ("a", "b", "c", "d"),
        *,
        images_per_identity: int = 3,
    ) -> None:
        self.training = False
        self.records: list[dict[str, object]] = []
        self.images: list[torch.Tensor] = []
        generator = torch.Generator().manual_seed(71)
        for identity_index, identity in enumerate(identities):
            for image_index in range(images_per_identity):
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
    def test_multiscale_hook_rows_are_restored_to_each_source_image(self):
        encoder = _MultiScaleSpatialEncoder()
        model = _model(encoder)
        images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        descriptors, pooled = model.image_encoder.encode_cacheable_features(
            images,
            require_spatial=True,
        )
        expected_first = torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))
        expected_second = torch.nn.functional.adaptive_avg_pool2d(images + 1.0, (2, 2))
        expected = torch.cat(
            (
                expected_first.flatten(2).transpose(1, 2),
                expected_second.flatten(2).transpose(1, 2),
            ),
            dim=2,
        )
        self.assertEqual(tuple(descriptors.shape), (2, 8))
        self.assertEqual(tuple(pooled.shape), (2, 4, 6))
        torch.testing.assert_close(pooled, expected)

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
        for query_index, target in zip(episode.query_indices, episode.targets.tolist()):
            identity = str(dataset.records[query_index]["identity"])
            self.assertEqual(episode.identity_names[target], identity)
            self.assertNotIn(query_index, episode.reference_indices[target])

    def test_rotating_validation_folds_cover_each_manifest_row_once(self):
        dataset = _ManifestDataset(images_per_identity=4)
        episodes = build_full_catalog_validation_episodes(
            dataset,
            reference_count=3,
            queries_per_identity=1,
            query_identities_per_batch=2,
            fold_count=0,
            seed=81,
        )
        self.assertEqual(len(episodes), 8)
        self.assertEqual(
            {episode.validation_fold_index for episode in episodes},
            {0, 1, 2, 3},
        )
        query_counts = Counter(
            index for episode in episodes for index in episode.query_indices
        )
        self.assertEqual(query_counts, Counter({index: 1 for index in range(16)}))
        for fold_index in range(4):
            fold = [
                episode
                for episode in episodes
                if episode.validation_fold_index == fold_index
            ]
            self.assertEqual(len(fold), 2)
            self.assertEqual(fold[0].reference_indices, fold[1].reference_indices)
            for episode in fold:
                for query_index, target in zip(
                    episode.query_indices,
                    episode.targets.tolist(),
                ):
                    self.assertNotIn(
                        query_index,
                        episode.reference_indices[target],
                    )

    def test_paired_error_summary_counts_fixes_regressions_and_gate_state(self):
        candidate = torch.tensor(
            [
                [0.9, 0.1],
                [0.9, 0.1],
                [0.9, 0.1],
                [0.9, 0.1],
            ]
        )
        baseline = torch.tensor(
            [
                [0.1, 0.9],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.8, 0.2],
            ]
        )
        summary = paired_retrieval_error_summary(
            candidate,
            baseline,
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([1.0, 0.5, 0.0, 0.0]),
        )
        self.assertEqual(summary["candidate_only_correct"], 1)
        self.assertEqual(summary["baseline_only_correct"], 1)
        self.assertEqual(summary["both_correct"], 1)
        self.assertEqual(summary["both_wrong"], 1)
        self.assertEqual(summary["net_candidate_corrections"], 0)
        active = summary["catalog_confidence_gate"]["active"]
        self.assertEqual(active["candidate_only_correct"], 1)
        self.assertEqual(active["baseline_only_correct"], 1)
        closed = summary["catalog_confidence_gate"]["closed"]
        self.assertEqual(closed["both_correct"], 1)
        self.assertEqual(closed["both_wrong"], 1)
        self.assertEqual(summary["margin"]["harmed_query_count"], 2)

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
            cached_batch = materialize_cached_reference_episode(cache, dataset, episode)
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
            all(
                parameter.grad is None
                for parameter in model.image_encoder.encoder.parameters()
            )
        )

    def test_full_catalog_validation_uses_nested_fixed_references(self):
        dataset = _ManifestDataset()
        model = _model().eval()
        model.freeze_encoder()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _write_manifest(manifest, dataset)
            cache = build_reference_spatial_feature_cache(
                model,
                dataset,
                manifest_path=manifest,
                base_checkpoint_sha256="a" * 64,
                device="cpu",
                batch_size=4,
            )
            report = evaluate_cached_reference_catalog(
                model,
                cache,
                dataset,
                reference_count=2,
                queries_per_identity=1,
                query_identities_per_batch=2,
                seed=101,
            )
            incomplete_report = evaluate_cached_reference_catalog(
                model,
                cache,
                dataset,
                reference_count=2,
                queries_per_identity=1,
                query_identities_per_batch=2,
                validation_fold_count=1,
                seed=101,
            )
        self.assertEqual(
            report["protocol"],
            "full_identity_catalog_nested_references",
        )
        self.assertEqual(report["candidate_identities"], 4)
        self.assertEqual(report["query_records"], 12)
        self.assertEqual(report["query_batches"], 6)
        self.assertEqual(len(report["folds"]), 3)
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["coverage"]["manifest_records"], 12)
        self.assertEqual(report["coverage"]["records_covered_once"], 12)
        self.assertEqual(report["coverage"]["duplicate_query_occurrences"], 0)
        self.assertEqual(report["coverage"]["uncovered_records"], 0)
        self.assertEqual(set(report["reference_counts"]), {"1", "2"})
        singleton = report["reference_counts"]["1"]
        self.assertEqual(singleton["learned"]["query_records"], 12)
        self.assertTrue(singleton["exact_centroid_match"])
        self.assertEqual(singleton["delta"]["top1_accuracy"], 0.0)
        self.assertEqual(singleton["delta"]["mean_reciprocal_rank"], 0.0)
        self.assertEqual(singleton["no_harm"]["harmed_margin_query_count"], 0)
        multi_reference = report["reference_counts"]["2"]
        self.assertEqual(multi_reference["paired_top1"]["query_records"], 12)
        self.assertEqual(
            multi_reference["paired_top1"]["candidate_only_correct"]
            - multi_reference["paired_top1"]["baseline_only_correct"],
            multi_reference["learned"]["top1_correct"]
            - multi_reference["centroid"]["top1_correct"],
        )
        self.assertEqual(multi_reference["delta"]["top1_accuracy"], 0.0)
        self.assertEqual(multi_reference["delta"]["mean_positive_margin"], 0.0)
        for row in report["reference_counts"].values():
            gate = row["catalog_confidence_gate"]
            self.assertGreaterEqual(gate["mean"], 0.0)
            self.assertLessEqual(gate["mean"], 1.0)
            self.assertAlmostEqual(
                gate["closed_fraction"] + gate["active_fraction"],
                1.0,
            )
        self.assertTrue(report["selection"]["eligible_for_best_learned"])
        self.assertTrue(report["selection"]["full_query_coverage"])
        self.assertTrue(report["selection"]["singleton_exact_centroid_match"])
        self.assertTrue(report["selection"]["multi_reference_top1_noninferior"])
        self.assertEqual(
            report["selection"]["tie_policy"],
            "keep_earliest_within_tolerance",
        )
        self.assertFalse(incomplete_report["coverage"]["complete"])
        self.assertFalse(
            incomplete_report["selection"]["eligible_for_best_learned"]
        )

    def test_learned_selection_uses_aggregate_top1_and_float_tolerance(self):
        def row(
            count: int,
            *,
            learned_correct: int,
            centroid_correct: int,
            learned_reciprocal: float,
            centroid_reciprocal: float,
            learned_margin: float,
            centroid_margin: float,
            exact: bool = False,
        ) -> dict:
            query_records = 100
            return {
                "reference_count": count,
                "exact_centroid_match": exact,
                "learned": {
                    "query_records": query_records,
                    "top1_correct": learned_correct,
                    "mean_reciprocal_rank": learned_reciprocal,
                    "mean_positive_margin": learned_margin,
                },
                "centroid": {
                    "query_records": query_records,
                    "top1_correct": centroid_correct,
                    "mean_reciprocal_rank": centroid_reciprocal,
                    "mean_positive_margin": centroid_margin,
                },
            }

        singleton = row(
            1,
            learned_correct=93,
            centroid_correct=93,
            learned_reciprocal=0.95,
            centroid_reciprocal=0.95,
            learned_margin=0.2,
            centroid_margin=0.2,
            exact=True,
        )
        incumbent_summary = reference_validation_selection_summary(
            {
                "1": singleton,
                "2": row(
                    2,
                    learned_correct=96,
                    centroid_correct=96,
                    learned_reciprocal=0.97,
                    centroid_reciprocal=0.97,
                    learned_margin=0.20,
                    centroid_margin=0.20,
                ),
                "3": row(
                    3,
                    learned_correct=96,
                    centroid_correct=96,
                    learned_reciprocal=0.97,
                    centroid_reciprocal=0.97,
                    learned_margin=0.20,
                    centroid_margin=0.20,
                ),
            }
        )
        candidate_summary = reference_validation_selection_summary(
            {
                "1": singleton,
                "2": row(
                    2,
                    learned_correct=95,
                    centroid_correct=96,
                    learned_reciprocal=0.9699995,
                    centroid_reciprocal=0.97,
                    learned_margin=0.22,
                    centroid_margin=0.20,
                ),
                "3": row(
                    3,
                    learned_correct=97,
                    centroid_correct=96,
                    learned_reciprocal=0.9699995,
                    centroid_reciprocal=0.97,
                    learned_margin=0.22,
                    centroid_margin=0.20,
                ),
            }
        )
        incumbent = {
            "protocol": "full_identity_catalog_nested_references",
            "selection": incumbent_summary,
        }
        candidate = {
            "protocol": "full_identity_catalog_nested_references",
            "selection": candidate_summary,
        }

        self.assertTrue(reference_validation_checkpoint_eligible(candidate))
        self.assertFalse(
            candidate_summary["all_multi_reference_counts_top1_noninferior"]
        )
        self.assertTrue(reference_validation_is_better(candidate, incumbent))
        self.assertEqual(
            candidate_summary["per_query_margin_non_degradation"],
            "diagnostic_only",
        )

    def test_learned_selection_rejects_top1_regression_or_singleton_drift(self):
        def validation(*, singleton_exact: bool, learned_correct: int) -> dict:
            def summary_row(count: int, correct: int, exact: bool = False) -> dict:
                return {
                    "reference_count": count,
                    "exact_centroid_match": exact,
                    "learned": {
                        "query_records": 100,
                        "top1_correct": correct,
                        "mean_reciprocal_rank": 0.98,
                        "mean_positive_margin": 0.3,
                    },
                    "centroid": {
                        "query_records": 100,
                        "top1_correct": 96 if count > 1 else 93,
                        "mean_reciprocal_rank": 0.97,
                        "mean_positive_margin": 0.2,
                    },
                }

            selection = reference_validation_selection_summary(
                {
                    "1": summary_row(1, 93, singleton_exact),
                    "2": summary_row(2, learned_correct),
                    "3": summary_row(3, learned_correct),
                }
            )
            return {
                "protocol": "full_identity_catalog_nested_references",
                "selection": selection,
            }

        self.assertFalse(
            reference_validation_checkpoint_eligible(
                validation(singleton_exact=True, learned_correct=95)
            )
        )
        self.assertFalse(
            reference_validation_checkpoint_eligible(
                validation(singleton_exact=False, learned_correct=97)
            )
        )

    def test_incomplete_query_coverage_cannot_select_a_checkpoint(self):
        def row(count: int, *, exact: bool = False) -> dict:
            return {
                "reference_count": count,
                "exact_centroid_match": exact,
                "learned": {
                    "query_records": 100,
                    "top1_correct": 98,
                    "mean_reciprocal_rank": 0.99,
                    "mean_positive_margin": 0.3,
                },
                "centroid": {
                    "query_records": 100,
                    "top1_correct": 96,
                    "mean_reciprocal_rank": 0.97,
                    "mean_positive_margin": 0.2,
                },
            }

        selection = reference_validation_selection_summary(
            {"1": row(1, exact=True), "2": row(2)},
            coverage={"complete": False},
        )
        report = {
            "protocol": "full_identity_catalog_nested_references",
            "selection": selection,
        }
        self.assertFalse(selection["full_query_coverage"])
        self.assertFalse(reference_validation_checkpoint_eligible(report))

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
