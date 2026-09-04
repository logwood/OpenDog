"""Tests for two-stage identity retrieval with per-reference evidence."""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from pet_id.gallery import sha256_file
from pet_id.identity_set_reranker import (
    IdentityReferenceSet,
    IdentitySetReranker,
    ModelReferenceEvidenceEncoder,
    QueryConditionedReferenceSelector,
    QueryEvidence,
    ReferenceEvidence,
)
from pet_id.gallery_service import (
    EncodedPetImage,
    EnrollmentRecord,
    GalleryModelMismatch,
    InvalidPetImage,
    PetGalleryStore,
    PetIdentificationService,
    UploadPayload,
    validate_upload,
)
from pet_id.reference_token_model import TokenConditionedReferenceMatcher


BASE_TOKENS = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def reference(
    reference_id: str,
    descriptor: tuple[float, float],
    *,
    viewpoint: str | None = None,
    tokens: np.ndarray | None = None,
) -> ReferenceEvidence:
    return ReferenceEvidence(
        reference_id=reference_id,
        descriptor=np.asarray(descriptor, dtype=np.float32),
        tokens=BASE_TOKENS.copy() if tokens is None else np.asarray(tokens),
        viewpoint=viewpoint,
    )


def identity(identity_id: str, *references: ReferenceEvidence) -> IdentityReferenceSet:
    return IdentityReferenceSet(identity_id=identity_id, references=references)


class RecordingSelector:
    descriptor_dim = 2
    token_dim = 2
    max_references = 4

    def __init__(
        self,
        scores: list[float] | None = None,
        attention: list[list[float]] | None = None,
    ) -> None:
        self.scores = scores
        self.attention = attention
        self.calls: list[dict[str, np.ndarray]] = []

    def select(
        self,
        query_descriptor,
        query_tokens,
        reference_descriptors,
        reference_tokens,
        reference_mask,
    ):
        mask = np.asarray(reference_mask, dtype=np.bool_)
        self.calls.append(
            {
                "query_descriptor": np.asarray(query_descriptor).copy(),
                "query_tokens": np.asarray(query_tokens).copy(),
                "reference_descriptors": np.asarray(reference_descriptors).copy(),
                "reference_tokens": np.asarray(reference_tokens).copy(),
                "reference_mask": mask.copy(),
            }
        )
        batch, width = mask.shape
        scores = np.asarray(
            self.scores if self.scores is not None else np.arange(batch, 0, -1),
            dtype=np.float32,
        )[:batch]
        if self.attention is None:
            weights = mask.astype(np.float32)
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            weights = np.zeros((batch, width), dtype=np.float32)
            for row, values in enumerate(self.attention[:batch]):
                weights[row, : min(width, len(values))] = values[:width]
        similarities = np.einsum(
            "d,bkd->bk", np.asarray(query_descriptor), reference_descriptors
        )
        return {
            "score": scores,
            "attention": weights,
            "similarities": similarities,
            "token_scores": np.full((batch, width), 0.25, dtype=np.float32),
            "novelty": mask.astype(np.float32) * 0.75,
            "coverage_gate": mask.astype(np.float32) * 0.5,
            "baseline_score": scores - 0.05,
            "residual": np.full(batch, 0.05, dtype=np.float32),
            "coverage_score": np.full(batch, 0.6, dtype=np.float32),
            "duplicate_score": np.full(batch, 0.4, dtype=np.float32),
            "centroid_score": scores - 0.1,
            "top_k_score": scores,
        }


class IdentitySetRerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = QueryEvidence(
            descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
            tokens=BASE_TOKENS.copy(),
        )

    def test_coarse_recall_keeps_one_strong_view_instead_of_averaging(self):
        selector = RecordingSelector()
        reranker = IdentitySetReranker(selector, candidate_count=1)
        result = reranker.rerank(
            self.query,
            [
                identity(
                    "varied",
                    reference("varied-front", (1.0, 0.0), viewpoint="front"),
                    reference("varied-away", (-1.0, 0.0), viewpoint="rear"),
                ),
                identity(
                    "uniform",
                    reference("uniform-a", (0.8, 0.6), viewpoint="front"),
                    reference("uniform-b", (0.8, -0.6), viewpoint="side"),
                ),
            ],
        )

        self.assertEqual(result["coarse_ranking"][0]["identity_id"], "varied")
        self.assertEqual(
            result["coarse_ranking"][0]["support_reference_ids"], ["varied-front"]
        )
        self.assertAlmostEqual(result["coarse_ranking"][0]["score"], 1.0)
        self.assertEqual(result["reranked_identities"], 1)
        self.assertEqual(selector.calls[0]["reference_descriptors"].shape[0], 1)

    def test_token_rerank_changes_order_and_preserves_reference_alignment(self):
        selector = RecordingSelector(
            scores=[0.1, 0.9],
            attention=[[0.2, 0.8], [1.0, 0.0]],
        )
        reranker = IdentitySetReranker(selector, candidate_count=2)
        result = reranker.rerank(
            self.query,
            [
                identity(
                    "coarse-first",
                    reference("first-a", (1.0, 0.0), viewpoint="front"),
                    reference("first-b", (0.9, 0.1), viewpoint="side"),
                ),
                identity(
                    "fine-first",
                    reference("second-a", (0.8, 0.6), viewpoint="front"),
                ),
            ],
        )

        self.assertEqual(result["coarse_ranking"][0]["identity_id"], "coarse-first")
        self.assertEqual(result["matches"][0]["identity_id"], "fine-first")
        coarse_first = next(
            match
            for match in result["matches"]
            if match["identity_id"] == "coarse-first"
        )
        self.assertEqual(
            [row["reference_id"] for row in coarse_first["reference_contributions"]],
            ["first-a", "first-b"],
        )
        np.testing.assert_allclose(
            [
                row["contribution_weight"]
                for row in coarse_first["reference_contributions"]
            ],
            [0.2, 0.8],
        )

    def test_shortlist_is_the_only_batch_and_padding_is_explicit(self):
        selector = RecordingSelector()
        reranker = IdentitySetReranker(selector, candidate_count=2)
        result = reranker.rerank(
            self.query,
            [
                identity("a", reference("a-1", (1.0, 0.0))),
                identity(
                    "b",
                    reference("b-1", (0.9, 0.1)),
                    reference("b-2", (0.8, 0.2)),
                ),
                identity("c", reference("c-1", (0.0, 1.0))),
            ],
        )

        self.assertEqual(len(selector.calls), 1)
        call = selector.calls[0]
        self.assertEqual(call["reference_descriptors"].shape, (2, 2, 2))
        np.testing.assert_array_equal(
            call["reference_mask"],
            np.asarray([[True, False], [True, True]]),
        )
        self.assertEqual(
            [
                row["identity_id"]
                for row in result["coarse_ranking"]
                if row["shortlisted"]
            ],
            ["a", "b"],
        )
        self.assertFalse(result["coarse_ranking"][2]["shortlisted"])

    def test_tokens_are_loaded_only_for_coarse_shortlist(self):
        selector = RecordingSelector()
        reranker = IdentitySetReranker(selector, candidate_count=1)
        requested: list[str] = []

        def load_tokens(reference_ids):
            requested.extend(reference_ids)
            return {reference_id: BASE_TOKENS.copy() for reference_id in reference_ids}

        result = reranker.rerank(
            self.query,
            [
                identity(
                    "shortlisted",
                    ReferenceEvidence(
                        reference_id="shortlisted-1",
                        descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
                        tokens=None,
                    ),
                ),
                identity(
                    "coarse-only",
                    ReferenceEvidence(
                        reference_id="coarse-only-1",
                        descriptor=np.asarray([0.0, 1.0], dtype=np.float32),
                        tokens=None,
                    ),
                ),
            ],
            token_loader=load_tokens,
        )

        self.assertEqual(requested, ["shortlisted-1"])
        self.assertEqual(result["reranked_identities"], 1)
        self.assertEqual(result["coarse_ranking"][1]["identity_id"], "coarse-only")

    def test_fewer_identities_than_candidate_capacity_is_stable(self):
        selector = RecordingSelector()
        result = IdentitySetReranker(selector, candidate_count=8).rerank(
            self.query,
            [
                identity("single", reference("single-1", (1.0, 0.0))),
            ],
            limit=5,
        )

        self.assertEqual(result["reranked_identities"], 1)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(
            result["matches"][0]["evidence"]["status"], "low_reference_support"
        )

    def test_evidence_diagnostics_use_structural_facts_not_thresholds(self):
        altered_tokens = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        selector = RecordingSelector()
        result = IdentitySetReranker(selector, candidate_count=5).rerank(
            self.query,
            [
                identity(
                    "duplicates",
                    reference("dup-a", (1.0, 0.0), viewpoint="front"),
                    reference("dup-b", (1.0, 0.0), viewpoint="front"),
                ),
                identity(
                    "same-view",
                    reference("same-a", (0.9, 0.1), viewpoint="front"),
                    reference(
                        "same-b",
                        (0.8, 0.2),
                        viewpoint="front",
                        tokens=altered_tokens,
                    ),
                ),
                identity(
                    "complementary",
                    reference("comp-a", (0.7, 0.3), viewpoint="front"),
                    reference(
                        "comp-b",
                        (0.6, 0.4),
                        viewpoint="side",
                        tokens=altered_tokens,
                    ),
                ),
                identity(
                    "unknown-views",
                    reference("unknown-a", (0.5, 0.5)),
                    reference("unknown-b", (0.4, 0.6), tokens=altered_tokens),
                ),
            ],
        )
        evidence = {
            match["identity_id"]: match["evidence"] for match in result["matches"]
        }

        self.assertEqual(evidence["duplicates"]["status"], "duplicate_references")
        self.assertEqual(
            evidence["duplicates"]["duplicate_reference_pairs"],
            [["dup-a", "dup-b"]],
        )
        self.assertEqual(evidence["same-view"]["status"], "missing_complementary_view")
        self.assertEqual(evidence["same-view"]["repeated_viewpoints"], ["front"])
        self.assertEqual(evidence["complementary"]["status"], "sufficient")
        self.assertEqual(
            evidence["complementary"]["distinct_viewpoints"], ["front", "side"]
        )
        self.assertEqual(
            evidence["unknown-views"]["status"], "viewpoint_metadata_unavailable"
        )
        self.assertEqual(
            evidence["unknown-views"]["model_signals"]["calibration"],
            "uncalibrated",
        )

    def test_real_token_selector_handles_mixed_reference_counts(self):
        torch.manual_seed(11)
        matcher = TokenConditionedReferenceMatcher(
            descriptor_dim=2,
            token_dim=2,
            hidden_dim=4,
            max_references=3,
            reference_top_k=2,
        )
        selector = QueryConditionedReferenceSelector(matcher)
        result = IdentitySetReranker(selector, candidate_count=2).rerank(
            self.query,
            [
                identity("a", reference("a-1", (1.0, 0.0))),
                identity(
                    "b",
                    reference("b-1", (0.8, 0.2)),
                    reference(
                        "b-2",
                        (0.2, 0.8),
                        tokens=np.asarray([[0.6, 0.4], [0.1, 0.9]], dtype=np.float32),
                    ),
                ),
            ],
        )

        self.assertEqual(len(result["matches"]), 2)
        self.assertTrue(all(np.isfinite(match["score"]) for match in result["matches"]))
        contributions = {
            match["identity_id"]: match["reference_contributions"]
            for match in result["matches"]
        }
        self.assertEqual(
            [row["reference_id"] for row in contributions["b"]],
            ["b-1", "b-2"],
        )
        self.assertAlmostEqual(
            sum(row["contribution_weight"] for row in contributions["b"]), 1.0
        )
        catalog_gates = [
            match["model_signals"]["catalog_confidence_gate"]
            for match in result["matches"]
        ]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in catalog_gates))
        self.assertAlmostEqual(catalog_gates[0], catalog_gates[1])

    def test_runtime_selector_returns_only_compact_rerank_evidence(self):
        torch.manual_seed(17)
        matcher = TokenConditionedReferenceMatcher(
            descriptor_dim=2,
            token_dim=2,
            hidden_dim=4,
            max_references=2,
            reference_top_k=1,
        )
        selector = QueryConditionedReferenceSelector(matcher)

        output = selector.select(
            self.query.descriptor,
            self.query.tokens,
            np.asarray([[[1.0, 0.0]]], dtype=np.float32),
            np.asarray([[[BASE_TOKENS[0], BASE_TOKENS[1]]]], dtype=np.float32),
            np.asarray([[True]]),
        )

        self.assertIn("score", output)
        self.assertIn("token_scores", output)
        self.assertIn("catalog_confidence_gate", output)
        self.assertNotIn("token_attention", output)
        self.assertNotIn("token_similarity", output)

    def test_capacity_and_duplicate_ids_fail_instead_of_dropping_evidence(self):
        selector = RecordingSelector()
        selector.max_references = 1
        reranker = IdentitySetReranker(selector)
        with self.assertRaisesRegex(ValueError, "selector capacity"):
            reranker.rerank(
                self.query,
                [
                    identity(
                        "too-many",
                        reference("one", (1.0, 0.0)),
                        reference("two", (0.0, 1.0)),
                    )
                ],
            )
        selector.max_references = 4
        reranker = IdentitySetReranker(selector)
        with self.assertRaisesRegex(ValueError, "duplicate reference_id"):
            reranker.rerank(
                self.query,
                [
                    identity("a", reference("shared", (1.0, 0.0))),
                    identity("b", reference("shared", (0.0, 1.0))),
                ],
            )


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


class _ServicePrimaryEncoder:
    def backend_info(self) -> dict[str, object]:
        return {
            "backend": "test-primary",
            "model_sha256": "primary-evidence-integration",
            "embedding_dim": 2,
        }

    def encode_file(self, path: Path) -> EncodedPetImage:
        with Image.open(path) as image:
            mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        # Keep the primary descriptor non-discriminative: this test must use
        # the separately persisted evidence descriptor.
        descriptor = np.asarray([1.0, 1.0], dtype=np.float32)
        return EncodedPetImage(
            fused=descriptor,
            nose=descriptor,
            face=descriptor,
            metadata={
                "detections": 1,
                "descriptor": {
                    "viewpoint_label": "front" if mean[0] > mean[2] else "side",
                    "viewpoint": [1.0, 0.0, 0.0, 0.0],
                },
            },
        )


class _ServiceEvidenceEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def backend_info(self) -> dict[str, object]:
        return {
            "type": "test-evidence",
            "model_sha256": "spatial-evidence-integration",
            "descriptor_dim": 2,
            "token_dim": 2,
            "token_count": 2,
        }

    def encode_file(self, path: Path) -> QueryEvidence:
        self.calls += 1
        with Image.open(path) as image:
            mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        descriptor = (
            np.asarray([1.0, 0.0], dtype=np.float32)
            if mean[0] > mean[2]
            else np.asarray([0.0, 1.0], dtype=np.float32)
        )
        tokens = (
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            if descriptor[0] > descriptor[1]
            else np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        )
        return QueryEvidence(descriptor=descriptor, tokens=tokens)


class _AlternateServiceEvidenceEncoder(_ServiceEvidenceEncoder):
    def backend_info(self) -> dict[str, object]:
        info = super().backend_info()
        info["model_sha256"] = "different-spatial-evidence"
        return info


class _WrongTokenCountEvidenceEncoder(_ServiceEvidenceEncoder):
    def encode_file(self, path: Path) -> QueryEvidence:
        evidence = super().encode_file(path)
        return QueryEvidence(
            descriptor=evidence.descriptor,
            tokens=evidence.tokens[:1],
        )


class _ServiceSelector:
    descriptor_dim = 2
    token_dim = 2
    max_references = 4

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def select(
        self,
        query_descriptor,
        query_tokens,
        reference_descriptors,
        reference_tokens,
        reference_mask,
    ):
        mask = np.asarray(reference_mask, dtype=np.bool_)
        self.calls.append(
            (
                int(mask.shape[0]),
                int(mask.shape[1]),
                int(mask.sum()),
            )
        )
        similarities = np.einsum(
            "d,bkd->bk",
            np.asarray(query_descriptor, dtype=np.float32),
            np.asarray(reference_descriptors, dtype=np.float32),
        )
        masked = np.where(mask, similarities, -2.0)
        scores = masked.max(axis=1).astype(np.float32)
        attention = mask.astype(np.float32)
        attention /= attention.sum(axis=1, keepdims=True)
        return {
            "score": scores,
            "attention": attention,
            "similarities": similarities,
            "token_scores": similarities,
            "novelty": mask.astype(np.float32),
            "coverage_gate": mask.astype(np.float32),
            "baseline_score": scores,
            "residual": np.zeros(mask.shape[0], dtype=np.float32),
            "coverage_score": np.ones(mask.shape[0], dtype=np.float32),
            "duplicate_score": np.zeros(mask.shape[0], dtype=np.float32),
            "centroid_score": scores,
            "top_k_score": scores,
        }


class _ServiceRerankerProxy:
    """Structural runtime adapter that deliberately does not subclass the reranker."""

    def __init__(self, reranker: IdentitySetReranker) -> None:
        self._reranker = reranker
        for field in (
            "descriptor_dim",
            "token_dim",
            "max_references",
            "candidate_count",
            "coarse_support_count",
        ):
            setattr(self, field, getattr(reranker, field))

    def configuration(self):
        return self._reranker.configuration()

    def rerank(self, query, identities, *, limit=None, token_loader=None):
        return self._reranker.rerank(
            query,
            identities,
            limit=limit,
            token_loader=token_loader,
        )


class IdentitySetServiceIntegrationTest(unittest.TestCase):
    def _service(self, root: Path, selector: _ServiceSelector):
        return PetIdentificationService(
            PetGalleryStore(root),
            _ServicePrimaryEncoder(),
            default_scoring_mode="identity_set_rerank",
            identity_set_reranker=_ServiceRerankerProxy(
                IdentitySetReranker(selector, candidate_count=2)
            ),
            reference_evidence_encoder=_ServiceEvidenceEncoder(),
        )

    def test_enrollment_persists_evidence_and_identification_uses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gallery"
            selector = _ServiceSelector()
            service = self._service(root, selector)
            red = UploadPayload("red.png", "image/png", _image_bytes((250, 10, 10)))
            red_two = UploadPayload(
                "red-two.png", "image/png", _image_bytes((235, 20, 10))
            )
            blue = UploadPayload("blue.png", "image/png", _image_bytes((10, 10, 250)))
            service.enroll("red", [red, red_two])
            service.enroll("blue", [blue])

            evidence_encoder = service.reference_evidence_encoder
            self.assertIsInstance(evidence_encoder, _ServiceEvidenceEncoder)
            self.assertEqual(evidence_encoder.calls, 3)
            service.identify(
                red,
                scoring_mode="centroid",
                top_k=2,
                record_history=False,
            )
            self.assertEqual(evidence_encoder.calls, 3)
            result = service.identify(red, top_k=2, record_history=False)

            self.assertEqual(result["predicted_pet_id"], "red")
            self.assertEqual(result["scoring"]["mode"], "identity_set_rerank")
            self.assertEqual(result["scoring"]["reranked_identities"], 2)
            self.assertIn(
                result["query"]["sha256"],
                {
                    row["reference_id"]
                    for row in result["candidates"][0]["reference_contributions"]
                },
            )
            self.assertEqual(len(result["candidates"][0]["reference_contributions"]), 2)
            self.assertEqual(service.store.summary()["reference_evidence"], 3)
            self.assertEqual(selector.calls, [(2, 2, 3)])
            self.assertEqual(evidence_encoder.calls, 4)

            reopened = PetGalleryStore(root)
            sets = reopened.identity_reference_sets()
            self.assertEqual([item.identity_id for item in sets], ["blue", "red"])
            self.assertEqual(len(sets[1].references), 2)
            self.assertIn(
                result["query"]["sha256"],
                {row.reference_id for row in sets[1].references},
            )
            for row in sets[1].references:
                np.testing.assert_allclose(
                    row.descriptor,
                    np.asarray([1.0, 0.0], dtype=np.float32),
                )
            descriptor_sets = reopened.identity_reference_sets(include_tokens=False)
            self.assertTrue(
                all(
                    row.tokens is None
                    for item in descriptor_sets
                    for row in item.references
                )
            )

    def test_incomplete_evidence_is_rejected_only_by_opt_in_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gallery"
            store = PetGalleryStore(root)
            upload = UploadPayload("red.png", "image/png", _image_bytes((250, 10, 10)))
            validated = validate_upload(
                upload,
                maximum_bytes=1024 * 1024,
                maximum_pixels=1_000_000,
            )
            store.enroll(
                "red",
                "red",
                [
                    EnrollmentRecord(
                        upload=validated,
                        encoded=EncodedPetImage(
                            fused=np.asarray([1.0, 0.0], dtype=np.float32),
                            nose=np.asarray([1.0, 0.0], dtype=np.float32),
                            face=np.asarray([1.0, 0.0], dtype=np.float32),
                            metadata={"detections": 1},
                        ),
                    )
                ],
            )
            # A compatibility gallery remains usable by the old descriptor path.
            compatibility = PetIdentificationService(
                PetGalleryStore(root),
                _ServicePrimaryEncoder(),
                model_fingerprint="primary-evidence-integration",
            )
            compatibility_result = compatibility.identify(upload, record_history=False)
            self.assertEqual(compatibility_result["predicted_pet_id"], "red")

            with self.assertRaisesRegex(GalleryModelMismatch, "reference evidence"):
                self._service(root, _ServiceSelector()).identify(
                    upload,
                    record_history=False,
                )

    def test_evidence_fingerprint_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gallery"
            service = self._service(root, _ServiceSelector())
            service.enroll(
                "red",
                [
                    UploadPayload(
                        "red.png",
                        "image/png",
                        _image_bytes((250, 10, 10)),
                    )
                ],
            )
            with self.assertRaisesRegex(GalleryModelMismatch, "different model"):
                PetIdentificationService(
                    PetGalleryStore(root),
                    _ServicePrimaryEncoder(),
                    default_scoring_mode="identity_set_rerank",
                    identity_set_reranker=IdentitySetReranker(
                        _ServiceSelector(), candidate_count=2
                    ),
                    reference_evidence_encoder=_AlternateServiceEvidenceEncoder(),
                )

    def test_evidence_output_must_match_declared_token_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PetIdentificationService(
                PetGalleryStore(Path(directory) / "gallery"),
                _ServicePrimaryEncoder(),
                default_scoring_mode="identity_set_rerank",
                identity_set_reranker=IdentitySetReranker(
                    _ServiceSelector(), candidate_count=2
                ),
                reference_evidence_encoder=_WrongTokenCountEvidenceEncoder(),
            )

            with self.assertRaisesRegex(InvalidPetImage, "token shape"):
                service.enroll(
                    "red",
                    [
                        UploadPayload(
                            "red.png",
                            "image/png",
                            _image_bytes((250, 10, 10)),
                        )
                    ],
                )

    def test_backup_restore_rebuilds_complete_reference_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._service(root / "source", _ServiceSelector())
            red = UploadPayload("red.png", "image/png", _image_bytes((250, 10, 10)))
            red_two = UploadPayload(
                "red-two.png", "image/png", _image_bytes((235, 20, 10))
            )
            blue = UploadPayload("blue.png", "image/png", _image_bytes((10, 10, 250)))
            source.enroll("red", [red, red_two])
            source.enroll("blue", [blue])
            _, backup = source.create_gallery_backup()

            target = self._service(root / "target", _ServiceSelector())
            restored = target.restore_gallery_backup(backup)

            self.assertEqual(restored["added_images"], 3)
            summary = target.store.summary()
            self.assertEqual(summary["reference_images"], 3)
            self.assertEqual(summary["reference_evidence"], 3)
            sets = target.store.identity_reference_sets()
            self.assertEqual(
                {item.identity_id: len(item.references) for item in sets},
                {"blue": 1, "red": 2},
            )
            result = target.identify(red, top_k=2, record_history=False)
            self.assertEqual(result["predicted_pet_id"], "red")

    def test_seed_gallery_import_generates_reference_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_image = root / "seed.png"
            source_image.write_bytes(_image_bytes((250, 10, 10)))
            feature_path = root / "gallery_features.npz"
            np.savez_compressed(
                feature_path,
                selected_fused_references=np.asarray([[1.0, 0.0]], dtype=np.float32),
                selected_nose_references=np.asarray([[1.0, 0.0]], dtype=np.float32),
                selected_face_references=np.asarray([[1.0, 0.0]], dtype=np.float32),
                reference_identity_indices=np.asarray([0], dtype=np.int64),
            )
            model_path = root / "gallery_model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "identities": ["seed-dog"],
                        "references": [
                            {
                                "path": str(source_image),
                                "sha256": sha256_file(source_image),
                                "selected_inference": {"detections": 1},
                            }
                        ],
                        "selected_backend": {
                            "model_sha256": "primary-evidence-integration"
                        },
                        "features_file": feature_path.name,
                        "features_sha256": sha256_file(feature_path),
                    }
                ),
                encoding="utf-8",
            )
            service = self._service(root / "gallery", _ServiceSelector())

            imported = service.import_gallery_model(model_path)

            self.assertEqual(imported["added"], 1)
            summary = service.store.summary()
            self.assertEqual(summary["reference_images"], 1)
            self.assertEqual(summary["reference_evidence"], 1)
            evidence = service.store.identity_reference_sets()[0].references[0]
            np.testing.assert_allclose(
                evidence.descriptor,
                np.asarray([1.0, 0.0], dtype=np.float32),
            )
            result = service.identify(
                UploadPayload(
                    "query.png",
                    "image/png",
                    _image_bytes((245, 15, 10)),
                ),
                record_history=False,
            )
            self.assertEqual(result["predicted_pet_id"], "seed-dog")


class _TinyEvidenceModel(torch.nn.Module):
    descriptor_dim = 3
    token_dim = 2
    token_grid = 2

    def encode_image_features(self, images):
        pooled = images.mean(dim=(2, 3))
        descriptor = torch.nn.functional.normalize(pooled, dim=-1)
        token = torch.nn.functional.normalize(
            images[:, :2, :2, :2].flatten(2).transpose(1, 2), dim=-1
        )
        return descriptor, token


class ModelEvidenceEncoderTest(unittest.TestCase):
    def test_model_adapter_uses_declared_preprocess_and_returns_complete_evidence(self):
        model = _TinyEvidenceModel()
        adapter = ModelReferenceEvidenceEncoder(
            model,
            lambda _path: np.ones((3, 2, 2), dtype=np.float32),
            model_fingerprint="tiny-evidence-model",
        )

        evidence = adapter.encode_file(Path("unused.png"))

        self.assertEqual(evidence.descriptor.shape, (3,))
        self.assertEqual(evidence.tokens.shape, (4, 2))
        self.assertEqual(adapter.backend_info()["model_sha256"], "tiny-evidence-model")


if __name__ == "__main__":
    unittest.main()
