# encoding: utf-8
"""Regression checks for the initial dog ArcFace integration."""

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from pet_id.arcface import DogArcFaceEncoder, fuse_main_and_arcface


CHECKPOINT = Path(__file__).resolve().parents[3] / "dog.pt"


class ArcFaceIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.is_file():
            raise unittest.SkipTest(f"ArcFace checkpoint not found: {CHECKPOINT}")
        cls.encoder = DogArcFaceEncoder(CHECKPOINT, load_classifier=True)

    def test_checkpoint_contract_and_cpu_forward(self):
        self.assertEqual(self.encoder.feature_dim, 512)
        self.assertEqual(self.encoder.num_pretraining_classes, 46755)
        self.assertFalse(self.encoder.training)
        self.assertFalse(any(p.requires_grad for p in self.encoder.parameters()))

        with torch.inference_mode():
            output = self.encoder(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 512))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.norm(dim=1), torch.ones(1))

    def test_complete_classifier_head(self):
        with torch.inference_mode():
            output = self.encoder.forward_with_classifier(
                torch.zeros(2, 3, 224, 224), topk=5
            )
        self.assertEqual(tuple(output["features"].shape), (2, 512))
        self.assertEqual(tuple(output["classifier_logits"].shape), (2, 46755))
        self.assertEqual(tuple(output["topk_indices"].shape), (2, 5))
        self.assertEqual(tuple(output["topk_scores"].shape), (2, 5))
        self.assertTrue((output["topk_scores"] <= 1.0).all())
        self.assertTrue((output["topk_scores"] >= -1.0).all())

    def test_frozen_batchnorm_stays_in_eval_mode(self):
        self.encoder.train()
        self.assertFalse(self.encoder.training)
        self.assertFalse(self.encoder.backbone.training)

    def test_weighted_fusion_has_expected_cosine(self):
        generator = torch.Generator().manual_seed(17)
        main = torch.randn(2, 2048, generator=generator)
        arcface = torch.randn(2, 512, generator=generator)
        arcface_weight = 0.3
        fused = fuse_main_and_arcface(
            main, arcface, arcface_weight=arcface_weight
        )

        actual = F.cosine_similarity(fused[0:1], fused[1:2]).item()
        expected = (
            (1.0 - arcface_weight)
            * F.cosine_similarity(main[0:1], main[1:2]).item()
            + arcface_weight
            * F.cosine_similarity(arcface[0:1], arcface[1:2]).item()
        )
        self.assertAlmostEqual(actual, expected, places=6)
        torch.testing.assert_close(fused.norm(dim=1), torch.ones(2))

    def test_selective_local_end_to_end_unfreeze(self):
        self.encoder.configure_trainable_parts(("layer4", "fc"))
        trainable = {
            name for name, parameter in self.encoder.backbone.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("layer4.", "fc.")) for name in trainable))
        self.encoder.train()
        self.assertFalse(self.encoder.backbone.layer3.training)
        self.assertTrue(self.encoder.backbone.layer4.training)
        self.encoder.configure_trainable_parts(())
        self.assertFalse(any(p.requires_grad for p in self.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
