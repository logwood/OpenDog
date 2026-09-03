from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from pet_id.unified_external_model import GradientControlledFusionRefiner


class GradientControlledFusionRefinerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.base = F.normalize(torch.randn(4, 512), dim=1)
        self.face = F.normalize(torch.randn(4, 512), dim=1)
        self.nose = F.normalize(torch.randn(4, 512), dim=1)
        self.confidence = torch.tensor([0.1, 0.4, 0.7, 0.9])

    def test_zero_initialization_is_exact_and_has_first_step_gradients(self):
        refiner = GradientControlledFusionRefiner()
        output = refiner(
            self.base, self.face, self.nose, self.confidence
        )
        self.assertTrue(torch.equal(output, self.base))
        loss = -(output * self.nose).sum(dim=1).mean()
        loss.backward()
        gradient = refiner.direction_gain_logit.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(abs(float(gradient)), 1e-6)
        interaction_gradient = refiner.interaction[-1].weight.grad
        self.assertIsNotNone(interaction_gradient)
        self.assertTrue(torch.isfinite(interaction_gradient).all())
        self.assertGreater(float(interaction_gradient.norm()), 1e-6)

    def test_nonzero_gain_is_normalized_and_bounded(self):
        refiner = GradientControlledFusionRefiner(
            maximum_residual_weight=0.10
        )
        with torch.no_grad():
            refiner.direction_gain_logit.fill_(0.7)
        output = refiner(
            self.base,
            self.face,
            self.nose,
            self.confidence,
            return_aux=True,
        )
        self.assertTrue(
            torch.allclose(
                output["embedding"].norm(dim=1),
                torch.ones(4),
                atol=1e-6,
            )
        )
        self.assertLessEqual(
            float(output["refiner_residual_weight"].detach().abs().max()), 0.10
        )
        self.assertEqual(tuple(output["refiner_signals"].shape), (4, 5))

    def test_interaction_residual_is_normalized_and_hard_bounded(self):
        maximum_interaction_norm = 0.03
        refiner = GradientControlledFusionRefiner(
            maximum_interaction_norm=maximum_interaction_norm
        )
        with torch.no_grad():
            refiner.interaction[-1].bias.fill_(3.0)
        output = refiner(
            self.base,
            self.face,
            self.nose,
            self.confidence,
            return_aux=True,
        )
        interaction_delta = (
            output["refiner_interaction_scale"][:, None]
            * output["refiner_interaction"]
        )
        self.assertGreater(
            float(interaction_delta.detach().norm(dim=1).min()), 0.0
        )
        self.assertLessEqual(
            float(interaction_delta.detach().norm(dim=1).max()),
            maximum_interaction_norm + 1e-7,
        )
        self.assertTrue(
            torch.equal(
                output["refiner_interaction_scale"],
                torch.full((4,), maximum_interaction_norm),
            )
        )
        self.assertTrue(
            torch.allclose(
                output["embedding"].norm(dim=1),
                torch.ones(4),
                atol=1e-6,
            )
        )

    def test_legacy_reliability_interaction_scale_remains_available(self):
        refiner = GradientControlledFusionRefiner(
            maximum_interaction_norm=0.04,
            interaction_scale_mode="reliability",
        )
        output = refiner(
            self.base,
            self.face,
            self.nose,
            self.confidence,
            return_aux=True,
        )
        self.assertTrue(
            torch.allclose(
                output["refiner_interaction_scale"],
                0.04 * output["refiner_reliability"],
            )
        )


if __name__ == "__main__":
    unittest.main()
