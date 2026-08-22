# encoding: utf-8
"""Regression tests for validation and frozen-layer train/eval modes."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fastreid.engine.hooks import LayerFreeze
from fastreid.evaluation.evaluator import inference_context
from fastreid.solver.build import _generate_optimizer_class_with_freeze_layer


class _MixedModeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
        self.heads = nn.Linear(4, 2)
        self.workspace = nn.Linear(4, 4)


class TrainingModeRestoreTest(unittest.TestCase):
    @staticmethod
    def _build_freeze_optimizer(freeze_iters=4):
        frozen = nn.Parameter(torch.tensor([1.0]))
        normal = nn.Parameter(torch.tensor([1.0]))
        optimizer_type = _generate_optimizer_class_with_freeze_layer(
            torch.optim.Adam, freeze_iters=freeze_iters
        )
        optimizer = optimizer_type(
            [
                {"params": [frozen], "freeze_status": "freeze"},
                {"params": [normal], "freeze_status": "normal"},
            ],
            lr=0.1,
        )
        return frozen, normal, optimizer

    @staticmethod
    def _optimizer_step(frozen, normal, optimizer):
        frozen.grad = torch.ones_like(frozen)
        normal.grad = torch.ones_like(normal)
        optimizer.step()

    def test_inference_restores_every_submodule_mode(self):
        model = _MixedModeModel().train()
        model.backbone.eval()
        modes_before = {
            name: module.training for name, module in model.named_modules()
        }

        with inference_context(model):
            self.assertTrue(all(not module.training for module in model.modules()))

        modes_after = {
            name: module.training for name, module in model.named_modules()
        }
        self.assertEqual(modes_after, modes_before)
        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.heads.training)
        self.assertTrue(model.workspace.training)

    def test_inference_restores_modes_after_exception(self):
        model = _MixedModeModel().train()
        model.backbone.eval()
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with inference_context(model):
                raise RuntimeError("evaluation failed")
        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.heads.training)

    def test_freeze_hook_reasserts_eval_until_boundary(self):
        model = _MixedModeModel().train()
        hook = LayerFreeze(model, ["backbone"], freeze_iters=100)
        hook.trainer = SimpleNamespace(iter=0)
        hook.before_step()
        self.assertTrue(hook.is_frozen)
        self.assertFalse(model.backbone.training)

        # Simulate an external recursive model.train() call.
        model.train()
        self.assertTrue(model.backbone.training)
        hook.trainer.iter = 50
        hook.before_step()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.heads.training)
        self.assertTrue(model.workspace.training)

        hook.trainer.iter = 100
        hook.before_step()
        self.assertFalse(hook.is_frozen)
        self.assertTrue(model.backbone.training)

    def test_optimizer_freeze_counter_survives_checkpoint_roundtrip(self):
        frozen, normal, optimizer = self._build_freeze_optimizer()
        self._optimizer_step(frozen, normal, optimizer)
        self._optimizer_step(frozen, normal, optimizer)
        state_dict = optimizer.state_dict()
        self.assertEqual(state_dict["_freeze_step_count"], 2)

        resumed_frozen, resumed_normal, resumed = self._build_freeze_optimizer()
        resumed.load_state_dict(state_dict)
        self.assertEqual(resumed.state_dict()["_freeze_step_count"], 2)

        frozen_before = resumed_frozen.detach().clone()
        self._optimizer_step(resumed_frozen, resumed_normal, resumed)
        self._optimizer_step(resumed_frozen, resumed_normal, resumed)
        torch.testing.assert_close(resumed_frozen, frozen_before)
        self._optimizer_step(resumed_frozen, resumed_normal, resumed)
        self.assertFalse(torch.equal(resumed_frozen, frozen_before))

    def test_optimizer_infers_counter_from_legacy_adam_checkpoint(self):
        frozen, normal, optimizer = self._build_freeze_optimizer()
        self._optimizer_step(frozen, normal, optimizer)
        self._optimizer_step(frozen, normal, optimizer)
        legacy_state_dict = copy.copy(optimizer.state_dict())
        legacy_state_dict.pop("_freeze_step_count")

        _, _, resumed = self._build_freeze_optimizer()
        resumed.load_state_dict(legacy_state_dict)
        self.assertEqual(resumed.state_dict()["_freeze_step_count"], 2)


if __name__ == "__main__":
    unittest.main()
