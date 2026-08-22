# encoding: utf-8
"""Training-time health telemetry for the persistent latent workspace."""

import torch
from torch.nn.parallel import DistributedDataParallel

from fastreid.engine.train_loop import HookBase
from fastreid.utils.events import get_event_storage


class LatentHealthHook(HookBase):
    """Record whether the workspace receives finite, non-zero gradients."""

    def __init__(self, model, period):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        if not hasattr(model, "workspace"):
            raise TypeError("LatentHealthHook requires model.workspace")
        self.workspace = model.workspace
        self.period = int(period)

    def after_step(self):
        if self.period <= 0 or self.trainer.iter % self.period != 0:
            return

        squared_norm = None
        nonzero_elements = 0
        finite_elements = 0
        total_elements = 0
        gradient_elements = 0
        parameters_with_grad = 0
        total_parameters = 0

        for parameter in self.workspace.parameters():
            if not parameter.requires_grad:
                continue
            total_parameters += 1
            gradient = parameter.grad
            if gradient is None:
                total_elements += parameter.numel()
                continue

            parameters_with_grad += 1
            gradient = gradient.detach().float()
            finite = torch.isfinite(gradient)
            finite_gradient = torch.where(finite, gradient, torch.zeros_like(gradient))
            term = finite_gradient.square().sum()
            squared_norm = term if squared_norm is None else squared_norm + term
            nonzero_elements += int((finite_gradient != 0).sum().item())
            finite_elements += int(finite.sum().item())
            total_elements += gradient.numel()
            gradient_elements += gradient.numel()

        gradient_norm = 0.0 if squared_norm is None else squared_norm.sqrt().item()
        element_denominator = max(total_elements, 1)
        parameter_denominator = max(total_parameters, 1)
        gradient_denominator = max(gradient_elements, 1)
        storage = get_event_storage()
        storage.put_scalar(
            "latent/grad_norm", gradient_norm, smoothing_hint=False
        )
        storage.put_scalar(
            "latent/grad_nonzero_fraction",
            nonzero_elements / element_denominator,
            smoothing_hint=False,
        )
        storage.put_scalar(
            "latent/grad_finite_fraction",
            finite_elements / gradient_denominator,
            smoothing_hint=False,
        )
        storage.put_scalar(
            "latent/parameters_with_grad_fraction",
            parameters_with_grad / parameter_denominator,
            smoothing_hint=False,
        )
