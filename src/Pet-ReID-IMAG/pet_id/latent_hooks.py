# encoding: utf-8
"""Training-time health telemetry for the persistent latent workspace."""

import math

import torch
from torch.nn.parallel import DistributedDataParallel

from fastreid.engine.train_loop import HookBase
from fastreid.utils.events import get_event_storage


class LatentHealthHook(HookBase):
    """Record latent health and stop persistently degenerate experiment runs."""

    def __init__(
        self,
        model,
        period,
        *,
        early_abort_enabled=False,
        early_abort_warmup_iters=100,
        early_abort_patience=2,
        slot_cosine_max=0.995,
        min_effective_rank=2.0,
        query_cosine_max=0.995,
        min_query_rank=1.5,
    ):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        if not hasattr(model, "workspace"):
            raise TypeError("LatentHealthHook requires model.workspace")
        self.workspace = model.workspace
        self.identity_query_head = getattr(model, "identity_query_head", None)
        fusion_weight_logit = getattr(model, "fusion_weight_logit", None)
        monitored_parameters = list(self.workspace.parameters())
        if self.identity_query_head is not None:
            monitored_parameters.extend(self.identity_query_head.parameters())
        if isinstance(fusion_weight_logit, torch.nn.Parameter):
            monitored_parameters.append(fusion_weight_logit)
        self.monitored_parameters = monitored_parameters
        self.period = int(period)
        self.early_abort_enabled = bool(early_abort_enabled)
        self.early_abort_warmup_iters = int(early_abort_warmup_iters)
        self.early_abort_patience = int(early_abort_patience)
        self.slot_cosine_max = float(slot_cosine_max)
        self.min_effective_rank = float(min_effective_rank)
        self.query_cosine_max = float(query_cosine_max)
        self.min_query_rank = float(min_query_rank)
        self._consecutive_bad_checks = 0
        if self.early_abort_patience < 1:
            raise ValueError("early abort patience must be positive")

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

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

        for parameter in self.monitored_parameters:
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
        storage.put_scalar("latent/grad_norm", gradient_norm, smoothing_hint=False)
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

        if (
            not self.early_abort_enabled
            or self.trainer.iter < self.early_abort_warmup_iters
        ):
            return

        diagnostics = self.workspace.diagnostics()
        problems = []
        nonfinite_names = []
        for name, value in diagnostics.items():
            scalar = self._scalar(value)
            if not math.isfinite(scalar):
                nonfinite_names.append(name)
        if nonfinite_names:
            problems.append(f"non-finite diagnostics: {nonfinite_names}")

        slot_cosine = diagnostics.get("slot_cosine_max")
        effective_rank = diagnostics.get("slot_effective_rank")
        if slot_cosine is None or effective_rank is None:
            problems.append("final slot diagnostics are missing")
        else:
            slot_cosine = self._scalar(slot_cosine)
            effective_rank = self._scalar(effective_rank)
            if slot_cosine >= self.slot_cosine_max:
                problems.append(
                    f"slot cosine max {slot_cosine:.6f} >= {self.slot_cosine_max:.6f}"
                )
            if effective_rank < self.min_effective_rank:
                problems.append(
                    f"slot effective rank {effective_rank:.4f} < {self.min_effective_rank:.4f}"
                )

        if self.identity_query_head is not None:
            query_cosine = diagnostics.get("identity_query_cosine_max")
            query_rank = diagnostics.get("identity_query_effective_rank")
            if query_cosine is None or query_rank is None:
                problems.append("identity query diagnostics are missing")
            else:
                query_cosine = self._scalar(query_cosine)
                query_rank = self._scalar(query_rank)
                if query_cosine >= self.query_cosine_max:
                    problems.append(
                        "identity query cosine max "
                        f"{query_cosine:.6f} >= {self.query_cosine_max:.6f}"
                    )
                if query_rank < self.min_query_rank:
                    problems.append(
                        "identity query effective rank "
                        f"{query_rank:.4f} < {self.min_query_rank:.4f}"
                    )

        finite_fraction = finite_elements / gradient_denominator
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            problems.append(f"invalid latent gradient norm {gradient_norm}")
        if finite_fraction < 1.0:
            problems.append(
                f"finite latent gradient fraction {finite_fraction:.6f} < 1"
            )

        if problems:
            self._consecutive_bad_checks += 1
        else:
            self._consecutive_bad_checks = 0
        storage.put_scalar(
            "latent/early_abort_bad_checks",
            self._consecutive_bad_checks,
            smoothing_hint=False,
        )
        if self._consecutive_bad_checks >= self.early_abort_patience:
            reason = "; ".join(problems)
            raise RuntimeError(
                "Latent workspace early-abort gate stopped this run at "
                f"iteration {self.trainer.iter}: {reason}"
            )
