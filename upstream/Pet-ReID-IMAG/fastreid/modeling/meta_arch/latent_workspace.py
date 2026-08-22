# encoding: utf-8
"""Persistent multi-stage latent workspace for ResNeSt image ReID."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.utils.events import get_event_storage

from .baseline import Baseline
from .build import META_ARCH_REGISTRY


def _sinkhorn_log_domain(
    cost,
    token_marginal,
    slot_marginal,
    *,
    iterations,
    temperature,
    dual_token=None,
    dual_slot=None,
):
    """Compute a rectangular entropy-regularized transport plan in log space."""
    if cost.ndim != 3:
        raise ValueError(f"cost must have shape [B, N, K], got {tuple(cost.shape)}")
    if temperature <= 0:
        raise ValueError("Sinkhorn temperature must be positive")

    log_kernel = -cost / float(temperature)
    log_token = token_marginal.clamp_min(1e-30).log()
    log_slot = slot_marginal.clamp_min(1e-30).log()
    if dual_token is None:
        dual_token = torch.zeros_like(token_marginal)
    if dual_slot is None:
        dual_slot = torch.zeros_like(slot_marginal)

    for _ in range(int(iterations)):
        dual_token = log_token - torch.logsumexp(
            log_kernel + dual_slot.unsqueeze(1), dim=2
        )
        dual_slot = log_slot - torch.logsumexp(
            log_kernel + dual_token.unsqueeze(2), dim=1
        )

    log_transport = (
        log_kernel + dual_token.unsqueeze(2) + dual_slot.unsqueeze(1)
    )
    return log_transport.exp(), dual_token, dual_slot


class _CompetitiveMeshRead(nn.Module):
    """SA-MESH read step adapted to a persistent multi-stage workspace.

    The implementation follows the first-order entropy minimization described in
    https://github.com/davzha/MESH while keeping all transport arithmetic in FP32.
    """

    def __init__(
        self,
        dim,
        num_slots,
        mlp_ratio,
        *,
        mesh_iterations,
        sinkhorn_iterations,
        mesh_lr,
        noise_std,
        temperature,
    ):
        super().__init__()
        if mesh_iterations < 1 or sinkhorn_iterations < 1:
            raise ValueError("MESH and Sinkhorn iteration counts must be positive")
        if mesh_lr <= 0 or noise_std < 0 or temperature <= 0:
            raise ValueError("invalid MESH learning rate, noise, or temperature")

        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.mesh_iterations = int(mesh_iterations)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.mesh_lr = float(mesh_lr)
        self.noise_std = float(noise_std)
        self.temperature = float(temperature)

        self.slot_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        # Keep a learned input marginal so the read can down-weight background
        # tokens. Slot capacity is deliberately uniform: persistent slots in a
        # supervised ReID model otherwise learn a winner-take-all marginal long
        # before their spatial roles have formed.
        self.token_marginal = nn.Linear(dim, 1, bias=False)
        self.update = nn.GRUCell(dim, dim)

    @staticmethod
    def _fixed_noise(cost):
        token_index = torch.arange(
            1, cost.shape[1] + 1, device=cost.device, dtype=cost.dtype
        ).view(1, -1, 1)
        slot_index = torch.arange(
            1, cost.shape[2] + 1, device=cost.device, dtype=cost.dtype
        ).view(1, 1, -1)
        noise = torch.sin(token_index * slot_index * 12.9898)
        noise = noise + torch.cos((token_index + slot_index) * 78.233)
        noise = noise - noise.mean(dim=(1, 2), keepdim=True)
        noise = noise / noise.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(
            1e-6
        )
        return noise.expand(cost.shape[0], -1, -1)

    def _noise(self, cost):
        if self.noise_std == 0:
            return torch.zeros_like(cost)
        if self.training:
            return torch.randn_like(cost) * self.noise_std
        return self._fixed_noise(cost) * self.noise_std

    @torch.enable_grad()
    def _minimize_sinkhorn_entropy(
        self, cost, token_marginal, slot_marginal, noise
    ):
        optimized_cost = cost + noise
        if not optimized_cost.requires_grad:
            optimized_cost.requires_grad_(True)
        dual_token = None
        dual_slot = None

        for _ in range(self.mesh_iterations):
            transport, dual_token, dual_slot = _sinkhorn_log_domain(
                optimized_cost,
                token_marginal,
                slot_marginal,
                iterations=self.sinkhorn_iterations,
                temperature=self.temperature,
                dual_token=dual_token,
                dual_slot=dual_slot,
            )
            entropy = -(
                transport.clamp_min(1e-20) * transport.clamp_min(1e-20).log()
            ).sum(dim=(1, 2)).mean()
            (gradient,) = torch.autograd.grad(
                entropy, optimized_cost, retain_graph=True
            )
            gradient_norm = gradient.square().sum(
                dim=(1, 2), keepdim=True
            ).sqrt().clamp_min(1e-20)
            optimized_cost = optimized_cost - self.mesh_lr * gradient / gradient_norm

        transport, _, _ = _sinkhorn_log_domain(
            optimized_cost,
            token_marginal,
            slot_marginal,
            iterations=self.sinkhorn_iterations,
            temperature=self.temperature,
            dual_token=dual_token,
            dual_slot=dual_slot,
        )
        return transport

    def forward(self, latents, context, collect_diagnostics=False):
        output_dtype = latents.dtype
        device_type = latents.device.type
        if device_type not in {"cpu", "cuda"}:
            device_type = "cpu"

        # MESH contains repeated log/exp and an inner gradient. Keeping this
        # compact branch in FP32 is substantially more stable than AMP transport.
        with torch.autocast(device_type=device_type, enabled=False):
            slots = latents.float()
            tokens = context.float()
            normalized_slots = self.slot_norm(slots)
            normalized_tokens = self.context_norm(tokens)
            query = self.query_projection(normalized_slots)
            key = self.key_projection(normalized_tokens)
            value = self.value_projection(normalized_tokens)
            similarity = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(self.dim)
            cost = -similarity
            token_marginal = self.token_marginal(normalized_tokens).squeeze(-1).softmax(
                dim=1
            )
            slot_marginal = torch.full(
                (slots.shape[0], self.num_slots),
                1.0 / self.num_slots,
                device=slots.device,
                dtype=slots.dtype,
            )
            transport = self._minimize_sinkhorn_entropy(
                cost,
                token_marginal,
                slot_marginal,
                self._noise(cost),
            )

            # Convert the transport plan into a weighted mean for each slot.
            slot_weights = transport.transpose(1, 2)
            slot_weights = slot_weights / slot_weights.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-12)
            slot_update = torch.bmm(slot_weights, value)
            updated_slots = self.update(
                slot_update.reshape(-1, self.dim), slots.reshape(-1, self.dim)
            ).view_as(slots)

            diagnostics = OrderedDict()
            if collect_diagnostics:
                assignment = transport / token_marginal.unsqueeze(2).clamp_min(1e-12)
                assignment = assignment / assignment.sum(
                    dim=2, keepdim=True
                ).clamp_min(1e-12)
                normalized_entropy = -(
                    assignment.clamp_min(1e-12) * assignment.clamp_min(1e-12).log()
                ).sum(dim=2).mean() / math.log(self.num_slots)
                slot_mass = assignment.mean(dim=1)
                effective_slots = torch.exp(
                    -(slot_mass.clamp_min(1e-12) * slot_mass.clamp_min(1e-12).log())
                    .sum(dim=1)
                    .mean()
                )
                normalized_maps = F.normalize(slot_weights, dim=2)
                map_similarity = normalized_maps @ normalized_maps.transpose(1, 2)
                off_diagonal = ~torch.eye(
                    self.num_slots, device=map_similarity.device, dtype=torch.bool
                )[None]
                diagnostics["assignment_entropy"] = normalized_entropy.detach()
                diagnostics["effective_slots"] = effective_slots.detach()
                diagnostics["slot_mass_min"] = slot_mass.min().detach()
                diagnostics["slot_mass_max"] = slot_mass.max().detach()
                diagnostics["attention_cosine"] = map_similarity.masked_select(
                    off_diagonal
                ).mean().detach()

        return updated_slots.to(dtype=output_dtype), diagnostics, slot_weights.detach()


class _SharedWorkspaceCell(nn.Module):
    """Shared Read-Mix-Write computation used recurrently at every stage."""

    def __init__(
        self,
        dim,
        num_slots,
        num_heads,
        mlp_ratio,
        dropout,
        *,
        read_mode,
        mesh_iterations,
        sinkhorn_iterations,
        mesh_lr,
        mesh_noise_std,
        mesh_temperature,
        mix_gate_init,
        mix_gate_trainable,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"workspace dim {dim} must be divisible by {num_heads} heads")
        if read_mode not in {"mha", "mesh"}:
            raise ValueError(f"unknown latent read mode {read_mode!r}")
        hidden_dim = int(round(dim * mlp_ratio))
        self.read_mode = read_mode

        if read_mode == "mha":
            self.read_query_norm = nn.LayerNorm(dim)
            self.read_context_norm = nn.LayerNorm(dim)
            self.read_attention = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
        else:
            self.mesh_read = _CompetitiveMeshRead(
                dim,
                num_slots,
                mlp_ratio,
                mesh_iterations=mesh_iterations,
                sinkhorn_iterations=sinkhorn_iterations,
                mesh_lr=mesh_lr,
                noise_std=mesh_noise_std,
                temperature=mesh_temperature,
            )
            if not 0.0 < mix_gate_init < 1.0:
                raise ValueError("MESH mix gate init must be between zero and one")
            mix_probability = torch.tensor(float(mix_gate_init))
            self.mix_gate_logit = nn.Parameter(
                torch.logit(mix_probability),
                requires_grad=bool(mix_gate_trainable),
            )

        self.mix_attention_norm = nn.LayerNorm(dim)
        self.mix_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mix_ffn_norm = nn.LayerNorm(dim)
        self.mix_ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.write_query_norm = nn.LayerNorm(dim)
        self.write_latent_norm = nn.LayerNorm(dim)
        self.write_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.write_output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _attention(module, query, key, value):
        output, _ = module(
            query,
            key,
            value,
            need_weights=False,
        )
        return output

    def read_and_mix(
        self, latents, context, *, collect_diagnostics=False, capture_attention=False
    ):
        diagnostics = OrderedDict()
        attention_map = None
        if self.read_mode == "mha":
            normalized_context = self.read_context_norm(context)
            read_output, attention_map = self.read_attention(
                self.read_query_norm(latents),
                normalized_context,
                normalized_context,
                need_weights=capture_attention,
                average_attn_weights=True,
            )
            latents = latents + read_output
        else:
            latents, diagnostics, attention_map = self.mesh_read(
                latents, context, collect_diagnostics=collect_diagnostics
            )

        normalized_latents = self.mix_attention_norm(latents)
        mixed = self._attention(
            self.mix_attention,
            normalized_latents,
            normalized_latents,
            normalized_latents,
        )
        if self.read_mode == "mesh":
            mix_gate = self.mix_gate_logit.sigmoid().to(dtype=mixed.dtype)
            latents = latents + mix_gate * mixed
            if collect_diagnostics:
                diagnostics["mix_gate"] = mix_gate.detach().float()
        else:
            latents = latents + mixed
        latents = latents + self.mix_ffn(self.mix_ffn_norm(latents))
        return latents, diagnostics, attention_map

    def write(self, context, latents):
        normalized_latents = self.write_latent_norm(latents)
        update = self._attention(
            self.write_attention,
            self.write_query_norm(context),
            normalized_latents,
            normalized_latents,
        )
        return self.write_output_norm(update)


class _StageAdapter(nn.Module):
    """Translate one ResNeSt stage to/from the shared workspace width."""

    def __init__(
        self,
        in_channels,
        workspace_dim,
        *,
        read_downsample=1,
        write_enabled=True,
        gate_min=0.05,
        gate_init=0.10,
    ):
        super().__init__()
        if not 0.0 <= gate_min < gate_init < 1.0:
            raise ValueError("gate values must satisfy 0 <= gate_min < gate_init < 1")

        self.read_pool = (
            nn.AvgPool2d(read_downsample, stride=read_downsample)
            if read_downsample > 1
            else nn.Identity()
        )
        self.read_projection = nn.Conv2d(
            in_channels, workspace_dim, kernel_size=1, bias=False
        )
        self.write_enabled = write_enabled
        if write_enabled:
            self.write_projection = nn.Conv2d(
                workspace_dim, in_channels, kernel_size=1, bias=False
            )
            gate_probability = (gate_init - gate_min) / (1.0 - gate_min)
            gate_logit = math.log(gate_probability / (1.0 - gate_probability))
            self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            self.gate_min = float(gate_min)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.read_projection.weight)
        if self.write_enabled:
            nn.init.xavier_uniform_(self.write_projection.weight)

    def read_tokens(self, features, stage_embedding):
        context = self.read_projection(self.read_pool(features))
        spatial_size = context.shape[-2:]
        context = context.flatten(2).transpose(1, 2)
        context = context + stage_embedding.to(dtype=context.dtype)
        return context, spatial_size

    def gate(self):
        if not self.write_enabled:
            raise RuntimeError("read-only stage has no write gate")
        return self.gate_min + (1.0 - self.gate_min) * self.gate_logit.sigmoid()

    def write_features(self, token_update, spatial_size):
        if not self.write_enabled:
            raise RuntimeError("read-only stage cannot write features")
        height, width = spatial_size
        batch, token_count, channels = token_update.shape
        if token_count != height * width:
            raise ValueError(
                f"write token count {token_count} does not match {height}x{width}"
            )
        update = token_update.transpose(1, 2).reshape(batch, channels, height, width)
        return self.write_projection(update)


class PersistentLatentWorkspace(nn.Module):
    """Carry one latent state from C2 through C5 while preserving the CNN path."""

    stage_names = ("c2", "c3", "c4", "c5")
    stage_channels = {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}

    def __init__(
        self,
        *,
        dim,
        num_slots,
        num_heads,
        mlp_ratio,
        gate_min,
        gate_init,
        dropout,
        read_mode="mha",
        mesh_iterations=4,
        sinkhorn_iterations=5,
        mesh_lr=1.0,
        mesh_noise_std=1e-3,
        mesh_temperature=1.0,
        mix_gate_init=0.05,
        mix_gate_trainable=True,
    ):
        super().__init__()
        if num_slots < 2:
            raise ValueError("persistent workspace requires at least two latent slots")
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.latent_slots = nn.Parameter(torch.empty(1, num_slots, dim))
        self.stage_embeddings = nn.Parameter(torch.empty(len(self.stage_names), 1, dim))
        self.read_mode = str(read_mode).lower()
        self.cell = _SharedWorkspaceCell(
            dim,
            num_slots,
            num_heads,
            mlp_ratio,
            dropout,
            read_mode=self.read_mode,
            mesh_iterations=mesh_iterations,
            sinkhorn_iterations=sinkhorn_iterations,
            mesh_lr=mesh_lr,
            mesh_noise_std=mesh_noise_std,
            mesh_temperature=mesh_temperature,
            mix_gate_init=mix_gate_init,
            mix_gate_trainable=mix_gate_trainable,
        )
        self.adapters = nn.ModuleDict(
            {
                "c2": _StageAdapter(
                    256,
                    dim,
                    read_downsample=2,
                    write_enabled=False,
                    gate_min=gate_min,
                    gate_init=gate_init,
                ),
                "c3": _StageAdapter(
                    512, dim, gate_min=gate_min, gate_init=gate_init
                ),
                "c4": _StageAdapter(
                    1024, dim, gate_min=gate_min, gate_init=gate_init
                ),
                "c5": _StageAdapter(
                    2048, dim, gate_min=gate_min, gate_init=gate_init
                ),
            }
        )
        self._last_diagnostics = OrderedDict()
        self._capture_attention = False
        self._last_attention_maps = OrderedDict()
        self.reset_parameters()

    @classmethod
    def from_config(cls, cfg):
        workspace = cfg.MODEL.LATENT_WORKSPACE
        return cls(
            dim=workspace.DIM,
            num_slots=workspace.NUM_SLOTS,
            num_heads=workspace.NUM_HEADS,
            mlp_ratio=workspace.MLP_RATIO,
            gate_min=workspace.GATE_MIN,
            gate_init=workspace.GATE_INIT,
            dropout=workspace.ATTN_DROPOUT,
            read_mode=workspace.READ_MODE,
            mesh_iterations=workspace.MESH_ITERS,
            sinkhorn_iterations=workspace.SINKHORN_ITERS,
            mesh_lr=workspace.MESH_LR,
            mesh_noise_std=workspace.MESH_NOISE_STD,
            mesh_temperature=workspace.MESH_TEMPERATURE,
            mix_gate_init=workspace.MIX_GATE_INIT,
            mix_gate_trainable=workspace.MIX_GATE_TRAINABLE,
        )

    def reset_parameters(self):
        nn.init.normal_(self.latent_slots, std=0.02)
        nn.init.normal_(self.stage_embeddings, std=0.02)

    def _initial_latents(self, context):
        return self.latent_slots.expand(context.shape[0], -1, -1).to(dtype=context.dtype)

    def forward_stage(self, stage_name, features, latents=None, collect_diagnostics=False):
        if stage_name not in self.adapters:
            raise KeyError(f"unknown latent workspace stage {stage_name}")
        expected_channels = self.stage_channels[stage_name]
        if features.ndim != 4 or features.shape[1] != expected_channels:
            raise ValueError(
                f"{stage_name} expects [B, {expected_channels}, H, W], got {tuple(features.shape)}"
            )

        stage_index = self.stage_names.index(stage_name)
        adapter = self.adapters[stage_name]
        context, read_spatial_size = adapter.read_tokens(
            features, self.stage_embeddings[stage_index]
        )
        if latents is None:
            latents = self._initial_latents(context)
        latents, read_diagnostics, attention_map = self.cell.read_and_mix(
            latents,
            context,
            collect_diagnostics=collect_diagnostics,
            capture_attention=self._capture_attention,
        )
        if collect_diagnostics:
            for name, value in read_diagnostics.items():
                self._last_diagnostics[f"{stage_name}_{name}"] = value
        if self._capture_attention and attention_map is not None:
            self._last_attention_maps[stage_name] = {
                "attention": attention_map.detach().float().cpu(),
                "spatial_size": tuple(read_spatial_size),
            }

        if not adapter.write_enabled:
            if collect_diagnostics:
                self._last_diagnostics["c2_slot_variance"] = (
                    latents.float().var(dim=1, unbiased=False).mean().detach()
                )
            return features, latents

        token_update = self.cell.write(context, latents)
        feature_update = adapter.write_features(token_update, features.shape[-2:])
        gate = adapter.gate().to(dtype=feature_update.dtype)
        gated_update = gate * feature_update
        enriched_features = features + gated_update

        if collect_diagnostics:
            feature_norm = features.detach().float().norm().clamp_min(1e-12)
            update_ratio = gated_update.detach().float().norm() / feature_norm
            self._last_diagnostics[f"{stage_name}_gate"] = gate.detach().float()
            self._last_diagnostics[f"{stage_name}_write_ratio"] = update_ratio
            if stage_name == "c5":
                normalized_slots = F.normalize(latents.detach().float(), dim=-1)
                similarity = normalized_slots @ normalized_slots.transpose(1, 2)
                mask = ~torch.eye(
                    self.num_slots, device=similarity.device, dtype=torch.bool
                )[None]
                self._last_diagnostics["slot_cosine"] = similarity.masked_select(mask).mean()
                self._last_diagnostics["slot_variance"] = (
                    latents.detach().float().var(dim=1, unbiased=False).mean()
                )
                normalized_initial = F.normalize(
                    self.latent_slots.detach().float(), dim=-1
                )
                initial_similarity = normalized_initial @ normalized_initial.transpose(1, 2)
                self._last_diagnostics["raw_slot_cosine"] = initial_similarity.masked_select(
                    mask
                ).mean()
        return enriched_features, latents

    def diagnostics(self):
        return OrderedDict(self._last_diagnostics)

    def set_attention_capture(self, enabled=True):
        self._capture_attention = bool(enabled)
        self._last_attention_maps.clear()

    def attention_maps(self):
        return OrderedDict(self._last_attention_maps)


@META_ARCH_REGISTRY.register()
class LatentWorkspaceBaseline(Baseline):
    """Baseline ReID model with a top-level, non-frozen latent workspace."""

    @configurable
    def __init__(
        self,
        *,
        backbone,
        heads,
        workspace,
        pixel_mean,
        pixel_std,
        loss_kwargs=None,
        health_period=200,
    ):
        """Experimental configurable constructor for the latent-workspace model."""
        super().__init__(
            backbone=backbone,
            heads=heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            loss_kwargs=loss_kwargs,
        )
        self.workspace = workspace
        self.health_period = int(health_period)

    @classmethod
    def from_config(cls, cfg):
        params = Baseline.from_config(cfg)
        params.update(
            workspace=PersistentLatentWorkspace.from_config(cfg),
            health_period=cfg.MODEL.LATENT_WORKSPACE.HEALTH_PERIOD,
        )
        return params

    def _should_collect_diagnostics(self):
        if not self.training or self.health_period <= 0:
            return False
        try:
            storage = get_event_storage()
        except (AssertionError, RuntimeError):
            # Keep direct model forwards usable in unit tests and notebooks.
            return False
        return storage.iter % self.health_period == 0

    def _forward_workspace_backbone(self, images, collect_diagnostics):
        if not hasattr(self.backbone, "forward_stem") or not hasattr(
            self.backbone, "forward_stage"
        ):
            raise TypeError("LatentWorkspaceBaseline currently requires a staged ResNeSt backbone")

        features = self.backbone.forward_stem(images)
        c2 = self.backbone.forward_stage(features, 1)
        c2, latents = self.workspace.forward_stage(
            "c2", c2, collect_diagnostics=collect_diagnostics
        )
        c3 = self.backbone.forward_stage(c2, 2)
        c3, latents = self.workspace.forward_stage(
            "c3", c3, latents, collect_diagnostics=collect_diagnostics
        )
        c4 = self.backbone.forward_stage(c3, 3)
        c4, latents = self.workspace.forward_stage(
            "c4", c4, latents, collect_diagnostics=collect_diagnostics
        )
        c5 = self.backbone.forward_stage(c4, 4)
        c5, _ = self.workspace.forward_stage(
            "c5", c5, latents, collect_diagnostics=collect_diagnostics
        )
        return c5

    def _write_workspace_metrics(self):
        storage = get_event_storage()
        for name, value in self.workspace.diagnostics().items():
            storage.put_scalar(f"latent/{name}", float(value), smoothing_hint=False)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        collect_diagnostics = self._should_collect_diagnostics()
        features = self._forward_workspace_backbone(images, collect_diagnostics)
        if collect_diagnostics:
            self._write_workspace_metrics()

        if self.training:
            if "targets" not in batched_inputs:
                raise KeyError("Person ID annotation are missing in training")
            targets = batched_inputs["targets"]
            if targets.sum() < 0:
                targets.zero_()
            outputs = self.heads(features, targets)
            return self.losses(outputs, targets)
        return self.heads(features)
