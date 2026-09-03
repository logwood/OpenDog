# encoding: utf-8
"""Persistent multi-stage latent workspace for ResNeSt image ReID."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import any_softmax
from fastreid.modeling.losses import (
    cross_entropy_loss,
    pairwise_circleloss,
    pairwise_cosface,
    triplet_loss,
)
from fastreid.utils.events import get_event_storage

from .baseline import Baseline
from .build import META_ARCH_REGISTRY


def _slot_statistics(slots):
    """Return scale-aware diversity statistics for a [B, K, D] slot tensor."""
    values = slots.detach().float()
    normalized = F.normalize(values, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    mask = ~torch.eye(similarity.shape[-1], device=similarity.device, dtype=torch.bool)[
        None
    ]
    pairwise = similarity.masked_select(mask)

    centered = values - values.mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probability = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-20)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-20).log()).sum(dim=1)
    ).mean()
    return OrderedDict(
        cosine_mean=pairwise.mean(),
        cosine_max=pairwise.max(),
        variance=values.var(dim=1, unbiased=False).mean(),
        effective_rank=effective_rank,
        norm=values.norm(dim=-1).mean(),
    )


def _orthogonal_role_anchors(num_slots, dim):
    """Build deterministic zero-mean DCT rows with equal norm and no parameters."""
    if num_slots > dim:
        raise ValueError("number of role anchors cannot exceed workspace dimension")
    positions = torch.arange(dim, dtype=torch.float32) + 0.5
    frequencies = torch.arange(1, num_slots + 1, dtype=torch.float32).unsqueeze(1)
    anchors = torch.cos(math.pi * frequencies * positions.unsqueeze(0) / float(dim))
    return F.normalize(anchors, dim=1) * math.sqrt(dim)


def _sincos_2d_position(height, width, dim, *, device, dtype):
    """Create a deterministic [1, H*W, D] two-dimensional position encoding."""
    if dim % 4 != 0:
        raise ValueError(
            "workspace dimension must be divisible by four for 2D position encoding"
        )
    quarter = dim // 4
    exponent = torch.arange(quarter, device=device, dtype=torch.float32)
    exponent = exponent / max(quarter - 1, 1)
    frequency = torch.exp(-math.log(10000.0) * exponent)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    y_phase = math.pi * y[:, None] * frequency[None]
    x_phase = math.pi * x[:, None] * frequency[None]
    y_embedding = torch.cat((y_phase.sin(), y_phase.cos()), dim=1)
    x_embedding = torch.cat((x_phase.sin(), x_phase.cos()), dim=1)
    position = torch.cat(
        (
            y_embedding[:, None].expand(-1, width, -1),
            x_embedding[None].expand(height, -1, -1),
        ),
        dim=2,
    )
    return position.reshape(1, height * width, dim).to(dtype=dtype)


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

    log_transport = log_kernel + dual_token.unsqueeze(2) + dual_slot.unsqueeze(1)
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
    def _minimize_sinkhorn_entropy(self, cost, token_marginal, slot_marginal, noise):
        optimized_cost = cost + noise
        if not optimized_cost.requires_grad:
            optimized_cost.requires_grad_(True)
        initial_entropy = None
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
            entropy = (
                -(transport.clamp_min(1e-20) * transport.clamp_min(1e-20).log())
                .sum(dim=(1, 2))
                .mean()
            )
            if initial_entropy is None:
                initial_entropy = entropy.detach()
            (gradient,) = torch.autograd.grad(
                entropy, optimized_cost, retain_graph=True
            )
            gradient_norm = (
                gradient.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-20)
            )
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
        final_entropy = (
            -(transport.clamp_min(1e-20) * transport.clamp_min(1e-20).log())
            .sum(dim=(1, 2))
            .mean()
        )
        return transport, initial_entropy, final_entropy.detach()

    def forward(
        self,
        latents,
        context,
        *,
        role_anchor=None,
        update_alpha=None,
        collect_diagnostics=False,
    ):
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
            query_input = normalized_slots
            if role_anchor is not None:
                query_input = query_input + role_anchor.float()
            query = self.query_projection(query_input)
            key = self.key_projection(normalized_tokens)
            value = self.value_projection(normalized_tokens)
            similarity = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(self.dim)
            cost = -similarity
            token_marginal = (
                self.token_marginal(normalized_tokens).squeeze(-1).softmax(dim=1)
            )
            slot_marginal = torch.full(
                (slots.shape[0], self.num_slots),
                1.0 / self.num_slots,
                device=slots.device,
                dtype=slots.dtype,
            )
            transport, initial_entropy, final_entropy = self._minimize_sinkhorn_entropy(
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
            proposal = self.update(
                slot_update.reshape(-1, self.dim), slots.reshape(-1, self.dim)
            ).view_as(slots)
            if update_alpha is None:
                updated_slots = proposal
            else:
                alpha = torch.as_tensor(
                    update_alpha, device=slots.device, dtype=slots.dtype
                )
                updated_slots = slots + alpha * (proposal - slots)

            diagnostics = OrderedDict()
            if collect_diagnostics:
                assignment = transport / token_marginal.unsqueeze(2).clamp_min(1e-12)
                assignment = assignment / assignment.sum(dim=2, keepdim=True).clamp_min(
                    1e-12
                )
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
                read_stats = _slot_statistics(slot_update)
                proposal_stats = _slot_statistics(proposal)
                updated_stats = _slot_statistics(updated_slots)
                state_norm = slots.norm().clamp_min(1e-12)
                diagnostics["transport_entropy_before"] = initial_entropy
                diagnostics["transport_entropy_after"] = final_entropy
                diagnostics["transport_entropy_delta"] = final_entropy - initial_entropy
                diagnostics["read_vector_cosine"] = read_stats["cosine_mean"]
                diagnostics["proposal_cosine"] = proposal_stats["cosine_mean"]
                diagnostics["post_update_cosine"] = updated_stats["cosine_mean"]
                diagnostics["post_update_cosine_max"] = updated_stats["cosine_max"]
                diagnostics["gru_proposal_delta_ratio"] = (
                    proposal - slots
                ).norm().detach() / state_norm
                diagnostics["applied_update_ratio"] = (
                    updated_slots - slots
                ).norm().detach() / state_norm
                diagnostics["assignment_entropy"] = normalized_entropy.detach()
                diagnostics["effective_slots"] = effective_slots.detach()
                diagnostics["slot_mass_min"] = slot_mass.min().detach()
                diagnostics["slot_mass_max"] = slot_mass.max().detach()
                diagnostics["attention_cosine"] = (
                    map_similarity.masked_select(off_diagonal).mean().detach()
                )

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
        mix_enabled,
        ffn_layer_scale,
        mix_role_conditioned=False,
        mix_centered=False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"workspace dim {dim} must be divisible by {num_heads} heads"
            )
        if read_mode not in {"mha", "mesh"}:
            raise ValueError(f"unknown latent read mode {read_mode!r}")
        if ffn_layer_scale < 0:
            raise ValueError("FFN layer scale cannot be negative")
        hidden_dim = int(round(dim * mlp_ratio))
        self.read_mode = read_mode
        self.mix_enabled = bool(mix_enabled)
        self.ffn_layer_scale = float(ffn_layer_scale)
        self.mix_role_conditioned = bool(mix_role_conditioned)
        self.mix_centered = bool(mix_centered)

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
            if self.mix_enabled and not 0.0 < mix_gate_init < 1.0:
                raise ValueError("MESH mix gate init must be between zero and one")
            if self.mix_enabled:
                mix_probability = torch.tensor(float(mix_gate_init))
                self.mix_gate_logit = nn.Parameter(
                    torch.logit(mix_probability),
                    requires_grad=bool(mix_gate_trainable),
                )

        if self.mix_enabled:
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
        self,
        latents,
        context,
        *,
        role_anchor=None,
        update_alpha=None,
        collect_diagnostics=False,
        capture_attention=False,
    ):
        diagnostics = OrderedDict()
        attention_map = None
        if collect_diagnostics:
            pre_read = _slot_statistics(latents)
            diagnostics["pre_read_cosine"] = pre_read["cosine_mean"]
            diagnostics["pre_read_cosine_max"] = pre_read["cosine_max"]
        if self.read_mode == "mha":
            normalized_context = self.read_context_norm(context)
            query = self.read_query_norm(latents)
            if role_anchor is not None:
                query = query + role_anchor.to(dtype=query.dtype)
            read_output, attention_map = self.read_attention(
                query,
                normalized_context,
                normalized_context,
                need_weights=capture_attention,
                average_attn_weights=True,
            )
            latents = latents + read_output
        else:
            latents, read_diagnostics, attention_map = self.mesh_read(
                latents,
                context,
                role_anchor=role_anchor,
                update_alpha=update_alpha,
                collect_diagnostics=collect_diagnostics,
            )
            diagnostics.update(read_diagnostics)

        if collect_diagnostics:
            post_read = _slot_statistics(latents)
            diagnostics["post_read_cosine"] = post_read["cosine_mean"]
        if self.mix_enabled:
            normalized_latents = self.mix_attention_norm(latents)
            mix_query = normalized_latents
            mix_key = normalized_latents
            if self.mix_role_conditioned and role_anchor is not None:
                role = role_anchor.to(dtype=normalized_latents.dtype)
                mix_query = mix_query + role
                mix_key = mix_key + role
            mixed = self._attention(
                self.mix_attention,
                mix_query,
                mix_key,
                normalized_latents,
            )
            if self.mix_centered:
                # Remove the common-mode message that previously pulled every
                # slot in the same direction while preserving inter-slot flow.
                mixed = mixed - mixed.mean(dim=1, keepdim=True)
            if self.read_mode == "mesh":
                mix_gate = self.mix_gate_logit.sigmoid().to(dtype=mixed.dtype)
                latents = latents + mix_gate * mixed
                if collect_diagnostics:
                    diagnostics["mix_gate"] = mix_gate.detach().float()
            else:
                latents = latents + mixed
        latents = latents + self.ffn_layer_scale * self.mix_ffn(
            self.mix_ffn_norm(latents)
        )
        if collect_diagnostics:
            post_ffn = _slot_statistics(latents)
            diagnostics["post_ffn_cosine"] = post_ffn["cosine_mean"]
        return latents, diagnostics, attention_map

    def write(self, context, latents, role_anchor=None):
        normalized_latents = self.write_latent_norm(latents)
        keys = normalized_latents
        if role_anchor is not None:
            keys = keys + role_anchor.to(dtype=keys.dtype)
        update = self._attention(
            self.write_attention,
            self.write_query_norm(context),
            keys,
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
        position_encoding_enabled=False,
        position_encoding_scale=1.0,
        spatial_write_gate_enabled=False,
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
        self.write_enabled = bool(write_enabled)
        self.spatial_write_gate_enabled = bool(spatial_write_gate_enabled)
        self.position_encoding_enabled = bool(position_encoding_enabled)
        self.position_encoding_scale = float(position_encoding_scale)
        self._position_cache = {}
        if write_enabled:
            self.write_projection = nn.Conv2d(
                workspace_dim, in_channels, kernel_size=1, bias=False
            )
            if self.spatial_write_gate_enabled:
                self.spatial_gate_norm = nn.LayerNorm(workspace_dim)
                self.spatial_gate_projection = nn.Linear(
                    workspace_dim, 1, bias=False
                )
            gate_probability = (gate_init - gate_min) / (1.0 - gate_min)
            gate_logit = math.log(gate_probability / (1.0 - gate_probability))
            self.gate_logit = nn.Parameter(
                torch.tensor(gate_logit, dtype=torch.float32)
            )
            self.gate_min = float(gate_min)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.read_projection.weight)
        if self.write_enabled:
            nn.init.xavier_uniform_(self.write_projection.weight)
            if self.spatial_write_gate_enabled:
                nn.init.zeros_(self.spatial_gate_projection.weight)

    def _position_encoding(self, context, spatial_size):
        height, width = spatial_size
        cache_key = (
            height,
            width,
            context.device.type,
            context.device.index,
            context.dtype,
        )
        position = self._position_cache.get(cache_key)
        if position is None:
            position = _sincos_2d_position(
                height,
                width,
                context.shape[-1],
                device=context.device,
                dtype=context.dtype,
            )
            self._position_cache[cache_key] = position
        return position

    def read_tokens(self, features, stage_embedding):
        context = self.read_projection(self.read_pool(features))
        spatial_size = context.shape[-2:]
        context = context.flatten(2).transpose(1, 2)
        context = context + stage_embedding.to(dtype=context.dtype)
        if self.position_encoding_enabled:
            context = context + self.position_encoding_scale * self._position_encoding(
                context, spatial_size
            )
        return context, spatial_size

    def write_tokens(self, features, stage_embedding):
        if not self.write_enabled:
            raise RuntimeError("read-only stage cannot form write tokens")
        context = self.read_projection(features)
        spatial_size = context.shape[-2:]
        context = context.flatten(2).transpose(1, 2)
        context = context + stage_embedding.to(dtype=context.dtype)
        if self.position_encoding_enabled:
            context = context + self.position_encoding_scale * self._position_encoding(
                context, spatial_size
            )
        return context, spatial_size

    def gate(self):
        if not self.write_enabled:
            raise RuntimeError("read-only stage has no write gate")
        return self.gate_min + (1.0 - self.gate_min) * self.gate_logit.sigmoid()

    def spatial_gate(self, context, token_update):
        if not self.write_enabled or not self.spatial_write_gate_enabled:
            raise RuntimeError("stage does not have a spatial write gate")
        if context.shape != token_update.shape:
            raise ValueError("spatial gate context and update must have equal shape")
        residual_logit = self.spatial_gate_projection(
            self.spatial_gate_norm(context + token_update)
        )
        probability = (self.gate_logit + residual_logit).sigmoid()
        return self.gate_min + (1.0 - self.gate_min) * probability

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
        role_anchor_enabled=False,
        role_anchor_scale=0.25,
        position_encoding_enabled=False,
        position_encoding_scale=1.0,
        update_mode="replace",
        update_alphas=(1.0, 1.0, 1.0, 1.0),
        mix_enabled=True,
        ffn_layer_scale=1.0,
        mix_role_conditioned=False,
        mix_centered=False,
        write_stages=("c3", "c4", "c5"),
        spatial_write_gate_enabled=False,
    ):
        super().__init__()
        if num_slots < 2:
            raise ValueError("persistent workspace requires at least two latent slots")
        update_mode = str(update_mode).lower()
        if update_mode not in {"replace", "residual"}:
            raise ValueError(f"unsupported latent update mode {update_mode}")
        if len(update_alphas) != len(self.stage_names):
            raise ValueError("UPDATE_ALPHAS must contain one value for C2 through C5")
        if any(float(alpha) < 0.0 or float(alpha) > 1.0 for alpha in update_alphas):
            raise ValueError("latent update alphas must be in [0, 1]")
        write_stages = tuple(str(stage).lower() for stage in write_stages)
        unknown_write_stages = set(write_stages).difference(self.stage_names)
        if unknown_write_stages:
            raise ValueError(
                f"unknown latent write stages: {sorted(unknown_write_stages)}"
            )
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.write_stages = write_stages
        self.latent_slots = nn.Parameter(torch.empty(1, num_slots, dim))
        self.stage_embeddings = nn.Parameter(torch.empty(len(self.stage_names), 1, dim))
        self.read_mode = str(read_mode).lower()
        self.role_anchor_enabled = bool(role_anchor_enabled)
        self.role_anchor_scale = float(role_anchor_scale)
        self.update_mode = update_mode
        self.update_alphas = tuple(float(alpha) for alpha in update_alphas)
        anchors = _orthogonal_role_anchors(num_slots, dim).unsqueeze(0)
        self.register_buffer("role_anchors", anchors, persistent=False)
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
            mix_enabled=mix_enabled,
            ffn_layer_scale=ffn_layer_scale,
            mix_role_conditioned=mix_role_conditioned,
            mix_centered=mix_centered,
        )
        self.adapters = nn.ModuleDict(
            {
                "c2": _StageAdapter(
                    256,
                    dim,
                    read_downsample=2,
                    write_enabled="c2" in write_stages,
                    gate_min=gate_min,
                    gate_init=gate_init,
                    position_encoding_enabled=position_encoding_enabled,
                    position_encoding_scale=position_encoding_scale,
                    spatial_write_gate_enabled=spatial_write_gate_enabled,
                ),
                "c3": _StageAdapter(
                    512,
                    dim,
                    write_enabled="c3" in write_stages,
                    gate_min=gate_min,
                    gate_init=gate_init,
                    position_encoding_enabled=position_encoding_enabled,
                    position_encoding_scale=position_encoding_scale,
                    spatial_write_gate_enabled=spatial_write_gate_enabled,
                ),
                "c4": _StageAdapter(
                    1024,
                    dim,
                    write_enabled="c4" in write_stages,
                    gate_min=gate_min,
                    gate_init=gate_init,
                    position_encoding_enabled=position_encoding_enabled,
                    position_encoding_scale=position_encoding_scale,
                    spatial_write_gate_enabled=spatial_write_gate_enabled,
                ),
                "c5": _StageAdapter(
                    2048,
                    dim,
                    write_enabled="c5" in write_stages,
                    gate_min=gate_min,
                    gate_init=gate_init,
                    position_encoding_enabled=position_encoding_enabled,
                    position_encoding_scale=position_encoding_scale,
                    spatial_write_gate_enabled=spatial_write_gate_enabled,
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
            role_anchor_enabled=workspace.ROLE_ANCHOR_ENABLED,
            role_anchor_scale=workspace.ROLE_ANCHOR_SCALE,
            position_encoding_enabled=workspace.POSITION_ENCODING_ENABLED,
            position_encoding_scale=workspace.POSITION_ENCODING_SCALE,
            update_mode=workspace.UPDATE_MODE,
            update_alphas=workspace.UPDATE_ALPHAS,
            mix_enabled=workspace.MIX_ENABLED,
            ffn_layer_scale=workspace.FFN_LAYER_SCALE,
            mix_role_conditioned=workspace.MIX_ROLE_CONDITIONED,
            mix_centered=workspace.MIX_CENTERED,
            write_stages=workspace.WRITE_STAGES,
            spatial_write_gate_enabled=workspace.SPATIAL_WRITE_GATE_ENABLED,
        )

    def reset_parameters(self):
        nn.init.normal_(self.latent_slots, std=0.02)
        nn.init.normal_(self.stage_embeddings, std=0.02)

    def _initial_latents(self, context):
        return self.latent_slots.expand(context.shape[0], -1, -1).to(
            dtype=context.dtype
        )

    def _role_anchor(self, context):
        if not self.role_anchor_enabled:
            return None
        anchors = self.role_anchors.to(device=context.device, dtype=context.dtype)
        return self.role_anchor_scale * anchors.expand(context.shape[0], -1, -1)

    def forward_stage(
        self, stage_name, features, latents=None, collect_diagnostics=False
    ):
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
        role_anchor = self._role_anchor(context)
        update_alpha = (
            self.update_alphas[stage_index] if self.update_mode == "residual" else None
        )
        latents, read_diagnostics, attention_map = self.cell.read_and_mix(
            latents,
            context,
            role_anchor=role_anchor,
            update_alpha=update_alpha,
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

        if collect_diagnostics:
            if stage_name == "c2":
                for name, value in _slot_statistics(latents).items():
                    self._last_diagnostics[f"c2_slot_{name}"] = value
            if stage_name == "c5":
                for name, value in _slot_statistics(latents).items():
                    self._last_diagnostics[f"slot_{name}"] = value
                initial = self.latent_slots.expand(latents.shape[0], -1, -1)
                for name, value in _slot_statistics(initial).items():
                    self._last_diagnostics[f"raw_slot_{name}"] = value

        if not adapter.write_enabled:
            return features, latents

        if tuple(read_spatial_size) == tuple(features.shape[-2:]):
            write_context = context
            write_spatial_size = read_spatial_size
        else:
            write_context, write_spatial_size = adapter.write_tokens(
                features, self.stage_embeddings[stage_index]
            )
        write_role_anchor = self._role_anchor(write_context)
        token_update = self.cell.write(
            write_context,
            latents,
            role_anchor=write_role_anchor,
        )
        feature_update = adapter.write_features(token_update, write_spatial_size)
        if adapter.spatial_write_gate_enabled:
            token_gate = adapter.spatial_gate(write_context, token_update)
            gate_map = token_gate.transpose(1, 2).reshape(
                features.shape[0], 1, *write_spatial_size
            )
            gated_update = gate_map.to(dtype=feature_update.dtype) * feature_update
            gate = token_gate.mean()
        else:
            gate = adapter.gate().to(dtype=feature_update.dtype)
            gated_update = gate * feature_update
        enriched_features = features + gated_update

        if collect_diagnostics:
            feature_norm = features.detach().float().norm().clamp_min(1e-12)
            update_ratio = gated_update.detach().float().norm() / feature_norm
            self._last_diagnostics[f"{stage_name}_gate"] = gate.detach().float()
            if adapter.spatial_write_gate_enabled:
                self._last_diagnostics[f"{stage_name}_gate_min"] = (
                    token_gate.detach().float().min()
                )
                self._last_diagnostics[f"{stage_name}_gate_max"] = (
                    token_gate.detach().float().max()
                )
            self._last_diagnostics[f"{stage_name}_write_ratio"] = update_ratio
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

    def _forward_workspace_backbone(
        self, images, collect_diagnostics, return_latents=False
    ):
        if not hasattr(self.backbone, "forward_stem") or not hasattr(
            self.backbone, "forward_stage"
        ):
            raise TypeError(
                "LatentWorkspaceBaseline currently requires a staged ResNeSt backbone"
            )

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
        c5, latents = self.workspace.forward_stage(
            "c5", c5, latents, collect_diagnostics=collect_diagnostics
        )
        if return_latents:
            return c5, latents
        return c5

    def _write_workspace_metrics(self):
        storage = get_event_storage()
        for name, value in self.workspace.diagnostics().items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
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


class _SlotSetHead(nn.Module):
    """Convert role-anchored slots into a compact retrieval descriptor."""

    def __init__(
        self,
        *,
        workspace_dim,
        num_slots,
        dim_per_slot,
        output_dim,
        num_classes,
        cls_type,
        scale,
        margin,
    ):
        super().__init__()
        descriptor_dim = int(num_slots) * int(dim_per_slot)
        self.slot_norm = nn.LayerNorm(workspace_dim)
        self.slot_projection = nn.Linear(workspace_dim, dim_per_slot, bias=False)
        self.descriptor_norm = nn.BatchNorm1d(descriptor_dim)
        self.descriptor_norm.bias.requires_grad_(False)
        self.fusion_projection = nn.Linear(descriptor_dim, output_dim, bias=False)
        self.weight = nn.Parameter(torch.empty(num_classes, descriptor_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.slot_projection.weight, mode="fan_out")
        nn.init.ones_(self.descriptor_norm.weight)
        nn.init.zeros_(self.descriptor_norm.bias)
        nn.init.zeros_(self.fusion_projection.weight)
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, latents, targets=None):
        projected = F.gelu(self.slot_projection(self.slot_norm(latents)))
        descriptor = self.descriptor_norm(projected.flatten(1))
        feature_delta = self.fusion_projection(descriptor)
        if not self.training:
            return descriptor, feature_delta
        if targets is None:
            raise KeyError("targets are required by the slot-set training head")
        if self.cls_layer.__class__.__name__ == "Linear":
            logits = F.linear(descriptor, self.weight)
        else:
            logits = F.linear(F.normalize(descriptor), F.normalize(self.weight))
        return {
            "cls_outputs": self.cls_layer(logits.clone(), targets),
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": descriptor,
            "feature_delta": feature_delta,
        }


@META_ARCH_REGISTRY.register()
class RoleAnchoredLatentWorkspace(LatentWorkspaceBaseline):
    """Role-preserving latent workspace with an explicit slot-set descriptor."""

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
        slot_set_head=None,
        slot_set_loss_weight=0.2,
        slot_feature_fusion_scale=0.2,
        slot_inference_fusion_weight=0.2,
    ):
        """Experimental configurable constructor for the role-anchored latent workspace."""
        super().__init__(
            backbone=backbone,
            heads=heads,
            workspace=workspace,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            loss_kwargs=loss_kwargs,
            health_period=health_period,
        )
        self.slot_set_head = slot_set_head
        self.slot_set_loss_weight = float(slot_set_loss_weight)
        self.slot_feature_fusion_scale = float(slot_feature_fusion_scale)
        self.slot_inference_fusion_weight = float(slot_inference_fusion_weight)
        if not 0.0 <= self.slot_inference_fusion_weight <= 1.0:
            raise ValueError("slot inference fusion weight must be in [0, 1]")

    @classmethod
    def from_config(cls, cfg):
        params = LatentWorkspaceBaseline.from_config(cfg)
        workspace = cfg.MODEL.LATENT_WORKSPACE
        slot_set_head = None
        if workspace.SLOT_SET_HEAD_ENABLED:
            cls_type = cfg.MODEL.HEADS.CLS_LAYER
            if not hasattr(any_softmax, cls_type):
                raise ValueError(f"unsupported slot-set classifier {cls_type}")
            slot_set_head = _SlotSetHead(
                workspace_dim=workspace.DIM,
                num_slots=workspace.NUM_SLOTS,
                dim_per_slot=workspace.SLOT_SET_DIM_PER_SLOT,
                output_dim=cfg.MODEL.BACKBONE.FEAT_DIM,
                num_classes=cfg.MODEL.HEADS.NUM_CLASSES,
                cls_type=cls_type,
                scale=cfg.MODEL.HEADS.SCALE,
                margin=cfg.MODEL.HEADS.MARGIN,
            )
        params.update(
            slot_set_head=slot_set_head,
            slot_set_loss_weight=workspace.SLOT_SET_LOSS_WEIGHT,
            slot_feature_fusion_scale=workspace.SLOT_FEATURE_FUSION_SCALE,
            slot_inference_fusion_weight=workspace.SLOT_INFERENCE_FUSION_WEIGHT,
        )
        return params

    def _slot_set_losses(self, outputs, targets):
        losses = {}
        loss_names = self.loss_kwargs["loss_names"]
        if "CrossEntropyLoss" in loss_names:
            kwargs = self.loss_kwargs["ce"]
            losses["loss_slot_cls"] = (
                cross_entropy_loss(
                    outputs["cls_outputs"],
                    targets,
                    kwargs["eps"],
                    kwargs["alpha"],
                    kwargs["use_pt"],
                )
                * kwargs["scale"]
            )
        if "TripletLoss" in loss_names:
            kwargs = self.loss_kwargs["tri"]
            losses["loss_slot_triplet"] = (
                triplet_loss(
                    outputs["features"],
                    targets,
                    kwargs["margin"],
                    kwargs["norm_feat"],
                    kwargs["hard_mining"],
                )
                * kwargs["scale"]
            )
        if "CircleLoss" in loss_names:
            kwargs = self.loss_kwargs["circle"]
            losses["loss_slot_circle"] = (
                pairwise_circleloss(
                    outputs["features"], targets, kwargs["margin"], kwargs["gamma"]
                )
                * kwargs["scale"]
            )
        if "Cosface" in loss_names:
            kwargs = self.loss_kwargs["cosface"]
            losses["loss_slot_cosface"] = (
                pairwise_cosface(
                    outputs["features"], targets, kwargs["margin"], kwargs["gamma"]
                )
                * kwargs["scale"]
            )
        return {
            name: value * self.slot_set_loss_weight for name, value in losses.items()
        }

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        collect_diagnostics = self._should_collect_diagnostics()
        features, latents = self._forward_workspace_backbone(
            images, collect_diagnostics, return_latents=True
        )

        targets = (
            batched_inputs.get("targets") if isinstance(batched_inputs, dict) else None
        )
        if self.training:
            if targets is None:
                raise KeyError("Person ID annotation are missing in training")
            if targets.sum() < 0:
                targets.zero_()

        slot_outputs = None
        if self.slot_set_head is not None:
            slot_outputs = self.slot_set_head(
                latents, targets if self.training else None
            )
            feature_delta = (
                slot_outputs["feature_delta"] if self.training else slot_outputs[1]
            )
            feature_update = (
                self.slot_feature_fusion_scale * feature_delta[..., None, None]
            )
            fused_features = features + feature_update
            if collect_diagnostics:
                denominator = features.detach().float().norm().clamp_min(1e-12)
                ratio = feature_update.detach().float().expand_as(features).norm()
                self.workspace._last_diagnostics["slot_set_fusion_ratio"] = (
                    ratio / denominator
                )
        else:
            fused_features = features

        if collect_diagnostics:
            self._write_workspace_metrics()
        if self.training:
            losses = self.losses(self.heads(fused_features, targets), targets)
            if slot_outputs is not None:
                losses.update(self._slot_set_losses(slot_outputs, targets))
            return losses

        main_descriptor = self.heads(fused_features)
        if slot_outputs is None or self.slot_inference_fusion_weight <= 0.0:
            return main_descriptor
        slot_descriptor = slot_outputs[0]
        weight = self.slot_inference_fusion_weight
        return torch.cat(
            (
                math.sqrt(1.0 - weight) * F.normalize(main_descriptor, dim=1),
                math.sqrt(weight) * F.normalize(slot_descriptor, dim=1),
            ),
            dim=1,
        )


class _IdentityQueryHead(nn.Module):
    """Decode an unordered slot set with several complementary identity queries."""

    def __init__(
        self,
        *,
        workspace_dim,
        num_slots,
        num_queries,
        dim_per_query,
        num_heads,
        ffn_scale,
        role_anchor_scale,
        num_classes,
        cls_type,
        scale,
        margin,
    ):
        super().__init__()
        if num_queries < 2:
            raise ValueError("identity query head requires at least two queries")
        if workspace_dim % num_heads != 0:
            raise ValueError("workspace dim must be divisible by identity query heads")
        if ffn_scale < 0:
            raise ValueError("identity query FFN scale cannot be negative")

        self.num_queries = int(num_queries)
        self.ffn_scale = float(ffn_scale)
        self.role_anchor_scale = float(role_anchor_scale)
        descriptor_dim = self.num_queries * int(dim_per_query)
        self.identity_queries = nn.Parameter(
            torch.empty(1, self.num_queries, workspace_dim)
        )
        self.query_norm = nn.LayerNorm(workspace_dim)
        self.slot_norm = nn.LayerNorm(workspace_dim)
        self.cross_attention = nn.MultiheadAttention(
            workspace_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(workspace_dim)
        self.ffn = nn.Sequential(
            nn.Linear(workspace_dim, 2 * workspace_dim),
            nn.GELU(),
            nn.Linear(2 * workspace_dim, workspace_dim),
        )
        self.query_projections = nn.ModuleList(
            [
                nn.Linear(workspace_dim, dim_per_query, bias=False)
                for _ in range(self.num_queries)
            ]
        )
        self.descriptor_norm = nn.BatchNorm1d(descriptor_dim)
        self.descriptor_norm.bias.requires_grad_(False)
        self.weight = nn.Parameter(torch.empty(num_classes, descriptor_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        anchors = _orthogonal_role_anchors(num_slots, workspace_dim).unsqueeze(0)
        self.register_buffer("slot_role_anchors", anchors, persistent=False)
        query_anchors = _orthogonal_role_anchors(num_queries, workspace_dim).unsqueeze(0)
        self.register_buffer("query_role_anchors", query_anchors, persistent=False)
        self._last_diagnostics = OrderedDict()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.identity_queries, std=0.02)
        for projection in self.query_projections:
            nn.init.kaiming_normal_(projection.weight, mode="fan_out")
        nn.init.ones_(self.descriptor_norm.weight)
        nn.init.zeros_(self.descriptor_norm.bias)
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, latents, targets=None, *, collect_diagnostics=False):
        normalized_slots = self.slot_norm(latents)
        role = self.role_anchor_scale * self.slot_role_anchors.to(
            device=latents.device, dtype=latents.dtype
        )
        keys = normalized_slots + role
        initial_queries = self.identity_queries.expand(latents.shape[0], -1, -1)
        query_role = self.role_anchor_scale * self.query_role_anchors.to(
            device=latents.device, dtype=latents.dtype
        )
        read, attention = self.cross_attention(
            self.query_norm(initial_queries) + query_role,
            keys,
            normalized_slots,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        # The shared component dominated all four query states in the first
        # smoke run. The base CNN already carries global evidence, so this head
        # deliberately retains only complementary query-specific messages.
        read = read - read.mean(dim=1, keepdim=True)
        query_states = initial_queries + read
        query_states = query_states + self.ffn_scale * self.ffn(
            self.ffn_norm(query_states)
        )
        projected = [
            F.gelu(projection(query_states[:, index]))
            for index, projection in enumerate(self.query_projections)
        ]
        descriptor = self.descriptor_norm(torch.cat(projected, dim=1))

        if collect_diagnostics:
            statistics = _slot_statistics(query_states)
            for name, value in statistics.items():
                self._last_diagnostics[f"query_{name}"] = value
            normalized_attention = F.normalize(attention.detach().float(), dim=2)
            attention_similarity = (
                normalized_attention @ normalized_attention.transpose(1, 2)
            )
            mask = ~torch.eye(
                self.num_queries,
                device=attention_similarity.device,
                dtype=torch.bool,
            )[None]
            self._last_diagnostics["query_attention_cosine"] = (
                attention_similarity.masked_select(mask).mean()
            )
            self._last_diagnostics["query_attention_entropy"] = (
                -(
                    attention.clamp_min(1e-12)
                    * attention.clamp_min(1e-12).log()
                ).sum(dim=2).mean()
                / math.log(attention.shape[2])
            ).detach()

        if not self.training:
            return descriptor
        if targets is None:
            raise KeyError("targets are required by the identity query head")
        if self.cls_layer.__class__.__name__ == "Linear":
            logits = F.linear(descriptor, self.weight)
        else:
            logits = F.linear(F.normalize(descriptor), F.normalize(self.weight))
        return {
            "cls_outputs": self.cls_layer(logits.clone(), targets),
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": descriptor,
        }

    def diagnostics(self):
        return OrderedDict(self._last_diagnostics)


@META_ARCH_REGISTRY.register()
class SpatialQueryLatentWorkspace(LatentWorkspaceBaseline):
    """Competitive multi-scale workspace with query-decoded identity evidence."""

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
        identity_query_head=None,
        identity_query_loss_weight=0.2,
        fusion_loss_weight=0.1,
        fusion_weight_min=0.0,
        fusion_weight_max=0.5,
        fusion_weight_init=0.05,
    ):
        """Experimental configurable constructor for the spatial-query latent workspace."""
        super().__init__(
            backbone=backbone,
            heads=heads,
            workspace=workspace,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            loss_kwargs=loss_kwargs,
            health_period=health_period,
        )
        if not 0.0 <= fusion_weight_min < fusion_weight_init < fusion_weight_max:
            raise ValueError("identity fusion weights must satisfy min < init < max")
        if fusion_weight_max > 1.0:
            raise ValueError("identity fusion maximum cannot exceed one")
        self.identity_query_head = identity_query_head
        self.identity_query_loss_weight = float(identity_query_loss_weight)
        self.fusion_loss_weight = float(fusion_loss_weight)
        self.fusion_weight_min = float(fusion_weight_min)
        self.fusion_weight_max = float(fusion_weight_max)
        probability = (fusion_weight_init - fusion_weight_min) / (
            fusion_weight_max - fusion_weight_min
        )
        self.fusion_weight_logit = nn.Parameter(torch.logit(torch.tensor(probability)))

    @classmethod
    def from_config(cls, cfg):
        params = LatentWorkspaceBaseline.from_config(cfg)
        workspace = cfg.MODEL.LATENT_WORKSPACE
        identity_query_head = None
        if workspace.IDENTITY_QUERY_HEAD_ENABLED:
            cls_type = cfg.MODEL.HEADS.CLS_LAYER
            if not hasattr(any_softmax, cls_type):
                raise ValueError(f"unsupported identity query classifier {cls_type}")
            identity_query_head = _IdentityQueryHead(
                workspace_dim=workspace.DIM,
                num_slots=workspace.NUM_SLOTS,
                num_queries=workspace.IDENTITY_QUERY_NUM,
                dim_per_query=workspace.IDENTITY_QUERY_DIM,
                num_heads=workspace.IDENTITY_QUERY_NUM_HEADS,
                ffn_scale=workspace.IDENTITY_QUERY_FFN_SCALE,
                role_anchor_scale=workspace.ROLE_ANCHOR_SCALE,
                num_classes=cfg.MODEL.HEADS.NUM_CLASSES,
                cls_type=cls_type,
                scale=cfg.MODEL.HEADS.SCALE,
                margin=cfg.MODEL.HEADS.MARGIN,
            )
        params.update(
            identity_query_head=identity_query_head,
            identity_query_loss_weight=workspace.IDENTITY_QUERY_LOSS_WEIGHT,
            fusion_loss_weight=workspace.IDENTITY_FUSION_LOSS_WEIGHT,
            fusion_weight_min=workspace.IDENTITY_FUSION_WEIGHT_MIN,
            fusion_weight_max=workspace.IDENTITY_FUSION_WEIGHT_MAX,
            fusion_weight_init=workspace.IDENTITY_FUSION_WEIGHT_INIT,
        )
        return params

    def _fusion_weight(self):
        span = self.fusion_weight_max - self.fusion_weight_min
        return self.fusion_weight_min + span * self.fusion_weight_logit.sigmoid()

    def _fused_descriptor(self, main_descriptor, query_descriptor):
        weight = self._fusion_weight().to(dtype=main_descriptor.dtype)
        return torch.cat(
            (
                torch.sqrt(1.0 - weight) * F.normalize(main_descriptor, dim=1),
                torch.sqrt(weight) * F.normalize(query_descriptor, dim=1),
            ),
            dim=1,
        )

    def _identity_query_losses(self, outputs, targets):
        losses = {}
        loss_names = self.loss_kwargs["loss_names"]
        if "CrossEntropyLoss" in loss_names:
            kwargs = self.loss_kwargs["ce"]
            losses["loss_query_cls"] = cross_entropy_loss(
                outputs["cls_outputs"],
                targets,
                kwargs["eps"],
                kwargs["alpha"],
                kwargs["use_pt"],
            ) * kwargs["scale"]
        if "TripletLoss" in loss_names:
            kwargs = self.loss_kwargs["tri"]
            losses["loss_query_triplet"] = triplet_loss(
                outputs["features"],
                targets,
                kwargs["margin"],
                kwargs["norm_feat"],
                kwargs["hard_mining"],
            ) * kwargs["scale"]
        if "CircleLoss" in loss_names:
            kwargs = self.loss_kwargs["circle"]
            losses["loss_query_circle"] = pairwise_circleloss(
                outputs["features"], targets, kwargs["margin"], kwargs["gamma"]
            ) * kwargs["scale"]
        if "Cosface" in loss_names:
            kwargs = self.loss_kwargs["cosface"]
            losses["loss_query_cosface"] = pairwise_cosface(
                outputs["features"], targets, kwargs["margin"], kwargs["gamma"]
            ) * kwargs["scale"]
        return {
            name: value * self.identity_query_loss_weight
            for name, value in losses.items()
        }

    def _fused_metric_losses(self, features, targets):
        losses = {}
        loss_names = self.loss_kwargs["loss_names"]
        if "TripletLoss" in loss_names:
            kwargs = self.loss_kwargs["tri"]
            losses["loss_fused_triplet"] = triplet_loss(
                features,
                targets,
                kwargs["margin"],
                kwargs["norm_feat"],
                kwargs["hard_mining"],
            ) * kwargs["scale"]
        if "CircleLoss" in loss_names:
            kwargs = self.loss_kwargs["circle"]
            losses["loss_fused_circle"] = pairwise_circleloss(
                features, targets, kwargs["margin"], kwargs["gamma"]
            ) * kwargs["scale"]
        if "Cosface" in loss_names:
            kwargs = self.loss_kwargs["cosface"]
            losses["loss_fused_cosface"] = pairwise_cosface(
                features, targets, kwargs["margin"], kwargs["gamma"]
            ) * kwargs["scale"]
        return {name: value * self.fusion_loss_weight for name, value in losses.items()}

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        collect_diagnostics = self._should_collect_diagnostics()
        features, latents = self._forward_workspace_backbone(
            images, collect_diagnostics, return_latents=True
        )
        targets = (
            batched_inputs.get("targets") if isinstance(batched_inputs, dict) else None
        )
        if self.training:
            if targets is None:
                raise KeyError("Person ID annotation are missing in training")
            if targets.sum() < 0:
                targets.zero_()

        query_outputs = None
        if self.identity_query_head is not None:
            query_outputs = self.identity_query_head(
                latents,
                targets if self.training else None,
                collect_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics:
                for name, value in self.identity_query_head.diagnostics().items():
                    self.workspace._last_diagnostics[f"identity_{name}"] = value
                self.workspace._last_diagnostics["identity_fusion_weight"] = (
                    self._fusion_weight().detach()
                )

        if collect_diagnostics:
            self._write_workspace_metrics()
        if self.training:
            main_outputs = self.heads(features, targets)
            losses = self.losses(main_outputs, targets)
            if query_outputs is not None:
                losses.update(self._identity_query_losses(query_outputs, targets))
                fused = self._fused_descriptor(
                    main_outputs["features"], query_outputs["features"]
                )
                losses.update(self._fused_metric_losses(fused, targets))
            return losses

        main_descriptor = self.heads(features)
        if query_outputs is None:
            return main_descriptor
        return self._fused_descriptor(main_descriptor, query_outputs)
