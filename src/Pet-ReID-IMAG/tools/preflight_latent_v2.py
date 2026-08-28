# encoding: utf-8
"""Zero-training structural gate for the persistent latent workspace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastreid.config import get_cfg  # noqa: E402
from fastreid.modeling.meta_arch.latent_workspace import (  # noqa: E402
    PersistentLatentWorkspace,
)
from pet_id import add_retri_config  # noqa: E402
from pet_id.workspace_paths import normalize_runtime_config  # noqa: E402


STAGE_CHANNELS = OrderedDict(c2=256, c3=512, c4=1024, c5=2048)
DEFAULT_SPATIAL_SIZES = dict(c2=16, c3=8, c4=4, c5=2)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic forward-only audit before spending time on training."
        )
    )
    parser.add_argument(
        "--config-file",
        default="configs/modern_latent_v2_s101_224.yaml",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional full-model or workspace checkpoint to audit.",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional torch file containing c2/c3/c4/c5 feature tensors.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the complete machine-readable report.",
    )
    return parser.parse_args()


def _resolve_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(choice)


def _load_cfg(config_file):
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_file))
    normalize_runtime_config(cfg)
    cfg.freeze()
    return cfg


def _load_feature_cache(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "features" in payload:
        payload = payload["features"]
    if not isinstance(payload, dict):
        raise TypeError(
            "feature cache must be a dict or contain a dict under 'features'"
        )
    return payload


def _load_workspace_weights(workspace, path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state dict")

    prefixed = {
        name[len("workspace.") :]: value
        for name, value in state.items()
        if name.startswith("workspace.")
    }
    if prefixed:
        state = prefixed
    else:
        known = set(workspace.state_dict())
        state = {name: value for name, value in state.items() if name in known}
    if not state:
        raise KeyError("checkpoint does not contain latent workspace parameters")

    incompatible = workspace.load_state_dict(state, strict=False)
    return {
        "path": str(path),
        "loaded_tensors": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _make_features(args, device):
    if args.feature_cache:
        payload = _load_feature_cache(args.feature_cache)
        missing = [name for name in STAGE_CHANNELS if name not in payload]
        if missing:
            raise KeyError(f"feature cache is missing stages: {missing}")
        features = OrderedDict()
        for name, channels in STAGE_CHANNELS.items():
            value = payload[name]
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ValueError(f"{name} cache entry must be a BCHW tensor")
            if value.shape[1] != channels:
                raise ValueError(
                    f"{name} cache entry has {value.shape[1]} channels, expected {channels}"
                )
            features[name] = value[: args.batch_size].to(
                device=device, dtype=torch.float32
            )
        batch_sizes = {value.shape[0] for value in features.values()}
        if batch_sizes != {args.batch_size}:
            raise ValueError(
                f"feature cache must contain at least {args.batch_size} samples per stage"
            )
        return features, "cache"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    features = OrderedDict()
    for name, channels in STAGE_CHANNELS.items():
        spatial = DEFAULT_SPATIAL_SIZES[name]
        value = torch.randn(
            args.batch_size,
            channels,
            spatial,
            spatial,
            generator=generator,
        )
        features[name] = value.to(device)
    return features, "synthetic"


def _run_stages(workspace, features, initial_latents=None, collect=False):
    latents = initial_latents
    for name, value in features.items():
        _, latents = workspace.forward_stage(
            name,
            value,
            latents,
            collect_diagnostics=collect,
        )
    return latents


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _build_report(cfg, workspace, features, source, checkpoint):
    workspace.eval()
    with torch.no_grad():
        final_latents = _run_stages(workspace, features, collect=True)

        initial = workspace.latent_slots.expand(features["c2"].shape[0], -1, -1)
        direction = workspace.role_anchors[:, :1].to(initial)
        direction = F.normalize(direction, dim=-1)
        epsilon = max(initial.detach().float().norm().item(), 1.0) * 1e-4
        perturbed = initial.clone()
        perturbed[:, :1] = perturbed[:, :1] + epsilon * direction
        reference_final = _run_stages(workspace, features, initial)
        perturbed_final = _run_stages(workspace, features, perturbed)

    initial_distance = (perturbed - initial).float().norm().clamp_min(1e-20)
    final_distance = (perturbed_final - reference_final).float().norm()
    sensitivity_ratio = _as_float(final_distance / initial_distance)

    normalized_anchors = F.normalize(workspace.role_anchors.detach().float(), dim=-1)
    anchor_gram = normalized_anchors @ normalized_anchors.transpose(1, 2)
    identity = torch.eye(workspace.num_slots, device=anchor_gram.device)[None]
    anchor_error = _as_float((anchor_gram - identity).abs().max())

    diagnostics = OrderedDict(
        (name, _as_float(value)) for name, value in workspace.diagnostics().items()
    )
    finite_diagnostics = all(math.isfinite(value) for value in diagnostics.values())
    entropy_deltas = [
        value
        for name, value in diagnostics.items()
        if name.endswith("transport_entropy_delta")
    ]

    checks = OrderedDict(
        finite_diagnostics=finite_diagnostics,
        orthogonal_role_anchors=anchor_error <= 1e-4,
        diverse_final_slots=diagnostics.get("slot_cosine_max", 1.0) < 0.995,
        usable_effective_rank=diagnostics.get("slot_effective_rank", 0.0) >= 2.0,
        mesh_entropy_not_increasing=bool(entropy_deltas)
        and max(entropy_deltas) <= 1e-3,
        perturbation_not_erased=sensitivity_ratio >= 0.01,
        perturbation_not_exploding=sensitivity_ratio <= 10.0,
        finite_final_latents=bool(torch.isfinite(final_latents).all()),
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config": str(cfg.OUTPUT_DIR),
        "feature_source": source,
        "checkpoint": checkpoint,
        "device": str(final_latents.device),
        "batch_size": int(final_latents.shape[0]),
        "anchor_max_orthogonality_error": anchor_error,
        "perturbation_sensitivity_ratio": sensitivity_ratio,
        "checks": checks,
        "diagnostics": diagnostics,
    }


def main():
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    cfg = _load_cfg(args.config_file)
    workspace = PersistentLatentWorkspace.from_config(cfg)
    checkpoint = (
        _load_workspace_weights(workspace, args.weights) if args.weights else None
    )
    workspace = workspace.to(device)
    features, source = _make_features(args, device)
    report = _build_report(cfg, workspace, features, source, checkpoint)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
