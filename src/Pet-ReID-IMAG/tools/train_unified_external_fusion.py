#!/usr/bin/env python3
"""Jointly fit the unified nose adapter and bounded fusion gate on external IDs."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_data import (  # noqa: E402
    UnifiedRawManifestDataset,
    identity_batches,
)
from pet_id.unified_external_protocol import sha256_file  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    build_external_joint_from_base_checkpoint,
    configure_strict_cuda_precision,
    create_external_joint_checkpoint,
    save_external_joint_checkpoint,
)
from pet_id.unified_training import (  # noqa: E402
    relational_distillation,
    supervised_contrastive_loss,
)
from pet_id.release_compatibility import acceptance_protocol_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--identities-per-batch", type=int, default=2)
    parser.add_argument("--adapter-lr", type=float, default=2e-5)
    parser.add_argument("--fusion-lr", type=float, default=5e-3)
    parser.add_argument("--refiner-hidden-dim", type=int, default=32)
    parser.add_argument("--maximum-residual-weight", type=float, default=0.10)
    parser.add_argument("--maximum-interaction-norm", type=float, default=0.05)
    parser.add_argument(
        "--interaction-scale-mode",
        choices=("constant", "reliability"),
        default="constant",
    )
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--cosine-weight", type=float, default=4.0)
    parser.add_argument("--relational-weight", type=float, default=2.0)
    parser.add_argument("--monotonic-weight", type=float, default=2.0)
    parser.add_argument("--drift-weight", type=float, default=1e-4)
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def initial_fusion(
    face: torch.Tensor,
    raw_nose: torch.Tensor,
    confidence: torch.Tensor,
    adapter: torch.nn.Module,
    policy: dict[str, float],
) -> torch.Tensor:
    with torch.no_grad():
        face = F.normalize(face.float(), dim=1)
        nose = F.normalize(adapter(raw_nose.float()).float(), dim=1)
        weight = float(policy["maximum_nose_weight"]) * torch.sigmoid(
            (
                float(policy["face_confidence_threshold"])
                - confidence.float().reshape(-1)
            )
            / float(policy["temperature"])
        )
        return F.normalize(
            face * (1.0 - weight[:, None]) + nose * weight[:, None], dim=1
        )


def monotonic_pair_loss(
    current: torch.Tensor, reference: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    current = F.normalize(current.float(), dim=1)
    reference = F.normalize(reference.float(), dim=1)
    current_similarity = current @ current.T
    reference_similarity = reference @ reference.T
    same = targets[:, None].eq(targets[None, :])
    diagonal = torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    positive = same & ~diagonal
    negative = ~same
    losses = []
    if positive.any():
        losses.append(
            F.relu(reference_similarity[positive] - current_similarity[positive]).mean()
        )
    if negative.any():
        losses.append(
            F.relu(current_similarity[negative] - reference_similarity[negative]).mean()
        )
    return sum(losses, current.new_zeros(()))


def collate(dataset: UnifiedRawManifestDataset, indices: list[int]) -> dict:
    rows = [dataset[index] for index in indices]
    return {
        "rgb": torch.stack([row["rgb"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
    }


def main() -> None:
    args = parse_args()
    precision = configure_strict_cuda_precision()
    initial_path = args.initial_checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (initial_path, acceptance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if (
        acceptance.get("protocol_name")
        != acceptance_protocol_name("external-development")
    ):
        raise RuntimeError("Unexpected external acceptance protocol")
    manifest_path = Path(acceptance["training"]["path"]).resolve()
    if sha256_file(manifest_path) != acceptance["training"]["sha256"]:
        raise RuntimeError("Training manifest differs from acceptance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_split") != "training_extension":
        raise RuntimeError("Fusion training only accepts training_extension")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    base_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    initial_policy = dict(base_payload["model_config"]["fusion"])
    model = build_external_joint_from_base_checkpoint(
        initial_path,
        hidden_dim=args.refiner_hidden_dim,
        maximum_residual_weight=args.maximum_residual_weight,
        maximum_interaction_norm=args.maximum_interaction_norm,
        interaction_scale_mode=args.interaction_scale_mode,
        device=device,
    )
    initial_adapter = copy.deepcopy(model.base_model.nose_adapter).to(device).eval()
    initial_adapter.requires_grad_(False)
    initial_adapter_state = {
        name: value.detach().clone()
        for name, value in model.base_model.nose_adapter.named_parameters()
    }
    model.configure_trainable(nose_adapter=True, refiner=True)
    model.eval()
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable_names or any(
        not (
            name.startswith("base_model.nose_adapter.")
            or name.startswith("refiner.")
        )
        for name in trainable_names
    ):
        raise RuntimeError(f"Unsafe trainable parameter scope: {trainable_names}")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.base_model.nose_adapter.parameters()),
                "lr": args.adapter_lr,
                "weight_decay": 1e-5,
            },
            {
                "params": list(model.refiner.parameters()),
                "lr": args.fusion_lr,
                "weight_decay": 0.0,
            },
        ]
    )
    dataset = UnifiedRawManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=True,
        horizontal_flip_probability=args.horizontal_flip,
        color_jitter=args.color_jitter,
        allow_letterbox_upscale=False,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    step = 0
    epoch = 0
    while step < args.steps:
        batches = identity_batches(
            dataset,
            identities_per_batch=args.identities_per_batch,
            seed=args.seed,
            epoch=epoch,
        )
        for indices in batches:
            if step >= args.steps:
                break
            batch = collate(dataset, indices)
            rgb = batch["rgb"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(rgb, return_aux=True)
                current = output["embedding"].float()
                reference = initial_fusion(
                    output["face_descriptor"],
                    output["raw_nose_descriptor"],
                    output["geometry_confidence"][:, 0],
                    initial_adapter,
                    initial_policy,
                )
                metric = supervised_contrastive_loss(
                    current,
                    targets,
                    temperature=args.temperature,
                )
                cosine = (1.0 - (current * reference).sum(dim=1)).mean()
                relational = relational_distillation(current, reference)
                monotonic = monotonic_pair_loss(current, reference, targets)
                drift = current.new_zeros(())
                for name, value in model.base_model.nose_adapter.named_parameters():
                    drift = drift + (
                        value.float() - initial_adapter_state[name].float()
                    ).square().mean()
                loss = (
                    metric
                    + args.cosine_weight * cosine
                    + args.relational_weight * relational
                    + args.monotonic_weight * monotonic
                    + args.drift_weight * drift
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fusion training loss")
            loss.backward()
            interaction_gradient = model.refiner.interaction[-1].weight.grad
            interaction_gradient_norm = (
                float(interaction_gradient.float().norm())
                if interaction_gradient is not None
                else 0.0
            )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad],
                    max_norm=1.0,
                )
            )
            optimizer.step()
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "metric": float(metric.detach()),
                "cosine": float(cosine.detach()),
                "relational": float(relational.detach()),
                "monotonic": float(monotonic.detach()),
                "drift": float(drift.detach()),
                "gradient_norm": gradient_norm,
                "interaction_final_gradient_norm": interaction_gradient_norm,
                "base_fusion": model.base_model.fusion.configuration(),
                "refiner_global_gain": float(
                    output["refiner_global_gain"].detach()
                ),
                "refiner_weight_mean": float(
                    output["refiner_residual_weight"].detach().mean()
                ),
                "refiner_reliability_mean": float(
                    output["refiner_reliability"].detach().mean()
                ),
                "refiner_interaction_norm_mean": float(
                    (
                        output["refiner_interaction_scale"].detach()[:, None]
                        * output["refiner_interaction"].detach()
                    )
                    .float()
                    .norm(dim=1)
                    .mean()
                ),
                "refiner_interaction_norm_maximum": float(
                    (
                        output["refiner_interaction_scale"].detach()[:, None]
                        * output["refiner_interaction"].detach()
                    )
                    .float()
                    .norm(dim=1)
                    .max()
                ),
            }
            history.append(row)
            if step == 1 or step % max(args.log_every, 1) == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        epoch += 1

    selection = {
        "step": args.steps,
        "rule": (
            "external training only; promotion still requires baseline and external "
            "development noninferiority plus one-shot external blind acceptance"
        ),
    }
    training = {
        "stage": "external_joint_fusion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(initial_path),
        "parent_checkpoint_sha256": sha256_file(initial_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(dataset),
        "identities": dataset.num_classes,
        "steps": args.steps,
        "epochs_started": epoch,
        "identities_per_batch": args.identities_per_batch,
        "images_per_identity": dataset.images_per_identity,
        "trainable_parameters": trainable_names,
        "adapter_lr": args.adapter_lr,
        "fusion_lr": args.fusion_lr,
        "refiner": model.refiner.configuration(),
        "loss_weights": {
            "metric": 1.0,
            "cosine": args.cosine_weight,
            "relational": args.relational_weight,
            "monotonic": args.monotonic_weight,
            "drift": args.drift_weight,
        },
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "cuda_precision": precision,
        "seed": args.seed,
        "blind_data_used": False,
    }
    trained_payload = create_external_joint_checkpoint(
        model,
        base_checkpoint=initial_path,
        training=training,
        selection=selection,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_external_joint_checkpoint(trained_payload, output_path)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "parent_checkpoint_sha256": sha256_file(initial_path),
        "training": training,
        "initial_fusion": initial_policy,
        "final_base_fusion": model.base_model.fusion.configuration(),
        "final_refiner": {
            **model.refiner.configuration(),
            "global_gain": float(
                args.maximum_residual_weight
                * model.refiner.direction_gain_logit.detach().tanh()
            ),
        },
        "first_step": history[0],
        "last_step": history[-1],
        "minimum_loss": min(row["loss"] for row in history),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "history": history,
    }
    summary_path = output_path.with_suffix(".training.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": summary["output_sha256"],
                "summary": str(summary_path),
                "initial_fusion": initial_policy,
                "final_base_fusion": summary["final_base_fusion"],
                "final_refiner": summary["final_refiner"],
                "last_step": history[-1],
                "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
