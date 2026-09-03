#!/usr/bin/env python3
"""Train the bounded V4 high-resolution detail refiner without touching V3."""

from __future__ import annotations

import argparse
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

from pet_id.unified_external_data import identity_batches  # noqa: E402
from pet_id.unified_external_model import configure_strict_cuda_precision  # noqa: E402
from pet_id.unified_fresh_protocol import sha256_file  # noqa: E402
from pet_id.unified_highres import (  # noqa: E402
    build_highres_from_parent_checkpoint,
    create_highres_checkpoint,
    save_highres_checkpoint,
)
from pet_id.unified_highres_data import (  # noqa: E402
    UnifiedHighResolutionTrainingDataset,
)
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402
from pet_id.unified_training import (  # noqa: E402
    different_identity_permutation,
    relational_distillation,
    supervised_contrastive_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--identities-per-batch", type=int, default=2)
    parser.add_argument("--training-size", type=int, default=2048)
    parser.add_argument("--degraded-detail-size", type=int, default=1280)
    parser.add_argument("--fusion-lr", type=float, default=2e-3)
    parser.add_argument("--refiner-hidden-dim", type=int, default=64)
    parser.add_argument("--maximum-detail-weight", type=float, default=0.08)
    parser.add_argument("--maximum-interaction-norm", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--cosine-weight", type=float, default=4.0)
    parser.add_argument("--relational-weight", type=float, default=2.0)
    parser.add_argument("--monotonic-weight", type=float, default=2.0)
    parser.add_argument("--cross-resolution-weight", type=float, default=3.0)
    parser.add_argument("--degraded-anchor-weight", type=float, default=2.0)
    parser.add_argument("--conflict-weight", type=float, default=1.0)
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def monotonic_pair_loss(
    current: torch.Tensor,
    reference: torch.Tensor,
    targets: torch.Tensor,
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


def collate(
    dataset: UnifiedHighResolutionTrainingDataset,
    indices: list[int],
) -> dict[str, torch.Tensor]:
    rows = [dataset[index] for index in indices]
    return {
        "high_rgb": torch.stack([row["high_rgb"] for row in rows]),
        "degraded_rgb": torch.stack([row["degraded_rgb"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
    }


def validate_protocol_lock(path: Path) -> tuple[dict, Path]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 protocol lock")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 protocol must remain locked and unscored")
    policy = lock.get("policy", {})
    for key in (
        "v4_identity_disjoint",
        "exact_image_disjoint",
        "blind_single_candidate_attempt",
        "blind_training_forbidden",
        "blind_model_selection_forbidden",
        "blind_features_must_not_be_persisted",
        "failed_candidate_keeps_v3_default",
    ):
        if policy.get(key) is not True:
            raise RuntimeError(f"V4 protocol policy is missing {key}")
    training = lock["splits"]["training_extension"]
    manifest_path = Path(training["path"]).expanduser().resolve()
    if sha256_file(manifest_path) != training["sha256"]:
        raise RuntimeError("V4 training manifest changed after lock")
    return lock, manifest_path


def main() -> None:
    args = parse_args()
    precision = configure_strict_cuda_precision()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    lock_path = args.protocol_lock.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (parent_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    lock, manifest_path = validate_protocol_lock(lock_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model = build_highres_from_parent_checkpoint(
        parent_path,
        refiner_hidden_dim=args.refiner_hidden_dim,
        maximum_detail_weight=args.maximum_detail_weight,
        maximum_interaction_norm=args.maximum_interaction_norm,
        device=device,
    )
    model.configure_trainable(refiner=True)
    model.train()
    trainable_names = [
        name for name, value in model.named_parameters() if value.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("refiner.") for name in trainable_names
    ):
        raise RuntimeError(f"Unsafe V4 trainable parameter scope: {trainable_names}")
    optimizer = torch.optim.AdamW(
        model.refiner.parameters(),
        lr=args.fusion_lr,
        weight_decay=0.0,
    )
    dataset = UnifiedHighResolutionTrainingDataset(
        manifest_path,
        training_size=args.training_size,
        degraded_detail_size=args.degraded_detail_size,
        horizontal_flip_probability=args.horizontal_flip,
        color_jitter=args.color_jitter,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict] = []
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
            high_rgb = batch["high_rgb"].to(device, non_blocking=True)
            degraded_rgb = batch["degraded_rgb"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                high = model(high_rgb, return_aux=True)
                degraded = model(degraded_rgb, return_aux=True)
                current = high["embedding"].float()
                parent = high["highres_parent_embedding"].float()
                degraded_current = degraded["embedding"].float()
                degraded_parent = degraded["highres_parent_embedding"].float()

                metric = supervised_contrastive_loss(
                    current,
                    targets,
                    temperature=args.temperature,
                )
                cosine = (1.0 - (current * parent).sum(dim=1)).mean()
                relational = relational_distillation(current, parent)
                monotonic = monotonic_pair_loss(current, parent, targets)
                cross_resolution = (
                    1.0 - (current * degraded_current).sum(dim=1)
                ).mean()
                degraded_anchor = (
                    1.0 - (degraded_current * degraded_parent).sum(dim=1)
                ).mean()

                permutation, conflict_valid = different_identity_permutation(targets)
                conflict = current.new_zeros(())
                if conflict_valid.any():
                    conflict_output = model.refiner(
                        high["highres_parent_embedding"],
                        high["detail_face_descriptor"],
                        high["detail_nose_descriptor"].index_select(0, permutation),
                        high["geometry_confidence"][:, 0],
                        high["detail_scale"],
                        high["detail_availability"],
                        high["detail_signals"][:, 8],
                        high["detail_signals"][:, 9].index_select(0, permutation),
                    ).float()
                    conflict = (
                        1.0
                        - (
                            conflict_output[conflict_valid]
                            * parent[conflict_valid]
                        ).sum(dim=1)
                    ).mean()
                loss = (
                    metric
                    + args.cosine_weight * cosine
                    + args.relational_weight * relational
                    + args.monotonic_weight * monotonic
                    + args.cross_resolution_weight * cross_resolution
                    + args.degraded_anchor_weight * degraded_anchor
                    + args.conflict_weight * conflict
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite V4 training loss")
            loss.backward()
            direction_gradient = model.refiner.direction_gain_logit.grad
            interaction_gradient = model.refiner.interaction[-1].weight.grad
            reliability_gradient = model.refiner.reliability[-1].weight.grad
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.0)
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
                "cross_resolution": float(cross_resolution.detach()),
                "degraded_anchor": float(degraded_anchor.detach()),
                "conflict": float(conflict.detach()),
                "gradient_norm": gradient_norm,
                "direction_gradient": (
                    float(direction_gradient.detach().abs())
                    if direction_gradient is not None
                    else 0.0
                ),
                "interaction_final_gradient": (
                    float(interaction_gradient.detach().float().norm())
                    if interaction_gradient is not None
                    else 0.0
                ),
                "reliability_final_gradient": (
                    float(reliability_gradient.detach().float().norm())
                    if reliability_gradient is not None
                    else 0.0
                ),
                "detail_global_gain": float(
                    model.refiner.maximum_detail_weight
                    * model.refiner.direction_gain_logit.detach().tanh()
                ),
                "detail_weight_face_mean": float(
                    high["detail_weights"][:, 0].detach().float().mean()
                ),
                "detail_weight_nose_mean": float(
                    high["detail_weights"][:, 1].detach().float().mean()
                ),
            }
            history.append(row)
            if step == 1 or step % max(args.log_every, 1) == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        epoch += 1

    training = {
        "stage": "v4_real_high_resolution_bounded_fusion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": sha256_file(parent_path),
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": sha256_file(lock_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(dataset),
        "identities": dataset.num_classes,
        "steps": args.steps,
        "epochs_started": epoch,
        "identities_per_batch": args.identities_per_batch,
        "images_per_identity": dataset.images_per_identity,
        "training_size": args.training_size,
        "degraded_detail_size": args.degraded_detail_size,
        "trainable_parameters": trainable_names,
        "fusion_lr": args.fusion_lr,
        "loss_weights": {
            "metric": 1.0,
            "cosine": args.cosine_weight,
            "relational": args.relational_weight,
            "monotonic": args.monotonic_weight,
            "cross_resolution": args.cross_resolution_weight,
            "degraded_anchor": args.degraded_anchor_weight,
            "conflict": args.conflict_weight,
        },
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "cuda_precision": precision,
        "seed": args.seed,
        "v3_blind_data_used": False,
        "v4_blind_data_used": False,
        "blind_data_used": False,
    }
    selection = {
        "step": args.steps,
        "rule": (
            "training-only output; selection requires locked V3 development, "
            "V4 high-resolution development, cross-resolution and legacy "
            "noninferiority before one-shot V4 blind"
        ),
    }
    payload = create_highres_checkpoint(
        model,
        parent_checkpoint=parent_path,
        training=training,
        selection=selection,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_highres_checkpoint(payload, output_path)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "training": training,
        "refiner": model.refiner.configuration(),
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
                "last_step": history[-1],
                "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
                "blind_data_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
