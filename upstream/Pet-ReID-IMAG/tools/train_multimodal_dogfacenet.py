#!/usr/bin/env python3
"""Train the locally end-to-end DogFaceNet nose + face fusion model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg  # noqa: E402

from pet_id import add_retri_config  # noqa: E402
from pet_id.dogfacenet_alignment import (  # noqa: E402
    PKBatchSampler,
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model  # noqa: E402


def _optimizer(model, *, encoder_lr, new_lr, weight_decay):
    encoder, new = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(
            (
                "gate.",
                "view_gate.",
                "nose_adapter.",
                "face_adapter.",
                "cross_modal_residual.",
                "joint_mix_logit",
            )
        ) or "classifier" in name:
            new.append(parameter)
        else:
            encoder.append(parameter)
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr, "name": "encoders"})
    if new:
        groups.append({"params": new, "lr": new_lr, "name": "fusion_heads"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay), len(encoder), len(new)


def _gradient_norm(module) -> float:
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().square().sum()
        total = squared if total is None else total + squared
    return float(total.sqrt()) if total is not None else 0.0


def _load_transferable_warm_start(
    model,
    checkpoint_path,
    *,
    scope="compatible",
) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_state = checkpoint.get("model", checkpoint)
    target_state = model.state_dict()
    if scope not in {"compatible", "encoders"}:
        raise ValueError(f"Unknown warm-start scope: {scope}")
    compatible = {
        name: value
        for name, value in source_state.items()
        if "classifier" not in name
        and (
            scope == "compatible"
            or name.startswith(("nose_encoder.", "face_encoder."))
        )
        and name in target_state
        and tuple(value.shape) == tuple(target_state[name].shape)
    }
    transferable_prefixes = (
        "gate.", "view_gate.", "nose_adapter.", "face_adapter.",
        "cross_modal_residual.", "joint_mix_logit",
    )
    transferred_fusion = sorted(
        name for name in compatible if name.startswith(transferable_prefixes)
    )
    transferred_encoders = sorted(
        name
        for name in compatible
        if name.startswith(("nose_encoder.", "face_encoder."))
    )
    if scope == "compatible" and not transferred_fusion:
        raise ValueError(f"Warm-start checkpoint has no compatible fusion tensors: {checkpoint_path}")
    if scope == "encoders" and not transferred_encoders:
        raise ValueError(f"Warm-start checkpoint has no compatible encoder tensors: {checkpoint_path}")
    model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "scope": scope,
        "source_num_classes": checkpoint.get("num_classes"),
        "loaded_tensors": len(compatible),
        "transferred_encoder_tensors": len(transferred_encoders),
        "transferred_fusion_tensors": len(transferred_fusion),
        "classification_heads_reinitialized": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="manifest.json from geometry preparation")
    parser.add_argument("--config-file", default="configs/multimodal_dogfacenet_train.yaml")
    parser.add_argument("--output-dir", default="logs/multimodal_dogfacenet_train")
    parser.add_argument(
        "--identity-weights",
        default="",
        help="optional joint checkpoint used to initialize a later fine-tuning phase",
    )
    parser.add_argument(
        "--warm-start-weights",
        default="",
        help=("transfer compatible encoders/fusion tensors from another identity set; "
              "local classification heads are deliberately reinitialized"),
    )
    parser.add_argument(
        "--warm-start-scope",
        choices=("compatible", "encoders"),
        default="compatible",
        help=(
            "load all compatible non-classifier tensors, or only the identity "
            "encoders; encoder-only is the safe v3 migration path"
        ),
    )
    parser.add_argument(
        "--joint-mix",
        type=float,
        default=None,
        help=(
            "override the legacy checkpoint joint residual share for a "
            "post-warmup phase; unavailable for shared-space fusion modes"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--identities-per-batch", type=int, default=4)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument(
        "--max-images-per-identity",
        type=int,
        default=0,
        help="cap training records per identity; useful for held-out image validation",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--new-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.12)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if args.identity_weights and args.warm_start_weights:
        raise ValueError("Use either --identity-weights or --warm-start-weights, not both")
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = PreparedDogFaceNetDataset(
        args.manifest,
        training=True,
        horizontal_flip_probability=args.flip_probability,
        color_jitter=args.color_jitter,
        min_images_per_identity=args.images_per_identity,
        max_images_per_identity=args.max_images_per_identity,
    )
    sampler = PKBatchSampler(
        dataset.targets,
        identities_per_batch=args.identities_per_batch,
        images_per_identity=args.images_per_identity,
        steps=args.steps,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prepared_dogfacenet,
    )

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE = str(device)
    cfg.MULTIMODAL.NUM_CLASSES = dataset.num_classes
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=True,
        identity_weights=args.identity_weights or None,
    )
    warm_start_report = (
        _load_transferable_warm_start(
            model,
            args.warm_start_weights,
            scope=args.warm_start_scope,
        )
        if args.warm_start_weights
        else None
    )
    if model.identity_to_label and model.identity_to_label != dataset.identity_to_label:
        raise ValueError("Checkpoint and current manifest use different identity labels")
    if args.joint_mix is not None:
        if not hasattr(model, "joint_mix_logit"):
            raise ValueError(
                "--joint-mix is only available for legacy joint-neck models"
            )
        if not 0.0 < args.joint_mix < 1.0:
            raise ValueError("--joint-mix must be strictly between 0 and 1")
        logit = math.log(args.joint_mix / (1.0 - args.joint_mix))
        with torch.no_grad():
            model.joint_mix_logit.fill_(logit)
    optimizer, encoder_parameters, new_parameters = _optimizer(
        model,
        encoder_lr=args.encoder_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    # The IMAG backbone can overflow in FP16 on high-contrast nose crops.
    # BF16 has FP32-like exponent range and is the safe default on supported GPUs.
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    history = []

    def save_checkpoint(step: int, name: str) -> Path:
        path = output_root / name
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_classes": dataset.num_classes,
                "identity_to_label": dataset.identity_to_label,
                "config_file": str(Path(args.config_file).resolve()),
                "manifest": str(Path(args.manifest).resolve()),
                "max_images_per_identity": args.max_images_per_identity,
                "history": history,
                "warm_start": warm_start_report,
                "architecture": {
                    "name": (
                        model.fusion_mode
                        if model.fusion_mode
                        in {"shared_space_v2", "semantic_residual_v3"}
                        else "residual_view_joint_v1"
                        if model.joint_enabled
                        else "legacy_concat_v1"
                    ),
                    "fusion_mode": model.fusion_mode,
                    "joint_enabled": model.joint_enabled,
                    "joint_dim": model.joint_dim,
                    "fused_dim": model.fused_dim,
                    "semantic_max_nose_weight": model.semantic_max_nose_weight,
                    "semantic_residual_scale": model.semantic_residual_scale,
                },
            },
            path,
        )
        return path

    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"identities", "source_paths"}
        }
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(**inputs)
            loss = sum(output["losses"].values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        branch_gradient_norms = {
            "nose_gradient_norm": _gradient_norm(model.nose_encoder),
            "face_gradient_norm": _gradient_norm(model.face_encoder),
            "gate_gradient_norm": _gradient_norm(model.gate),
        }
        if model.joint_enabled:
            branch_gradient_norms.update(
                {
                    "adapter_gradient_norm": _gradient_norm(model.nose_adapter)
                    + _gradient_norm(model.face_adapter),
                    "interaction_gradient_norm": _gradient_norm(
                        model.cross_modal_residual
                    ),
                }
            )
            if hasattr(model, "view_gate"):
                branch_gradient_norms["view_gate_gradient_norm"] = _gradient_norm(
                    model.view_gate
                )
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite training step: loss={float(loss.detach())}, "
                f"gradient_norm={gradient_norm}, amp_dtype={amp_dtype}"
            )
        scaler.step(optimizer)
        scaler.update()
        metrics = {
            "step": step,
            "fusion_mode": model.fusion_mode,
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "nose_weight": float(output["fusion_weights"][:, 0].detach().mean()),
            "face_weight": float(output["fusion_weights"][:, 1].detach().mean()),
            "joint_mix": float(output["joint_mix"].detach()),
            "joint_nose_weight": (
                float(output["joint_weights"][:, 0].detach().mean())
                if output["joint_weights"] is not None
                else 0.0
            ),
            "semantic_cosine": (
                float(output["semantic_agreement"][:, 0].detach().mean())
                if output["semantic_agreement"] is not None
                else 0.0
            ),
            "semantic_mean_abs_difference": (
                float(output["semantic_agreement"][:, 1].detach().mean())
                if output["semantic_agreement"] is not None
                else 0.0
            ),
            "conflict_nose_weight": (
                float(output["conflict_nose_weight"])
                if output["conflict_nose_weight"] is not None
                else 0.0
            ),
            "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
            **branch_gradient_norms,
            **{
                name: float(value.detach()) for name, value in output["losses"].items()
            },
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(step, f"checkpoint_{step:07d}.pth")

    final_path = save_checkpoint(args.steps, "model_final.pth")
    summary = {
        "steps": args.steps,
        "records": len(dataset),
        "classes": dataset.num_classes,
        "encoder_parameter_tensors": encoder_parameters,
        "new_parameter_tensors": new_parameters,
        "final_checkpoint": str(final_path),
        "warm_start": warm_start_report,
        "last_metrics": history[-1] if history else None,
    }
    (output_root / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
