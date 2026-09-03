#!/usr/bin/env python3
"""Train the RGB end-to-end structural nose interface over a detail model.

The 800 training identities are split into an identity-disjoint development
partition and an optimization partition.  Epoch selection never reads the
locked 200-identity validation partition; that partition is scored once after
the epoch is fixed and the selected model is retrained on all 800 identities.

Only the new structural bridges/residual and the final nose backbone stage are
trainable.  The old identity-specific FastReID classifier is explicitly kept
frozen and is not part of the new path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_unified_detail_interface import (  # noqa: E402
    atomic_json_dump,
    atomic_text_dump,
)
from pet_id.unified_highres_structural import (  # noqa: E402
    UnifiedHighResolutionStructuralPetReID,
    build_structural_from_detail_checkpoint,
    create_structural_checkpoint,
    save_structural_checkpoint,
)
from pet_id.release_compatibility import locked_protocol_paths  # noqa: E402
from train_unified_nose_detail import (  # noqa: E402
    LockedDetailDataset,
    batch_hard_triplet,
    circle_loss,
    collate,
    evaluate,
    freeze_batch_norm_statistics,
    identity_batches,
    learning_rate_scale,
    read_json,
    sha256_file,
    workspace_path,
    verify_protocol_sources,
)


TRAINABLE_NOSE_PARTS = ("layer4", "heads")
TRAINABLE_NEW_PREFIXES = (
    "global_bridge.",
    "detail_bridge.",
    "structural_residual.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detail-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-initialization", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--internal-dev-identities", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--identities-per-microbatch", type=int, default=4)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--nose-learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--global-aux-weight", type=float, default=0.20)
    parser.add_argument("--detail-aux-weight", type=float, default=0.10)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=0.20)
    parser.add_argument("--consistency-every-microbatches", type=int, default=4)
    parser.add_argument("--smoke-microbatches", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_identity_targets(
    targets: list[int],
    *,
    dev_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 1 <= int(dev_count) < len(targets):
        raise ValueError("internal-dev-identities must leave optimization identities")
    shuffled = list(targets)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[int(dev_count) :]), sorted(shuffled[: int(dev_count)])


def configure_trainable_scope(
    model: UnifiedHighResolutionStructuralPetReID,
) -> list[tuple[str, nn.Parameter]]:
    model.configure_trainable(
        nose_encoder_parts=TRAINABLE_NOSE_PARTS,
        structural=True,
    )
    # The released FastReID classification matrix has zero classes in this
    # integration.  Keep it explicitly frozen even when the heads container is
    # enabled, so no identity-specific classifier can receive gradients.
    encoder = model.detail_model.parent_model.base_model.nose_encoder
    encoder.model.heads.weight.requires_grad_(False)
    for parameter in encoder.model.heads.cls_layer.parameters():
        parameter.requires_grad_(False)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable parameters selected")
    forbidden_fragments = (
        "geometry_frontend",
        "identity_encoder",
        "global_model",
        "semantic_fusion",
        "nose_adapter",
        ".fusion.",
        "parent_model.refiner",
        ".detail_model.refiner",
        "heads.weight",
        "heads.cls_layer",
    )
    forbidden = [
        name
        for name, parameter in trainable
        if any(fragment in name for fragment in forbidden_fragments)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden parameter became trainable: {forbidden}")
    expected_new = [
        name
        for name, _ in trainable
        if any(name.startswith(prefix) for prefix in TRAINABLE_NEW_PREFIXES)
    ]
    if not expected_new:
        raise RuntimeError("New structural bridge has no trainable parameters")
    return trainable


def parameter_audit(
    model: nn.Module,
    trainable: list[tuple[str, nn.Parameter]],
) -> dict[str, Any]:
    names = [name for name, _ in trainable]
    nose_names = [name for name in names if "nose_encoder.model" in name]
    new_names = [
        name
        for name in names
        if any(name.startswith(prefix) for prefix in TRAINABLE_NEW_PREFIXES)
    ]
    return {
        "trainable_names": names,
        "trainable_tensors": len(names),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "new_structural_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if any(name.startswith(prefix) for prefix in TRAINABLE_NEW_PREFIXES)
        ),
        "nose_tail_parameters": sum(
            parameter.numel() for name, parameter in trainable if name in nose_names
        ),
        "new_structural_tensors": len(new_names),
        "frozen_geometry": not any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "geometry_frontend" in name
        ),
        "frozen_arcface": not any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "identity_encoder" in name
        ),
        "frozen_legacy_nose_adapter": not any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "nose_adapter" in name or ".fusion." in name
        ),
        "frozen_identity_specific_classifier": not any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "heads.weight" in name or "heads.cls_layer" in name
        ),
    }


def initialize_from_bridge(
    model: UnifiedHighResolutionStructuralPetReID,
    path: Path | None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    source = workspace_path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    # The source checkpoint contains one shared bridge. Start both global/detail
    # bridges from it; the end-to-end run specializes the detail copy on real crops.
    global_state = model.global_bridge.state_dict()
    required = set(global_state)
    if not required <= set(state):
        missing = sorted(required - set(state))
        raise RuntimeError(f"Bridge initialization is missing tensors: {missing[:8]}")
    selected = {name: state[name] for name in required}
    model.global_bridge.load_state_dict(selected, strict=True)
    model.detail_bridge.load_state_dict(selected, strict=True)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "source_model_type": payload.get("model_type"),
    }


def pairwise_geometry_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=1)
    right = F.normalize(right.float(), dim=1)
    if left.shape[0] < 2:
        return left.new_zeros(())
    diagonal = torch.eye(left.shape[0], device=left.device, dtype=torch.bool)
    return F.smooth_l1_loss((left @ left.T)[~diagonal], (right @ right.T).detach()[~diagonal])


def ranking_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    student = F.normalize(student.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)
    student_scores = student @ student.T
    teacher_scores = teacher @ teacher.T
    off_diagonal = ~torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    # Preserve only the relative ordering of non-self peers, not an absolute
    # logits; this allows the new branch to correct errors while protecting the
    # broad ArcFace geometry.
    return F.smooth_l1_loss(
        student_scores[off_diagonal],
        teacher_scores.detach()[off_diagonal],
    )


def compute_loss(
    model: UnifiedHighResolutionStructuralPetReID,
    classifier: nn.Module,
    batch: dict[str, Any],
    *,
    args: argparse.Namespace,
    use_amp: bool,
    amp_dtype: torch.dtype,
    microbatch_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    with torch.autocast(
        device_type=batch["high"].device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        output = model(batch["high"], return_aux=True)
        embedding = output["embedding"].float()
        logits = classifier(embedding)
        ce = F.cross_entropy(
            logits,
            batch["target"],
            label_smoothing=0.1,
        )
        triplet = batch_hard_triplet(embedding, batch["target"])
        circle = circle_loss(
            embedding,
            batch["target"],
            margin=0.35,
            gamma=64.0,
        )
        anchor = (
            1.0
            - F.cosine_similarity(
                embedding,
                F.normalize(output["protected_anchor"].float(), dim=1),
                dim=1,
            )
        ).mean()
        global_logits = classifier(output["global_structural"].float())
        detail_logits = classifier(output["detail_structural"].float())
        global_aux = F.cross_entropy(global_logits, batch["target"], label_smoothing=0.1)
        detail_aux = F.cross_entropy(detail_logits, batch["target"], label_smoothing=0.1)
        geometry = pairwise_geometry_loss(
            output["global_nose_post_descriptor"],
            output["global_structural"],
        )
        if "detail_nose_post_descriptor" in output:
            geometry = 0.5 * (
                geometry
                + pairwise_geometry_loss(
                    output["detail_nose_post_descriptor"],
                    output["detail_structural"],
                )
            )
        distill = ranking_distillation_loss(
            embedding,
            output["protected_anchor"].detach(),
            batch["target"],
        )
        consistency = embedding.new_zeros(())
        if (
            int(args.consistency_every_microbatches) > 0
            and microbatch_index % int(args.consistency_every_microbatches) == 0
        ):
            degraded = model(batch["degraded"])
            consistency = (
                1.0
                - F.cosine_similarity(
                    embedding,
                    F.normalize(degraded.float(), dim=1),
                    dim=1,
                )
            ).mean()
        total = (
            ce
            + triplet
            + circle
            + float(args.anchor_weight) * anchor
            + float(args.global_aux_weight) * global_aux
            + float(args.detail_aux_weight) * detail_aux
            + float(args.geometry_weight) * geometry
            + float(args.distill_weight) * distill
            + float(args.consistency_weight) * consistency
        )
    scalars = {
        "loss": total.detach(),
        "cross_entropy": ce.detach(),
        "triplet": triplet.detach(),
        "circle": circle.detach(),
        "anchor": anchor.detach(),
        "global_aux": global_aux.detach(),
        "detail_aux": detail_aux.detach(),
        "geometry": geometry.detach(),
        "distill": distill.detach(),
        "consistency": consistency.detach(),
    }
    diagnostics = {
        "embedding_anchor_cosine": float(
            F.cosine_similarity(
                embedding.detach(),
                F.normalize(output["protected_anchor"].detach().float(), dim=1),
                dim=1,
            ).mean()
        ),
        "structural_residual_gain": float(output["global_gain"].detach()),
        "nose_tail_gradient_path": True,
    }
    return total, scalars, diagnostics


def aggregate_scalars(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }


def run_training(
    model: UnifiedHighResolutionStructuralPetReID,
    dataset: LockedDetailDataset,
    train_targets: list[int],
    *,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    epochs: int,
    validation_dataset: LockedDetailDataset | None,
    validation_targets: list[int] | None,
    amp_dtype: torch.dtype,
    use_amp: bool,
    mode_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accumulation = max(int(args.gradient_accumulation_steps), 1)
    batches_per_epoch = len(train_targets) // int(args.identities_per_microbatch)
    optimizer_steps_per_epoch = max(1, (batches_per_epoch + accumulation - 1) // accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * max(epochs, 1))
    warmup_steps = max(1, optimizer_steps_per_epoch)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    global_step = 0
    seen_microbatches = 0
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "epoch": 0,
        "top1_accuracy": -1.0,
        "top5_accuracy": -1.0,
        "mrr": -1.0,
        "auc": -1.0,
    }
    smoke_initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for epoch in range(epochs):
        model.train()
        freeze_batch_norm_statistics(model)
        classifier.train()
        batches = identity_batches(
            dataset,
            identities_per_batch=int(args.identities_per_microbatch),
            images_per_identity=int(args.images_per_identity),
            seed=seed,
            epoch=epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        scalar_rows: list[dict[str, torch.Tensor]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        for microbatch, indices in enumerate(batches):
            batch = collate(dataset, indices, device)
            total, scalars, diagnostics = compute_loss(
                model,
                classifier,
                batch,
                args=args,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                microbatch_index=microbatch,
            )
            (total / accumulation).backward()
            scalar_rows.append(scalars)
            diagnostic_rows.append(diagnostics)
            should_step = (
                (microbatch + 1) % accumulation == 0
                or microbatch + 1 == len(batches)
                or (
                    int(args.smoke_microbatches) > 0
                    and seen_microbatches + 1 >= int(args.smoke_microbatches)
                )
            )
            if should_step:
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                trainable += [parameter for parameter in classifier.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, float(args.gradient_clip_norm))
                ratio = learning_rate_scale(
                    global_step,
                    total_steps,
                    warmup_steps,
                    0.05,
                )
                for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base_lr * ratio
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            seen_microbatches += 1
            if microbatch == 0 or (microbatch + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "epoch": epoch + 1,
                            "microbatch": microbatch + 1,
                            "microbatches": len(batches),
                            "optimizer_step": global_step,
                            "loss": float(total.detach()),
                            "anchor_cosine": diagnostics["embedding_anchor_cosine"],
                            "cuda_memory_gib": (
                                torch.cuda.max_memory_allocated(device) / 1024**3
                                if device.type == "cuda"
                                else 0.0
                            ),
                        }
                    ),
                    flush=True,
                )
            if int(args.smoke_microbatches) > 0 and seen_microbatches >= int(args.smoke_microbatches):
                break
        if int(args.smoke_microbatches) > 0:
            changed = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not torch.equal(parameter.detach(), smoke_initial[name])
            ]
            if not changed:
                raise RuntimeError("Smoke run changed no trainable structural parameters")
            return history, {
                "smoke_complete": True,
                "microbatches": seen_microbatches,
                "changed_trainable_tensors": len(changed),
            }
        validation = None
        if validation_dataset is not None and validation_targets is not None:
            validation = evaluate(
                model,
                validation_dataset,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                gallery_images=2,
            )
            key = (
                float(validation["top1_accuracy"]),
                float(validation["top5_accuracy"]),
                float(validation["mrr"]),
                float(validation["auc"]),
                -float(epoch + 1),
            )
            best_key = (
                float(best["top1_accuracy"]),
                float(best["top5_accuracy"]),
                float(best["mrr"]),
                float(best["auc"]),
                -float(best["epoch"] or 10**9),
            )
            if key > best_key:
                best = {
                    "epoch": epoch + 1,
                    "top1_accuracy": float(validation["top1_accuracy"]),
                    "top5_accuracy": float(validation["top5_accuracy"]),
                    "mrr": float(validation["mrr"]),
                    "auc": float(validation["auc"]),
                }
        row = {
            "mode": mode_name,
            "epoch": epoch + 1,
            "optimizer_step": global_step,
            "training": aggregate_scalars(scalar_rows),
            "diagnostics": {
                "mean_anchor_cosine": sum(item["embedding_anchor_cosine"] for item in diagnostic_rows)
                / max(len(diagnostic_rows), 1),
                "mean_structural_residual_gain": sum(item["structural_residual_gain"] for item in diagnostic_rows)
                / max(len(diagnostic_rows), 1),
            },
            "internal_validation": validation,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return history, best


def build_optimizer(
    model: UnifiedHighResolutionStructuralPetReID,
    classifier: nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    structural = []
    nose = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "nose_encoder.model" in name:
            nose.append(parameter)
        else:
            structural.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": structural, "lr": float(args.learning_rate), "name": "structural"},
            {"params": nose, "lr": float(args.nose_learning_rate), "name": "nose_tail"},
            {"params": list(classifier.parameters()), "lr": float(args.learning_rate), "name": "classifier"},
        ],
        weight_decay=float(args.weight_decay),
    )


def load_protocol(config: dict[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    lock_path, train_manifest, validation_manifest = locked_protocol_paths(
        ROOT.parents[1], config["protocol"]
    )
    lock = read_json(lock_path)
    audit = verify_protocol_sources(
        lock,
        {"train": train_manifest, "validation": validation_manifest},
    )
    return lock_path, train_manifest, validation_manifest, audit


def metrics_table(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "queries",
            "gallery_identities",
            "top1_correct",
            "top1_accuracy",
            "top5_correct",
            "top5_accuracy",
            "mrr",
            "auc",
        )
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Unified high-resolution structural nose training",
        "",
        "The old FastReID identity classifier and serial nose adapter are frozen; only the new structural interface and the nose tail are trainable.",
        "",
        "## Internal identity-disjoint selection",
        "",
        f"Optimization identities: {report['protocol']['optimization_identities']}; internal development identities: {report['protocol']['internal_dev_identities']}; locked validation identities: {report['protocol']['locked_validation_identities']}.",
        "",
        f"Selected epoch: **{report['selection']['epoch']}** (chosen without reading the locked validation partition).",
        "",
        "## Locked validation (not blind test)",
        "",
        "| Model | Top-1 | Top-5 | MRR | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["locked_validation_comparisons"].items():
        if metrics is None:
            continue
        lines.append(
            f"| `{name}` | {metrics['top1_correct']}/{metrics['queries']} ({metrics['top1_accuracy']:.4f}) | {metrics['top5_correct']}/{metrics['queries']} ({metrics['top5_accuracy']:.4f}) | {metrics['mrr']:.6f} | {metrics['auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Parameter audit",
            "",
            "```json",
            json.dumps(report["parameter_audit"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    output_dir = workspace_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, train_manifest, validation_manifest, source_audit = load_protocol(config)
    training_size = int(config["model"]["training_size"])
    degraded_size = int(config["model"]["degraded_detail_size"])
    augmentation = config["augmentation"]
    train_dataset = LockedDetailDataset(
        train_manifest,
        training_size=training_size,
        degraded_size=degraded_size,
        training=True,
        horizontal_flip=float(augmentation["horizontal_flip"]),
        color_jitter=float(augmentation["color_jitter"]),
    )
    dev_dataset = LockedDetailDataset(
        train_manifest,
        training_size=training_size,
        degraded_size=degraded_size,
        training=False,
        horizontal_flip=0.0,
        color_jitter=0.0,
    )
    locked_dataset = LockedDetailDataset(
        validation_manifest,
        training_size=training_size,
        degraded_size=degraded_size,
        training=False,
        horizontal_flip=0.0,
        color_jitter=0.0,
    )
    all_targets = sorted(train_dataset.indices_by_target)
    optimization_targets, internal_targets = split_identity_targets(
        all_targets,
        dev_count=int(args.internal_dev_identities),
        seed=seed,
    )
    locked_targets = sorted(locked_dataset.indices_by_target)
    amp_name = str(config["training"].get("amp", "bf16")).casefold()
    use_amp = (
        device.type == "cuda"
        and not args.no_amp
        and amp_name != "float32"
    )
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    detail_checkpoint = workspace_path(args.detail_checkpoint)
    model, detail_payload = build_structural_from_detail_checkpoint(
        detail_checkpoint,
        device=device,
        verify_sources=True,
        bridge_variant="dual_consensus",
        bridge_token_dim=256,
        bridge_bottleneck_dim=128,
        bridge_hidden_dim=256,
        bridge_attention_heads=4,
        bridge_dropout=0.10,
    )
    bridge_source = initialize_from_bridge(model, args.bridge_initialization)
    trainable = configure_trainable_scope(model)
    audit = parameter_audit(model, trainable)
    print(json.dumps({"parameter_audit": audit}, ensure_ascii=False, indent=2), flush=True)

    classifier = nn.Linear(model.descriptor_dim, len(all_targets), bias=False).to(device)
    nn.init.normal_(classifier.weight, std=0.01)
    optimizer = build_optimizer(model, classifier, args)
    started = time.monotonic()
    internal_history, internal_best = run_training(
        model,
        train_dataset,
        optimization_targets,
        classifier=classifier,
        optimizer=optimizer,
        args=args,
        device=device,
        seed=seed,
        epochs=int(args.epochs),
        validation_dataset=dev_dataset,
        validation_targets=internal_targets,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        mode_name="internal_selection",
    )
    if int(args.smoke_microbatches) > 0:
        print(json.dumps({"smoke": internal_best}, ensure_ascii=False, indent=2), flush=True)
        return
    selected_epoch = int(internal_best["epoch"])
    if selected_epoch <= 0:
        raise RuntimeError("Internal selection did not select a positive epoch")

    # Rebuild from the protected anchor and retrain the selected structure/epoch on all
    # 800 identities.  No locked image is read before this point.
    del model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(seed + 77_777)
    final_model, _ = build_structural_from_detail_checkpoint(
        detail_checkpoint,
        device=device,
        verify_sources=True,
        bridge_variant="dual_consensus",
        bridge_token_dim=256,
        bridge_bottleneck_dim=128,
        bridge_hidden_dim=256,
        bridge_attention_heads=4,
        bridge_dropout=0.10,
    )
    initialize_from_bridge(final_model, args.bridge_initialization)
    configure_trainable_scope(final_model)
    final_classifier = nn.Linear(final_model.descriptor_dim, len(all_targets), bias=False).to(device)
    nn.init.normal_(final_classifier.weight, std=0.01)
    final_optimizer = build_optimizer(final_model, final_classifier, args)
    retrain_history, retrain_summary = run_training(
        final_model,
        train_dataset,
        all_targets,
        classifier=final_classifier,
        optimizer=final_optimizer,
        args=args,
        device=device,
        seed=seed + 77_777,
        epochs=selected_epoch,
        validation_dataset=None,
        validation_targets=None,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        mode_name="full_800_retrain",
    )
    # This is the first and only score of the locked 200 identities in this
    # run, after all choices and retraining are complete.
    locked_metrics = evaluate(
        final_model,
        locked_dataset,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        gallery_images=2,
    )

    checkpoint_path = output_dir / "model_structural_end_to_end.pth"
    checkpoint_payload = create_structural_checkpoint(
        final_model,
        detail_checkpoint=detail_checkpoint,
        training={
            "stage": "unified_highres_structural_nose_end_to_end",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "protocol_lock": str(lock_path),
            "protocol_lock_sha256": sha256_file(lock_path),
            "protocol_source_audit": source_audit,
            "optimization_identities": len(optimization_targets),
            "internal_dev_identities": len(internal_targets),
            "full_retrain_identities": len(all_targets),
            "nose_trainable_parts": list(TRAINABLE_NOSE_PARTS),
            "amp": use_amp,
            "amp_dtype": str(amp_dtype),
            "seed": seed,
        },
        selection={
            "source": "identity_disjoint_internal_development",
            "epoch": selected_epoch,
            "internal_best": internal_best,
            "locked_validation_used_for_selection": False,
        },
    )
    save_structural_checkpoint(checkpoint_payload, checkpoint_path)

    baseline = None
    baseline_path = output_dir.parent / "baseline_before_stage1.json"
    if baseline_path.is_file():
        try:
            baseline_payload = read_json(baseline_path)
            baseline = baseline_payload.get("validation")
        except Exception:
            baseline = None
    report = {
        "schema_version": 1,
        "experiment": "unified_highres_structural_nose_end_to_end",
        "design": {
            "structure_level_interface": True,
            "native_nose_spaces": ["pre_bn_2048", "post_bn_2048"],
            "separate_native_necks": True,
            "face_conditioned_dual_consensus": True,
            "legacy_nose_adapter_used_for_new_path": False,
            "legacy_identity_classifier_trainable": False,
            "arcface_trainable": False,
            "geometry_trainable": False,
            "nose_tail_trainable": list(TRAINABLE_NOSE_PARTS),
            "protected_anchor_zero_initialized_residual": True,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "epochs": int(args.epochs),
            "selected_epoch": selected_epoch,
            "learning_rate": float(args.learning_rate),
            "nose_learning_rate": float(args.nose_learning_rate),
            "identities_per_microbatch": int(args.identities_per_microbatch),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "amp": use_amp,
            "amp_dtype": str(amp_dtype),
        },
        "protocol": {
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "source_audit": source_audit,
            "optimization_identities": len(optimization_targets),
            "internal_dev_identities": len(internal_targets),
            "locked_validation_identities": len(locked_targets),
            "locked_validation_used_for_selection": False,
            "locked_validation_is_blind_test": False,
        },
        "parameter_audit": audit,
        "bridge_initialization": bridge_source,
        "detail_source_checkpoint": {
            "path": str(detail_checkpoint),
            "sha256": sha256_file(detail_checkpoint),
        },
        "internal_selection": {
            "history": internal_history,
            "best": internal_best,
        },
        "full_retrain": {
            "history": retrain_history,
            "summary": retrain_summary,
        },
        "locked_validation": metrics_table(locked_metrics),
        "locked_validation_comparisons": {
            "structural_end_to_end": metrics_table(locked_metrics),
            "previous_detail_epoch7": baseline,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    atomic_json_dump(report, report_path)
    atomic_text_dump(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "internal_best": internal_best,
                "locked_validation": locked_metrics,
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
