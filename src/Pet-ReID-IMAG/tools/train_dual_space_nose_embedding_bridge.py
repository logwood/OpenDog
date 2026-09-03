#!/usr/bin/env python3
"""Select and train a structure-level dual-space nose/face embedding bridge.

All architecture and epoch selection happens with identity-disjoint folds made
from the 800-identity training split.  The locked 200-identity validation split
is loaded for final scoring only after one variant and one epoch are fixed.
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
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_unified_detail_interface import (  # noqa: E402
    atomic_json_dump,
    atomic_text_dump,
)
from pet_id.dual_space_embedding import (  # noqa: E402
    STRUCTURAL_VARIANTS,
    DualSpaceNoseEmbeddingBridge,
)
from train_dual_space_nose_score_bridge import (  # noqa: E402
    build_or_load_cache,
    indices_by_target,
    score_metrics,
    validation_baselines,
    validate_protocol,
)
from train_unified_nose_detail import (  # noqa: E402
    atomic_torch_save,
    read_json,
    sha256_file,
    workspace_path,
)
from pet_id.release_compatibility import detail_final_endpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-funnel", type=Path)
    parser.add_argument("--score-bridge-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", default=",".join(STRUCTURAL_VARIANTS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--identities-per-batch", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--nose-aux-weight", type=float, default=0.15)
    parser.add_argument("--geometry-weight", type=float, default=0.10)
    parser.add_argument("--face-distill-weight", type=float, default=0.02)
    parser.add_argument("--face-anchor-weight", type=float, default=0.02)
    return parser.parse_args()


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one structural variant is required")
    unknown = sorted(set(variants) - set(STRUCTURAL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown structural variants: {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Structural variants must not be repeated")
    return variants


def stable_variant_seed(variant: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(variant))


def make_identity_folds(
    targets: list[int],
    *,
    fold_count: int,
    seed: int,
) -> list[list[int]]:
    if fold_count < 2 or fold_count > len(targets):
        raise ValueError("folds must be between 2 and the identity count")
    shuffled = list(targets)
    random.Random(seed).shuffle(shuffled)
    folds = [sorted(shuffled[index::fold_count]) for index in range(fold_count)]
    if set().union(*(set(fold) for fold in folds)) != set(targets):
        raise RuntimeError("Identity fold assignment is incomplete")
    if sum(len(fold) for fold in folds) != len(targets):
        raise RuntimeError("Identity fold assignment contains duplicates")
    return folds


def episode_layout(
    split: dict[str, Any],
    targets: list[int],
    *,
    generator: random.Random | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = indices_by_target(split)
    ordered_indices: list[int] = []
    gallery_positions: list[list[int]] = []
    query_positions: list[int] = []
    truth: list[int] = []
    for column, target in enumerate(targets):
        rows = list(grouped[target])
        if generator is not None:
            generator.shuffle(rows)
        offset = len(ordered_indices)
        ordered_indices.extend(rows)
        gallery_positions.append([offset, offset + 1])
        query_positions.extend((offset + 2, offset + 3))
        truth.extend((column, column))
    return (
        torch.tensor(ordered_indices, dtype=torch.long),
        torch.tensor(gallery_positions, dtype=torch.long),
        torch.tensor(query_positions, dtype=torch.long),
        torch.tensor(truth, dtype=torch.long),
    )


def load_episode_features(
    split: dict[str, Any],
    targets: list[int],
    *,
    device: torch.device,
    generator: random.Random | None,
) -> dict[str, torch.Tensor]:
    rows, gallery, queries, truth = episode_layout(
        split,
        targets,
        generator=generator,
    )
    return {
        "face": split["face"].index_select(0, rows).float().to(device),
        "nose_pre": split["nose_pre"].index_select(0, rows).float().to(device),
        "nose_post": split["nose_post"].index_select(0, rows).float().to(device),
        "gallery": gallery.to(device),
        "queries": queries.to(device),
        "truth": truth.to(device),
    }


def cosine_episode_scores(
    descriptors: torch.Tensor,
    gallery: torch.Tensor,
    queries: torch.Tensor,
) -> torch.Tensor:
    prototypes = F.normalize(descriptors[gallery].mean(dim=1), dim=1)
    query_descriptors = F.normalize(descriptors[queries], dim=1)
    return query_descriptors @ prototypes.T


def hardest_negative_ranking(scores: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    positive = scores[torch.arange(len(truth), device=scores.device), truth]
    negative = scores.masked_fill(
        F.one_hot(truth, scores.shape[1]).bool(),
        torch.finfo(scores.dtype).min,
    ).max(dim=1).values
    return F.softplus(negative - positive + 0.10).mean()


def geometry_preservation_loss(
    raw: torch.Tensor,
    projected: torch.Tensor,
) -> torch.Tensor:
    raw = F.normalize(raw.float(), dim=1)
    projected = F.normalize(projected.float(), dim=1)
    raw_geometry = raw @ raw.T
    projected_geometry = projected @ projected.T
    diagonal = torch.eye(len(raw), device=raw.device, dtype=torch.bool)
    return F.smooth_l1_loss(
        projected_geometry[~diagonal],
        raw_geometry.detach()[~diagonal],
    )


def training_objective(
    model: DualSpaceNoseEmbeddingBridge,
    batch: dict[str, torch.Tensor],
    *,
    ranking_weight: float,
    nose_aux_weight: float,
    geometry_weight: float,
    face_distill_weight: float,
    face_anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(
        batch["face"],
        batch["nose_pre"],
        batch["nose_post"],
        return_aux=True,
    )
    raw_scores = cosine_episode_scores(
        output["embedding"],
        batch["gallery"],
        batch["queries"],
    )
    scores = raw_scores * output["score_scale"]
    identity = F.cross_entropy(scores, batch["truth"])
    ranking = hardest_negative_ranking(scores, batch["truth"])

    auxiliary_tokens = [output["nose_post_token"]]
    if output["nose_pre_token"] is not None:
        auxiliary_tokens.append(output["nose_pre_token"])
        auxiliary_tokens.append(output["nose_context_token"])
    auxiliary_losses = [
        F.cross_entropy(
            12.0
            * cosine_episode_scores(
                token,
                batch["gallery"],
                batch["queries"],
            ),
            batch["truth"],
        )
        for token in auxiliary_tokens
    ]
    nose_auxiliary = torch.stack(auxiliary_losses).mean()

    geometry_terms = [
        geometry_preservation_loss(
            batch["nose_post"],
            output["nose_post_token"],
        )
    ]
    if output["nose_pre_token"] is not None:
        geometry_terms.append(
            geometry_preservation_loss(
                batch["nose_pre"],
                output["nose_pre_token"],
            )
        )
    geometry = torch.stack(geometry_terms).mean()

    face_scores = cosine_episode_scores(
        batch["face"],
        batch["gallery"],
        batch["queries"],
    )
    face_distribution = F.softmax(16.0 * face_scores.detach(), dim=1)
    face_distillation = F.kl_div(
        F.log_softmax(scores, dim=1),
        face_distribution,
        reduction="batchmean",
    )
    face_anchor = (
        1.0
        - F.cosine_similarity(
            output["embedding"].float(),
            output["face_descriptor"].float(),
            dim=1,
        )
    ).mean()
    total = (
        identity
        + float(ranking_weight) * ranking
        + float(nose_aux_weight) * nose_auxiliary
        + float(geometry_weight) * geometry
        + float(face_distill_weight) * face_distillation
        + float(face_anchor_weight) * face_anchor
    )
    components = {
        "total": float(total.detach()),
        "identity": float(identity.detach()),
        "ranking": float(ranking.detach()),
        "nose_auxiliary": float(nose_auxiliary.detach()),
        "geometry_preservation": float(geometry.detach()),
        "face_distillation": float(face_distillation.detach()),
        "face_anchor": float(face_anchor.detach()),
    }
    return total, components


def build_model(args: argparse.Namespace, variant: str) -> DualSpaceNoseEmbeddingBridge:
    return DualSpaceNoseEmbeddingBridge(
        variant=variant,
        token_dim=int(args.token_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        hidden_dim=int(args.hidden_dim),
        attention_heads=int(args.attention_heads),
        dropout=float(args.dropout),
    )


def train_one_epoch(
    model: DualSpaceNoseEmbeddingBridge,
    optimizer: torch.optim.Optimizer,
    split: dict[str, Any],
    train_targets: list[int],
    *,
    device: torch.device,
    epoch: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    generator = random.Random(seed + 1_000_003 * epoch)
    shuffled = list(train_targets)
    generator.shuffle(shuffled)
    rows: list[dict[str, float]] = []
    batch_size = int(args.identities_per_batch)
    for start in range(0, len(shuffled), batch_size):
        chosen = shuffled[start : start + batch_size]
        if len(chosen) < 2:
            continue
        batch = load_episode_features(
            split,
            chosen,
            device=device,
            generator=generator,
        )
        loss, components = training_objective(
            model,
            batch,
            ranking_weight=float(args.ranking_weight),
            nose_aux_weight=float(args.nose_aux_weight),
            geometry_weight=float(args.geometry_weight),
            face_distill_weight=float(args.face_distill_weight),
            face_anchor_weight=float(args.face_anchor_weight),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        rows.append(components)
    if not rows:
        raise RuntimeError("No training batches were produced")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


@torch.inference_mode()
def evaluate_model(
    model: DualSpaceNoseEmbeddingBridge,
    split: dict[str, Any],
    targets: list[int],
    *,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    model.eval()
    batch = load_episode_features(
        split,
        targets,
        device=device,
        generator=None,
    )
    output = model(
        batch["face"],
        batch["nose_pre"],
        batch["nose_post"],
        return_aux=True,
    )
    scores = cosine_episode_scores(
        output["embedding"],
        batch["gallery"],
        batch["queries"],
    )
    attention = output["attention_weights"]
    diagnostics: dict[str, Any] = {
        "residual_scale": float(output["residual_scale"]),
        "score_scale": float(output["score_scale"]),
        "mean_identity_residual_norm": float(
            output["identity_residual"].float().norm(dim=1).mean()
        ),
        "mean_embedding_face_cosine": float(
            F.cosine_similarity(
                output["embedding"].float(),
                output["face_descriptor"].float(),
                dim=1,
            ).mean()
        ),
    }
    if attention is not None:
        diagnostics["mean_attention"] = {
            "nose_pre_bn": float(attention[:, 0].mean()),
            "nose_post_bn": float(attention[:, 1].mean()),
        }
    return score_metrics(scores, batch["truth"]), diagnostics


def model_parameter_count(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def create_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )


def train_fold(
    variant: str,
    fold_index: int,
    split: dict[str, Any],
    train_targets: list[int],
    dev_targets: list[int],
    *,
    device: torch.device,
    base_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_seed = base_seed + stable_variant_seed(variant) + 10_007 * fold_index
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    model = build_model(args, variant).to(device)
    optimizer = create_optimizer(model, args)
    history: list[dict[str, Any]] = []
    initial_metrics, initial_diagnostics = evaluate_model(
        model,
        split,
        dev_targets,
        device=device,
    )
    history.append(
        {
            "epoch": 0,
            "training": None,
            "identity_disjoint_dev": initial_metrics,
            "diagnostics": initial_diagnostics,
        }
    )
    for epoch in range(1, int(args.epochs) + 1):
        training = train_one_epoch(
            model,
            optimizer,
            split,
            train_targets,
            device=device,
            epoch=epoch,
            seed=run_seed,
            args=args,
        )
        metrics, diagnostics = evaluate_model(
            model,
            split,
            dev_targets,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "training": training,
                "identity_disjoint_dev": metrics,
                "diagnostics": diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "fold": fold_index + 1,
                    "epoch": epoch,
                    "loss": training["total"],
                    "dev_top1": metrics["top1_accuracy"],
                    "dev_top5": metrics["top5_accuracy"],
                    "dev_mrr": metrics["mrr"],
                }
            ),
            flush=True,
        )
    return {
        "fold": fold_index + 1,
        "train_identities": len(train_targets),
        "dev_identities": len(dev_targets),
        "seed": run_seed,
        "history": history,
    }


def aggregate_epoch_metrics(fold_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epochs = [row["epoch"] for row in fold_runs[0]["history"]]
    if any([row["epoch"] for row in run["history"]] != epochs for run in fold_runs):
        raise RuntimeError("Fold histories do not contain the same epochs")
    aggregate = []
    for row_index, epoch in enumerate(epochs):
        metrics = [
            run["history"][row_index]["identity_disjoint_dev"]
            for run in fold_runs
        ]
        queries = sum(int(value["queries"]) for value in metrics)
        top1 = sum(int(value["top1_correct"]) for value in metrics)
        top5 = sum(int(value["top5_correct"]) for value in metrics)
        aggregate.append(
            {
                "epoch": epoch,
                "folds": len(metrics),
                "queries": queries,
                "top1_correct": top1,
                "top1_accuracy": top1 / queries,
                "top5_correct": top5,
                "top5_accuracy": top5 / queries,
                "mean_mrr": sum(float(value["mrr"]) for value in metrics)
                / len(metrics),
                "mean_auc": sum(float(value["auc"]) for value in metrics)
                / len(metrics),
            }
        )
    return aggregate


def epoch_selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["top1_accuracy"]),
        float(row["top5_accuracy"]),
        float(row["mean_mrr"]),
        float(row["mean_auc"]),
        -float(row["epoch"]),
    )


def select_cross_validated_variant(
    cross_validation: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    candidates = []
    for variant, result in cross_validation.items():
        best = max(result["aggregate_by_epoch"], key=epoch_selection_key)
        result["selected_epoch"] = int(best["epoch"])
        result["selected_metrics"] = best
        candidates.append(
            (
                epoch_selection_key(best)
                + (-float(result["parameter_count"]["trainable"]),),
                variant,
                best,
            )
        )
    _key, variant, best = max(candidates, key=lambda item: item[0])
    return variant, best


def train_all_identities(
    variant: str,
    selected_epoch: int,
    split: dict[str, Any],
    targets: list[int],
    *,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
) -> tuple[DualSpaceNoseEmbeddingBridge, list[dict[str, Any]]]:
    run_seed = seed + stable_variant_seed(variant) + 900_001
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    model = build_model(args, variant).to(device)
    optimizer = create_optimizer(model, args)
    history = []
    for epoch in range(1, selected_epoch + 1):
        training = train_one_epoch(
            model,
            optimizer,
            split,
            targets,
            device=device,
            epoch=epoch,
            seed=run_seed,
            args=args,
        )
        history.append({"epoch": epoch, "training": training})
        print(
            json.dumps(
                {
                    "final_retrain": True,
                    "variant": variant,
                    "epoch": epoch,
                    "epochs": selected_epoch,
                    "loss": training["total"],
                }
            ),
            flush=True,
        )
    return model, history


def protected_detail_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(workspace_path(path))
    after = next(
        checkpoint
        for checkpoint in payload["checkpoints"]
        if checkpoint["name"] == "after"
    )
    return dict(detail_final_endpoint(after["endpoints"]))


def score_bridge_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(workspace_path(path))
    return {
        name: result["locked_validation"]
        for name, result in payload["variants"].items()
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dual-space structural embedding bridge",
        "",
        (
            "Architecture and epoch were selected by identity-disjoint cross-validation "
            "inside the 800 training identities. The locked 200-identity validation "
            "set was evaluated once after selection and full-training retraining."
        ),
        "",
        "## Cross-validation selection",
        "",
        "| Variant | Parameters | Epoch | Top-1 | Top-5 | Mean MRR | Mean AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in report["cross_validation"].items():
        metrics = result["selected_metrics"]
        lines.append(
            f"| `{variant}` | {result['parameter_count']['trainable']:,} | "
            f"{result['selected_epoch']} | "
            f"{metrics['top1_correct']}/{metrics['queries']} "
            f"({metrics['top1_accuracy']:.4f}) | "
            f"{metrics['top5_correct']}/{metrics['queries']} "
            f"({metrics['top5_accuracy']:.4f}) | "
            f"{metrics['mean_mrr']:.6f} | {metrics['mean_auc']:.6f} |"
        )
    selected = report["selection"]
    lines.extend(
        [
            "",
            f"Selected: `{selected['variant']}` at epoch {selected['epoch']}.",
            "",
            "## Locked validation (not blind test)",
            "",
            "| Model | Top-1 | Top-5 | MRR | AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    comparisons = report["locked_validation_comparisons"]
    for name, metrics in comparisons.items():
        if metrics is None:
            continue
        lines.append(
            f"| `{name}` | {metrics['top1_correct']}/{metrics['queries']} "
            f"({metrics['top1_accuracy']:.4f}) | "
            f"{metrics['top5_correct']}/{metrics['queries']} "
            f"({metrics['top5_accuracy']:.4f}) | "
            f"{metrics['mrr']:.6f} | {metrics['auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selected-model diagnostics",
            "",
            "```json",
            json.dumps(
                report["selected_locked_diagnostics"],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = workspace_path(args.checkpoint)
    cache_path = workspace_path(args.cache)
    output_dir = workspace_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, train_manifest, validation_manifest, source_audit = validate_protocol(
        config
    )
    seed = int(config["experiment"]["seed"])
    device = torch.device(args.device)
    amp_name = str(config["training"]["amp"]).casefold()
    use_amp = device.type == "cuda" and amp_name != "float32"
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    cache = build_or_load_cache(
        cache_path,
        checkpoint,
        config,
        train_manifest,
        validation_manifest,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )
    train_split = cache["splits"]["train"]
    locked_split = cache["splits"]["validation"]
    train_targets = sorted(indices_by_target(train_split))
    locked_targets = sorted(indices_by_target(locked_split))
    folds = make_identity_folds(
        train_targets,
        fold_count=int(args.folds),
        seed=seed,
    )

    started = time.monotonic()
    cross_validation: dict[str, Any] = {}
    for variant in variants:
        prototype = build_model(args, variant)
        fold_runs = []
        for fold_index, dev_targets in enumerate(folds):
            train_fold_targets = sorted(set(train_targets) - set(dev_targets))
            fold_runs.append(
                train_fold(
                    variant,
                    fold_index,
                    train_split,
                    train_fold_targets,
                    dev_targets,
                    device=device,
                    base_seed=seed,
                    args=args,
                )
            )
        cross_validation[variant] = {
            "configuration": prototype.configuration(),
            "parameter_count": model_parameter_count(prototype),
            "fold_runs": fold_runs,
            "aggregate_by_epoch": aggregate_epoch_metrics(fold_runs),
        }
        del prototype
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_variant, selected_metrics = select_cross_validated_variant(
        cross_validation
    )
    selected_epoch = int(selected_metrics["epoch"])
    selected_model, retraining_history = train_all_identities(
        selected_variant,
        selected_epoch,
        train_split,
        train_targets,
        device=device,
        seed=seed,
        args=args,
    )
    selected_locked_metrics, selected_locked_diagnostics = evaluate_model(
        selected_model,
        locked_split,
        locked_targets,
        device=device,
    )

    checkpoint_path = output_dir / "selected_structural_bridge.pth"
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in selected_model.state_dict().items()
    }
    atomic_torch_save(
        {
            "schema_version": 2,
            "model_type": "dual_space_nose_embedding_bridge",
            "configuration": selected_model.configuration(),
            "model": state,
            "selection": {
                "source": "identity_disjoint_cross_validation_on_training_split",
                "variant": selected_variant,
                "epoch": selected_epoch,
                "metrics": selected_metrics,
                "locked_validation_used_for_selection": False,
            },
            "source_feature_checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "feature_cache_identity": cache["identity"],
        },
        checkpoint_path,
    )

    score_bridge = score_bridge_metrics(args.score_bridge_report)
    protected_detail = protected_detail_metrics(args.reference_funnel)
    baseline = validation_baselines(
        locked_split,
        locked_targets,
        device=device,
    )
    comparisons: dict[str, Any] = {
        "face_only": baseline["face_only"],
        "protected_detail_reference": protected_detail,
    }
    if score_bridge:
        comparisons.update(
            {
                f"score_bridge_{name}": metrics
                for name, metrics in score_bridge.items()
            }
        )
    comparisons[f"selected_{selected_variant}"] = selected_locked_metrics

    report = {
        "schema_version": 2,
        "experiment": "dual_space_nose_embedding_bridge",
        "design": {
            "structure_level_fusion": True,
            "native_nose_spaces": ["pre_bn_2048", "post_bn_2048"],
            "separate_native_necks": True,
            "face_conditioned_token_fusion": True,
            "bounded_face_residual": True,
            "fixed_identity_classifier": False,
            "geometry_preservation_auxiliary": True,
            "frozen_feature_extractor_during_structure_selection": True,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "seed": seed,
            "variants": variants,
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "identities_per_batch": int(args.identities_per_batch),
            "token_dim": int(args.token_dim),
            "bottleneck_dim": int(args.bottleneck_dim),
            "hidden_dim": int(args.hidden_dim),
            "attention_heads": int(args.attention_heads),
            "dropout": float(args.dropout),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "loss_weights": {
                "ranking": float(args.ranking_weight),
                "nose_auxiliary": float(args.nose_aux_weight),
                "geometry_preservation": float(args.geometry_weight),
                "face_distillation": float(args.face_distill_weight),
                "face_anchor": float(args.face_anchor_weight),
            },
        },
        "protocol": {
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "source_audit": source_audit,
            "cross_validation_train_identities": len(train_targets),
            "cross_validation_fold_sizes": [len(fold) for fold in folds],
            "locked_validation_identities": len(locked_targets),
            "locked_validation_used_for_selection": False,
            "locked_validation_is_blind_test": False,
        },
        "feature_cache": {
            "path": str(cache_path),
            "identity": cache["identity"],
        },
        "cross_validation": cross_validation,
        "selection": {
            "variant": selected_variant,
            "epoch": selected_epoch,
            "metrics": selected_metrics,
            "tie_break_order": ["top1", "top5", "mrr", "auc", "earlier_epoch"],
        },
        "full_training_retrain": {
            "identities": len(train_targets),
            "history": retraining_history,
        },
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "locked_validation_comparisons": comparisons,
        "selected_locked_diagnostics": selected_locked_diagnostics,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    atomic_json_dump(report, report_path)
    atomic_text_dump(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "selected_variant": selected_variant,
                "selected_epoch": selected_epoch,
                "cross_validation": selected_metrics,
                "locked_validation": selected_locked_metrics,
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
