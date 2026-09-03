#!/usr/bin/env python3
"""Train a headless dual-space nose/face score bridge on frozen detail endpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_unified_detail_interface import (  # noqa: E402
    atomic_json_dump,
    atomic_text_dump,
    combine_descriptor_pair,
    normalized_row,
)
from pet_id.unified_highres import build_highres_from_checkpoint  # noqa: E402
from train_unified_nose_detail import (  # noqa: E402
    LockedDetailDataset,
    atomic_torch_save,
    read_json,
    sha256_file,
    verify_protocol_sources,
    workspace_path,
)
from pet_id.release_compatibility import (  # noqa: E402
    detail_final_endpoint,
    locked_protocol_paths,
)


BRANCH_NAMES = ("face_global_512", "nose_pre_bn_2048", "nose_post_bn_2048")
QUALITY_NAMES = (
    "face_geometry_confidence",
    "global_pre_post_bn_cosine",
    "global_detail_pre_bn_cosine",
    "global_detail_post_bn_cosine",
    "global_detail_face_cosine",
    "nose_detail_energy",
)


class DualSpaceScoreBridge(nn.Module):
    """Fuse native cosine score matrices without projecting nose into face space."""

    def __init__(self, quality_dim: int, *, quality_gated: bool) -> None:
        super().__init__()
        initial_weights = torch.tensor((0.90, 0.05, 0.05), dtype=torch.float32)
        self.global_weight_logits = nn.Parameter(initial_weights.log())
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(10.0)))
        self.quality_gated = bool(quality_gated)
        if self.quality_gated:
            self.quality_norm = nn.LayerNorm(int(quality_dim))
            self.gate = nn.Sequential(
                nn.Linear(int(quality_dim), 16),
                nn.GELU(),
                nn.Linear(16, len(BRANCH_NAMES)),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def standardize_gallery(scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-5)
        return centered / scale

    def forward(
        self,
        raw_scores: torch.Tensor,
        query_quality: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        if raw_scores.ndim != 3 or raw_scores.shape[2] != len(BRANCH_NAMES):
            raise ValueError("raw_scores must have shape [queries, gallery, 3]")
        if query_quality.shape != (raw_scores.shape[0], len(QUALITY_NAMES)):
            raise ValueError("query_quality has the wrong shape")
        standardized = torch.stack(
            [
                self.standardize_gallery(raw_scores[:, :, branch])
                for branch in range(raw_scores.shape[2])
            ],
            dim=2,
        )
        logits = self.global_weight_logits[None].expand(raw_scores.shape[0], -1)
        if self.quality_gated:
            logits = logits + self.gate(self.quality_norm(query_quality.float()))
        weights = logits.softmax(dim=1)
        scale = self.logit_scale_log.clamp(math.log(1.0), math.log(64.0)).exp()
        fused = (standardized * weights[:, None]).sum(dim=2) * scale
        if not return_aux:
            return fused
        return {
            "scores": fused,
            "weights": weights,
            "standardized_branch_scores": standardized,
            "logit_scale": scale,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-funnel", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--identities-per-batch", type=int, default=20)
    parser.add_argument("--internal-dev-identities", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser.parse_args()


def validate_protocol(
    config: dict[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    lock_path, train_manifest, validation_manifest = locked_protocol_paths(
        ROOT.parents[1], config["protocol"]
    )
    lock = read_json(lock_path)
    for split, path in (("train", train_manifest), ("validation", validation_manifest)):
        if sha256_file(path) != lock["splits"][split]["sha256"]:
            raise RuntimeError(f"Locked {split} manifest hash mismatch")
    source_audit = verify_protocol_sources(
        lock,
        {"train": train_manifest, "validation": validation_manifest},
    )
    return lock_path, train_manifest, validation_manifest, source_audit


@torch.inference_mode()
def extract_split(
    model: nn.Module,
    dataset: LockedDetailDataset,
    *,
    split_name: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, Any]:
    pool_layer = model.parent_model.base_model.nose_encoder.model.heads.pool_layer
    face_rows: list[torch.Tensor] = []
    pre_rows: list[torch.Tensor] = []
    post_rows: list[torch.Tensor] = []
    quality_rows: list[torch.Tensor] = []
    started = time.monotonic()
    for index in range(len(dataset.records)):
        captured_pool: list[torch.Tensor] = []

        def capture_pool(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured_pool.append(output.detach())

        handle = pool_layer.register_forward_hook(capture_pool)
        try:
            row = dataset.load(index)
            rgb = row["high"].unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(rgb, return_aux=True)
        finally:
            handle.remove()
        if len(captured_pool) != 4:
            raise RuntimeError(
                f"Expected four nose-pool calls per image, got {len(captured_pool)}"
            )
        global_pre = combine_descriptor_pair(captured_pool[0], captured_pool[1])
        detail_pre = combine_descriptor_pair(captured_pool[2], captured_pool[3])
        global_post = normalized_row(output["raw_nose_descriptor"])
        detail_post = normalized_row(output["raw_detail_nose_descriptor"])
        global_face = normalized_row(output["face_descriptor"])
        detail_face = normalized_row(output["detail_face_descriptor"])
        quality = torch.tensor(
            [
                float(output["geometry_confidence"][0, 0]),
                float(F.cosine_similarity(global_pre[None], global_post[None])),
                float(F.cosine_similarity(global_pre[None], detail_pre[None])),
                float(F.cosine_similarity(global_post[None], detail_post[None])),
                float(F.cosine_similarity(global_face[None], detail_face[None])),
                float(output["detail_signals"][0, 9]),
            ],
            dtype=torch.float32,
        )
        face_rows.append(global_face)
        pre_rows.append(global_pre)
        post_rows.append(global_post)
        quality_rows.append(quality)
        if index == 0 or (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "feature_cache_split": split_name,
                        "images_complete": index + 1,
                        "images_total": len(dataset.records),
                    }
                ),
                flush=True,
            )
    return {
        "face": torch.stack(face_rows),
        "nose_pre": torch.stack(pre_rows),
        "nose_post": torch.stack(post_rows),
        "quality": torch.stack(quality_rows),
        "targets": torch.tensor(dataset.targets, dtype=torch.long),
        "identities": [str(row["identity"]).casefold() for row in dataset.records],
        "source_paths": [str(row["source_path"]) for row in dataset.records],
        "elapsed_seconds": time.monotonic() - started,
    }


def build_or_load_cache(
    cache_path: Path,
    checkpoint: Path,
    config: dict[str, Any],
    train_manifest: Path,
    validation_manifest: Path,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, Any]:
    identity = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "training_size": int(config["model"]["training_size"]),
        "amp": str(config["training"]["amp"]).casefold(),
    }
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("identity") != identity:
            raise RuntimeError("Existing feature cache identity does not match this run")
        return payload

    model, _ = build_highres_from_checkpoint(
        checkpoint,
        device=device,
        verify_sources=True,
    )
    model.eval()
    training_size = int(config["model"]["training_size"])
    degraded_size = int(config["model"]["degraded_detail_size"])
    datasets = {
        "train": LockedDetailDataset(
            train_manifest,
            training_size=training_size,
            degraded_size=degraded_size,
            training=False,
            horizontal_flip=0.0,
            color_jitter=0.0,
        ),
        "validation": LockedDetailDataset(
            validation_manifest,
            training_size=training_size,
            degraded_size=degraded_size,
            training=False,
            horizontal_flip=0.0,
            color_jitter=0.0,
        ),
    }
    payload = {
        "schema_version": 1,
        "feature_cache": "unified_detail_dual_space_native_endpoints",
        "identity": identity,
        "branches": list(BRANCH_NAMES),
        "quality_names": list(QUALITY_NAMES),
        "splits": {
            split: extract_split(
                model,
                dataset,
                split_name=split,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )
            for split, dataset in datasets.items()
        },
    }
    atomic_torch_save(payload, cache_path)
    return payload


def indices_by_target(split: dict[str, Any]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(split["targets"].tolist()):
        result[int(target)].append(index)
    if any(len(indices) != 4 for indices in result.values()):
        raise RuntimeError("Every cached identity must have exactly four images")
    return result


def episode(
    split: dict[str, Any],
    targets: list[int],
    *,
    device: torch.device,
    generator: random.Random | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = indices_by_target(split)
    gallery_indices: list[list[int]] = []
    query_indices: list[int] = []
    truth: list[int] = []
    for column, target in enumerate(targets):
        rows = list(grouped[target])
        if generator is not None:
            generator.shuffle(rows)
        gallery_indices.append(rows[:2])
        query_indices.extend(rows[2:])
        truth.extend((column, column))
    gallery_flat = torch.tensor(gallery_indices, dtype=torch.long)
    query_index = torch.tensor(query_indices, dtype=torch.long)

    branch_scores: list[torch.Tensor] = []
    for key in ("face", "nose_pre", "nose_post"):
        values = split[key].float()
        prototypes = F.normalize(values[gallery_flat].mean(dim=1), dim=1)
        queries = F.normalize(values[query_index], dim=1)
        branch_scores.append(queries @ prototypes.T)
    raw_scores = torch.stack(branch_scores, dim=2).to(device)
    quality = split["quality"][query_index].float().to(device)
    return raw_scores, quality, torch.tensor(truth, dtype=torch.long, device=device)


def score_metrics(scores: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int]:
    scores = scores.detach().float().cpu()
    truth = truth.detach().long().cpu()
    order = scores.argsort(dim=1, descending=True)
    top_k = min(5, scores.shape[1])
    top1 = int(order[:, 0].eq(truth).sum())
    top5 = int((order[:, :top_k] == truth[:, None]).any(dim=1).sum())
    labels = torch.zeros_like(scores, dtype=torch.int64)
    labels[torch.arange(len(truth)), truth] = 1
    ranks = (order == truth[:, None]).nonzero()[:, 1] + 1
    same = scores[torch.arange(len(truth)), truth]
    different = scores[~labels.bool()]
    return {
        "queries": len(truth),
        "gallery_identities": scores.shape[1],
        "top1_correct": top1,
        "top1_accuracy": top1 / len(truth),
        "top5_correct": top5,
        "top5_accuracy": top5 / len(truth),
        "mrr": float((1.0 / ranks.float()).mean()),
        "auc": float(roc_auc_score(labels.flatten().numpy(), scores.flatten().numpy())),
        "same_score_mean": float(same.mean()),
        "different_score_mean": float(different.mean()),
        "same_different_gap": float(same.mean() - different.mean()),
    }


@torch.inference_mode()
def evaluate_bridge(
    model: DualSpaceScoreBridge,
    split: dict[str, Any],
    targets: list[int],
    *,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    model.eval()
    raw_scores, quality, truth = episode(
        split,
        targets,
        device=device,
        generator=None,
    )
    output = model(raw_scores, quality, return_aux=True)
    weights = output["weights"].detach().float().cpu()
    return score_metrics(output["scores"], truth), {
        "mean_weights": {
            branch: float(weights[:, index].mean())
            for index, branch in enumerate(BRANCH_NAMES)
        },
        "p05_weights": {
            branch: float(torch.quantile(weights[:, index], 0.05))
            for index, branch in enumerate(BRANCH_NAMES)
        },
        "p95_weights": {
            branch: float(torch.quantile(weights[:, index], 0.95))
            for index, branch in enumerate(BRANCH_NAMES)
        },
        "logit_scale": float(output["logit_scale"]),
    }


def train_variant(
    variant: str,
    train_split: dict[str, Any],
    train_targets: list[int],
    dev_targets: list[int],
    *,
    output_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    identities_per_batch: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[DualSpaceScoreBridge, dict[str, Any]]:
    quality_gated = variant == "quality_gated"
    model = DualSpaceScoreBridge(
        len(QUALITY_NAMES),
        quality_gated=quality_gated,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best = {"top1_accuracy": -1.0, "auc": -1.0, "epoch": -1}
    generator = random.Random(seed + (1 if quality_gated else 0))
    for epoch_index in range(epochs):
        model.train()
        shuffled = list(train_targets)
        generator.shuffle(shuffled)
        losses: list[float] = []
        for start in range(0, len(shuffled), identities_per_batch):
            chosen = shuffled[start : start + identities_per_batch]
            if len(chosen) < 2:
                continue
            raw_scores, quality, truth = episode(
                train_split,
                chosen,
                device=device,
                generator=generator,
            )
            output = model(raw_scores, quality, return_aux=True)
            scores = output["scores"]
            cross_entropy = F.cross_entropy(scores, truth)
            positive = scores[torch.arange(len(truth), device=device), truth]
            negative = scores.masked_fill(
                F.one_hot(truth, scores.shape[1]).bool(),
                torch.finfo(scores.dtype).min,
            ).max(dim=1).values
            ranking = F.softplus(negative - positive + 0.20).mean()
            loss = cross_entropy + 0.25 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_metrics, dev_policy = evaluate_bridge(
            model,
            train_split,
            dev_targets,
            device=device,
        )
        row = {
            "epoch": epoch_index + 1,
            "training_loss": sum(losses) / max(len(losses), 1),
            "internal_dev": dev_metrics,
            "policy": dev_policy,
        }
        history.append(row)
        better = (
            float(dev_metrics["top1_accuracy"]) > float(best["top1_accuracy"])
            or (
                float(dev_metrics["top1_accuracy"]) == float(best["top1_accuracy"])
                and float(dev_metrics["auc"]) > float(best["auc"])
            )
        )
        if better:
            best = {
                "top1_accuracy": float(dev_metrics["top1_accuracy"]),
                "auc": float(dev_metrics["auc"]),
                "epoch": epoch_index + 1,
            }
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {
                    "variant": variant,
                    "epoch": epoch_index + 1,
                    "loss": row["training_loss"],
                    "internal_dev_top1": dev_metrics["top1_accuracy"],
                    "internal_dev_auc": dev_metrics["auc"],
                }
            ),
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("No bridge checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    atomic_torch_save(
        {
            "schema_version": 1,
            "model_type": "dual_space_nose_score_bridge",
            "variant": variant,
            "branches": BRANCH_NAMES,
            "quality_names": QUALITY_NAMES,
            "model": best_state,
            "selection": best,
        },
        output_dir / f"bridge_{variant}_best.pth",
    )
    return model, {"best": best, "history": history}


def validation_baselines(
    split: dict[str, Any],
    targets: list[int],
    *,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    raw_scores, _quality, truth = episode(
        split,
        targets,
        device=device,
        generator=None,
    )
    standardized = torch.stack(
        [
            DualSpaceScoreBridge.standardize_gallery(raw_scores[:, :, branch])
            for branch in range(raw_scores.shape[2])
        ],
        dim=2,
    )
    return {
        "face_only": score_metrics(raw_scores[:, :, 0], truth),
        "nose_pre_bn_only": score_metrics(raw_scores[:, :, 1], truth),
        "nose_post_bn_only": score_metrics(raw_scores[:, :, 2], truth),
        "equal_standardized_three_branch": score_metrics(
            standardized.mean(dim=2),
            truth,
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dual-space nose score bridge",
        "",
        (
            "Selection used only an identity-disjoint internal split of the 800 "
            "training identities; the locked 200-identity validation set was scored "
            "after bridge selection."
        ),
        "",
        "## Locked validation",
        "",
        "| Model | Top-1 | Top-5 | MRR | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["validation_baselines"].items():
        lines.append(
            f"| `{name}` | {metrics['top1_correct']}/{metrics['queries']} "
            f"({metrics['top1_accuracy']:.4f}) | "
            f"{metrics['top5_correct']}/{metrics['queries']} "
            f"({metrics['top5_accuracy']:.4f}) | "
            f"{metrics['mrr']:.6f} | {metrics['auc']:.6f} |"
        )
    for name, result in report["variants"].items():
        metrics = result["locked_validation"]
        lines.append(
            f"| `bridge_{name}` | {metrics['top1_correct']}/{metrics['queries']} "
            f"({metrics['top1_accuracy']:.4f}) | "
            f"{metrics['top5_correct']}/{metrics['queries']} "
            f"({metrics['top5_accuracy']:.4f}) | "
            f"{metrics['mrr']:.6f} | {metrics['auc']:.6f} |"
        )
    reference = report.get("protected_detail_reference")
    if reference:
        lines.append(
            f"| `protected_detail_reference` | "
            f"{reference['top1_correct']}/{reference['queries']} "
            f"({reference['top1_accuracy']:.4f}) | "
            f"{reference['top5_correct']}/{reference['queries']} "
            f"({reference['top5_accuracy']:.4f}) | "
            f"{reference['mrr']:.6f} | {reference['auc']:.6f} |"
        )
    lines.extend(["", "## Learned policies", ""])
    for name, result in report["variants"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Selected epoch: {result['training']['best']['epoch']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result["policy"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
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
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    validation_split = cache["splits"]["validation"]
    all_train_targets = sorted(indices_by_target(train_split))
    validation_targets = sorted(indices_by_target(validation_split))
    split_generator = random.Random(seed)
    split_generator.shuffle(all_train_targets)
    dev_count = int(args.internal_dev_identities)
    if not 1 <= dev_count < len(all_train_targets):
        raise ValueError("internal-dev-identities must leave identities for training")
    dev_targets = sorted(all_train_targets[:dev_count])
    bridge_train_targets = sorted(all_train_targets[dev_count:])
    if len(bridge_train_targets) % int(args.identities_per_batch) != 0:
        raise RuntimeError("Bridge-train identities must divide evenly into batches")

    variants: dict[str, Any] = {}
    for variant in ("global", "quality_gated"):
        model, training_report = train_variant(
            variant,
            train_split,
            bridge_train_targets,
            dev_targets,
            output_dir=output_dir,
            device=device,
            seed=seed,
            epochs=int(args.epochs),
            identities_per_batch=int(args.identities_per_batch),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        locked_metrics, policy = evaluate_bridge(
            model,
            validation_split,
            validation_targets,
            device=device,
        )
        variants[variant] = {
            "training": training_report,
            "locked_validation": locked_metrics,
            "policy": policy,
        }

    reference = None
    if args.reference_funnel is not None:
        reference_path = workspace_path(args.reference_funnel)
        reference_payload = read_json(reference_path)
        after = next(
            checkpoint_report
            for checkpoint_report in reference_payload["checkpoints"]
            if checkpoint_report["name"] == "after"
        )
        reference = detail_final_endpoint(after["endpoints"])
    report = {
        "schema_version": 1,
        "experiment": "dual_space_nose_score_bridge",
        "design": {
            "fixed_identity_classifier": False,
            "episodic_cosine_training": True,
            "nose_projection_to_face_space": False,
            "native_nose_spaces": ["pre_bn_2048", "post_bn_2048"],
            "single_score_fusion": True,
            "frozen_detail_feature_extractor": True,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "seed": seed,
            "epochs": int(args.epochs),
            "identities_per_batch": int(args.identities_per_batch),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "protocol": {
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "source_audit": source_audit,
            "bridge_train_identities": len(bridge_train_targets),
            "internal_dev_identities": len(dev_targets),
            "locked_validation_identities": len(validation_targets),
            "locked_validation_used_for_selection": False,
        },
        "feature_cache": {
            "path": str(cache_path),
            "identity": cache["identity"],
            "train_elapsed_seconds": train_split["elapsed_seconds"],
            "validation_elapsed_seconds": validation_split["elapsed_seconds"],
        },
        "validation_baselines": validation_baselines(
            validation_split,
            validation_targets,
            device=device,
        ),
        "variants": variants,
        "protected_detail_reference": reference,
    }
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    atomic_json_dump(report, report_path)
    atomic_text_dump(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "locked_validation": {
                    name: result["locked_validation"]
                    for name, result in variants.items()
                },
                "protected_detail_reference": reference,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
