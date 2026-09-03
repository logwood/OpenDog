#!/usr/bin/env python3
"""Train a face-anchored semantic fusion head on locked development caches."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.config import add_retri_config
from pet_id.dogfacenet_alignment import PKBatchSampler
from pet_id.multimodal import build_local_identity_model
from pet_id.model_profiles import get_runtime_profile
from pet_id.onnx_export import PreCroppedPetEmbeddingModel
from pet_id.unified_semantic import FaceAnchoredSemanticFusion
from pet_id.unified_training import (
    atomic_torch_save,
    batch_hard_metric_violation,
    cosine_distillation,
    different_identity_permutation,
    retrieval_metrics,
    sha256_file,
    supervised_contrastive_loss,
)
from pet_id.workspace_paths import normalize_runtime_config


class CachedFusionDataset(Dataset):
    REQUIRED = (
        "face_embedding",
        "nose_embedding",
        "adapted_nose_embedding",
        "quality_signals",
        "viewpoint_signals",
        "geometry_confidence",
        "identities",
        "source_paths",
    )

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        payload = np.load(self.path, allow_pickle=False)
        missing = sorted(set(self.REQUIRED).difference(payload.files))
        if missing:
            raise ValueError(f"Fusion cache is missing arrays: {missing}")
        self.face = torch.from_numpy(payload["face_embedding"]).float()
        self.nose = torch.from_numpy(payload["nose_embedding"]).float()
        self.adapted_nose = torch.from_numpy(
            payload["adapted_nose_embedding"]
        ).float()
        self.quality = torch.from_numpy(payload["quality_signals"]).float()
        self.viewpoint = torch.from_numpy(payload["viewpoint_signals"]).float()
        self.confidence = torch.from_numpy(payload["geometry_confidence"]).float()
        self.identities = payload["identities"].astype(str).tolist()
        self.source_paths = payload["source_paths"].astype(str).tolist()
        identity_names = sorted({value.casefold() for value in self.identities})
        identity_to_target = {
            identity: index for index, identity in enumerate(identity_names)
        }
        self.targets = torch.tensor(
            [identity_to_target[value.casefold()] for value in self.identities],
            dtype=torch.long,
        )
        count = len(self.identities)
        tensors = (
            self.face,
            self.nose,
            self.adapted_nose,
            self.quality,
            self.viewpoint,
            self.confidence,
            self.targets,
        )
        if any(tensor.shape[0] != count for tensor in tensors):
            raise ValueError("Fusion cache arrays have inconsistent record counts")
        if self.face.shape != self.adapted_nose.shape or self.face.ndim != 2:
            raise ValueError("Cached face and adapted nose descriptors must match")
        if self.nose.ndim != 2:
            raise ValueError("Cached raw nose descriptors must be two-dimensional")
        if not all(torch.isfinite(tensor).all() for tensor in tensors[:-1]):
            raise FloatingPointError("Fusion cache contains non-finite values")
        self.num_classes = len(identity_names)

    def __len__(self) -> int:
        return self.face.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "face": self.face[index],
            "nose": self.nose[index],
            "quality": self.quality[index],
            "viewpoint": self.viewpoint[index],
            "confidence": self.confidence[index],
            "target": self.targets[index],
        }


class CosineIdentityClassifier(nn.Module):
    def __init__(self, classes: int, dimension: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(
            F.normalize(features.float(), dim=1),
            F.normalize(self.weight.float(), dim=1),
        )


def parse_args() -> argparse.Namespace:
    identity_profile = get_runtime_profile("legacy-semantic")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_cache", type=Path)
    parser.add_argument("validation_cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=identity_profile.config,
    )
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=identity_profile.identity_weights,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--classifier-scale", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--classification-weight", type=float, default=0.50)
    parser.add_argument("--contrastive-weight", type=float, default=0.50)
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-tolerance", type=float, default=0.01)
    parser.add_argument("--dominance-weight", type=float, default=0.50)
    parser.add_argument("--dominance-tolerance", type=float, default=0.01)
    parser.add_argument("--conflict-weight", type=float, default=0.75)
    parser.add_argument("--conflict-margin", type=float, default=0.03)
    parser.add_argument("--gain-target", type=float, default=0.55)
    parser.add_argument("--gain-prior-weight", type=float, default=0.02)
    parser.add_argument("--maximum-nose-weight", type=float, default=0.35)
    parser.add_argument("--semantic-residual-scale", type=float, default=0.05)
    parser.add_argument("--train-nose-adapter", action="store_true")
    parser.add_argument("--minimum-top1-correct", type=int, default=193)
    parser.add_argument("--minimum-top5-correct", type=int, default=198)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def build_semantic_wrapper(
    config_path: Path,
    checkpoint_path: Path,
) -> PreCroppedPetEmbeddingModel:
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_path.resolve()))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    identity_model = build_local_identity_model(
        cfg,
        device=torch.device("cpu"),
        for_training=False,
        identity_weights=checkpoint_path.resolve(),
    )
    return PreCroppedPetEmbeddingModel(identity_model).eval()


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def pairwise_noninferiority_loss(
    fused: torch.Tensor,
    face: torch.Tensor,
    targets: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    fused_similarity = F.normalize(fused.float(), dim=1) @ F.normalize(
        fused.float(), dim=1
    ).T
    face_similarity = F.normalize(face.float(), dim=1) @ F.normalize(
        face.float(), dim=1
    ).T
    eye = torch.eye(
        targets.shape[0], dtype=torch.bool, device=targets.device
    )
    positive = targets[:, None].eq(targets[None, :]) & ~eye
    negative = targets[:, None].ne(targets[None, :])
    positive_loss = F.relu(
        face_similarity - fused_similarity - float(tolerance)
    )[positive]
    negative_loss = F.relu(
        fused_similarity - face_similarity - float(tolerance)
    )[negative]
    parts = []
    if positive_loss.numel():
        parts.append(positive_loss.mean())
    if negative_loss.numel():
        parts.append(negative_loss.mean())
    return sum(parts) if parts else fused_similarity.sum() * 0.0


def compute_losses(
    fusion: FaceAnchoredSemanticFusion,
    classifier: CosineIdentityClassifier,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = fusion(
        batch["face"],
        batch["nose"],
        batch["quality"],
        batch["viewpoint"],
        batch["confidence"],
        return_aux=True,
    )
    fused = output["embedding"]
    face = output["face_descriptor"]
    targets = batch["target"]
    losses = {
        "loss_classification": args.classification_weight
        * F.cross_entropy(classifier(fused), targets),
        "loss_contrastive": args.contrastive_weight
        * supervised_contrastive_loss(
            fused,
            targets,
            temperature=args.temperature,
        ),
        "loss_anchor": args.anchor_weight
        * cosine_distillation(fused, face.detach()),
        "loss_pairwise": args.pairwise_weight
        * pairwise_noninferiority_loss(
            fused,
            face.detach(),
            targets,
            tolerance=args.pairwise_tolerance,
        ),
        "loss_gain_prior": args.gain_prior_weight
        * (output["direction_gain"] - args.gain_target).square(),
    }
    fused_violation, fused_valid = batch_hard_metric_violation(fused, targets)
    face_violation, face_valid = batch_hard_metric_violation(face, targets)
    dominance_valid = fused_valid & face_valid
    if dominance_valid.any():
        losses["loss_dominance"] = args.dominance_weight * F.relu(
            fused_violation
            - face_violation.detach()
            - args.dominance_tolerance
        )[dominance_valid].mean()

    permutation, conflict_valid = different_identity_permutation(targets)
    if conflict_valid.any():
        corrupted = fusion.forward_adapted(
            face,
            output["adapted_nose_descriptor"].index_select(0, permutation),
            batch["quality"],
            batch["viewpoint"],
            batch["confidence"],
            return_aux=True,
        )
        clean_weight = output["effective_nose_weight"].abs()
        corrupted_weight = corrupted["effective_nose_weight"].abs()
        conflict_loss = (
            corrupted_weight[conflict_valid].mean()
            + F.relu(
                args.conflict_margin
                - clean_weight[conflict_valid]
                + corrupted_weight[conflict_valid]
            ).mean()
            + cosine_distillation(
                corrupted["embedding"][conflict_valid],
                face[conflict_valid].detach(),
            )
        )
        losses["loss_conflict"] = args.conflict_weight * conflict_loss
    return losses, output


@torch.inference_mode()
def evaluate(
    fusion: FaceAnchoredSemanticFusion,
    dataset: CachedFusionDataset,
    *,
    device: torch.device,
    include_queries: bool = False,
) -> dict[str, Any]:
    fusion.eval()
    features = []
    proposed_weights = []
    effective_weights = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = fusion(
            batch["face"],
            batch["nose"],
            batch["quality"],
            batch["viewpoint"],
            batch["confidence"],
            return_aux=True,
        )
        features.append(output["embedding"].float().cpu())
        proposed_weights.append(output["proposed_nose_weight"].float().cpu())
        effective_weights.append(output["effective_nose_weight"].float().cpu())
    feature_tensor = torch.cat(features)
    proposed = torch.cat(proposed_weights)
    effective = torch.cat(effective_weights)
    return {
        "retrieval": retrieval_metrics(
            feature_tensor,
            dataset.identities,
            dataset.source_paths,
            gallery_images_per_identity=2,
            include_queries=include_queries,
        ),
        "weights": {
            "proposed_mean": float(proposed.mean()),
            "proposed_minimum": float(proposed.min()),
            "proposed_maximum": float(proposed.max()),
            "effective_mean": float(effective.mean()),
            "effective_minimum": float(effective.min()),
            "effective_maximum": float(effective.max()),
            "direction_gain": float(fusion.direction_gain_logit.tanh()),
        },
    }


def selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    metrics = report["retrieval"]
    return (
        float(metrics["top1_correct"]),
        float(metrics["top5_correct"]),
        float(metrics["mean_reciprocal_rank"]),
        float(metrics["auc"]),
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.minimum_top1_correct < 0 or args.minimum_top5_correct < 0:
        raise ValueError("Retrieval thresholds must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = CachedFusionDataset(args.train_cache)
    validation_dataset = CachedFusionDataset(args.validation_cache)
    if train_dataset.face.shape[1] != 512:
        raise ValueError("The face anchor must be 512-dimensional")
    if set(value.casefold() for value in train_dataset.identities) & set(
        value.casefold() for value in validation_dataset.identities
    ):
        raise RuntimeError("Training and validation identity sets overlap")

    source_wrapper = build_semantic_wrapper(
        args.semantic_config.expanduser().resolve(),
        args.semantic_checkpoint.expanduser().resolve(),
    )
    fusion = FaceAnchoredSemanticFusion(
        nose_dim=train_dataset.nose.shape[1],
        descriptor_dim=train_dataset.face.shape[1],
        maximum_nose_weight=args.maximum_nose_weight,
        semantic_residual_scale=args.semantic_residual_scale,
    )
    fusion.load_semantic_residual_initialization(source_wrapper)
    del source_wrapper
    gc.collect()

    device = torch.device(args.device)
    fusion.to(device)
    fusion.requires_grad_(True)
    if not args.train_nose_adapter:
        fusion.nose_adapter.requires_grad_(False)
    classifier = CosineIdentityClassifier(
        train_dataset.num_classes,
        fusion.descriptor_dim,
        args.classifier_scale,
    ).to(device)
    with torch.inference_mode():
        adapted_check = fusion.adapt_nose(
            validation_dataset.nose[:32].to(device)
        ).cpu()
    torch.testing.assert_close(
        adapted_check,
        validation_dataset.adapted_nose[:32],
        rtol=2e-5,
        atol=2e-6,
    )
    fusion_parameters = [
        parameter for parameter in fusion.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": fusion_parameters, "lr": args.learning_rate},
            {
                "params": classifier.parameters(),
                "lr": args.classifier_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.05,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    face_baseline = retrieval_metrics(
        validation_dataset.face,
        validation_dataset.identities,
        validation_dataset.source_paths,
        gallery_images_per_identity=2,
    )
    initial = evaluate(fusion, validation_dataset, device=device)
    ranking_keys = ("top1_correct", "top5_correct")
    if any(
        initial["retrieval"][key] != face_baseline[key]
        for key in ranking_keys
    ):
        raise RuntimeError(
            "Zero-gain initialization changed face retrieval ranks"
        )

    history = []
    best_key = None
    best_diagnostic_key = None
    started = time.time()
    batch_size = args.identities_per_batch * args.images_per_identity
    default_steps = math.ceil(len(train_dataset) / batch_size)
    for epoch in range(1, args.epochs + 1):
        fusion.train()
        classifier.train()
        sampler = PKBatchSampler(
            train_dataset.targets.tolist(),
            identities_per_batch=args.identities_per_batch,
            images_per_identity=args.images_per_identity,
            steps=args.steps_per_epoch or default_steps,
            seed=args.seed + epoch,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        sums: dict[str, float] = {}
        samples = 0
        epoch_started = time.perf_counter()
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses, output = compute_losses(fusion, classifier, batch, args)
            total = sum(losses.values())
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*fusion_parameters, *classifier.parameters()],
                args.grad_clip,
            )
            if not torch.isfinite(total) or not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite fusion training step")
            optimizer.step()
            count = int(batch["target"].shape[0])
            samples += count
            values = {
                "loss": float(total.detach()),
                **{name: float(value.detach()) for name, value in losses.items()},
                "effective_weight": float(
                    output["effective_nose_weight"].detach().mean()
                ),
                "direction_gain": float(output["direction_gain"].detach()),
            }
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + value * count
        scheduler.step()

        validation = evaluate(fusion, validation_dataset, device=device)
        key = selection_key(validation)
        metrics = validation["retrieval"]
        eligible = (
            metrics["top1_correct"] >= args.minimum_top1_correct
            and metrics["top5_correct"] >= args.minimum_top5_correct
        )
        row = {
            "epoch": epoch,
            "samples": samples,
            "wall_seconds": time.perf_counter() - epoch_started,
            "training": {name: value / samples for name, value in sums.items()},
            "validation": validation,
            "promotion_eligible": eligible,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(row)
        checkpoint = {
            "schema_version": 1,
            "model_type": "face_anchored_semantic_fusion",
            "epoch": epoch,
            "model_config": fusion.configuration(),
            "model": fusion.state_dict(),
            "semantic_checkpoint": str(args.semantic_checkpoint.resolve()),
            "semantic_checkpoint_sha256": sha256_file(
                args.semantic_checkpoint.resolve()
            ),
            "semantic_config": str(args.semantic_config.resolve()),
            "semantic_config_sha256": sha256_file(args.semantic_config.resolve()),
            "train_cache": str(train_dataset.path),
            "train_cache_sha256": sha256_file(train_dataset.path),
            "validation_cache": str(validation_dataset.path),
            "validation_cache_sha256": sha256_file(validation_dataset.path),
            "face_baseline": face_baseline,
            "minimum_top1_correct": args.minimum_top1_correct,
            "minimum_top5_correct": args.minimum_top5_correct,
            "promotion_eligible": eligible,
            "validation": validation,
            "optimizer": optimizer.state_dict(),
            "classifier": classifier.state_dict(),
            "history": history,
            "arguments": vars(args),
        }
        atomic_torch_save(checkpoint, output_dir / "model_last.pth")
        if best_diagnostic_key is None or key > best_diagnostic_key:
            best_diagnostic_key = key
            atomic_torch_save(checkpoint, output_dir / "model_best_diagnostic.pth")
        if eligible and (best_key is None or key > best_key):
            best_key = key
            atomic_torch_save(checkpoint, output_dir / "model_best.pth")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "training": row["training"],
                    "validation": {
                        key: metrics[key]
                        for key in (
                            "top1_correct",
                            "top5_correct",
                            "mean_reciprocal_rank",
                            "auc",
                        )
                    },
                    "weights": validation["weights"],
                    "promotion_eligible": eligible,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    best_path = output_dir / "model_best.pth"
    diagnostic_path = output_dir / "model_best_diagnostic.pth"
    final_report = {
        "schema_version": 1,
        "purpose": "locked_development_face_anchored_fusion_training",
        "face_baseline": face_baseline,
        "initial": initial,
        "minimum_top1_correct": args.minimum_top1_correct,
        "minimum_top5_correct": args.minimum_top5_correct,
        "model_best": str(best_path) if best_path.exists() else None,
        "model_best_sha256": sha256_file(best_path) if best_path.exists() else None,
        "model_best_diagnostic": str(diagnostic_path),
        "model_best_diagnostic_sha256": sha256_file(diagnostic_path),
        "best_selection_key": list(best_key) if best_key is not None else None,
        "best_diagnostic_key": list(best_diagnostic_key),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
