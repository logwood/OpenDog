#!/usr/bin/env python3
"""Train the complete full-resolution graph with the standard ReID recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import _read_bgr  # noqa: E402
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.unified_data import letterbox_rgb  # noqa: E402
from pet_id.unified_highres import (  # noqa: E402
    build_highres_from_checkpoint,
    create_highres_checkpoint,
    save_highres_checkpoint,
)
from pet_id.release_compatibility import (  # noqa: E402
    locked_protocol_paths,
    parent_checkpoint_source,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class LockedDetailDataset:
    def __init__(
        self,
        manifest_path: Path,
        *,
        training_size: int,
        degraded_size: int,
        training: bool,
        horizontal_flip: float,
        color_jitter: float,
    ) -> None:
        payload = read_json(manifest_path)
        self.records = list(payload["records"])
        self.training_size = int(training_size)
        self.degraded_size = int(degraded_size)
        self.training = bool(training)
        self.horizontal_flip = float(horizontal_flip)
        self.color_jitter = float(color_jitter)
        identities = sorted({str(row["identity"]).casefold() for row in self.records})
        self.identity_to_target = {identity: index for index, identity in enumerate(identities)}
        self.targets = [self.identity_to_target[str(row["identity"]).casefold()] for row in self.records]
        self.indices_by_target: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(self.targets):
            self.indices_by_target[target].append(index)
        expected = int(payload["images_per_identity"])
        if expected != 4 or any(len(rows) != expected for rows in self.indices_by_target.values()):
            raise RuntimeError("The locked protocol requires four images per identity")

    @property
    def num_classes(self) -> int:
        return len(self.indices_by_target)

    def _source(self, row: dict[str, Any]) -> Path:
        path = workspace_path(row["source_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image = cv2.cvtColor(_read_bgr(self._source(row)), cv2.COLOR_BGR2RGB)
        if self.training and random.random() < self.horizontal_flip:
            image = np.ascontiguousarray(image[:, ::-1])
        if self.training and self.color_jitter > 0:
            alpha = random.uniform(1.0 - self.color_jitter, 1.0 + self.color_jitter)
            beta = random.uniform(-24.0, 24.0) * self.color_jitter
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        high, _, _ = letterbox_rgb(
            image,
            size=self.training_size,
            fill_value=0,
            allow_upscale=True,
        )
        degraded, _, _ = letterbox_rgb(
            image,
            size=self.degraded_size,
            fill_value=0,
            allow_upscale=True,
        )
        if self.degraded_size != self.training_size:
            degraded = cv2.resize(
                degraded,
                (self.training_size, self.training_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return {
            "high": torch.from_numpy(high.transpose(2, 0, 1).copy()).float(),
            "degraded": torch.from_numpy(degraded.transpose(2, 0, 1).copy()).float(),
            "target": self.targets[index],
            "identity": str(row["identity"]).casefold(),
            "source_path": str(self._source(row)),
        }


def identity_batches(
    dataset: LockedDetailDataset,
    *,
    identities_per_batch: int,
    images_per_identity: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    generator = random.Random(seed + 1_000_003 * epoch)
    targets = list(dataset.indices_by_target)
    generator.shuffle(targets)
    batches = []
    for start in range(0, len(targets), identities_per_batch):
        chosen = targets[start : start + identities_per_batch]
        if len(chosen) != identities_per_batch:
            continue
        indices = []
        for target in chosen:
            rows = list(dataset.indices_by_target[target])
            generator.shuffle(rows)
            indices.extend(rows[:images_per_identity])
        batches.append(indices)
    return batches


def collate(
    dataset: LockedDetailDataset,
    indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    rows = [dataset.load(index) for index in indices]
    return {
        "high": torch.stack([row["high"] for row in rows]).to(device, non_blocking=True),
        "degraded": torch.stack([row["degraded"] for row in rows]).to(device, non_blocking=True),
        "target": torch.tensor([row["target"] for row in rows], device=device),
    }


def batch_hard_triplet(embedding: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    distance = 1.0 - F.normalize(embedding.float(), dim=1) @ F.normalize(embedding.float(), dim=1).T
    same = targets[:, None].eq(targets[None, :])
    diagonal = torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    positive = same & ~diagonal
    negative = ~same
    if not positive.any() or not negative.any():
        return embedding.new_zeros(())
    hardest_positive = distance.masked_fill(~positive, -1.0).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative, float("inf")).min(dim=1).values
    return F.softplus(hardest_positive - hardest_negative).mean()


def circle_loss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
    gamma: float,
) -> torch.Tensor:
    similarity = F.normalize(embedding.float(), dim=1) @ F.normalize(embedding.float(), dim=1).T
    same = targets[:, None].eq(targets[None, :])
    diagonal = torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    positive = same & ~diagonal
    negative = ~same
    losses = []
    for index in range(len(targets)):
        sp = similarity[index][positive[index]]
        sn = similarity[index][negative[index]]
        if sp.numel() == 0 or sn.numel() == 0:
            continue
        ap = torch.clamp_min(-sp.detach() + 1.0 + margin, 0.0)
        an = torch.clamp_min(sn.detach() + margin, 0.0)
        logit_p = -gamma * ap * (sp - (1.0 - margin))
        logit_n = gamma * an * (sn - margin)
        losses.append(F.softplus(torch.logsumexp(logit_n, 0) + torch.logsumexp(logit_p, 0)))
    return torch.stack(losses).mean() if losses else embedding.new_zeros(())


def freeze_batch_norm_statistics(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def configure_parameter_scope(model: nn.Module) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    model.requires_grad_(False)
    heads: list[tuple[str, nn.Parameter]] = []
    tails: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        arcface_tail = "identity_encoder.backbone.layer4" in name or "identity_encoder.backbone.fc" in name
        nose_tail = "nose_encoder.model.backbone.layer4" in name or "nose_encoder.model.heads" in name
        any_backbone = "identity_encoder.backbone" in name or "nose_encoder.model.backbone" in name
        if arcface_tail or nose_tail:
            tails.append((name, parameter))
        elif not any_backbone:
            parameter.requires_grad_(True)
            heads.append((name, parameter))
    return heads, tails


def set_tail_trainable(tails: list[tuple[str, nn.Parameter]], enabled: bool) -> None:
    for _, parameter in tails:
        parameter.requires_grad_(enabled)


def learning_rate_scale(step: int, total_steps: int, warmup_steps: int, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return 0.1 + 0.9 * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (1.0 - minimum) * cosine


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: LockedDetailDataset,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    gallery_images: int,
) -> dict[str, float | int]:
    model.eval()
    embeddings: dict[int, list[torch.Tensor]] = defaultdict(list)
    for index, target in enumerate(dataset.targets):
        row = dataset.load(index)
        rgb = row["high"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            value = model(rgb)
        embeddings[target].append(F.normalize(value.float(), dim=1).cpu()[0])
    prototypes = []
    queries = []
    query_targets = []
    for target in sorted(embeddings):
        values = embeddings[target]
        prototype = F.normalize(torch.stack(values[:gallery_images]).mean(dim=0), dim=0)
        prototypes.append(prototype)
        for value in values[gallery_images:]:
            queries.append(value)
            query_targets.append(target)
    gallery = torch.stack(prototypes)
    query = torch.stack(queries)
    scores = query @ gallery.T
    order = scores.argsort(dim=1, descending=True)
    truth = torch.tensor(query_targets)
    top1 = int(order[:, 0].eq(truth).sum())
    top5 = int((order[:, :5] == truth[:, None]).any(dim=1).sum())
    labels = torch.zeros_like(scores, dtype=torch.int64)
    labels[torch.arange(len(truth)), truth] = 1
    auc = float(roc_auc_score(labels.flatten().numpy(), scores.flatten().numpy()))
    ranks = (order == truth[:, None]).nonzero()[:, 1] + 1
    return {
        "queries": len(truth),
        "gallery_identities": len(gallery),
        "top1_correct": top1,
        "top1_accuracy": top1 / len(truth),
        "top5_correct": top5,
        "top5_accuracy": top5 / len(truth),
        "mrr": float((1.0 / ranks.float()).mean()),
        "auc": auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-microbatches", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    profile_name = str(config["model"]["profile"])
    model_path = get_runtime_profile(profile_name).package_checkpoint
    lock_path, train_manifest, validation_manifest = locked_protocol_paths(
        WORKSPACE, config["protocol"]
    )
    lock = read_json(lock_path)
    for split, path in (("train", train_manifest), ("validation", validation_manifest)):
        expected = lock["splits"][split]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"Locked {split} manifest hash mismatch")
    model, model_payload = build_highres_from_checkpoint(
        model_path,
        device=device,
        verify_sources=True,
    )
    parent_model_path = Path(
        parent_checkpoint_source(model_payload["sources"])["path"]
    ).expanduser().resolve()
    heads, tails = configure_parameter_scope(model)
    set_tail_trainable(tails, False)
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
    validation_dataset = LockedDetailDataset(
        validation_manifest,
        training_size=training_size,
        degraded_size=degraded_size,
        training=False,
        horizontal_flip=0.0,
        color_jitter=0.0,
    )
    if train_dataset.num_classes != int(config["protocol"]["train_identities"]):
        raise RuntimeError("Training identity count does not match the locked config")
    classifier = nn.Linear(int(config["model"]["descriptor_dim"]), train_dataset.num_classes, bias=False).to(device)
    nn.init.normal_(classifier.weight, std=0.001)
    optimizer_cfg = config["optimizer"]
    optimizer = torch.optim.Adam(
        [
            {"params": [p for _, p in heads], "lr": float(optimizer_cfg["head_lr"]), "name": "heads"},
            {"params": [p for _, p in tails], "lr": float(optimizer_cfg["backbone_tail_lr"]), "name": "tails"},
            {"params": classifier.parameters(), "lr": float(optimizer_cfg["classifier_lr"]), "name": "classifier"},
        ],
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    training = config["training"]
    epochs = int(training["epochs"])
    identities_per_batch = int(training["identities_per_microbatch"])
    images_per_identity = int(training["images_per_identity"])
    accumulation = int(training["gradient_accumulation_steps"])
    batches_per_epoch = len(train_dataset.indices_by_target) // identities_per_batch
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = optimizer_steps_per_epoch * int(optimizer_cfg["warmup_epochs"])
    tail_start_epoch = int(config["model"]["trainable_scope"]["tail_finetune"]["start_epoch"])
    use_amp = device.type == "cuda" and str(training["amp"]).casefold() != "float32"
    amp_dtype = torch.bfloat16 if str(training["amp"]).casefold() == "bf16" else torch.float16
    output_dir = workspace_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    recent_path = output_dir / "checkpoint_recent.pth"
    start_epoch = 0
    global_step = 0
    best = {"top1_accuracy": -1.0, "auc": -1.0, "epoch": -1}
    if bool(training.get("resume", True)) and not args.no_resume and recent_path.is_file():
        resume = torch.load(recent_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        classifier.load_state_dict(resume["classifier"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume["global_step"])
        best = dict(resume["best"])
    loss_cfg = config["loss"]
    smoke_limit = max(int(args.smoke_microbatches), 0)
    seen_microbatches = 0
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        tail_enabled = epoch >= tail_start_epoch
        set_tail_trainable(tails, tail_enabled)
        model.train()
        classifier.train()
        if bool(config["model"]["trainable_scope"]["freeze_batch_norm_statistics"]):
            freeze_batch_norm_statistics(model)
        batches = identity_batches(
            train_dataset,
            identities_per_batch=identities_per_batch,
            images_per_identity=images_per_identity,
            seed=seed,
            epoch=epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        epoch_sums: dict[str, float] = defaultdict(float)
        epoch_rows = 0
        for microbatch, indices in enumerate(batches):
            batch = collate(train_dataset, indices, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(batch["high"], return_aux=True)
                embedding = output["embedding"].float()
                logits = classifier(embedding)
                ce = F.cross_entropy(
                    logits,
                    batch["target"],
                    label_smoothing=float(loss_cfg["cross_entropy"]["label_smoothing"]),
                )
                triplet = batch_hard_triplet(embedding, batch["target"])
                circle = circle_loss(
                    embedding,
                    batch["target"],
                    margin=float(loss_cfg["circle"]["margin"]),
                    gamma=float(loss_cfg["circle"]["gamma"]),
                )
                parent = output["highres_parent_embedding"].detach().float()
                parent_anchor = (1.0 - (F.normalize(embedding, dim=1) * F.normalize(parent, dim=1)).sum(dim=1)).mean()
                consistency = embedding.new_zeros(())
                consistency_every = int(loss_cfg["highres_degraded_consistency"]["every_microbatches"])
                if consistency_every > 0 and microbatch % consistency_every == 0:
                    degraded = model(batch["degraded"])
                    consistency = (1.0 - (F.normalize(embedding, dim=1) * F.normalize(degraded.float(), dim=1)).sum(dim=1)).mean()
                loss = (
                    float(loss_cfg["cross_entropy"]["weight"]) * ce
                    + float(loss_cfg["triplet"]["weight"]) * triplet
                    + float(loss_cfg["circle"]["weight"]) * circle
                    + float(loss_cfg["parent_anchor"]["weight"]) * parent_anchor
                    + float(loss_cfg["highres_degraded_consistency"]["weight"]) * consistency
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite full-resolution loss")
            (loss / accumulation).backward()
            for key, value in {
                "loss": loss,
                "cross_entropy": ce,
                "triplet": triplet,
                "circle": circle,
                "parent_anchor": parent_anchor,
                "degraded_consistency": consistency,
            }.items():
                epoch_sums[key] += float(value.detach())
            epoch_rows += 1
            should_step = (microbatch + 1) % accumulation == 0 or microbatch + 1 == len(batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in list(model.parameters()) + list(classifier.parameters()) if parameter.requires_grad],
                    float(training["gradient_clip_norm"]),
                )
                scale = learning_rate_scale(
                    global_step,
                    total_steps,
                    warmup_steps,
                    float(optimizer_cfg["minimum_lr_ratio"]),
                )
                for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base_lr * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            seen_microbatches += 1
            if microbatch == 0 or (microbatch + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "microbatch": microbatch + 1,
                            "microbatches": len(batches),
                            "optimizer_step": global_step,
                            "loss": float(loss.detach()),
                            "tail_trainable": tail_enabled,
                            "cuda_memory_gib": (
                                torch.cuda.max_memory_allocated(device) / 1024**3
                                if device.type == "cuda"
                                else 0.0
                            ),
                        }
                    ),
                    flush=True,
                )
            if smoke_limit and seen_microbatches >= smoke_limit:
                break
        if smoke_limit:
            print(json.dumps({"smoke_complete": True, "microbatches": seen_microbatches}, indent=2))
            return
        validation = evaluate(
            model,
            validation_dataset,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            gallery_images=int(config["protocol"]["gallery_images_per_identity"]),
        )
        row = {
            "epoch": epoch + 1,
            "optimizer_step": global_step,
            "tail_trainable": tail_enabled,
            "training": {key: value / max(epoch_rows, 1) for key, value in epoch_sums.items()},
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        is_best = (
            float(validation["top1_accuracy"]) > float(best["top1_accuracy"])
            or (
                float(validation["top1_accuracy"]) == float(best["top1_accuracy"])
                and float(validation["auc"]) > float(best["auc"])
            )
        )
        if is_best and epoch + 1 >= int(config["selection"]["save_best_only_after_epoch"]):
            best = {
                "top1_accuracy": float(validation["top1_accuracy"]),
                "auc": float(validation["auc"]),
                "epoch": epoch + 1,
            }
            payload = create_highres_checkpoint(
                model,
                parent_checkpoint=parent_model_path,
                training={
                    "stage": "unified_full_resolution_standard35",
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "protocol_lock": str(lock_path),
                    "protocol_lock_sha256": sha256_file(lock_path),
                    "epoch": epoch + 1,
                    "optimizer_step": global_step,
                    "trainable_heads": [name for name, _ in heads],
                    "trainable_tails": [name for name, _ in tails],
                    "tail_finetune_started_epoch": tail_start_epoch + 1,
                    "standard_recipe": "35 epochs / Adam / CE + triplet + circle / identity-disjoint validation",
                    "seed": seed,
                },
                selection={"epoch": epoch + 1, "validation": validation},
            )
            save_highres_checkpoint(payload, output_dir / "model_best.pth")
        atomic_torch_save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best": best,
                "config_sha256": sha256_file(config_path),
            },
            recent_path,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    final_payload = create_highres_checkpoint(
        model,
        parent_checkpoint=parent_model_path,
        training={
            "stage": "unified_full_resolution_standard35",
            "epochs": epochs,
            "optimizer_steps": global_step,
            "protocol_lock": str(lock_path),
            "seed": seed,
        },
        selection={"best": best},
    )
    save_highres_checkpoint(final_payload, output_dir / "model_final.pth")


if __name__ == "__main__":
    main()
