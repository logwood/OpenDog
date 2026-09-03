#!/usr/bin/env python3
"""Jointly distill final descriptors into the single-graph geometry frontend."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import (  # noqa: E402
    UnifiedManifestDataset,
    UnifiedTeacherCache,
)
from pet_id.dogfacenet_alignment import PKBatchSampler  # noqa: E402
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)
from pet_id.unified_training import (  # noqa: E402
    atomic_torch_save,
    cosine_distillation,
    geometry_losses,
    relational_distillation,
    retrieval_metrics,
    sha256_file,
    supervised_contrastive_loss,
)
from pet_id.release_compatibility import (  # noqa: E402
    acceptance_path,
    historical_run_path,
    source_weight_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=acceptance_path(WORKSPACE, "baseline-training"),
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=historical_run_path(WORKSPACE, "semantic-teacher-training"),
    )
    parser.add_argument(
        "--development-teacher-cache",
        type=Path,
        default=historical_run_path(WORKSPACE, "semantic-teacher-development"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pk-sampling", action="store_true")
    parser.add_argument("--identities-per-batch", type=int, default=4)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--geometry-lr", type=float, default=3e-5)
    parser.add_argument("--calibration-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--embedding-weight", type=float, default=2.0)
    parser.add_argument("--face-weight", type=float, default=1.0)
    parser.add_argument("--relational-weight", type=float, default=0.25)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--metric-weight", type=float, default=0.0)
    parser.add_argument("--face-metric-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--flip-probability", type=float, default=0.25)
    parser.add_argument("--color-jitter", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_metadata(cache_path: Path) -> dict[str, Any]:
    metadata_path = cache_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("archive_sha256") != sha256_file(cache_path):
        raise RuntimeError(f"Teacher cache hash mismatch: {cache_path}")
    return metadata


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def configure_geometry_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze identities/fusion while retaining their input gradients."""

    model.requires_grad_(False)
    frontend = model.geometry_frontend
    frontend.geometry_adapter.requires_grad_(True)
    frontend.geometry.requires_grad_(True)
    frontend.geometry_calibration.requires_grad_(True)
    model.eval()
    frontend.geometry_adapter.train()
    frontend.geometry.train()
    frontend.geometry_calibration.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No geometry parameters were enabled")
    return parameters


def trainable_counts(model: torch.nn.Module) -> dict[str, int]:
    frontend = model.geometry_frontend
    return {
        "geometry_adapter": sum(
            parameter.numel()
            for parameter in frontend.geometry_adapter.parameters()
            if parameter.requires_grad
        ),
        "geometry_head": sum(
            parameter.numel()
            for parameter in frontend.geometry.parameters()
            if parameter.requires_grad
        ),
        "geometry_calibration": sum(
            parameter.numel()
            for parameter in frontend.geometry_calibration.parameters()
            if parameter.requires_grad
        ),
        "all_model": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def validate(
    model: torch.nn.Module,
    dataset: UnifiedManifestDataset,
    *,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    features = []
    teacher_features = []
    faces = []
    teacher_faces = []
    identities: list[str] = []
    source_paths: list[str] = []
    geometry_sums = {
        "geometry_center": 0.0,
        "geometry_size": 0.0,
        "geometry_angle": 0.0,
        "geometry_containment": 0.0,
        "geometry_total": 0.0,
    }
    records = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["rgb"], return_aux=True)
            losses = geometry_losses(
                output["raw_boxes_cxcywh"].float(),
                output["raw_angle_radians"].float(),
                batch["boxes_cxcywh"].float(),
                batch["angle_radians"].float(),
            )
            count = int(batch["rgb"].shape[0])
            records += count
            for name in geometry_sums:
                geometry_sums[name] += float(losses[name]) * count
            features.append(output["embedding"].float().cpu())
            faces.append(output["face_descriptor"].float().cpu())
            teacher_features.append(batch["teacher_embedding"].float().cpu())
            teacher_faces.append(batch["teacher_face_embedding"].float().cpu())
            identities.extend(raw_batch["identity"])
            source_paths.extend(raw_batch["source_path"])
    feature = torch.cat(features)
    teacher = torch.cat(teacher_features)
    face = torch.cat(faces)
    teacher_face = torch.cat(teacher_faces)
    cosine = (F.normalize(feature, dim=1) * F.normalize(teacher, dim=1)).sum(dim=1)
    face_cosine = (
        F.normalize(face, dim=1) * F.normalize(teacher_face, dim=1)
    ).sum(dim=1)
    retrieval = retrieval_metrics(feature, identities, source_paths)
    return {
        "records": records,
        "geometry": {name: value / records for name, value in geometry_sums.items()},
        "retrieval": retrieval,
        "teacher_parity": {
            "minimum_cosine": float(cosine.min()),
            "mean_cosine": float(cosine.mean()),
            "face_minimum_cosine": float(face_cosine.min()),
            "face_mean_cosine": float(face_cosine.mean()),
        },
    }


def checkpoint_payload(
    source: dict[str, Any],
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    acceptance_path: Path,
    teacher_path: Path,
    history: list[dict[str, Any]],
    epoch: int,
    selection_key: tuple[Any, ...],
    promotion_eligible: bool,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in source.items()
        if key not in {"model", "optimizer", "history"}
    }
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "geometry_training": {
                "acceptance": str(acceptance_path),
                "acceptance_sha256": sha256_file(acceptance_path),
                "teacher_cache": str(teacher_path),
                "teacher_cache_sha256": sha256_file(teacher_path),
                "epoch": epoch,
                "history": history,
                "selection_key": list(selection_key),
                "promotion_eligible": promotion_eligible,
                "trainable_sections": (
                    "geometry_adapter, geometry_head, geometry_calibration only"
                ),
                "identity_and_fusion_parameters_frozen": True,
                "blind_data_used": False,
                "configuration": {
                    key: value
                    for key, value in vars(args).items()
                    if isinstance(value, (str, int, float, bool))
                },
            },
            "promotion_status": (
                "development_noninferiority_passed_blind_not_run"
                if promotion_eligible
                else "experimental_development_noninferiority_not_met"
            ),
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.identities_per_batch < 2 or args.images_per_identity < 2:
        raise ValueError("PK sampling needs at least two identities and two images")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.metric_weight < 0 or args.face_metric_weight < 0:
        raise ValueError("metric weights must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resume_path = args.resume.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    teacher_path = args.teacher_cache.expanduser().resolve()
    development_teacher_path = args.development_teacher_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (
        resume_path,
        acceptance_path,
        teacher_path,
        development_teacher_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    training_manifest = Path(acceptance["training"]["path"]).expanduser().resolve()
    development_manifest = Path(acceptance["development"]["path"]).expanduser().resolve()
    if sha256_file(training_manifest) != acceptance["training"]["sha256"]:
        raise RuntimeError("Training manifest hash mismatch")
    if sha256_file(development_manifest) != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest hash mismatch")
    if any(
        token in training_manifest.name.casefold()
        for token in ("blind", "test")
    ):
        raise RuntimeError("Protected data cannot train the joint geometry model")
    teacher_metadata = load_metadata(teacher_path)
    development_metadata = load_metadata(development_teacher_path)
    if not teacher_metadata.get("training_eligible", False):
        raise RuntimeError("Training teacher cache is marked evaluation-only")
    if teacher_metadata.get("manifest_sha256") != acceptance["training"]["sha256"]:
        raise RuntimeError("Training teacher cache uses the wrong manifest")
    if development_metadata.get("training_eligible", True):
        raise RuntimeError("Development teacher cache is not protected")
    if development_metadata.get("manifest_sha256") != acceptance["development"][
        "sha256"
    ]:
        raise RuntimeError("Development teacher cache uses the wrong manifest")
    semantic_lock = source_weight_lock(acceptance, "semantic-checkpoint")
    config_lock = source_weight_lock(acceptance, "semantic-config")
    for name, metadata, expected in (
        ("checkpoint", teacher_metadata, semantic_lock["sha256"]),
        ("config", teacher_metadata, config_lock["sha256"]),
    ):
        key = "checkpoint_sha256" if name == "checkpoint" else "config_sha256"
        if metadata.get(key) != expected:
            raise RuntimeError(f"Training teacher {name} differs from acceptance")

    device = torch.device(args.device)
    model, source_payload = build_unified_semantic_from_checkpoint(
        resume_path, device=device, verify_sources=True
    )
    parameters = configure_geometry_only(model)
    counts = trainable_counts(model)
    if counts["all_model"] != sum(
        counts[name]
        for name in ("geometry_adapter", "geometry_head", "geometry_calibration")
    ):
        raise RuntimeError("An identity or fusion parameter was accidentally enabled")
    frontend = model.geometry_frontend
    optimizer = torch.optim.AdamW(
        (
            {
                "params": list(frontend.geometry_adapter.parameters())
                + list(frontend.geometry.parameters()),
                "lr": args.geometry_lr,
            },
            {
                "params": list(frontend.geometry_calibration.parameters()),
                "lr": args.calibration_lr,
            },
        ),
        weight_decay=args.weight_decay,
    )
    teacher_cache = UnifiedTeacherCache(teacher_path)
    development_teacher_cache = UnifiedTeacherCache(development_teacher_path)
    training_dataset = UnifiedManifestDataset(
        training_manifest,
        input_size=model.input_size,
        training=True,
        horizontal_flip_probability=args.flip_probability,
        color_jitter=args.color_jitter,
        min_images_per_identity=2,
        teacher_cache=teacher_cache,
        allow_letterbox_upscale=False,
    )
    development_dataset = UnifiedManifestDataset(
        development_manifest,
        input_size=model.input_size,
        training=False,
        teacher_cache=development_teacher_cache,
        allow_letterbox_upscale=False,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    baseline_lock = json.loads(
        Path(acceptance["baseline_lock"]["path"]).read_text(encoding="utf-8")
    )
    development_baseline = baseline_lock["reports"]["development"]["metrics"]
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None

    for epoch in range(1, args.epochs + 1):
        if args.pk_sampling:
            pk_batch_size = args.identities_per_batch * args.images_per_identity
            steps = args.steps_per_epoch or math.ceil(
                len(training_dataset) / pk_batch_size
            )
            loader = DataLoader(
                training_dataset,
                batch_sampler=PKBatchSampler(
                    training_dataset.targets,
                    identities_per_batch=args.identities_per_batch,
                    images_per_identity=args.images_per_identity,
                    steps=steps,
                    seed=args.seed + epoch,
                ),
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
        else:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(
                training_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
        model.eval()
        frontend.geometry_adapter.train()
        frontend.geometry.train()
        frontend.geometry_calibration.train()
        sums: dict[str, float] = {}
        samples = 0
        started = time.perf_counter()
        for step, raw_batch in enumerate(loader, 1):
            if args.steps_per_epoch and step > args.steps_per_epoch:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                output = model(batch["rgb"], return_aux=True)
                raw_geometry = geometry_losses(
                    output["raw_boxes_cxcywh"].float(),
                    output["raw_angle_radians"].float(),
                    batch["boxes_cxcywh"].float(),
                    batch["angle_radians"].float(),
                )
                losses = {
                    "embedding": args.embedding_weight
                    * cosine_distillation(
                        output["embedding"], batch["teacher_embedding"]
                    ),
                    "face": args.face_weight
                    * cosine_distillation(
                        output["face_descriptor"], batch["teacher_face_embedding"]
                    ),
                    "geometry": args.geometry_weight * raw_geometry["geometry_total"],
                }
                if batch["rgb"].shape[0] > 1 and args.relational_weight:
                    losses["relational"] = args.relational_weight * relational_distillation(
                        output["embedding"], batch["teacher_embedding"]
                    )
                if args.metric_weight:
                    losses["metric"] = args.metric_weight * supervised_contrastive_loss(
                        output["embedding"],
                        batch["target"],
                        temperature=args.temperature,
                    )
                if args.face_metric_weight:
                    losses["face_metric"] = (
                        args.face_metric_weight
                        * supervised_contrastive_loss(
                            output["face_descriptor"],
                            batch["target"],
                            temperature=args.temperature,
                        )
                    )
                total = sum(losses.values())
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_square = 0.0
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient_square += float(
                        parameter.grad.detach().float().square().sum()
                    )
            gradient = math.sqrt(gradient_square)
            if not math.isfinite(float(total.detach())) or not math.isfinite(gradient):
                raise FloatingPointError("Non-finite joint geometry training step")
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch["rgb"].shape[0])
            samples += count
            values = {
                "loss": float(total.detach()),
                **{f"loss_{name}": float(value.detach()) for name, value in losses.items()},
                "geometry_total": float(raw_geometry["geometry_total"].detach()),
            }
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + value * count
            if step == 1 or step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "samples": samples,
                            **values,
                            "gradient_norm": gradient,
                        }
                    ),
                    flush=True,
                )
        validation = validate(model, development_dataset, device=device)
        retrieval = validation["retrieval"]
        parity = validation["teacher_parity"]
        selection_key = (
            retrieval["top1_correct"],
            retrieval["top5_correct"],
            parity["mean_cosine"],
            -validation["geometry"]["geometry_total"],
        )
        promotion_eligible = (
            retrieval["top1_correct"] >= development_baseline["top1_correct"]
            and retrieval["top5_correct"] >= development_baseline["top5_correct"]
        )
        row = {
            "epoch": epoch,
            "samples": samples,
            "wall_seconds": time.perf_counter() - started,
            "training": {name: value / samples for name, value in sums.items()},
            "validation": validation,
            "selection_key": list(selection_key),
            "promotion_eligible": promotion_eligible,
        }
        history.append(row)
        payload = checkpoint_payload(
            source_payload,
            model,
            args=args,
            acceptance_path=acceptance_path,
            teacher_path=teacher_path,
            history=history,
            epoch=epoch,
            selection_key=selection_key,
            promotion_eligible=promotion_eligible,
        )
        atomic_torch_save(payload, output_dir / "model_last.pth")
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            atomic_torch_save(payload, output_dir / "model_best.pth")
        state = {
            "schema_version": 1,
            "epoch": epoch,
            "trainable_parameters": counts,
            "best_selection_key": list(best_key),
            "current_selection_key": list(selection_key),
            "promotion_eligible": promotion_eligible,
            "model_last": str(output_dir / "model_last.pth"),
            "model_last_sha256": sha256_file(output_dir / "model_last.pth"),
            "model_best": str(output_dir / "model_best.pth"),
            "model_best_sha256": sha256_file(output_dir / "model_best.pth"),
            "history": history,
        }
        (output_dir / "training_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
