#!/usr/bin/env python3
"""Train the query/reference image-set model on identity-disjoint episodes.

The base single-image model is loaded from an existing checkpoint.  By
default only the new set head is optimized; ``--unfreeze-identity`` enables a
small joint fine-tune of the identity encoder's final blocks.  This keeps the
experiment reversible and avoids silently changing the production descriptor
space.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.reference_aware_model import (  # noqa: E402
    ReferenceAwarePetReID,
    build_reference_aware_encoder_from_checkpoint,
    build_reference_aware_model_from_checkpoint,
    save_reference_aware_model,
)
from pet_id.reference_aware_training import (  # noqa: E402
    AllIdentityReferenceEpisodeSampler,
    ReferenceImageEpisodeSampler,
    ReferenceSpatialFeatureCache,
    build_reference_spatial_feature_cache,
    cached_reference_episode_loss,
    evaluate_cached_reference_catalog,
    load_reference_spatial_feature_cache,
    materialize_cached_reference_episode,
    materialize_reference_image_episode,
    reference_episode_loss,
    reference_validation_selection_key,
    save_reference_spatial_feature_cache,
    score_reference_image_episode,
    validate_reference_image_manifest,
)
from pet_id.reference_set_model import QueryConditionedReferenceMatcher  # noqa: E402
from pet_id.reference_token_model import (  # noqa: E402
    TokenConditionedReferenceMatcher,
    TokenReferenceAwarePetReID,
    build_token_reference_aware_model_from_checkpoint,
    save_token_reference_aware_model,
)
from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_highres_data import (  # noqa: E402
    UnifiedHighResolutionReferenceDataset,
    load_highres_manifest,
)
from pet_id.unified_training import sha256_file  # noqa: E402


FEATURE_CACHE_BATCH_SIZE = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
        help=(
            "legacy ArcFace source for un-packaged base checkpoints; packaged "
            "checkpoints restore their verified source chain automatically"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "artifacts/runs/reference_aware_model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--reference-count", type=int, default=3)
    parser.add_argument("--queries-per-identity", type=int, default=1)
    parser.add_argument("--max-references", type=int, default=4)
    parser.add_argument(
        "--all-identity-negatives",
        action="store_true",
        help=(
            "score each sampled query against every training identity using "
            "frozen encoder features"
        ),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help=(
            "persistent descriptor and pre-projection spatial feature cache "
            "required by --all-identity-negatives"
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reference-top-k", type=int, default=3)
    parser.add_argument("--reference-score-weight", type=float, default=0.4)
    parser.add_argument("--attention-temperature", type=float, default=0.10)
    parser.add_argument("--maximum-residual", type=float, default=0.25)
    parser.add_argument(
        "--interaction-level",
        choices=("descriptor", "token"),
        default="descriptor",
        help=(
            "descriptor keeps the existing set head; token performs spatial "
            "cross-view matching before descriptor pooling"
        ),
    )
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--token-grid", type=int, default=4)
    parser.add_argument("--coverage-weight", type=float, default=0.35)
    parser.add_argument(
        "--reference-set-schedule",
        choices=("nested", "sampled"),
        default="nested",
        help=(
            "nested trains every 1..reference-count prefix from one shared set; "
            "sampled retains independently sampled set sizes"
        ),
    )
    parser.add_argument(
        "--highres-image-size",
        type=int,
        default=2048,
        help=(
            "square canvas used when a high-resolution base checkpoint is "
            "selected; fixed-size manifests keep their encoder input size"
        ),
    )
    parser.add_argument(
        "--highres-maximum-side",
        type=int,
        default=4096,
        help="maximum raw source side accepted by the high-resolution adapter",
    )
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--residual-regularization", type=float, default=0.01)
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=0.0,
        help="optional weight for the strongest non-target identity margin loss",
    )
    parser.add_argument(
        "--hard-negative-margin",
        type=float,
        default=0.15,
        help="required score gap over the strongest non-target identity",
    )
    parser.add_argument(
        "--view-coverage-weight",
        type=float,
        default=0.0,
        help=(
            "optional weight for continuous viewpoint/quality coverage supervision; "
            "manifests without those signals safely skip it"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--unfreeze-identity",
        action="store_true",
        help="also fine-tune the base encoder's final identity blocks",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="disable flip/color augmentation in the training manifest",
    )
    return parser


def resolve_device(value: str) -> torch.device:
    requested = str(value).casefold()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "identities_per_batch",
        "reference_count",
        "queries_per_identity",
        "max_references",
        "hidden_dim",
        "token_dim",
        "token_grid",
        "highres_image_size",
        "highres_maximum_side",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.reference_count > args.max_references:
        raise ValueError("--reference-count cannot exceed --max-references")
    if args.steps_per_epoch < 0:
        raise ValueError("--steps-per-epoch cannot be negative")
    for name in (
        "attention_temperature",
        "maximum_residual",
        "temperature",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "hard_negative_margin",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not 0.0 <= float(args.reference_score_weight) <= 1.0:
        raise ValueError("--reference-score-weight must be between 0 and 1")
    if args.reference_top_k < 1 or args.reference_top_k > args.max_references:
        raise ValueError("--reference-top-k must fit within --max-references")
    if args.residual_regularization < 0.0 or not math.isfinite(
        float(args.residual_regularization)
    ):
        raise ValueError("--residual-regularization must be non-negative and finite")
    for name in ("hard_negative_weight", "view_coverage_weight"):
        value = float(getattr(args, name))
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(
                f"--{name.replace('_', '-')} must be non-negative and finite"
            )
    if args.coverage_weight < 0.0 or not math.isfinite(float(args.coverage_weight)):
        raise ValueError("--coverage-weight must be finite and non-negative")
    if args.all_identity_negatives:
        if args.interaction_level != "token":
            raise ValueError(
                "--all-identity-negatives requires --interaction-level token"
            )
        if args.unfreeze_identity:
            raise ValueError(
                "--all-identity-negatives requires a fully frozen base encoder"
            )
        if not args.no_augmentation:
            raise ValueError("--all-identity-negatives requires --no-augmentation")
        if args.feature_cache is None:
            raise ValueError("--all-identity-negatives requires --feature-cache")
    elif args.feature_cache is not None:
        raise ValueError("--feature-cache is only used with --all-identity-negatives")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _uses_highres_protocol(path: Path) -> bool:
    """Return whether a manifest follows the dynamic spatial-detail schema."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read manifest: {path}") from error
    protocol = str(payload.get("protocol_name", "")).casefold()
    return "high_resolution" in protocol or "highres" in protocol


def _dataset_has_view_supervision(dataset: Any) -> bool:
    records = getattr(dataset, "records", ())
    return bool(
        any(
            isinstance(record, dict)
            and isinstance(record.get("viewpoint_signals"), (list, tuple))
            for record in records
        )
    )


def _configure_identity_tail(image_encoder: nn.Module) -> None:
    """Unfreeze only a nested identity tail when explicitly requested.

    Packaged encoders may be wrapped several levels deep (external-joint or
    high-resolution).  Walking their modules finds the same conservative
    ``UnifiedPetReID.configure_identity_trainable`` hook without unfreezing
    geometry, detail refiners, or unrelated heads by accident.
    """

    for module in image_encoder.modules():
        configure = getattr(module, "configure_identity_trainable", None)
        if callable(configure):
            configure(("layer4", "fc"))
            return
    raise RuntimeError(
        "--unfreeze-identity requested, but the selected encoder exposes no "
        "identity-tail configuration hook"
    )


def train_one_epoch(
    model: ReferenceAwarePetReID,
    dataset: UnifiedManifestDataset,
    sampler: ReferenceImageEpisodeSampler,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    steps: int,
    device: torch.device,
    temperature: float,
    residual_regularization: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
    view_coverage_weight: float,
    nested_reference_counts: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "retrieval_loss": 0.0,
        "residual_penalty": 0.0,
        "hard_negative_loss": 0.0,
        "observed_hard_negative_margin": 0.0,
        "baseline_no_harm_loss": 0.0,
        "view_coverage_loss": 0.0,
        "attention_alignment_loss": 0.0,
        "token_alignment_loss": 0.0,
        "novelty_alignment_loss": 0.0,
        "reliability_alignment_loss": 0.0,
        "coverage_target_entropy": 0.0,
        "coverage_pred_entropy": 0.0,
        "coverage_valid_fraction": 0.0,
    }
    for step in range(1, steps + 1):
        episode = sampler.sample(epoch=epoch, step=step)
        batch = materialize_reference_image_episode(dataset, episode, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, details = reference_episode_loss(
            model,
            batch,
            temperature=temperature,
            residual_regularization=residual_regularization,
            hard_negative_weight=hard_negative_weight,
            hard_negative_margin=hard_negative_margin,
            view_coverage_weight=view_coverage_weight,
            nested_reference_counts=nested_reference_counts,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            grad_clip,
        )
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("non-finite reference-aware loss")
        optimizer.step()
        for name in totals:
            totals[name] += float(details[name].detach())
    return {name: value / steps for name, value in totals.items()}


def train_one_cached_epoch(
    model: TokenReferenceAwarePetReID,
    dataset: UnifiedManifestDataset,
    sampler: AllIdentityReferenceEpisodeSampler,
    cache: ReferenceSpatialFeatureCache,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    steps: int,
    device: torch.device,
    temperature: float,
    residual_regularization: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
    view_coverage_weight: float,
    nested_reference_counts: bool,
    grad_clip: float,
) -> dict[str, float]:
    """Train against the full identity catalog without rerunning the encoder."""

    model.train()
    totals = {
        "loss": 0.0,
        "retrieval_loss": 0.0,
        "residual_penalty": 0.0,
        "hard_negative_loss": 0.0,
        "observed_hard_negative_margin": 0.0,
        "baseline_no_harm_loss": 0.0,
        "view_coverage_loss": 0.0,
        "attention_alignment_loss": 0.0,
        "token_alignment_loss": 0.0,
        "novelty_alignment_loss": 0.0,
        "reliability_alignment_loss": 0.0,
        "coverage_target_entropy": 0.0,
        "coverage_pred_entropy": 0.0,
        "coverage_valid_fraction": 0.0,
    }
    for step in range(1, steps + 1):
        episode = sampler.sample(epoch=epoch, step=step)
        batch = materialize_cached_reference_episode(
            cache,
            dataset,
            episode,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, details = cached_reference_episode_loss(
            model,
            batch,
            temperature=temperature,
            residual_regularization=residual_regularization,
            hard_negative_weight=hard_negative_weight,
            hard_negative_margin=hard_negative_margin,
            view_coverage_weight=view_coverage_weight,
            nested_reference_counts=nested_reference_counts,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            grad_clip,
        )
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("non-finite cached reference-aware loss")
        optimizer.step()
        for name in totals:
            totals[name] += float(details[name].detach())
    return {name: value / steps for name, value in totals.items()}


def evaluate_episodes(
    model: ReferenceAwarePetReID,
    dataset: UnifiedManifestDataset,
    sampler: ReferenceImageEpisodeSampler,
    *,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    """Run deterministic held-out episodes and report retrieval accuracy."""

    model.eval()
    top1 = 0
    top5 = 0
    total = 0
    reciprocal: list[float] = []
    with torch.inference_mode():
        for step in range(1, steps + 1):
            episode = sampler.sample(epoch=0, step=step)
            batch = materialize_reference_image_episode(dataset, episode, device=device)
            scores = reference_episode_scores(model, batch)
            ranking = scores.argsort(dim=1, descending=True)
            targets = batch.targets.to(device=scores.device)
            for row, target in zip(ranking, targets):
                rank = int((row == target).nonzero(as_tuple=False)[0].item()) + 1
                top1 += rank == 1
                top5 += rank <= 5
                reciprocal.append(1.0 / rank)
            total += int(targets.numel())
    if total < 1:
        raise RuntimeError("validation produced no query records")
    return {
        "episodes": steps,
        "query_records": total,
        "top1_correct": int(top1),
        "top1_accuracy": float(top1 / total),
        "top5_correct": int(top5),
        "top5_accuracy": float(top5 / total),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
    }


def reference_episode_scores(model, batch):
    return score_reference_image_episode(model, batch)


def validation_selection_key(validation: dict[str, Any]) -> tuple[float, ...]:
    """Use catalog-relative selection when available, with legacy fallback."""

    if validation.get("protocol") == "full_identity_catalog_nested_references":
        return reference_validation_selection_key(validation)
    values = (
        validation.get("top1_accuracy"),
        validation.get("top5_accuracy"),
        validation.get("mean_reciprocal_rank"),
    )
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError("validation report has no supported selection metrics")
    key = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in key):
        raise ValueError("validation selection metrics must be finite")
    return key


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    train_manifest = args.train_manifest.expanduser().resolve()
    validation_manifest = args.validation_manifest.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    arcface_checkpoint = args.arcface_checkpoint.expanduser().resolve()
    for path in (train_manifest, validation_manifest, base_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"required input is missing: {path}")
    base_checkpoint_sha256 = sha256_file(base_checkpoint)
    image_encoder, base_payload = build_reference_aware_encoder_from_checkpoint(
        base_checkpoint, arcface_checkpoint, device=device
    )
    if not isinstance(image_encoder, nn.Module):
        raise TypeError("base checkpoint did not produce a torch image encoder")
    base_is_highres = (
        base_payload.get("model_type") == "unified_high_resolution_pet_reid"
    )
    train_manifest_is_highres = _uses_highres_protocol(train_manifest)
    validation_manifest_is_highres = _uses_highres_protocol(validation_manifest)
    if train_manifest_is_highres != validation_manifest_is_highres:
        raise ValueError(
            "training and validation manifests must use the same data schema"
        )
    if train_manifest_is_highres and not base_is_highres:
        raise ValueError(
            "high-resolution reference manifests require a high-resolution base checkpoint"
        )
    highres_mode = base_is_highres and train_manifest_is_highres
    minimum_records = args.reference_count + args.queries_per_identity
    if highres_mode:
        # The spatial-detail candidate has a dynamic raw-image contract. Keep the
        # episode tensors stackable by letterboxing source images to one
        # high-resolution canvas, while retaining the detail branch's pixels.
        load_highres_manifest(train_manifest, expected_split="training_extension")
        load_highres_manifest(validation_manifest, expected_split="development")
        train_dataset = UnifiedHighResolutionReferenceDataset(
            train_manifest,
            image_size=args.highres_image_size,
            expected_split="training_extension",
            training=not args.no_augmentation,
            horizontal_flip_probability=0.5 if not args.no_augmentation else 0.0,
            color_jitter=0.08 if not args.no_augmentation else 0.0,
            maximum_side=args.highres_maximum_side,
        )
        validation_dataset = UnifiedHighResolutionReferenceDataset(
            validation_manifest,
            image_size=args.highres_image_size,
            expected_split="development",
            training=False,
            maximum_side=args.highres_maximum_side,
        )
    else:
        validate_reference_image_manifest(train_manifest)
        validate_reference_image_manifest(validation_manifest)
        encoder_input_size = getattr(image_encoder, "input_size", None)
        if encoder_input_size is None:
            raise TypeError(
                "selected image encoder does not expose input_size; "
                "use a fixed-size packaged encoder"
            )
        train_dataset = UnifiedManifestDataset(
            train_manifest,
            input_size=int(encoder_input_size),
            training=not args.no_augmentation,
            horizontal_flip_probability=0.5 if not args.no_augmentation else 0.0,
            color_jitter=0.08 if not args.no_augmentation else 0.0,
            min_images_per_identity=minimum_records,
        )
        validation_dataset = UnifiedManifestDataset(
            validation_manifest,
            input_size=int(encoder_input_size),
            training=False,
            min_images_per_identity=minimum_records,
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if args.interaction_level == "token":
            model, resume_payload = build_token_reference_aware_model_from_checkpoint(
                args.resume.expanduser().resolve(), image_encoder, device=device
            )
        else:
            model, resume_payload = build_reference_aware_model_from_checkpoint(
                args.resume.expanduser().resolve(), image_encoder, device=device
            )
        start_epoch = int(resume_payload.get("training", {}).get("epoch", 0)) + 1
    else:
        descriptor_dim = int(getattr(image_encoder, "descriptor_dim", 512))
        if args.interaction_level == "token":
            matcher = TokenConditionedReferenceMatcher(
                descriptor_dim=descriptor_dim,
                token_dim=args.token_dim,
                hidden_dim=args.hidden_dim,
                max_references=args.max_references,
                reference_top_k=args.reference_top_k,
                attention_temperature=args.attention_temperature,
                maximum_residual=args.maximum_residual,
                coverage_weight=args.coverage_weight,
            )
            model = TokenReferenceAwarePetReID(
                image_encoder,
                matcher,
                token_dim=args.token_dim,
                token_grid=args.token_grid,
            ).to(device)
        else:
            matcher = QueryConditionedReferenceMatcher(
                descriptor_dim=descriptor_dim,
                hidden_dim=args.hidden_dim,
                max_references=args.max_references,
                reference_top_k=args.reference_top_k,
                reference_score_weight=args.reference_score_weight,
                attention_temperature=args.attention_temperature,
                maximum_residual=args.maximum_residual,
            )
            model = ReferenceAwarePetReID(image_encoder, matcher).to(device)
        start_epoch = 1

    model.freeze_encoder()
    if args.unfreeze_identity:
        _configure_identity_tail(model.image_encoder)
    training_cache: ReferenceSpatialFeatureCache | None = None
    feature_cache_path: Path | None = None
    feature_cache_sha256: str | None = None
    if args.all_identity_negatives:
        if not isinstance(model, TokenReferenceAwarePetReID):
            raise TypeError("all-identity candidates require a token reference model")
        feature_hook = model.image_encoder.feature_hook_name
        if not isinstance(feature_hook, str):
            raise RuntimeError(
                "all-identity candidates require a real spatial feature hook"
            )
        feature_cache_path = args.feature_cache.expanduser().resolve()
        if feature_cache_path.exists():
            print(
                json.dumps(
                    {
                        "stage": "loading_spatial_feature_cache",
                        "path": str(feature_cache_path),
                    }
                ),
                flush=True,
            )
            training_cache = load_reference_spatial_feature_cache(
                feature_cache_path,
                dataset=train_dataset,
                manifest_path=train_manifest,
                base_checkpoint_sha256=base_checkpoint_sha256,
                feature_hook=feature_hook,
                token_grid=model.token_grid,
                descriptor_dim=model.descriptor_dim,
            )
        else:
            print(
                json.dumps(
                    {
                        "stage": "building_spatial_feature_cache",
                        "records": len(train_dataset),
                        "feature_hook": feature_hook,
                        "token_grid": model.token_grid,
                        "path": str(feature_cache_path),
                    }
                ),
                flush=True,
            )
            training_cache = build_reference_spatial_feature_cache(
                model,
                train_dataset,
                manifest_path=train_manifest,
                base_checkpoint_sha256=base_checkpoint_sha256,
                device=device,
                batch_size=FEATURE_CACHE_BATCH_SIZE,
            )
            save_reference_spatial_feature_cache(training_cache, feature_cache_path)
        feature_cache_sha256 = sha256_file(feature_cache_path)
        training_cache = training_cache.to(device)
        with torch.no_grad():
            model.tokens_from_pooled_features(
                training_cache.pooled_spatial_features[:1]
            )
        print(
            json.dumps(
                {
                    "stage": "spatial_feature_cache_ready",
                    "records": int(training_cache.descriptors.shape[0]),
                    "pooled_shape": list(training_cache.pooled_spatial_features.shape),
                    "sha256": feature_cache_sha256,
                }
            ),
            flush=True,
        )
    validation_cache: ReferenceSpatialFeatureCache | None = None
    if isinstance(model, TokenReferenceAwarePetReID) and not args.unfreeze_identity:
        validation_feature_hook = model.image_encoder.feature_hook_name
        if isinstance(validation_feature_hook, str):
            print(
                json.dumps(
                    {
                        "stage": "building_validation_spatial_feature_cache",
                        "records": len(validation_dataset),
                        "feature_hook": validation_feature_hook,
                        "token_grid": model.token_grid,
                    }
                ),
                flush=True,
            )
            validation_cache = build_reference_spatial_feature_cache(
                model,
                validation_dataset,
                manifest_path=validation_manifest,
                base_checkpoint_sha256=base_checkpoint_sha256,
                device=device,
                batch_size=FEATURE_CACHE_BATCH_SIZE,
            )
            with torch.no_grad():
                model.tokens_from_pooled_features(
                    validation_cache.pooled_spatial_features[:1].to(device)
                )
            print(
                json.dumps(
                    {
                        "stage": "validation_spatial_feature_cache_ready",
                        "records": int(validation_cache.descriptors.shape[0]),
                        "candidate_identities": len(
                            validation_dataset.identity_to_label
                        ),
                    }
                ),
                flush=True,
            )
    train_sampler_type = (
        AllIdentityReferenceEpisodeSampler
        if args.all_identity_negatives
        else ReferenceImageEpisodeSampler
    )
    train_sampler = train_sampler_type(
        train_dataset,
        identities_per_batch=args.identities_per_batch,
        reference_count=args.reference_count,
        queries_per_identity=args.queries_per_identity,
        max_references=args.max_references,
        variable_reference_count=args.reference_set_schedule == "sampled",
        seed=args.seed,
    )
    validation_sampler = ReferenceImageEpisodeSampler(
        validation_dataset,
        identities_per_batch=min(
            args.identities_per_batch, len(validation_dataset.identity_to_label)
        ),
        reference_count=args.reference_count,
        queries_per_identity=args.queries_per_identity,
        max_references=args.max_references,
        variable_reference_count=False,
        seed=args.seed + 101,
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("reference-aware model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.resume:
        optimizer_state = resume_payload.get("optimizer")
        if isinstance(optimizer_state, dict) and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
    save_model = (
        save_token_reference_aware_model
        if isinstance(model, TokenReferenceAwarePetReID)
        else save_reference_aware_model
    )
    steps = args.steps_per_epoch or max(
        1, math.ceil(len(train_sampler.identity_names) / args.identities_per_batch)
    )
    validation_steps = max(
        1,
        math.ceil(
            len(validation_sampler.identity_names)
            / validation_sampler.identities_per_batch
        ),
    )
    history: list[dict[str, Any]] = []
    resume_training: dict[str, Any] = {}
    if args.resume:
        training_payload = resume_payload.get("training")
        if isinstance(training_payload, dict):
            resume_training = training_payload
        resumed_history = resume_training.get("history")
        if isinstance(resumed_history, list):
            history = [dict(item) for item in resumed_history if isinstance(item, dict)]
    best_path = output_dir / "model_best.pth"
    best_selection_key: tuple[float, ...] | None = None
    stored_best_key = resume_training.get("best_selection_key")
    if best_path.exists() and isinstance(stored_best_key, list):
        try:
            candidate_key = tuple(float(value) for value in stored_best_key)
            if candidate_key and all(math.isfinite(value) for value in candidate_key):
                best_selection_key = candidate_key
        except (TypeError, ValueError):
            best_selection_key = None
    if best_path.exists() and best_selection_key is None:
        for item in history:
            validation_row = item.get("validation")
            if not isinstance(validation_row, dict):
                continue
            try:
                candidate_key = validation_selection_key(validation_row)
            except (TypeError, ValueError):
                continue
            if best_selection_key is None or candidate_key > best_selection_key:
                best_selection_key = candidate_key
    initial_validation: dict[str, Any] | None = None
    if validation_cache is not None and (not args.resume or not best_path.exists()):
        initial_validation = evaluate_cached_reference_catalog(
            model,
            validation_cache,
            validation_dataset,
            reference_count=args.reference_count,
            queries_per_identity=args.queries_per_identity,
            query_identities_per_batch=validation_sampler.identities_per_batch,
            seed=args.seed + 101,
            device=device,
        )
        best_selection_key = validation_selection_key(initial_validation)
        initial_metadata = {
            "epoch": start_epoch - 1,
            "stage": "untrained_centroid_baseline",
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "train_manifest": str(train_manifest),
            "validation_manifest": str(validation_manifest),
            "validation": initial_validation,
            "history": history,
            "seed": args.seed,
            "interaction_level": args.interaction_level,
            "reference_set_schedule": args.reference_set_schedule,
            "validation_protocol": initial_validation["protocol"],
            "best_selection_key": list(best_selection_key),
            "checkpoint_selection": {
                "key": list(best_selection_key),
                "replaces_best": True,
                "tie_policy": "keep_earliest",
                "initial_centroid_baseline": True,
            },
        }
        save_model(
            model,
            best_path,
            base_encoder_checkpoint=base_checkpoint,
            encoder_fingerprint=base_checkpoint_sha256,
            training=initial_metadata,
            optimizer_state=optimizer.state_dict(),
        )
        print(
            json.dumps(
                {
                    "stage": "initial_centroid_baseline_saved",
                    "model_best": str(best_path),
                    "validation": initial_validation,
                }
            ),
            flush=True,
        )
    started = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if training_cache is None:
            training = train_one_epoch(
                model,
                train_dataset,
                train_sampler,
                optimizer,
                epoch=epoch,
                steps=steps,
                device=device,
                temperature=args.temperature,
                residual_regularization=args.residual_regularization,
                hard_negative_weight=args.hard_negative_weight,
                hard_negative_margin=args.hard_negative_margin,
                view_coverage_weight=args.view_coverage_weight,
                nested_reference_counts=args.reference_set_schedule == "nested",
                grad_clip=args.grad_clip,
            )
        else:
            if not isinstance(model, TokenReferenceAwarePetReID) or not isinstance(
                train_sampler, AllIdentityReferenceEpisodeSampler
            ):
                raise RuntimeError("cached all-identity training state is inconsistent")
            training = train_one_cached_epoch(
                model,
                train_dataset,
                train_sampler,
                training_cache,
                optimizer,
                epoch=epoch,
                steps=steps,
                device=device,
                temperature=args.temperature,
                residual_regularization=args.residual_regularization,
                hard_negative_weight=args.hard_negative_weight,
                hard_negative_margin=args.hard_negative_margin,
                view_coverage_weight=args.view_coverage_weight,
                nested_reference_counts=args.reference_set_schedule == "nested",
                grad_clip=args.grad_clip,
            )
        if validation_cache is not None:
            validation = evaluate_cached_reference_catalog(
                model,
                validation_cache,
                validation_dataset,
                reference_count=args.reference_count,
                queries_per_identity=args.queries_per_identity,
                query_identities_per_batch=validation_sampler.identities_per_batch,
                seed=args.seed + 101,
                device=device,
            )
        else:
            validation = evaluate_episodes(
                model,
                validation_dataset,
                validation_sampler,
                device=device,
                steps=validation_steps,
            )
        current_selection_key = validation_selection_key(validation)
        replaces_best = (
            best_selection_key is None or current_selection_key > best_selection_key
        )
        if replaces_best:
            best_selection_key = current_selection_key
        row = {
            "epoch": epoch,
            "training": training,
            "validation": validation,
            "selection_key": list(current_selection_key),
            "replaces_best": replaces_best,
        }
        history.append(row)
        metadata = {
            "epoch": epoch,
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": sha256_file(base_checkpoint),
            "train_manifest": str(train_manifest),
            "validation_manifest": str(validation_manifest),
            "training": training,
            "validation": validation,
            "history": history,
            "seed": args.seed,
            "unfreeze_identity": args.unfreeze_identity,
            "interaction_level": args.interaction_level,
            "all_identity_negatives": args.all_identity_negatives,
            "feature_cache": (
                str(feature_cache_path) if feature_cache_path is not None else None
            ),
            "feature_cache_sha256": feature_cache_sha256,
            "candidate_identity_count": len(train_sampler.identity_names),
            "hard_negative_weight": float(args.hard_negative_weight),
            "hard_negative_margin": float(args.hard_negative_margin),
            "view_coverage_weight": float(args.view_coverage_weight),
            "reference_set_schedule": args.reference_set_schedule,
            "validation_protocol": validation.get("protocol", "sampled_episodes"),
            "initial_validation": initial_validation,
            "best_selection_key": list(best_selection_key),
            "checkpoint_selection": {
                "key": list(current_selection_key),
                "replaces_best": replaces_best,
                "tie_policy": "keep_earliest",
            },
            "view_supervision_available": _dataset_has_view_supervision(train_dataset),
            "data_mode": "highres" if highres_mode else "fixed",
            "image_size": (
                int(args.highres_image_size)
                if highres_mode
                else int(getattr(train_dataset, "input_size", 0))
            ),
        }
        save_model(
            model,
            output_dir / "model_last.pth",
            base_encoder_checkpoint=base_checkpoint,
            encoder_fingerprint=base_checkpoint_sha256,
            training=metadata,
            optimizer_state=optimizer.state_dict(),
        )
        if replaces_best or not best_path.exists():
            save_model(
                model,
                best_path,
                base_encoder_checkpoint=base_checkpoint,
                encoder_fingerprint=base_checkpoint_sha256,
                training=metadata,
                optimizer_state=optimizer.state_dict(),
            )
        summary = {
            "epoch": epoch,
            "training": training,
            "validation": validation,
            "selection_key": list(current_selection_key),
            "replaces_best": replaces_best,
        }
        (output_dir / "training_state.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(_jsonable(summary), ensure_ascii=False), flush=True)

    final = {
        "format": "reference-aware-training",
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "elapsed_seconds": time.time() - started,
        "model_best": str((output_dir / "model_best.pth").resolve()),
        "model_last": str((output_dir / "model_last.pth").resolve()),
        "history": history,
        "configuration": model.configuration(),
        "initial_validation": initial_validation,
        "best_selection_key": (
            list(best_selection_key) if best_selection_key is not None else None
        ),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(_jsonable(final), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(final), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
