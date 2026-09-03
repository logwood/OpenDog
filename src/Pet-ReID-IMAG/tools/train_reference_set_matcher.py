#!/usr/bin/env python3
"""Train a query-conditioned matcher on cached single-image descriptors.

This experiment changes only the reference-set scoring component.  The RGB
encoder and its deployment artifact remain frozen, so a failed experiment can
be discarded without re-encoding the gallery.
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

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.reference_set_model import (  # noqa: E402
    QueryConditionedReferenceMatcher,
    build_reference_set_matcher_from_checkpoint,
    save_reference_set_matcher,
)
from pet_id.reference_set_training import (  # noqa: E402
    DescriptorTable,
    ReferenceEpisodeSampler,
    episode_retrieval_loss,
    evaluate_reference_matcher,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "artifacts/runs/reference_set_matcher",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--encoder-fingerprint")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--reference-count", type=int, default=2)
    parser.add_argument("--queries-per-identity", type=int, default=1)
    parser.add_argument("--max-references", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reference-top-k", type=int, default=3)
    parser.add_argument("--reference-score-weight", type=float, default=0.4)
    parser.add_argument("--attention-temperature", type=float, default=0.10)
    parser.add_argument("--maximum-residual", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--residual-regularization", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--fixed-reference-count",
        action="store_true",
        help="do not randomly vary the number of references during training",
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
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.steps_per_epoch < 0:
        raise ValueError("--steps-per-epoch cannot be negative")
    for name in (
        "identities_per_batch",
        "reference_count",
        "queries_per_identity",
        "max_references",
        "hidden_dim",
        "batch_size",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.reference_count > args.max_references:
        raise ValueError("--reference-count cannot exceed --max-references")
    for name in (
        "temperature",
        "attention_temperature",
        "learning_rate",
        "maximum_residual",
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
        raise ValueError("--residual-regularization must be finite and non-negative")
    if args.grad_clip <= 0.0 or not math.isfinite(float(args.grad_clip)):
        raise ValueError("--grad-clip must be positive and finite")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def train_one_epoch(
    model: QueryConditionedReferenceMatcher,
    sampler: ReferenceEpisodeSampler,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    steps: int,
    device: torch.device,
    temperature: float,
    residual_regularization: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "retrieval_loss": 0.0, "residual_penalty": 0.0}
    for step in range(1, steps + 1):
        episode = sampler.sample(epoch=epoch, step=step)
        queries = episode.queries.to(device)
        references = episode.references.to(device)
        mask = episode.reference_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(queries, references, mask, return_aux=True)
        query_count = sampler.identities_per_batch * sampler.queries_per_identity
        scores = output["score"].reshape(query_count, sampler.identities_per_batch)
        retrieval_loss = episode_retrieval_loss(
            scores,
            episode.targets.to(device),
            temperature=temperature,
        )
        residual_penalty = output["residual"].square().mean()
        loss = retrieval_loss + residual_regularization * residual_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("non-finite matcher loss")
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["retrieval_loss"] += float(retrieval_loss.detach())
        totals["residual_penalty"] += float(residual_penalty.detach())
    return {name: value / steps for name, value in totals.items()}


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_path = args.train_features.expanduser().resolve()
    validation_path = args.validation_features.expanduser().resolve()
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("train/validation descriptor cache is missing")
    train_table = DescriptorTable.from_npz(train_path)
    validation_table = DescriptorTable.from_npz(validation_path)
    if train_table.descriptor_dim != validation_table.descriptor_dim:
        raise ValueError("train and validation descriptor dimensions differ")
    train_table.require_records(args.reference_count + args.queries_per_identity)
    validation_table.require_records(args.reference_count + 1)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model, resume_payload = build_reference_set_matcher_from_checkpoint(
            args.resume.expanduser().resolve(), device=device
        )
        expected = {
            "descriptor_dim": train_table.descriptor_dim,
            "hidden_dim": args.hidden_dim,
            "max_references": args.max_references,
            "reference_top_k": args.reference_top_k,
            "reference_score_weight": args.reference_score_weight,
            "attention_temperature": args.attention_temperature,
            "maximum_residual": args.maximum_residual,
        }
        actual = {
            key: model.configuration()[key]
            for key in expected
            if key in model.configuration()
        }
        if actual != expected:
            raise ValueError(f"resume matcher configuration differs: {actual} != {expected}")
        start_epoch = int(resume_payload.get("training", {}).get("epoch", 0)) + 1
    else:
        model = QueryConditionedReferenceMatcher(
            descriptor_dim=train_table.descriptor_dim,
            hidden_dim=args.hidden_dim,
            max_references=args.max_references,
            reference_top_k=args.reference_top_k,
            reference_score_weight=args.reference_score_weight,
            attention_temperature=args.attention_temperature,
            maximum_residual=args.maximum_residual,
        ).to(device)
        start_epoch = 1

    sampler = ReferenceEpisodeSampler(
        train_table,
        identities_per_batch=args.identities_per_batch,
        reference_count=args.reference_count,
        queries_per_identity=args.queries_per_identity,
        max_references=args.max_references,
        variable_reference_count=not args.fixed_reference_count,
        seed=args.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps = args.steps_per_epoch or max(
        1, math.ceil(train_table.num_identities / args.identities_per_batch)
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    started = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_metrics = train_one_epoch(
            model,
            sampler,
            optimizer,
            epoch=epoch,
            steps=steps,
            device=device,
            temperature=args.temperature,
            residual_regularization=args.residual_regularization,
            grad_clip=args.grad_clip,
        )
        validation = evaluate_reference_matcher(
            model,
            validation_table,
            reference_count=args.reference_count,
            max_references=args.max_references,
            device=device,
            batch_size=args.batch_size,
        )
        learned = validation["learned"]
        selection_key = (
            float(learned["top1_accuracy"]),
            float(learned["top5_accuracy"]),
            float(learned["mean_reciprocal_rank"]),
        )
        row = {
            "epoch": epoch,
            "steps": steps,
            "training": train_metrics,
            "validation": validation,
            "selection_key": list(selection_key),
        }
        history.append(row)
        training_meta = {
            "epoch": epoch,
            "train_features": str(train_path),
            "train_features_sha256": sha256_file(train_path),
            "validation_features": str(validation_path),
            "validation_features_sha256": sha256_file(validation_path),
            "train_identities": train_table.num_identities,
            "validation_identities": validation_table.num_identities,
            "steps_per_epoch": steps,
            "training": train_metrics,
            "validation": validation,
            "history": history,
            "seed": args.seed,
        }
        save_reference_set_matcher(
            model,
            output_dir / "model_last.pth",
            encoder_fingerprint=args.encoder_fingerprint,
            training=training_meta,
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            save_reference_set_matcher(
                model,
                output_dir / "model_best.pth",
                encoder_fingerprint=args.encoder_fingerprint,
                training=training_meta,
            )
        summary = {
            "epoch": epoch,
            "training": train_metrics,
            "validation": validation,
            "best_selection_key": list(best_key) if best_key else None,
            "model_last": str((output_dir / "model_last.pth").resolve()),
            "model_best": str((output_dir / "model_best.pth").resolve()),
        }
        (output_dir / "training_state.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(_jsonable(summary), ensure_ascii=False), flush=True)

    final_summary = {
        "format": "reference-set-matcher-training",
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "elapsed_seconds": time.time() - started,
        "model_best": str((output_dir / "model_best.pth").resolve()),
        "model_last": str((output_dir / "model_last.pth").resolve()),
        "best_selection_key": list(best_key) if best_key else None,
        "history": history,
        "configuration": model.configuration(),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(_jsonable(final_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(final_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
