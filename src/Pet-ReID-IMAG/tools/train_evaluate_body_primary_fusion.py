#!/usr/bin/env python3
"""Train a small body-primary fusion neck on frozen 100/20 features."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.body_fusion import BodyPrimaryFusionNeck
from pet_id.multimodal import cross_modal_supervised_contrastive_loss
from pet_id.workspace_paths import SELECTED_MODELS_ROOT


def _path_key(value: str) -> str:
    return str(Path(value).resolve()).replace("\\", "/").casefold()


def _tensor(array: np.ndarray, *, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype)


class FrozenFeatureSet:
    def __init__(self, multimodal_path: Path, body_path: Path):
        multimodal = np.load(multimodal_path, allow_pickle=False)
        body = np.load(body_path, allow_pickle=False)
        body_rows = {
            _path_key(path): index
            for index, path in enumerate(body["source_paths"].tolist())
        }
        requested_paths = multimodal["source_paths"].tolist()
        missing = [path for path in requested_paths if _path_key(path) not in body_rows]
        if missing:
            raise ValueError(f"Body archive is missing {len(missing)} source paths")
        order = np.asarray([body_rows[_path_key(path)] for path in requested_paths])
        multimodal_identities = [value.casefold() for value in multimodal["identities"].tolist()]
        body_identities = [
            value.casefold() for value in body["identities"][order].tolist()
        ]
        if multimodal_identities != body_identities:
            raise ValueError("Aligned multimodal/body identities differ")

        unique_identities = sorted(set(multimodal_identities))
        identity_to_label = {
            identity: index for index, identity in enumerate(unique_identities)
        }
        self.identities = multimodal_identities
        self.source_paths = [str(Path(path).resolve()) for path in requested_paths]
        self.labels = torch.tensor(
            [identity_to_label[identity] for identity in self.identities],
            dtype=torch.long,
        )
        self.nose = _tensor(multimodal["adapted_nose_features"])
        self.face = _tensor(multimodal["adapted_face_features"])
        self.body = _tensor(body["body_features"][order])
        self.nose_quality = _tensor(multimodal["gate_quality_signals"])
        self.body_quality = _tensor(
            np.stack(
                (
                    body["body_detection_scores"][order],
                    body["body_box_area_ratios"][order],
                    body["face_to_body_area_ratios"][order],
                    body["imagenet_dog_probabilities"][order],
                ),
                axis=1,
            )
        )
        multimodal_available = np.asarray(multimodal["branch_available"], dtype=np.bool_)
        body_available = np.asarray(body["body_detected"][order], dtype=np.bool_)
        self.available = torch.from_numpy(
            np.concatenate((multimodal_available, body_available[:, None]), axis=1)
        ).bool()
        self.baseline = _tensor(multimodal["baseline_features"])
        self.num_classes = len(unique_identities)
        self.multimodal_path = multimodal_path.resolve()
        self.body_path = body_path.resolve()

    def __len__(self) -> int:
        return len(self.identities)

    def to(self, device: torch.device) -> "FrozenFeatureSet":
        for name in (
            "labels",
            "nose",
            "face",
            "body",
            "nose_quality",
            "body_quality",
            "available",
            "baseline",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def inputs(self, indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        def select(value: torch.Tensor) -> torch.Tensor:
            return value if indices is None else value.index_select(0, indices)

        return {
            "nose_features": select(self.nose),
            "face_features": select(self.face),
            "body_features": select(self.body),
            "nose_quality_signals": select(self.nose_quality),
            "body_quality_signals": select(self.body_quality),
            "branch_available": select(self.available),
        }


def exact_auc(positive: list[float], negative: list[float]) -> float:
    negative_sorted = sorted(float(value) for value in negative)
    wins = 0.0
    for value in positive:
        left = bisect_left(negative_sorted, float(value))
        right = bisect_right(negative_sorted, float(value))
        wins += left + 0.5 * (right - left)
    return float(wins / (len(positive) * len(negative_sorted)))


def best_balanced_threshold(positive: list[float], negative: list[float]) -> dict:
    rows = sorted(
        [(float(value), 1) for value in positive]
        + [(float(value), 0) for value in negative],
        reverse=True,
    )
    positive_count = len(positive)
    negative_count = len(negative)
    true_positives = false_positives = 0
    best = (0.5, 0.0, 1.0, rows[0][0] + 1e-6)
    cursor = 0
    while cursor < len(rows):
        score = rows[cursor][0]
        next_cursor = cursor
        while next_cursor < len(rows) and rows[next_cursor][0] == score:
            if rows[next_cursor][1]:
                true_positives += 1
            else:
                false_positives += 1
            next_cursor += 1
        next_score = rows[next_cursor][0] if next_cursor < len(rows) else score - 2e-6
        threshold = 0.5 * (score + next_score)
        true_positive_rate = true_positives / positive_count
        true_negative_rate = (negative_count - false_positives) / negative_count
        candidate = (
            0.5 * (true_positive_rate + true_negative_rate),
            true_positive_rate,
            true_negative_rate,
            threshold,
        )
        if candidate > best:
            best = candidate
        cursor = next_cursor
    return {
        "threshold": best[3],
        "balanced_accuracy": best[0],
        "same_recall": best[1],
        "different_recall": best[2],
    }


def evaluate_features(
    features: torch.Tensor,
    identities: list[str],
    source_paths: list[str],
    *,
    gallery_images_per_identity: int = 2,
) -> dict:
    features = F.normalize(features.float().cpu(), dim=1)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities):
        grouped[identity].append(index)
    prototype_identities = sorted(grouped)
    prototypes = []
    query_indices: list[int] = []
    for identity in prototype_identities:
        indices = grouped[identity]
        if len(indices) <= gallery_images_per_identity:
            raise ValueError(f"Identity {identity} has no held-out query")
        gallery_indices = indices[:gallery_images_per_identity]
        query_indices.extend(indices[gallery_images_per_identity:])
        prototypes.append(F.normalize(features[gallery_indices].mean(dim=0), dim=0))
    prototype_matrix = torch.stack(prototypes)
    similarities = features[query_indices] @ prototype_matrix.T
    rankings = similarities.argsort(dim=1, descending=True)
    identity_to_column = {
        identity: column for column, identity in enumerate(prototype_identities)
    }
    positives: list[float] = []
    negatives: list[float] = []
    margins: list[float] = []
    query_rows = []
    top1_correct = top5_correct = 0
    for row, query_index in enumerate(query_indices):
        true_column = identity_to_column[identities[query_index]]
        ranking = rankings[row].tolist()
        true_rank = ranking.index(true_column) + 1
        positive = float(similarities[row, true_column])
        negative_columns = [column for column in range(len(prototypes)) if column != true_column]
        negative_values = similarities[row, negative_columns]
        strongest_negative = float(negative_values.max())
        positives.append(positive)
        negatives.extend(float(value) for value in negative_values.tolist())
        margins.append(positive - strongest_negative)
        top1_correct += int(true_rank == 1)
        top5_correct += int(true_rank <= 5)
        query_rows.append(
            {
                "query_index": query_index,
                "query_identity": identities[query_index],
                "query_source_path": source_paths[query_index],
                "true_identity_rank": true_rank,
                "correct": true_rank == 1,
                "positive_score": positive,
                "strongest_negative_score": strongest_negative,
                "margin": margins[-1],
                "predicted_identity": prototype_identities[ranking[0]],
            }
        )

    loo_correct = 0
    for query_index, query_identity in enumerate(identities):
        loo_prototypes = []
        for identity in prototype_identities:
            indices = grouped[identity]
            if identity == query_identity:
                indices = [index for index in indices if index != query_index]
            loo_prototypes.append(F.normalize(features[indices].mean(dim=0), dim=0))
        scores = features[query_index] @ torch.stack(loo_prototypes).T
        prediction = int(scores.argmax())
        loo_correct += int(prototype_identities[prediction] == query_identity)

    query_count = len(query_indices)
    threshold = best_balanced_threshold(positives, negatives)
    return {
        "retrieval_unit": "l2_normalized_mean_identity_prototype",
        "gallery_identities": len(prototype_identities),
        "gallery_images_per_identity": gallery_images_per_identity,
        "query_records": query_count,
        "gallery_rank1_correct": top1_correct,
        "gallery_rank1": top1_correct / query_count,
        "gallery_rank5_correct": top5_correct,
        "gallery_rank5": top5_correct / query_count,
        "leave_one_out_correct": loo_correct,
        "leave_one_out_rank1": loo_correct / len(identities),
        "auc": exact_auc(positives, negatives),
        "best_balanced_threshold": threshold,
        "mean_gap": float(np.mean(margins)),
        "worst_gap": float(np.min(margins)),
        "queries": query_rows,
    }


def load_semantic_initialization(
    model: BodyPrimaryFusionNeck,
    checkpoint_path: Path,
) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError("Semantic checkpoint does not contain a model state")

    def substate(prefix: str) -> dict[str, torch.Tensor]:
        direct = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if direct:
            return direct
        module_prefix = "module." + prefix
        return {
            key[len(module_prefix) :]: value
            for key, value in state.items()
            if key.startswith(module_prefix)
        }

    gate_state = substate("gate.")
    interaction_state = substate("cross_modal_residual.")
    if not gate_state or not interaction_state:
        raise ValueError("Checkpoint is missing semantic-v3 gate/interaction tensors")
    model.nose_gate.load_state_dict(gate_state, strict=True)
    model.nose_interaction.load_state_dict(interaction_state, strict=True)
    model.nose_gate.requires_grad_(False)
    model.nose_interaction.requires_grad_(False)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "gate_tensors": len(gate_state),
        "interaction_tensors": len(interaction_state),
        "nose_fusion_frozen": True,
    }


def initialize_body_projection_ridge(
    model: BodyPrimaryFusionNeck,
    dataset: FrozenFeatureSet,
    *,
    regularization: float,
) -> dict:
    valid = dataset.available[:, 2] & dataset.available[:, 1]
    indices = valid.nonzero(as_tuple=False).flatten()
    with torch.no_grad():
        body = dataset.body.index_select(0, indices)
        normalized_body = model.body_adapter.input_norm(body.float())
        labels = dataset.labels.index_select(0, indices)
        prototypes = []
        for label in range(dataset.num_classes):
            selected = dataset.face[dataset.labels == label]
            prototypes.append(F.normalize(selected.mean(dim=0), dim=0))
        targets = torch.stack(prototypes).index_select(0, labels)
        gram = normalized_body @ normalized_body.T
        gram.diagonal().add_(float(regularization))
        dual = torch.linalg.solve(gram, targets)
        weight = dual.T @ normalized_body
        model.body_adapter.projection.weight.copy_(weight)
        model.body_adapter.residual[-1].weight.zero_()
    return {
        "method": "dual_ridge_to_training_identity_face_prototype",
        "valid_records": int(indices.numel()),
        "regularization": float(regularization),
    }


def batch_hard_triplet(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.30,
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=1)
    distances = 1.0 - features @ features.T
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & ~eye
    negatives = ~same
    hardest_positive = distances.masked_fill(~positives, -float("inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~negatives, float("inf")).min(dim=1).values
    valid = positives.any(dim=1) & negatives.any(dim=1)
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def cosine_logits(
    features: torch.Tensor,
    classifier: torch.Tensor,
    *,
    scale: float = 32.0,
) -> torch.Tensor:
    return float(scale) * F.linear(
        F.normalize(features, dim=1),
        F.normalize(classifier, dim=1),
    )


def sample_pk_batch(
    label_rows: dict[int, list[int]],
    *,
    identities_per_batch: int,
    images_per_identity: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    labels = rng.sample(
        sorted(label_rows),
        min(int(identities_per_batch), len(label_rows)),
    )
    indices: list[int] = []
    for label in labels:
        rows = label_rows[label]
        if len(rows) >= images_per_identity:
            indices.extend(rng.sample(rows, images_per_identity))
        else:
            indices.extend(rng.choices(rows, k=images_per_identity))
    rng.shuffle(indices)
    return torch.tensor(indices, dtype=torch.long, device=device)


def feature_outputs(
    model: BodyPrimaryFusionNeck,
    dataset: FrozenFeatureSet,
    *,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            indices = torch.arange(
                start,
                min(start + batch_size, len(dataset)),
                device=dataset.face.device,
            )
            output = model(**dataset.inputs(indices))
            for name in (
                "features",
                "primary_features",
                "adapted_body_features",
                "body_weights",
                "nose_weights",
            ):
                rows[name].append(output[name].detach().cpu())
    return {name: torch.cat(values) for name, values in rows.items()}


def selection_key(row: dict) -> tuple:
    metrics = row["metrics"]
    return (
        metrics["gallery_rank1"],
        metrics["leave_one_out_rank1"],
        metrics["auc"],
        metrics["best_balanced_threshold"]["balanced_accuracy"],
        metrics["mean_gap"],
        metrics["worst_gap"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-multimodal", type=Path, required=True)
    parser.add_argument("--train-body", type=Path, required=True)
    parser.add_argument("--validation-multimodal", type=Path, required=True)
    parser.add_argument("--validation-body", type=Path, required=True)
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "model_final.pth",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-regularization", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train = FrozenFeatureSet(args.train_multimodal, args.train_body).to(device)
    validation = FrozenFeatureSet(
        args.validation_multimodal, args.validation_body
    ).to(device)
    if train.num_classes != 100 or validation.num_classes != 20:
        raise ValueError(
            f"Expected 100/20 identities, got {train.num_classes}/{validation.num_classes}"
        )
    if set(train.identities) & set(validation.identities):
        raise ValueError("Train and validation identities overlap")

    model = BodyPrimaryFusionNeck(
        body_dim=train.body.shape[1],
        embedding_dim=train.face.shape[1],
        body_quality_dim=train.body_quality.shape[1],
        nose_quality_dim=train.nose_quality.shape[1],
    ).to(device)
    semantic_initialization = load_semantic_initialization(
        model, args.semantic_checkpoint
    )

    compatibility_available = train.available.clone()
    compatibility_available[:, 2] = False
    with torch.inference_mode():
        fallback_inputs = train.inputs()
        fallback_inputs["branch_available"] = compatibility_available
        fallback = model(**fallback_inputs)["features"]
        baseline_cosine = F.cosine_similarity(fallback, train.baseline, dim=1)
        compatibility = {
            "body_missing_mean_cosine_to_semantic_v3": float(baseline_cosine.mean()),
            "body_missing_min_cosine_to_semantic_v3": float(baseline_cosine.min()),
            "body_missing_max_absolute_difference": float(
                (fallback - train.baseline).abs().max()
            ),
        }
    ridge_initialization = initialize_body_projection_ridge(
        model,
        train,
        regularization=args.ridge_regularization,
    )

    classifiers = nn.ParameterDict(
        {
            name: nn.Parameter(torch.empty(train.num_classes, model.embedding_dim, device=device))
            for name in ("final", "primary", "body")
        }
    )
    for parameter in classifiers.values():
        nn.init.normal_(parameter, std=0.01)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ] + list(classifiers.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    label_rows: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(train.labels.tolist()):
        label_rows[int(label)].append(index)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = evaluate_features(
        validation.baseline,
        validation.identities,
        validation.source_paths,
    )
    raw_body_metrics = evaluate_features(
        validation.body,
        validation.identities,
        validation.source_paths,
    )
    candidates: list[dict] = []
    history: list[dict] = []

    def save_and_evaluate(step: int) -> None:
        outputs = feature_outputs(model, validation)
        metrics = evaluate_features(
            outputs["features"],
            validation.identities,
            validation.source_paths,
        )
        checkpoint_path = args.output_dir / f"checkpoint_{step:07d}.pth"
        checkpoint = {
            "schema_version": 1,
            "step": step,
            "neck": model.state_dict(),
            "classifiers": classifiers.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": {
                "name": "body_primary_semantic_residual_v1",
                "input": "one dog image through frozen nose/face/body encoders",
                "output_dim": model.embedding_dim,
                "body_dim": model.body_dim,
                "body_classifier_head": None,
                "max_body_weight": model.body_gate.max_body_weight,
                "max_nose_weight": model.nose_gate.max_nose_weight,
            },
            "semantic_initialization": semantic_initialization,
            "ridge_initialization": ridge_initialization,
            "seed": args.seed,
        }
        torch.save(checkpoint, checkpoint_path)
        both = validation.available[:, 1] & validation.available[:, 2]
        nose_primary = validation.available[:, 0] & (
            validation.available[:, 1] | validation.available[:, 2]
        )
        candidate = {
            "step": step,
            "checkpoint": str(checkpoint_path.resolve()),
            "metrics": metrics,
            "mean_body_weight_when_face_and_body_available": float(
                outputs["body_weights"][both.cpu(), 0].mean()
            ),
            "mean_nose_weight_when_primary_and_nose_available": float(
                outputs["nose_weights"][nose_primary.cpu(), 0].mean()
            ),
        }
        candidates.append(candidate)
        concise = {
            "step": step,
            "rank1": metrics["gallery_rank1"],
            "loo_rank1": metrics["leave_one_out_rank1"],
            "auc": metrics["auc"],
            "mean_body_weight": candidate[
                "mean_body_weight_when_face_and_body_available"
            ],
        }
        print(json.dumps(concise), flush=True)

    save_and_evaluate(0)
    for step in range(1, args.steps + 1):
        model.train()
        indices = sample_pk_batch(
            label_rows,
            identities_per_batch=args.identities_per_batch,
            images_per_identity=args.images_per_identity,
            rng=rng,
            device=device,
        )
        labels = train.labels.index_select(0, indices)
        output = model(**train.inputs(indices))
        final_features = output["features"]
        primary_features = output["primary_features"]
        adapted_body = output["adapted_body_features"]
        final_ce = F.cross_entropy(
            cosine_logits(final_features, classifiers["final"]), labels
        )
        primary_ce = F.cross_entropy(
            cosine_logits(primary_features, classifiers["primary"]), labels
        )
        final_triplet = batch_hard_triplet(final_features, labels)
        primary_triplet = batch_hard_triplet(primary_features, labels)
        detected = train.available.index_select(0, indices)[:, 2]
        if detected.any():
            body_ce = F.cross_entropy(
                cosine_logits(adapted_body[detected], classifiers["body"]),
                labels[detected],
            )
            cross_modal = cross_modal_supervised_contrastive_loss(
                adapted_body[detected],
                train.face.index_select(0, indices)[detected],
                labels[detected],
                temperature=0.10,
            )
        else:
            body_ce = final_ce * 0.0
            cross_modal = final_ce * 0.0
        both = train.available.index_select(0, indices)[:, 1:].all(dim=1)
        body_prior_penalty = (
            output["body_weights"][both, 0].mean() - model.initial_body_weight
        ).square()
        loss = (
            final_ce
            + 0.50 * final_triplet
            + 0.50 * primary_ce
            + 0.25 * primary_triplet
            + 0.25 * body_ce
            + 0.10 * cross_modal
            + 0.05 * body_prior_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "final_ce": float(final_ce.detach()),
                "final_triplet": float(final_triplet.detach()),
                "primary_ce": float(primary_ce.detach()),
                "body_ce": float(body_ce.detach()),
                "cross_modal": float(cross_modal.detach()),
                "mean_body_weight": float(output["body_weights"][both, 0].mean().detach()),
            }
        )
        if step % args.eval_interval == 0 or step == args.steps:
            save_and_evaluate(step)

    ranked = sorted(candidates, key=selection_key, reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    selected = ranked[0]
    selected_checkpoint = Path(selected["checkpoint"])
    model_final = args.output_dir / "model_final.pth"
    shutil.copy2(selected_checkpoint, model_final)
    selected_payload = torch.load(model_final, map_location=device, weights_only=False)
    model.load_state_dict(selected_payload["neck"])
    selected_outputs = feature_outputs(model, validation)
    np.savez_compressed(
        args.output_dir / "selected_validation_features.npz",
        features=selected_outputs["features"].numpy(),
        primary_features=selected_outputs["primary_features"].numpy(),
        adapted_body_features=selected_outputs["adapted_body_features"].numpy(),
        body_weights=selected_outputs["body_weights"].numpy(),
        nose_weights=selected_outputs["nose_weights"].numpy(),
        identities=np.asarray(validation.identities),
        source_paths=np.asarray(validation.source_paths),
    )

    baseline_queries = {
        row["query_source_path"]: row for row in baseline_metrics["queries"]
    }
    selected_queries = {
        row["query_source_path"]: row
        for row in selected["metrics"]["queries"]
    }
    rescued = []
    lost = []
    for path, baseline_row in baseline_queries.items():
        selected_row = selected_queries[path]
        comparison = {
            "query_source_path": path,
            "identity": baseline_row["query_identity"],
            "baseline_prediction": baseline_row["predicted_identity"],
            "body_fusion_prediction": selected_row["predicted_identity"],
            "baseline_margin": baseline_row["margin"],
            "body_fusion_margin": selected_row["margin"],
        }
        if not baseline_row["correct"] and selected_row["correct"]:
            rescued.append(comparison)
        elif baseline_row["correct"] and not selected_row["correct"]:
            lost.append(comparison)

    report = {
        "schema_version": 1,
        "protocol": {
            "train_identities": train.num_classes,
            "train_records": len(train),
            "validation_identities": validation.num_classes,
            "validation_records": len(validation),
            "identity_overlap": 0,
            "model_selection_data": "validation identities only",
            "gallery_images_per_identity": 2,
        },
        "interface": {
            "input": "one dog image; body crop is internal",
            "output": "512-D L2-normalized identity embedding",
            "retrieval": "unchanged cosine similarity/prototype gallery",
        },
        "architecture": {
            "primary": "bounded face + headless Swin-V2-B body fusion",
            "residual": "pretrained semantic-v3 bounded nose residual",
            "body_backbone_classifier_head": "removed",
            "body_backbone_frozen": True,
            "nose_face_encoders_frozen": True,
            "nose_gate_and_interaction_frozen": True,
        },
        "sources": {
            "train_multimodal": str(train.multimodal_path),
            "train_body": str(train.body_path),
            "validation_multimodal": str(validation.multimodal_path),
            "validation_body": str(validation.body_path),
        },
        "semantic_initialization": semantic_initialization,
        "ridge_initialization": ridge_initialization,
        "compatibility": compatibility,
        "selection_rule": (
            "lexicographic descending: gallery_rank1, leave_one_out_rank1, "
            "auc, balanced_accuracy, mean_gap, worst_gap"
        ),
        "baseline_semantic_v3_nose_face": baseline_metrics,
        "raw_body_backbone": raw_body_metrics,
        "candidates_ranked": ranked,
        "selected": selected,
        "selected_model": str(model_final.resolve()),
        "delta_vs_nose_face": {
            "gallery_rank1": selected["metrics"]["gallery_rank1"]
            - baseline_metrics["gallery_rank1"],
            "leave_one_out_rank1": selected["metrics"]["leave_one_out_rank1"]
            - baseline_metrics["leave_one_out_rank1"],
            "auc": selected["metrics"]["auc"] - baseline_metrics["auc"],
            "balanced_accuracy": selected["metrics"]["best_balanced_threshold"][
                "balanced_accuracy"
            ]
            - baseline_metrics["best_balanced_threshold"]["balanced_accuracy"],
            "mean_gap": selected["metrics"]["mean_gap"]
            - baseline_metrics["mean_gap"],
        },
        "query_outcomes_vs_nose_face": {
            "rescued_count": len(rescued),
            "lost_count": len(lost),
            "rescued": rescued,
            "lost": lost,
        },
        "history": history,
    }
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        "output": str(report_path.resolve()),
        "selected_step": selected["step"],
        "baseline": {
            key: baseline_metrics[key]
            for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
        },
        "body_fusion": {
            key: selected["metrics"][key]
            for key in ("gallery_rank1", "leave_one_out_rank1", "auc")
        },
        "rescued": len(rescued),
        "lost": len(lost),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
