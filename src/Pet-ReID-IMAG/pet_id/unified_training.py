# encoding: utf-8
"""Losses, retrieval metrics, and checkpoint helpers for UnifiedPetReID."""

from __future__ import annotations

import hashlib
import json
import math
import os
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .unified import UnifiedPetReID
from .release_compatibility import acceptance_schema_for_protocol


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_losses(
    predicted_boxes: torch.Tensor,
    predicted_angles: torch.Tensor,
    target_boxes: torch.Tensor,
    target_angles: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Stable normalized-box and periodic-angle supervision."""

    if predicted_boxes.shape != target_boxes.shape or predicted_boxes.ndim != 3:
        raise ValueError("predicted and target boxes must have the same [B,P,4] shape")
    center = F.smooth_l1_loss(
        predicted_boxes[..., :2],
        target_boxes[..., :2],
        beta=0.02,
    )
    size = F.smooth_l1_loss(
        predicted_boxes[..., 2:].clamp_min(1e-4).log(),
        target_boxes[..., 2:].clamp_min(1e-4).log(),
        beta=0.10,
    )
    angle = (1.0 - torch.cos(predicted_angles - target_angles)).mean()

    face = predicted_boxes[:, 0]
    nose = predicted_boxes[:, 1]
    face_left = face[:, 0] - face[:, 2] * 0.5
    face_top = face[:, 1] - face[:, 3] * 0.5
    face_right = face[:, 0] + face[:, 2] * 0.5
    face_bottom = face[:, 1] + face[:, 3] * 0.5
    nose_left = nose[:, 0] - nose[:, 2] * 0.5
    nose_top = nose[:, 1] - nose[:, 3] * 0.5
    nose_right = nose[:, 0] + nose[:, 2] * 0.5
    nose_bottom = nose[:, 1] + nose[:, 3] * 0.5
    containment = torch.stack(
        (
            F.relu(face_left - nose_left),
            F.relu(face_top - nose_top),
            F.relu(nose_right - face_right),
            F.relu(nose_bottom - face_bottom),
        ),
        dim=1,
    ).mean()
    total = 4.0 * center + 2.0 * size + 0.5 * angle + 0.5 * containment
    return {
        "geometry_center": center,
        "geometry_size": size,
        "geometry_angle": angle,
        "geometry_containment": containment,
        "geometry_total": total,
    }


def cosine_distillation(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    return (
        1.0
        - F.cosine_similarity(student.float(), teacher.float(), dim=1)
    ).mean()


def relational_distillation(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    student = F.normalize(student.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)
    student_similarity = student @ student.T
    teacher_similarity = teacher @ teacher.T
    if student.shape[0] <= 1:
        return student_similarity.sum() * 0.0
    eye = torch.eye(
        student.shape[0], dtype=torch.bool, device=student.device
    )
    return F.smooth_l1_loss(
        student_similarity[~eye],
        teacher_similarity[~eye],
        beta=0.05,
    )


def supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=1)
    targets = targets.reshape(-1)
    logits = features @ features.T / float(temperature)
    eye = torch.eye(
        logits.shape[0], dtype=torch.bool, device=logits.device
    )
    positives = targets[:, None].eq(targets[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_probabilities = logits - torch.logsumexp(
        logits, dim=1, keepdim=True
    )
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_row = -(
        log_probabilities.masked_fill(~positives, 0.0).sum(dim=1)
        / positive_count
    )
    return per_row[valid].mean()


def cross_modal_supervised_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Align two branches by identity without making same-ID views negatives."""

    first = F.normalize(first.float(), dim=1)
    second = F.normalize(second.float(), dim=1)
    targets = targets.reshape(-1)
    logits = first @ second.T / float(temperature)
    positives = targets[:, None].eq(targets[None, :])
    row = torch.logsumexp(
        logits.masked_fill(~positives, -float("inf")), dim=1
    )
    row = row - torch.logsumexp(logits, dim=1)
    column = torch.logsumexp(
        logits.masked_fill(~positives, -float("inf")), dim=0
    )
    column = column - torch.logsumexp(logits, dim=0)
    return -0.5 * (row.mean() + column.mean())


def batch_hard_metric_violation(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample hardest-positive/negative margin violations."""

    features = F.normalize(features.float(), dim=1)
    targets = targets.reshape(-1)
    similarities = features @ features.T
    eye = torch.eye(
        similarities.shape[0], dtype=torch.bool, device=similarities.device
    )
    positives = targets[:, None].eq(targets[None, :]) & ~eye
    negatives = targets[:, None].ne(targets[None, :])
    valid = positives.any(dim=1) & negatives.any(dim=1)
    hardest_positive = similarities.masked_fill(
        ~positives, float("inf")
    ).min(dim=1).values
    hardest_negative = similarities.masked_fill(
        ~negatives, -float("inf")
    ).max(dim=1).values
    violation = F.relu(float(margin) + hardest_negative - hardest_positive)
    return torch.where(valid, violation, torch.zeros_like(violation)), valid


def different_identity_permutation(
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one deterministic different-identity partner for every sample."""

    targets = targets.reshape(-1)
    indices = torch.arange(targets.shape[0], device=targets.device)
    permutation = indices.clone()
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for offset in range(1, targets.shape[0]):
        candidate = (indices + offset) % targets.shape[0]
        take = ~valid & targets[candidate].ne(targets)
        permutation = torch.where(take, candidate, permutation)
        valid |= take
    return permutation, valid


def exact_auc(positive: list[float], negative: list[float]) -> float:
    negative_sorted = sorted(float(value) for value in negative)
    wins = 0.0
    for value in positive:
        left = bisect_left(negative_sorted, float(value))
        right = bisect_right(negative_sorted, float(value))
        wins += left + 0.5 * (right - left)
    return float(wins / (len(positive) * len(negative_sorted)))


def retrieval_metrics(
    features: torch.Tensor,
    identities: list[str],
    source_paths: list[str],
    *,
    gallery_images_per_identity: int = 2,
    include_queries: bool = False,
) -> dict[str, Any]:
    """Evaluate deterministic identity prototypes in manifest record order."""

    features = F.normalize(features.float().cpu(), dim=1)
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    incomplete = {
        identity: len(indices)
        for identity, indices in grouped.items()
        if len(indices) <= gallery_images_per_identity
    }
    if incomplete:
        raise ValueError(f"Identities without held-out queries: {incomplete}")
    prototype_identities = sorted(grouped)
    prototypes = []
    query_indices = []
    for identity in prototype_identities:
        indices = grouped[identity]
        gallery = indices[:gallery_images_per_identity]
        query_indices.extend(indices[gallery_images_per_identity:])
        prototypes.append(
            F.normalize(features[gallery].mean(dim=0), dim=0)
        )
    prototype_matrix = torch.stack(prototypes)
    query_matrix = features[query_indices]
    similarities = query_matrix @ prototype_matrix.T
    rankings = similarities.argsort(dim=1, descending=True)
    identity_to_column = {
        identity: column
        for column, identity in enumerate(prototype_identities)
    }

    ranks = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    queries = []
    for row, query_index in enumerate(query_indices):
        identity = identities[query_index].casefold()
        true_column = identity_to_column[identity]
        ranking = rankings[row].tolist()
        rank = ranking.index(true_column) + 1
        ranks.append(rank)
        positive_scores.append(float(similarities[row, true_column]))
        negative_scores.extend(
            float(value)
            for column, value in enumerate(similarities[row].tolist())
            if column != true_column
        )
        if include_queries:
            queries.append(
                {
                    "query_index": query_index,
                    "query_identity": identity,
                    "query_source_path": source_paths[query_index],
                    "true_identity_rank": rank,
                    "correct": rank == 1,
                    "top5": [
                        {
                            "identity": prototype_identities[column],
                            "score": float(similarities[row, column]),
                        }
                        for column in ranking[:5]
                    ],
                }
            )
    query_count = len(ranks)
    result: dict[str, Any] = {
        "retrieval_unit": "l2_normalized_mean_identity_prototype",
        "gallery_identities": len(prototype_identities),
        "gallery_images_per_identity": gallery_images_per_identity,
        "gallery_records": len(prototype_identities)
        * gallery_images_per_identity,
        "query_records": query_count,
        "top1_correct": sum(rank == 1 for rank in ranks),
        "top1_accuracy": sum(rank == 1 for rank in ranks) / query_count,
        "top5_correct": sum(rank <= 5 for rank in ranks),
        "top5_accuracy": sum(rank <= 5 for rank in ranks) / query_count,
        "mean_reciprocal_rank": float(
            np.mean([1.0 / rank for rank in ranks])
        ),
        "auc": exact_auc(positive_scores, negative_scores),
        "same_score_mean": float(np.mean(positive_scores)),
        "different_score_mean": float(np.mean(negative_scores)),
    }
    if include_queries:
        result["queries"] = queries
    return result


def model_configuration(model: UnifiedPetReID) -> dict[str, Any]:
    return {
        "input_size": model.input_size,
        "localization_size": model.localization_size,
        "crop_size": model.crop_size,
        "geometry_hidden_channels": model.geometry.hidden_channels,
        "fusion_hidden_dim": int(
            model.semantic_fusion.context_projection[1].out_features
        ),
        "geometry_feature_mode": model.geometry_feature_mode,
        "maximum_residual_scale": model.semantic_fusion.maximum_residual_scale,
        "geometry_minimum_sizes": list(model.geometry.minimum_sizes),
        "geometry_maximum_sizes": list(model.geometry.maximum_sizes),
    }


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    arcface_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[UnifiedPetReID, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported UnifiedPetReID checkpoint schema")
    model = UnifiedPetReID.from_arcface_checkpoint(
        arcface_checkpoint,
        **payload["model_config"],
    )
    state = dict(payload["model"])
    migrations = []
    for name, value in model.state_dict().items():
        if name.startswith("geometry_calibration.") and name not in state:
            state[name] = value.detach().clone()
            migrations.append(f"initialized_identity:{name}")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unified checkpoint mismatch: {incompatible}")
    if migrations:
        payload = dict(payload)
        payload["load_migrations"] = [
            *payload.get("load_migrations", []),
            *migrations,
        ]
    return model.to(device), payload


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".exporting")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_acceptance(
    path: str | Path,
    *,
    expected_protocol: str | None = None,
) -> dict[str, Any]:
    """Load one known unified acceptance contract.

    ``expected_protocol`` lets protocol-specific tools fail before interpreting
    fields from the wrong schema. It remains optional for compatibility callers.
    """

    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol_name")
    expected_schema = acceptance_schema_for_protocol(protocol)
    if expected_schema is None:
        raise ValueError(f"Unexpected acceptance protocol: {path}")
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            f"Acceptance schema/protocol mismatch for {path}: "
            f"{payload.get('schema_version')!r} != {expected_schema}"
        )
    if expected_protocol is not None and protocol != expected_protocol:
        raise ValueError(
            f"Expected acceptance protocol {expected_protocol!r}, got "
            f"{protocol!r}: {path}"
        )
    return payload


def verify_file_lock(
    workspace: str | Path,
    item: dict[str, Any],
) -> Path:
    path = Path(workspace) / item["path"]
    path = path.resolve()
    actual = sha256_file(path)
    if actual != item["sha256"]:
        raise RuntimeError(
            f"Locked file hash mismatch for {path}: expected {item['sha256']}, got {actual}"
        )
    return path


def gradient_norm(module_or_parameters) -> float:
    parameters = (
        module_or_parameters.parameters()
        if isinstance(module_or_parameters, torch.nn.Module)
        else module_or_parameters
    )
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().square().sum())
        found = True
    return math.sqrt(total) if found else 0.0
