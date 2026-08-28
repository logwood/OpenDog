#!/usr/bin/env python3
"""Evaluate a frozen whole-dog semantic branch on a prepared Re-ID manifest.

The branch deliberately contains no project-trained model:

1. Torchvision Faster R-CNN selects the COCO ``dog`` box associated with the
   manifest's target face.
2. Torchvision Swin V2-B emits its frozen ImageNet feature and logits.
3. The contiguous ImageNet dog classes are exposed as a 118-way soft breed
   distribution rather than a brittle hard breed prediction.

The script exports reusable features and reports identity-prototype retrieval
only as a diagnostic of whether the frozen representation carries useful
same-dog evidence. It does not modify or fuse with the joint nose/face model.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.models import Swin_V2_B_Weights, swin_v2_b
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms import functional as TVF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def exact_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Return exact Mann-Whitney AUC without a quadratic allocation."""
    negative_sorted = sorted(float(value) for value in negative)
    wins = 0.0
    for value in positive:
        left = bisect_left(negative_sorted, float(value))
        right = bisect_right(negative_sorted, float(value))
        wins += left + 0.5 * (right - left)
    return float(wins / (len(positive) * len(negative_sorted)))


def summarize(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def evaluate_branch(
    features: torch.Tensor,
    identities: Sequence[str],
    source_paths: Sequence[str],
    gallery_per_identity: int,
) -> dict:
    """Run the same mean-prototype retrieval used by the joint-model evaluator."""
    features = F.normalize(features.float(), dim=1)
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity, []).append(index)
    incomplete = {
        identity: len(indices)
        for identity, indices in grouped.items()
        if len(indices) <= gallery_per_identity
    }
    if incomplete:
        raise ValueError(f"Identities without held-out queries: {incomplete}")

    prototype_identities = sorted(grouped)
    prototypes: list[torch.Tensor] = []
    query_indices: list[int] = []
    for identity in prototype_identities:
        indices = grouped[identity]
        gallery_indices = indices[:gallery_per_identity]
        query_indices.extend(indices[gallery_per_identity:])
        prototype = features[gallery_indices].mean(dim=0)
        prototypes.append(F.normalize(prototype, dim=0))

    prototype_matrix = torch.stack(prototypes)
    query_matrix = features[query_indices]
    similarities = query_matrix @ prototype_matrix.T
    rankings = similarities.argsort(dim=1, descending=True)
    identity_to_column = {
        identity: column for column, identity in enumerate(prototype_identities)
    }

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    query_results: list[dict] = []
    for row, query_index in enumerate(query_indices):
        query_identity = identities[query_index]
        true_column = identity_to_column[query_identity]
        ranking = rankings[row].tolist()
        true_rank = ranking.index(true_column) + 1
        positive_scores.append(float(similarities[row, true_column]))
        negative_scores.extend(
            float(value)
            for column, value in enumerate(similarities[row].tolist())
            if column != true_column
        )
        query_results.append(
            {
                "query_index": query_index,
                "query_identity": query_identity,
                "query_source_path": source_paths[query_index],
                "true_identity_rank": true_rank,
                "correct": true_rank == 1,
                "top5": [
                    {
                        "identity": prototype_identities[column],
                        "score": float(similarities[row, column]),
                    }
                    for column in ranking[:5]
                ],
            }
        )

    ranks = [row["true_identity_rank"] for row in query_results]
    query_count = len(query_results)
    return {
        "retrieval_unit": "l2_normalized_mean_identity_prototype",
        "gallery_identities": len(prototype_identities),
        "gallery_images_per_identity": gallery_per_identity,
        "gallery_records": len(prototype_identities) * gallery_per_identity,
        "query_records": query_count,
        "top1_correct": sum(rank <= 1 for rank in ranks),
        "top1_accuracy": sum(rank <= 1 for rank in ranks) / query_count,
        "top5_correct": sum(rank <= 5 for rank in ranks),
        "top5_accuracy": sum(rank <= 5 for rank in ranks) / query_count,
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "true_rank": summarize(ranks),
        "same_score": summarize(positive_scores),
        "different_score": summarize(negative_scores),
        "auc": exact_auc(positive_scores, negative_scores),
        "queries": query_results,
    }


def box_area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(
        float(box[3]) - float(box[1]), 0.0
    )


def intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    return max(min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])), 0.0) * max(
        min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])), 0.0
    )


def select_target_dog_box(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    face_box: Sequence[float],
    *,
    dog_label: int,
    score_threshold: float,
) -> tuple[list[float] | None, float]:
    """Select the detected dog most strongly associated with the target face."""
    face_area = max(box_area(face_box), 1e-6)
    face_center = (
        0.5 * (float(face_box[0]) + float(face_box[2])),
        0.5 * (float(face_box[1]) + float(face_box[3])),
    )
    candidates: list[tuple[tuple[float, float, float], list[float], float]] = []
    for box_tensor, label, score_tensor in zip(boxes, labels, scores):
        score = float(score_tensor)
        if int(label) != dog_label or score < score_threshold:
            continue
        box = [float(value) for value in box_tensor.tolist()]
        contains_center = float(
            box[0] <= face_center[0] <= box[2]
            and box[1] <= face_center[1] <= box[3]
        )
        face_coverage = intersection_area(box, face_box) / face_area
        candidates.append(((contains_center, face_coverage, score), box, score))
    if not candidates:
        return None, 0.0
    _, selected_box, selected_score = max(candidates, key=lambda row: row[0])
    return selected_box, selected_score


def expand_and_clip_box(
    box: Sequence[float], width: int, height: int, expansion: float
) -> list[int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    pad_x = (x2 - x1) * expansion
    pad_y = (y2 - y1) * expansion
    return [
        max(int(math.floor(x1 - pad_x)), 0),
        max(int(math.floor(y1 - pad_y)), 0),
        min(int(math.ceil(x2 + pad_x)), width),
        min(int(math.ceil(y2 + pad_y)), height),
    ]


def forward_swin_features(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror Torchvision Swin.forward while retaining the pre-head feature."""
    features = model.features(images)
    features = model.norm(features)
    features = model.permute(features)
    features = model.avgpool(features)
    features = model.flatten(features)
    logits = model.head(features)
    return features, logits


def dog_breed_class_range(categories: Sequence[str]) -> tuple[int, int]:
    start = categories.index("Chihuahua")
    end = categories.index("Mexican hairless") + 1
    if end - start != 118:
        raise RuntimeError(
            f"Unexpected ImageNet dog breed range [{start}, {end}) with {end - start} labels"
        )
    return start, end


def normalize_path(value: str) -> str:
    return str(Path(value).resolve()).casefold()


def compare_with_reference(body_evaluation: dict, reference_path: Path) -> dict:
    """Count complementarity without fitting a score-level fusion on the test set."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    body_queries = {
        normalize_path(row["query_source_path"]): row
        for row in body_evaluation["queries"]
    }
    comparisons = {}
    for branch_name, branch in reference.get("branches", {}).items():
        reference_queries = {
            normalize_path(row["query_source_path"]): row
            for row in branch.get("queries", [])
        }
        shared_paths = sorted(set(body_queries) & set(reference_queries))
        both_correct = body_only = reference_only = both_wrong = 0
        for path in shared_paths:
            body_correct = bool(body_queries[path]["correct"])
            reference_correct = bool(reference_queries[path]["correct"])
            if body_correct and reference_correct:
                both_correct += 1
            elif body_correct:
                body_only += 1
            elif reference_correct:
                reference_only += 1
            else:
                both_wrong += 1
        comparisons[branch_name] = {
            "shared_queries": len(shared_paths),
            "both_correct": both_correct,
            "body_only_correct_potential_rescues": body_only,
            "reference_only_correct": reference_only,
            "both_wrong": both_wrong,
            "oracle_union_correct": both_correct + body_only + reference_only,
        }
    return {
        "reference_evaluation": str(reference_path.resolve()),
        "branches": comparisons,
        "note": "Oracle counts measure complementarity only; no fusion was fitted.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-batch-size", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    parser.add_argument("--detection-score-threshold", type=float, default=0.5)
    parser.add_argument("--crop-expansion", type=float, default=0.04)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument(
        "--evaluation-purpose",
        choices=("development", "spent_test_diagnostic", "locked_final"),
        default="development",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.detection_score_threshold <= 1.0:
        raise ValueError("--detection-score-threshold must be in [0, 1]")
    if args.detector_batch_size < 1 or args.classifier_batch_size < 1:
        raise ValueError("Batch sizes must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    if not records:
        raise RuntimeError("Manifest contains no records")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    detector_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector_categories = detector_weights.meta["categories"]
    dog_label = detector_categories.index("dog")
    detector_transform = detector_weights.transforms()
    detector = fasterrcnn_resnet50_fpn_v2(weights=detector_weights).to(device).eval()

    classifier_weights = Swin_V2_B_Weights.DEFAULT
    classifier_categories = classifier_weights.meta["categories"]
    breed_start, breed_end = dog_breed_class_range(classifier_categories)
    breed_names = list(classifier_categories[breed_start:breed_end])
    classifier_transform = classifier_weights.transforms()
    classifier = swin_v2_b(weights=classifier_weights).to(device).eval()

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    identities: list[str] = []
    source_paths: list[str] = []
    body_crops: list[torch.Tensor] = []
    body_boxes: list[list[int]] = []
    body_detected: list[bool] = []
    detection_scores: list[float] = []
    body_box_area_ratios: list[float] = []
    face_to_body_area_ratios: list[float] = []

    for batch_start in range(0, len(records), args.detector_batch_size):
        batch_records = records[batch_start : batch_start + args.detector_batch_size]
        images: list[torch.Tensor] = []
        face_boxes: list[list[float]] = []
        for record in batch_records:
            image = read_image(record["source_path"], mode=ImageReadMode.RGB)
            target_width, target_height = (int(value) for value in record["resized_size"])
            if tuple(image.shape[-2:]) != (target_height, target_width):
                image = TVF.resize(
                    image,
                    [target_height, target_width],
                    antialias=True,
                )
            images.append(image)
            face_boxes.append([float(value) for value in record["face_roi_xyxy"]])

        detector_inputs = [detector_transform(image).to(device) for image in images]
        with torch.inference_mode():
            predictions = detector(detector_inputs)

        for record, image, face_box, prediction in zip(
            batch_records, images, face_boxes, predictions
        ):
            height, width = image.shape[-2:]
            selected_box, selected_score = select_target_dog_box(
                prediction["boxes"].detach().cpu(),
                prediction["labels"].detach().cpu(),
                prediction["scores"].detach().cpu(),
                face_box,
                dog_label=dog_label,
                score_threshold=args.detection_score_threshold,
            )
            detected = selected_box is not None
            if selected_box is None:
                selected_box = [0.0, 0.0, float(width), float(height)]
            crop_box = expand_and_clip_box(
                selected_box,
                width,
                height,
                args.crop_expansion if detected else 0.0,
            )
            x1, y1, x2, y2 = crop_box
            body_crop = image[:, y1:y2, x1:x2]
            if body_crop.numel() == 0:
                raise RuntimeError(f"Empty body crop for {record['source_path']}: {crop_box}")
            body_crops.append(classifier_transform(body_crop))
            body_boxes.append(crop_box)
            body_detected.append(detected)
            detection_scores.append(selected_score)
            body_area = max(box_area(crop_box), 1.0)
            body_box_area_ratios.append(body_area / float(width * height))
            face_to_body_area_ratios.append(box_area(face_box) / body_area)
            identities.append(record["identity"].casefold())
            source_paths.append(record["source_path"])

        processed = min(batch_start + len(batch_records), len(records))
        print(
            f"detector: {processed}/{len(records)} "
            f"({sum(body_detected)}/{processed} target dogs found)",
            flush=True,
        )

    feature_rows: list[torch.Tensor] = []
    logit_rows: list[torch.Tensor] = []
    for batch_start in range(0, len(body_crops), args.classifier_batch_size):
        batch = torch.stack(
            body_crops[batch_start : batch_start + args.classifier_batch_size]
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            features, logits = forward_swin_features(classifier, batch)
        feature_rows.append(F.normalize(features.float(), dim=1).cpu())
        logit_rows.append(logits.float().cpu())
        processed = min(batch_start + len(batch), len(body_crops))
        print(f"classifier: {processed}/{len(body_crops)}", flush=True)

    body_features = torch.cat(feature_rows)
    imagenet_logits = torch.cat(logit_rows)
    imagenet_probabilities = imagenet_logits.softmax(dim=1)
    dog_probabilities = imagenet_probabilities[:, breed_start:breed_end].sum(dim=1)
    breed_probabilities = imagenet_logits[:, breed_start:breed_end].softmax(dim=1)
    breed_entropy = -(
        breed_probabilities
        * breed_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=1) / math.log(len(breed_names))

    body_evaluation = evaluate_branch(
        body_features,
        identities,
        source_paths,
        args.gallery_images_per_identity,
    )
    breed_evaluation = evaluate_branch(
        F.normalize(breed_probabilities, dim=1),
        identities,
        source_paths,
        args.gallery_images_per_identity,
    )

    top_values, top_indices = breed_probabilities.topk(5, dim=1)
    record_diagnostics = []
    for index, record in enumerate(records):
        record_diagnostics.append(
            {
                "source_path": source_paths[index],
                "identity": identities[index],
                "body_box_xyxy": body_boxes[index],
                "body_detected": body_detected[index],
                "body_detection_score": detection_scores[index],
                "body_box_area_ratio": body_box_area_ratios[index],
                "face_to_body_area_ratio": face_to_body_area_ratios[index],
                "imagenet_dog_probability": float(dog_probabilities[index]),
                "breed_entropy_normalized": float(breed_entropy[index]),
                "breed_top5": [
                    {
                        "breed": breed_names[int(column)],
                        "probability": float(value),
                    }
                    for value, column in zip(top_values[index], top_indices[index])
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "body_semantic_features.npz"
    np.savez_compressed(
        features_path,
        body_features=body_features.numpy(),
        breed_probabilities=breed_probabilities.numpy(),
        imagenet_logits=imagenet_logits.numpy(),
        identities=np.asarray(identities),
        source_paths=np.asarray(source_paths),
        body_boxes_xyxy=np.asarray(body_boxes, dtype=np.int32),
        body_detected=np.asarray(body_detected, dtype=np.bool_),
        body_detection_scores=np.asarray(detection_scores, dtype=np.float32),
        body_box_area_ratios=np.asarray(body_box_area_ratios, dtype=np.float32),
        face_to_body_area_ratios=np.asarray(
            face_to_body_area_ratios, dtype=np.float32
        ),
        imagenet_dog_probabilities=dog_probabilities.numpy(),
        breed_entropy_normalized=breed_entropy.numpy(),
        breed_names=np.asarray(breed_names),
    )

    summary = {
        "schema_version": 1,
        "purpose": args.evaluation_purpose,
        "manifest": str(args.manifest.resolve()),
        "protocol_split": manifest.get("protocol_split"),
        "records": len(records),
        "identities": len(set(identities)),
        "model_policy": "official_frozen_pretrained_weights_only",
        "detector": {
            "name": "torchvision/fasterrcnn_resnet50_fpn_v2",
            "weights": detector_weights.name,
            "dog_label": dog_label,
            "score_threshold": args.detection_score_threshold,
            "crop_expansion": args.crop_expansion,
            "detected_records": sum(body_detected),
            "detection_rate": sum(body_detected) / len(body_detected),
        },
        "classifier": {
            "name": "torchvision/swin_v2_b",
            "weights": classifier_weights.name,
            "feature_dimensions": int(body_features.shape[1]),
            "breed_dimensions": int(breed_probabilities.shape[1]),
            "breed_class_start_inclusive": breed_start,
            "breed_class_end_exclusive": breed_end,
            "breed_names": breed_names,
            "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        },
        "quality": {
            "body_detection_score": summarize(detection_scores),
            "body_box_area_ratio": summarize(body_box_area_ratios),
            "face_to_body_area_ratio": summarize(face_to_body_area_ratios),
            "imagenet_dog_probability": summarize(dog_probabilities.tolist()),
            "breed_entropy_normalized": summarize(breed_entropy.tolist()),
        },
        "branches": {
            "body_feature": body_evaluation,
            "breed_probability": breed_evaluation,
        },
        "record_diagnostics": record_diagnostics,
        "feature_archive": str(features_path.resolve()),
    }
    if args.reference_evaluation:
        summary["reference_comparison"] = compare_with_reference(
            body_evaluation, args.reference_evaluation
        )

    output_path = args.output_dir / "evaluation.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        "output": str(output_path.resolve()),
        "feature_archive": str(features_path.resolve()),
        "records": len(records),
        "identities": len(set(identities)),
        "detection_rate": summary["detector"]["detection_rate"],
        "branches": {
            name: {
                key: metrics[key]
                for key in (
                    "top1_correct",
                    "top1_accuracy",
                    "top5_correct",
                    "top5_accuracy",
                    "auc",
                    "mean_reciprocal_rank",
                )
            }
            for name, metrics in summary["branches"].items()
        },
    }
    if "reference_comparison" in summary:
        concise["reference_comparison"] = summary["reference_comparison"]
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
