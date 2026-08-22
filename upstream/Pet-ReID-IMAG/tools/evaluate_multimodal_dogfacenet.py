#!/usr/bin/env python3
"""Evaluate a trained multimodal checkpoint with same/different dog pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _auc(positive: list[float], negative: list[float]) -> float:
    positive_array = np.asarray(positive, dtype=np.float64)[:, None]
    negative_array = np.asarray(negative, dtype=np.float64)[None, :]
    return float(
        ((positive_array > negative_array).mean())
        + 0.5 * ((positive_array == negative_array).mean())
    )


def _best_balanced_threshold(positive: list[float], negative: list[float]) -> dict:
    positive_array = np.asarray(positive, dtype=np.float64)
    negative_array = np.asarray(negative, dtype=np.float64)
    values = np.unique(np.concatenate((positive_array, negative_array)))
    thresholds = np.concatenate(
        (
            [values[0] - 1e-6],
            (values[:-1] + values[1:]) / 2.0,
            [values[-1] + 1e-6],
        )
    )
    best = None
    for threshold in thresholds:
        true_positive_rate = float((positive_array >= threshold).mean())
        true_negative_rate = float((negative_array < threshold).mean())
        balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
        candidate = (
            balanced_accuracy,
            true_positive_rate,
            true_negative_rate,
            float(threshold),
        )
        if best is None or candidate > best:
            best = candidate
    return {
        "threshold": best[3],
        "balanced_accuracy": best[0],
        "same_recall": best[1],
        "different_recall": best[2],
    }


def _font(size: int):
    candidates = (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _face_thumbnail(record: dict, size=(300, 220)) -> Image.Image:
    with Image.open(record["source_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image = image.resize(tuple(record["resized_size"]), Image.Resampling.LANCZOS)
    face = image.crop(tuple(record["face_roi_xyxy"]))
    face.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#11151c")
    canvas.paste(face, ((size[0] - face.width) // 2, (size[1] - face.height) // 2))
    return canvas


def _representative_pairs(pairs: list[dict]) -> list[dict]:
    selected = []
    for same in (True, False):
        group = sorted(
            (pair for pair in pairs if pair["same_identity"] is same),
            key=lambda pair: pair["fused"],
        )
        indices = (0, len(group) // 2, len(group) - 1)
        selected.extend(group[index] for index in indices)
    return selected


def _render_pairs(
    records: list[dict], pairs: list[dict], output_path: Path, *, title: str
) -> None:
    selected = _representative_pairs(pairs)
    width, row_height, header_height = 860, 280, 74
    sheet = Image.new("RGB", (width, header_height + row_height * len(selected)), "#0b0f15")
    draw = ImageDraw.Draw(sheet)
    title_font, row_font, small_font = _font(28), _font(21), _font(17)
    draw.text((24, 18), title, fill="white", font=title_font)
    for row, pair in enumerate(selected):
        top = header_height + row * row_height
        same = pair["same_identity"]
        color = "#35d07f" if same else "#ff6b6b"
        left = _face_thumbnail(records[pair["left_index"]])
        right = _face_thumbnail(records[pair["right_index"]])
        sheet.paste(left, (18, top + 48))
        sheet.paste(right, (542, top + 48))
        relation = "SAME" if same else "DIFFERENT"
        caption = (
            f"{relation}: {pair['left_identity']} vs {pair['right_identity']}   "
            f"fused={pair['fused']:.3f}"
        )
        draw.text((22, top + 12), caption, fill=color, font=row_font)
        draw.text(
            (329, top + 104),
            f"nose  {pair['nose']:.3f}\nface  {pair['face']:.3f}",
            fill="#d8dee9",
            font=small_font,
            spacing=9,
        )
        draw.rectangle((12, top + 4, width - 12, top + row_height - 5), outline=color, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="",
        help="optional trained fusion checkpoint; omit for frozen pretrained encoders",
    )
    parser.add_argument("--config-file", default="configs/multimodal_dogfacenet_train.yaml")
    parser.add_argument("--output-dir", default="logs/multimodal_dogfacenet_eval")
    parser.add_argument("--visualization", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-identities", type=int, default=0)
    parser.add_argument("--max-images-per-identity", type=int, default=0)
    parser.add_argument(
        "--gallery-images-per-identity",
        type=int,
        default=0,
        help="use the first N images as gallery and the remainder as held-out queries",
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    full_dataset = PreparedDogFaceNetDataset(args.manifest, training=False)
    selected_indices = []
    selected_identities = []
    identity_counts = {}
    for index, record in enumerate(full_dataset.records):
        identity = record["identity"].casefold()
        if identity not in identity_counts:
            if args.max_identities > 0 and len(selected_identities) >= args.max_identities:
                continue
            selected_identities.append(identity)
            identity_counts[identity] = 0
        if (
            args.max_images_per_identity > 0
            and identity_counts[identity] >= args.max_images_per_identity
        ):
            continue
        selected_indices.append(index)
        identity_counts[identity] += 1
    dataset = Subset(full_dataset, selected_indices)
    if len(dataset) < 2:
        raise RuntimeError("Evaluation selection needs at least two images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prepared_dogfacenet,
    )
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=args.checkpoint,
    )
    model.eval()

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    features, nose_features, face_features, probabilities = [], [], [], []
    fusion_weight_rows, joint_weight_rows, viewpoint_rows = [], [], []
    viewpoint_frontality_rows = []
    joint_mix_value = 0.0
    identities, source_paths = [], []
    for batch in loader:
        inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"targets", "identities", "source_paths"}
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output = model(**inputs)
        features.append(output["features"].float().cpu())
        nose_features.append(output["nose_features"].float().cpu())
        face_features.append(output["face_features"].float().cpu())
        fusion_weight_rows.append(output["fusion_weights"].float().cpu())
        if output["joint_weights"] is not None:
            joint_weight_rows.append(output["joint_weights"].float().cpu())
        viewpoint_rows.append(inputs["viewpoint_signals"].float().cpu())
        viewpoint_frontality_rows.append(output["viewpoint_frontality"].float().cpu())
        joint_mix_value = float(output["joint_mix"].detach())
        if "probabilities" in output:
            probabilities.append(output["probabilities"].float().cpu())
        identities.extend(identity.casefold() for identity in batch["identities"])
        source_paths.extend(batch["source_paths"])

    feature_sets = {
        "fused": torch.cat(features),
        "nose": torch.cat(nose_features),
        "face": torch.cat(face_features),
    }
    probability_matrix = torch.cat(probabilities) if probabilities else None
    fusion_weight_matrix = torch.cat(fusion_weight_rows)
    joint_weight_matrix = torch.cat(joint_weight_rows) if joint_weight_rows else None
    viewpoint_matrix = torch.cat(viewpoint_rows)
    viewpoint_frontality_vector = torch.cat(viewpoint_frontality_rows)
    label_to_identity = model.label_to_identity
    predictions = (
        [label_to_identity[int(label)] for label in probability_matrix.argmax(dim=1)]
        if probability_matrix is not None
        else [None] * len(identities)
    )
    records = []
    for index, (identity, source_path, prediction) in enumerate(
        zip(identities, source_paths, predictions)
    ):
        records.append(
            {
                "index": index,
                "identity": identity,
                "source_path": source_path,
                "predicted_identity": prediction,
                "predicted_probability": (
                    float(probability_matrix[index].max())
                    if probability_matrix is not None
                    else None
                ),
                "viewpoint_signals": viewpoint_matrix[index].tolist(),
                "viewpoint_frontality": float(viewpoint_frontality_vector[index]),
                "fusion_weights": fusion_weight_matrix[index].tolist(),
                "joint_weights": (
                    joint_weight_matrix[index].tolist()
                    if joint_weight_matrix is not None
                    else None
                ),
            }
        )

    similarity_matrices = {
        name: values @ values.T for name, values in feature_sets.items()
    }
    pairs = []
    for left in range(len(dataset)):
        for right in range(left + 1, len(dataset)):
            pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "left_identity": identities[left],
                    "right_identity": identities[right],
                    "same_identity": identities[left] == identities[right],
                    **{
                        name: float(matrix[left, right])
                        for name, matrix in similarity_matrices.items()
                    },
                }
            )

    branch_metrics = {}
    for name in feature_sets:
        positive = [pair[name] for pair in pairs if pair["same_identity"]]
        negative = [pair[name] for pair in pairs if not pair["same_identity"]]
        branch_metrics[name] = {
            "same": _summary(positive),
            "different": _summary(negative),
            "auc": _auc(positive, negative),
            "pilot_best_threshold": _best_balanced_threshold(positive, negative),
        }

    fused_matrix = similarity_matrices["fused"].clone()
    fused_matrix.fill_diagonal_(-math.inf)
    nearest = fused_matrix.argmax(dim=1).tolist()
    rank1_correct = sum(
        identities[index] == identities[neighbor]
        for index, neighbor in enumerate(nearest)
    )
    classifier_accuracy = None
    if probability_matrix is not None:
        classifier_correct = sum(
            prediction == identity for prediction, identity in zip(predictions, identities)
        )
        classifier_accuracy = classifier_correct / len(records)
    gallery_query = None
    if args.gallery_images_per_identity > 0:
        gallery_indices, query_indices = [], []
        seen_counts = {}
        for index, identity in enumerate(identities):
            count = seen_counts.get(identity, 0)
            if count < args.gallery_images_per_identity:
                gallery_indices.append(index)
            else:
                query_indices.append(index)
            seen_counts[identity] = count + 1
        gallery_identities = {identities[index] for index in gallery_indices}
        query_indices = [
            index for index in query_indices if identities[index] in gallery_identities
        ]
        if not query_indices:
            raise RuntimeError("Gallery/query split produced no held-out queries")
        query_features = feature_sets["fused"].index_select(
            0, torch.tensor(query_indices)
        )
        gallery_features = feature_sets["fused"].index_select(
            0, torch.tensor(gallery_indices)
        )
        gallery_similarities = query_features @ gallery_features.T
        nearest_gallery = gallery_similarities.argmax(dim=1).tolist()
        query_results = []
        for row, (query_index, nearest_column) in enumerate(
            zip(query_indices, nearest_gallery)
        ):
            gallery_index = gallery_indices[nearest_column]
            query_results.append(
                {
                    "query_index": query_index,
                    "query_identity": identities[query_index],
                    "matched_gallery_index": gallery_index,
                    "matched_identity": identities[gallery_index],
                    "score": float(gallery_similarities[row, nearest_column]),
                    "correct": identities[query_index] == identities[gallery_index],
                }
            )
        gallery_query = {
            "gallery_images_per_identity": args.gallery_images_per_identity,
            "gallery_records": len(gallery_indices),
            "query_records": len(query_indices),
            "rank1_accuracy": sum(row["correct"] for row in query_results)
            / len(query_results),
            "queries": query_results,
        }
    summary = {
        "model_source": "trained_fusion_checkpoint" if args.checkpoint else "frozen_pretrained",
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "manifest": str(Path(args.manifest).resolve()),
        "records": len(records),
        "identities": len(set(identities)),
        "same_pairs": sum(pair["same_identity"] for pair in pairs),
        "different_pairs": sum(not pair["same_identity"] for pair in pairs),
        "rank1_leave_one_out_accuracy": rank1_correct / len(records),
        "closed_set_classifier_accuracy": classifier_accuracy,
        "joint_mix": joint_mix_value,
        "viewpoint_gate": None,
        "gallery_query": gallery_query,
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "branches": branch_metrics,
        "records_detail": records,
    }
    if joint_weight_matrix is not None:
        pose_magnitude = viewpoint_matrix[:, :3].norm(dim=1)
        nose_weight = joint_weight_matrix[:, 0]
        correlation = None
        if float(pose_magnitude.std()) > 1e-8 and float(nose_weight.std()) > 1e-8:
            correlation = float(torch.corrcoef(torch.stack((pose_magnitude, nose_weight)))[0, 1])
        summary["viewpoint_gate"] = {
            "nose_weight_min": float(nose_weight.min()),
            "nose_weight_mean": float(nose_weight.mean()),
            "nose_weight_max": float(nose_weight.max()),
            "pose_magnitude_nose_weight_correlation": correlation,
        }
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_root / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    if args.visualization:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest_by_path = {record["source_path"]: record for record in manifest["records"]}
        visual_records = [manifest_by_path[path] for path in source_paths]
        _render_pairs(
            visual_records,
            pairs,
            Path(args.visualization),
            title=(
                "Trained fusion: cosine pair sanity check"
                if args.checkpoint
                else "Frozen pretrained fusion: cosine pair sanity check"
            ),
        )
        summary["visualization"] = str(Path(args.visualization).resolve())
        (output_root / "evaluation.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
