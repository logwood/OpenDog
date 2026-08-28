#!/usr/bin/env python3
"""Validate labeled images or identify arbitrary new images with a gallery model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import (
    build_pipeline,
    collect_images,
    encode_primary,
    load_gallery_model,
    normalized_array,
    sha256_file,
)
from pet_id.workspace_paths import EVALUATIONS_ROOT, resolve_legacy_path


VARIANTS = (
    "selected_fused",
    "frozen_fused",
    "selected_nose",
    "selected_face",
    "frozen_nose",
    "frozen_face",
)


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    pos = np.asarray(positive, dtype=np.float64)[:, None]
    neg = np.asarray(negative, dtype=np.float64)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def load_queries(args) -> list[dict]:
    rows = []
    if args.manifest:
        manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
        rows.extend(
            {
                "path": resolve_legacy_path(record["library_path"]),
                "expected_identity": record["identity"],
                "source": "manifest_validation",
            }
            for record in manifest["records"]
            if record["split"] == "validation"
        )
    for path in collect_images(args.images) if args.images else []:
        rows.append(
            {
                "path": path,
                "expected_identity": path.parent.name.casefold() if args.labels_from_parent else None,
                "source": "input",
            }
        )
    unique = []
    seen = set()
    for row in rows:
        key = str(row["path"]).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if not unique:
        raise ValueError("provide --manifest and/or image files/directories")
    return unique


def encode_queries(
    queries: list[dict],
    config: Path,
    checkpoint: Path | None,
    device: str,
    *,
    backend: str = "pytorch",
    onnx_model: Path | None = None,
    onnx_provider: str = "cuda",
    onnx_warmup_batches: tuple[int, ...] = (),
):
    pipeline = build_pipeline(
        config,
        checkpoint,
        device,
        backend=backend,
        onnx_model=onnx_model,
        onnx_provider=onnx_provider,
        onnx_warmup_batches=onnx_warmup_batches,
    )
    backend_info = (
        pipeline.identity_model.backend_info()
        if hasattr(pipeline.identity_model, "backend_info")
        else {"backend": "pytorch", "device": str(pipeline.device)}
    )
    encoded = []
    try:
        for index, row in enumerate(queries, 1):
            descriptor, inference = encode_primary(pipeline, row["path"])
            encoded.append(
                {
                    "fused": normalized_array(descriptor.fused_feature),
                    "nose": normalized_array(descriptor.nose_feature),
                    "face": normalized_array(descriptor.face_feature),
                    "inference": inference,
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "selected" if checkpoint or backend == "onnx" else "frozen",
                        "backend": backend_info,
                        "query": index,
                        "total": len(queries),
                        "path": str(row["path"]),
                        "detections": inference["detections"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return encoded, backend_info


def feature_rows(
    selected: list[dict], frozen: list[dict] | None = None
) -> dict[str, np.ndarray]:
    source = {
        "selected_fused": (selected, "fused"),
        "selected_nose": (selected, "nose"),
        "selected_face": (selected, "face"),
    }
    if frozen is not None:
        source.update(
            frozen_fused=(frozen, "fused"),
            frozen_nose=(frozen, "nose"),
            frozen_face=(frozen, "face"),
        )
    return {
        name: np.stack([row[feature] for row in rows]).astype(np.float32)
        for name, (rows, feature) in source.items()
    }


def evaluate_variant(
    name: str,
    query_features: np.ndarray,
    queries: list[dict],
    identities: list[str],
    arrays: dict[str, np.ndarray],
) -> dict:
    prototypes = arrays[f"{name}_prototypes"]
    references = arrays[f"{name}_references"]
    reference_identity_indices = arrays["reference_identity_indices"].astype(np.int64)
    prototype_scores = query_features @ prototypes.T
    reference_scores = query_features @ references.T
    details = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    correct_count = 0
    labeled_count = 0
    for index, query in enumerate(queries):
        scores = prototype_scores[index]
        order = np.argsort(-scores)
        top_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else top_index
        expected = query["expected_identity"]
        expected_index = identities.index(expected) if expected in identities else None
        if expected_index is not None:
            impostor_indices = [item for item in range(len(identities)) if item != expected_index]
            strongest_impostor_index = max(impostor_indices, key=lambda item: float(scores[item]))
            correct_score = float(scores[expected_index])
            impostor_score = float(scores[strongest_impostor_index])
            margin = correct_score - impostor_score
            correct = top_index == expected_index
            correct_count += int(correct)
            labeled_count += 1
            for ref_index, ref_identity_index in enumerate(reference_identity_indices):
                score = float(reference_scores[index, ref_index])
                if int(ref_identity_index) == expected_index:
                    positive_scores.append(score)
                else:
                    negative_scores.append(score)
        else:
            strongest_impostor_index = second_index
            correct_score = None
            impostor_score = float(scores[second_index])
            margin = float(scores[top_index] - scores[second_index])
            correct = None
        margins.append(margin)
        details.append(
            {
                "query_index": index,
                "path": str(query["path"]),
                "expected_identity": expected,
                "predicted_identity": identities[top_index],
                "top1_score": float(scores[top_index]),
                "runner_up_identity": identities[second_index],
                "runner_up_score": float(scores[second_index]),
                "correct_score": correct_score,
                "strongest_impostor_identity": identities[strongest_impostor_index],
                "strongest_impostor_score": impostor_score,
                "margin": margin,
                "correct": correct,
            }
        )
    return {
        "variant": name,
        "queries": len(queries),
        "labeled_queries": labeled_count,
        "rank1_accuracy": correct_count / labeled_count if labeled_count else None,
        "pair_auc": auc(positive_scores, negative_scores),
        "same_reference_mean": float(np.mean(positive_scores)) if positive_scores else None,
        "different_reference_mean": float(np.mean(negative_scores)) if negative_scores else None,
        "minimum_margin": min(margins) if margins else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "details": details,
    }


def comparison(selected: dict, baseline: dict) -> dict:
    metrics = ("rank1_accuracy", "pair_auc", "minimum_margin", "mean_margin")
    delta = {
        metric: (
            selected[metric] - baseline[metric]
            if selected[metric] is not None and baseline[metric] is not None
            else None
        )
        for metric in metrics
    }
    selected_key = tuple(selected[metric] if selected[metric] is not None else -math.inf for metric in metrics)
    baseline_key = tuple(baseline[metric] if baseline[metric] is not None else -math.inf for metric in metrics)
    return {
        "rule": "lexicographic: rank1, pair AUC, minimum margin, mean margin",
        "selected_fused_better": selected_key > baseline_key,
        "tie": selected_key == baseline_key,
        "delta": delta,
    }


def font(size: int, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size=size)


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#0b1018")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def render_results(
    output: Path,
    queries: list[dict],
    results: dict[str, dict],
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    model: dict,
) -> None:
    selected_details = results["selected_fused"]["details"]
    reference_scores = features["selected_fused"] @ arrays["selected_fused_references"].T
    references = model["references"]
    cols = 2
    tile_w, tile_h = 880, 360
    rows = math.ceil(len(queries) / cols)
    image = Image.new("RGB", (1800, 120 + rows * 375 + 55), "#090d14")
    draw = ImageDraw.Draw(image)
    draw.text((42, 25), "本地新图：图库 Top-1 验证", font=font(38, True), fill="#f5f7fa")
    subtitle = "每格：验证新图 → 联合模型 Top-1 图库参考；列出可用分支余量"
    if "frozen_fused" in results:
        subtitle += "和冻结基线"
    draw.text(
        (44, 78), subtitle, font=font(19), fill="#aab6c5"
    )
    for index, query in enumerate(queries):
        x = 20 + (index % cols) * 890
        y = 120 + (index // cols) * 375
        draw.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=18, fill="#111923", outline="#2a394b", width=2)
        detail = selected_details[index]
        ref_order = np.argsort(-reference_scores[index])
        top_ref_index = int(ref_order[0])
        query_image = thumbnail(query["path"], (250, 230))
        reference_image = thumbnail(
            resolve_legacy_path(references[top_ref_index]["path"]), (250, 230)
        )
        image.paste(query_image, (x + 18, y + 58))
        image.paste(reference_image, (x + 315, y + 58))
        color = "#49d17d" if detail["correct"] is not False else "#ff6b6b"
        draw.text((x + 18, y + 14), query["path"].name, font=font(18, True), fill="#f5f7fa")
        draw.text((x + 143, y + 302), "验证新图", font=font(16, True), fill="#53a7ff", anchor="ma")
        draw.text((x + 440, y + 302), f"Top-1：{detail['predicted_identity']}", font=font(16, True), fill=color, anchor="ma")
        draw.text((x + 282, y + 135), "→", font=font(30, True), fill="#ffd166", anchor="mm")
        value_x = x + 590
        expected = detail["expected_identity"] or "未知"
        draw.text((value_x, y + 62), f"期望：{expected}", font=font(18, True), fill="#f5f7fa")
        draw.text((value_x, y + 102), f"联合 margin {detail['margin']:+.3f}", font=font(18, True), fill=color)
        comparison_variants = tuple(
            variant
            for variant in ("frozen_fused", "selected_nose", "selected_face")
            if variant in results
        )
        labels = {
            "frozen_fused": "冻结融合",
            "selected_nose": "鼻部分支",
            "selected_face": "脸部分支",
        }
        for offset, variant in enumerate(comparison_variants):
            row = results[variant]["details"][index]
            label = labels[variant]
            draw.text((value_x, y + 145 + offset * 39), f"{label} {row['margin']:+.3f}", font=font(16), fill="#aab6c5")
        draw.text((value_x, y + 278), f"联合 Top-1 {detail['top1_score']:.3f}", font=font(17, True), fill=color)
    draw.text((30, image.height - 38), "验证图未用于建图库原型或选择神经 checkpoint；两只犬外观差异较大，结果仅属外部小样本验证。", font=font(17), fill="#aab6c5")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*")
    parser.add_argument("--gallery-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels-from-parent", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=EVALUATIONS_ROOT / "local_pet_gallery_validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("pytorch", "onnx"))
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1,4,8")
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="run only selected fused/nose/face variants and skip frozen PyTorch",
    )
    parser.add_argument("--no-visualization", action="store_true")
    args = parser.parse_args()
    from pet_id.onnx_runtime import parse_warmup_batches

    model_path = args.gallery_model.resolve()
    model, arrays = load_gallery_model(model_path)
    recorded_backend = model.get("selected_backend", {})
    backend = args.backend
    if backend is None:
        backend = (
            "onnx"
            if recorded_backend.get("backend") == "onnxruntime"
            else "pytorch"
        )
    recorded_onnx_model = recorded_backend.get("model")
    onnx_model = (
        resolve_legacy_path(args.onnx_model)
        if args.onnx_model is not None
        else resolve_legacy_path(
            recorded_onnx_model
            or "models/selected/dogfacenet_joint800_v1/onnx/pet_embedding.onnx"
        )
    )
    recorded_onnx_hash = recorded_backend.get("model_sha256")
    if backend == "onnx" and recorded_onnx_hash:
        actual_onnx_hash = sha256_file(onnx_model)
        if actual_onnx_hash.casefold() != str(recorded_onnx_hash).casefold():
            raise ValueError(
                "The requested ONNX model does not match the model used to build "
                f"this gallery: expected {recorded_onnx_hash}, got {actual_onnx_hash}"
            )
    warmup_batches = parse_warmup_batches(args.onnx_warmup_batches)
    identities = list(model["identities"])
    queries = load_queries(args)
    config = resolve_legacy_path(model["config_file"])
    checkpoint = resolve_legacy_path(model["selected_checkpoint"])
    selected, selected_backend = encode_queries(
        queries,
        config,
        checkpoint,
        args.device,
        backend=backend,
        onnx_model=onnx_model,
        onnx_provider=args.onnx_provider,
        onnx_warmup_batches=warmup_batches,
    )
    frozen = None
    frozen_backend = None
    frozen_gallery_available = all(
        f"{name}_prototypes" in arrays
        for name in ("frozen_fused", "frozen_nose", "frozen_face")
    )
    if not args.production_only and frozen_gallery_available:
        frozen, frozen_backend = encode_queries(
            queries, config, None, args.device, backend="pytorch"
        )
    query_features = feature_rows(selected, frozen)
    results = {
        name: evaluate_variant(name, query_features[name], queries, identities, arrays)
        for name in VARIANTS
        if name in query_features and f"{name}_prototypes" in arrays
    }
    compare = (
        comparison(results["selected_fused"], results["frozen_fused"])
        if "frozen_fused" in results
        else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization = None
    if not args.no_visualization:
        visualization = output_dir / "top1_results.png"
        render_results(visualization, queries, results, query_features, arrays, model)
    payload = {
        "schema_version": 1,
        "gallery_model": str(model_path),
        "queries": [
            ({
                "path": str(row["path"]),
                "expected_identity": row["expected_identity"],
                "source": row["source"],
                "selected_inference": selected[index]["inference"],
            } | (
                {"frozen_inference": frozen[index]["inference"]}
                if frozen is not None
                else {}
            ))
            for index, row in enumerate(queries)
        ],
        "selected_backend": selected_backend,
        "frozen_backend": frozen_backend,
        "production_only": args.production_only or not frozen_gallery_available,
        "results": results,
        "selected_vs_frozen": compare,
        "visualization": str(visualization) if visualization else None,
        "limitations": [
            "only two new identities are present",
            "images appear to come from one capture environment per identity",
            "this is an external held-out small set, not a new blind benchmark",
        ],
    }
    result_path = output_dir / "validation.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "queries.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["variant", "path", "expected_identity", "predicted_identity", "top1_score", "runner_up_identity", "runner_up_score", "margin", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant, result in results.items():
            for row in result["details"]:
                writer.writerow({name: row.get(name) for name in fieldnames} | {"variant": variant})
    print(
        json.dumps(
            {
                "selected_backend": selected_backend,
                "validation": str(result_path),
                "csv": str(csv_path),
                "visualization": str(visualization) if visualization else None,
                "metrics": {
                    name: {key: value for key, value in result.items() if key != "details"}
                    for name, result in results.items()
                },
                "selected_vs_frozen": compare,
                "predictions": results["selected_fused"]["details"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
