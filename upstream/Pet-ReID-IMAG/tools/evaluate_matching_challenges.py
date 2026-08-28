#!/usr/bin/env python3
"""Stress-test pet matching with sparse, polluted, degraded, and mixed galleries."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import mimetypes
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import (  # noqa: E402
    build_pipeline,
    encode_primary,
    load_gallery_model,
    normalized_array,
)
from pet_id.gallery_service import (  # noqa: E402
    EncodedPetImage,
    EnrollmentRecord,
    PetGalleryStore,
    UploadPayload,
    validate_upload,
)
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402


VARIANTS = (
    "jpeg_q15",
    "lowres_128",
    "dark_25",
    "rotate_25",
    "center_occlusion_40",
    "hard_combo",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resize_long_side(image: Image.Image, limit: int) -> Image.Image:
    if max(image.size) <= limit:
        return image.copy()
    scale = limit / max(image.size)
    size = tuple(max(1, round(value * scale)) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def create_variant(source: Path, destination: Path, variant: str) -> None:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if variant == "jpeg_q15":
        image = resize_long_side(image, 1280)
        quality = 15
    elif variant == "lowres_128":
        image = resize_long_side(image, 128)
        quality = 80
    elif variant == "dark_25":
        image = resize_long_side(image, 1280)
        image = ImageEnhance.Brightness(image).enhance(0.25)
        quality = 90
    elif variant == "rotate_25":
        image = resize_long_side(image, 1280)
        image = image.rotate(
            25,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(32, 32, 32),
        )
        quality = 90
    elif variant == "center_occlusion_40":
        image = resize_long_side(image, 1280)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        draw.rectangle(
            (0.2 * width, 0.3 * height, 0.8 * width, 0.7 * height),
            fill=(20, 20, 20),
        )
        quality = 90
    elif variant == "hard_combo":
        image = resize_long_side(image, 192)
        image = ImageEnhance.Brightness(image).enhance(0.40)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        quality = 20
    else:
        raise ValueError(f"unsupported variant: {variant}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=quality, optimize=True)


def fit_panel(source: Path, width: int, height: int) -> Image.Image:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (28, 28, 28))
    panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return panel


def create_composite(
    left: Path,
    right: Path,
    destination: Path,
    left_width: int,
) -> None:
    width, height = 1200, 800
    right_width = width - left_width
    canvas = Image.new("RGB", (width, height), (28, 28, 28))
    canvas.paste(fit_panel(left, left_width, height), (0, 0))
    canvas.paste(fit_panel(right, right_width, height), (left_width, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=92, optimize=True)


def encode_sample(pipeline, sample: dict[str, Any], index: int, total: int) -> None:
    started = time.perf_counter()
    try:
        descriptor, inference = encode_primary(pipeline, sample["path"])
        sample["feature"] = normalized_array(descriptor.fused_feature)
        sample["inference"] = inference
        sample["error"] = None
    except Exception as error:  # stress tests must retain detector failures
        sample["feature"] = None
        sample["inference"] = None
        sample["error"] = f"{type(error).__name__}: {error}"
    sample["latency_ms"] = (time.perf_counter() - started) * 1000.0
    print(
        json.dumps(
            {
                "stage": "encode",
                "index": index,
                "total": total,
                "key": sample["key"],
                "variant": sample["variant"],
                "latency_ms": round(sample["latency_ms"], 2),
                "detections": (
                    sample["inference"].get("detections")
                    if sample["inference"]
                    else None
                ),
                "error": sample["error"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def load_inputs(manifest_path: Path, model_path: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model, arrays = load_gallery_model(model_path)
    reference_counts: dict[str, int] = defaultdict(int)
    references: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(model["references"]):
        identity = row["identity"]
        reference_counts[identity] += 1
        key = f"{identity}/g{reference_counts[identity]}"
        references[key] = {
            "key": key,
            "identity": identity,
            "path": Path(row["path"]).resolve(),
            "fused": arrays["selected_fused_references"][index],
            "nose": arrays["selected_nose_references"][index],
            "face": arrays["selected_face_references"][index],
            "inference": row["selected_inference"],
        }
    validation = {}
    for row in manifest["records"]:
        if row["split"] != "validation":
            continue
        path = Path(row["library_path"]).resolve()
        key = f"{row['identity']}/validation/{path.name}"
        validation[key] = {
            "key": key,
            "identity": row["identity"],
            "path": path,
            "variant": "original",
            "source_key": key,
        }
    return model, references, validation


def upload_record(reference: dict[str, Any]) -> EnrollmentRecord:
    path = reference["path"]
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    upload = validate_upload(
        UploadPayload(path.name, content_type, path.read_bytes()),
        maximum_bytes=100 * 1024 * 1024,
        maximum_pixels=30_000_000,
    )
    return EnrollmentRecord(
        upload=upload,
        encoded=EncodedPetImage(
            fused=np.asarray(reference["fused"], dtype=np.float32),
            nose=np.asarray(reference["nose"], dtype=np.float32),
            face=np.asarray(reference["face"], dtype=np.float32),
            metadata=dict(reference["inference"]),
        ),
    )


def clear_gallery(store: PetGalleryStore) -> None:
    for pet in store.list_pets():
        store.delete_pet(pet["pet_id"])


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["status"] == "scored"]
    known = [row for row in rows if row["expected_identity"] is not None]
    known_scored = [row for row in known if row["status"] == "scored"]
    unknown = [row for row in rows if row["unknown"]]
    unknown_scored = [row for row in unknown if row["status"] == "scored"]
    correct = sum(row["correct"] is True for row in known_scored)
    margins = [row["margin"] for row in scored if row["margin"] is not None]
    scores = [row["top1_score"] for row in scored]
    return {
        "attempts": len(rows),
        "scored": len(scored),
        "encoding_errors": len(rows) - len(scored),
        "known_attempts": len(known),
        "known_correct": correct,
        "rank1_scored": correct / len(known_scored) if known_scored else None,
        "end_to_end_accuracy": correct / len(known) if known else None,
        "unknown_attempts": len(unknown),
        "unknown_false_accepts_default": len(unknown_scored),
        "unknown_false_accept_rate_default": (
            len(unknown_scored) / len(unknown) if unknown else None
        ),
        "top1_mean": float(np.mean(scores)) if scores else None,
        "top1_min": min(scores) if scores else None,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_min": min(margins) if margins else None,
    }


def evaluate_scenario(
    store: PetGalleryStore,
    references: dict[str, dict[str, Any]],
    name: str,
    category: str,
    assignments: dict[str, list[str]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    clear_gallery(store)
    enrollment = []
    try:
        for pet_id, keys in assignments.items():
            result = store.enroll(
                pet_id,
                pet_id,
                [upload_record(references[key]) for key in keys],
            )
            enrollment.append(
                {
                    "pet_id": pet_id,
                    "reference_keys": keys,
                    "added": len(result["added_image_ids"]),
                }
            )
        prototypes = store.prototypes()
        rows = []
        for query in queries:
            sample = query["sample"]
            expected = query.get("expected_identity", sample.get("identity"))
            unknown = bool(query.get("unknown", False))
            base = {
                "sample_key": sample["key"],
                "path": str(sample["path"]),
                "variant": sample["variant"],
                "actual_identity": sample.get("identity"),
                "expected_identity": expected,
                "unknown": unknown,
                "latency_ms": sample.get("latency_ms"),
                "inference": sample.get("inference"),
            }
            if sample.get("feature") is None:
                rows.append(
                    base
                    | {
                        "status": "encoding_error",
                        "error": sample.get("error"),
                        "predicted_identity": None,
                        "top1_score": None,
                        "runner_up_identity": None,
                        "runner_up_score": None,
                        "margin": None,
                        "correct": False if expected is not None else None,
                        "accepted_default": False,
                    }
                )
                continue
            scores = np.asarray(
                [float(sample["feature"] @ item["prototype"]) for item in prototypes],
                dtype=np.float32,
            )
            order = np.argsort(-scores)
            best_index = int(order[0])
            second_index = int(order[1]) if len(order) > 1 else None
            predicted = prototypes[best_index]["pet_id"]
            runner_up_score = (
                float(scores[second_index]) if second_index is not None else None
            )
            top1_score = float(scores[best_index])
            rows.append(
                base
                | {
                    "status": "scored",
                    "error": None,
                    "predicted_identity": predicted,
                    "top1_score": top1_score,
                    "runner_up_identity": (
                        prototypes[second_index]["pet_id"]
                        if second_index is not None
                        else None
                    ),
                    "runner_up_score": runner_up_score,
                    "margin": (
                        top1_score - runner_up_score
                        if runner_up_score is not None
                        else None
                    ),
                    "correct": predicted == expected if expected is not None else None,
                    "accepted_default": True,
                }
            )
        return {
            "name": name,
            "category": category,
            "assignments": assignments,
            "enrollment": enrollment,
            "summary": summarize_rows(rows),
            "rows": rows,
        }
    finally:
        clear_gallery(store)


def pooled_summary(scenarios: list[dict], category: str) -> dict[str, Any]:
    rows = [
        row
        for scenario in scenarios
        if scenario["category"] == category
        for row in scenario["rows"]
    ]
    return summarize_rows(rows)


def threshold_grid(clean_rows: list[dict], unknown_rows: list[dict]) -> list[dict]:
    results = []
    for threshold in np.arange(0.0, 0.651, 0.05):
        for minimum_margin in np.arange(0.0, 0.301, 0.05):
            def accepted(row: dict[str, Any]) -> bool:
                if row["status"] != "scored" or row["top1_score"] < threshold:
                    return False
                return row["margin"] is None or row["margin"] >= minimum_margin

            true_accept = sum(accepted(row) for row in clean_rows) / len(clean_rows)
            false_accept = sum(accepted(row) for row in unknown_rows) / len(unknown_rows)
            results.append(
                {
                    "match_threshold": round(float(threshold), 2),
                    "minimum_margin": round(float(minimum_margin), 2),
                    "clean_true_accept_rate": true_accept,
                    "unknown_false_accept_rate": false_accept,
                    "diagnostic_balanced_accuracy": (
                        true_accept + (1.0 - false_accept)
                    )
                    / 2.0,
                }
            )
    return sorted(
        results,
        key=lambda row: (
            -row["diagnostic_balanced_accuracy"],
            -row["clean_true_accept_rate"],
            row["unknown_false_accept_rate"],
            row["match_threshold"],
            row["minimum_margin"],
        ),
    )


def write_csv(path: Path, scenarios: list[dict]) -> None:
    fields = [
        "scenario",
        "category",
        "sample_key",
        "variant",
        "actual_identity",
        "expected_identity",
        "unknown",
        "status",
        "predicted_identity",
        "top1_score",
        "runner_up_identity",
        "runner_up_score",
        "margin",
        "correct",
        "latency_ms",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in scenarios:
            for row in scenario["rows"]:
                writer.writerow(
                    {
                        field: (
                            scenario["name"]
                            if field == "scenario"
                            else scenario["category"]
                            if field == "category"
                            else row.get(field)
                        )
                        for field in fields
                    }
                )


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Pet ReID 困难组合测试",
        "",
        f"- 时间：{report['finished_at']}",
        f"- 后端：{report['backend']['provider']}",
        f"- 临时图库：`{report['temporary_gallery']['root']}`",
        "",
        "| 场景 | 类别 | 尝试 | 编码失败 | Rank-1 / E2E | 最小 margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        summary = scenario["summary"]
        accuracy = summary["end_to_end_accuracy"]
        accuracy_text = "—" if accuracy is None else f"{accuracy:.1%}"
        margin = summary["margin_min"]
        margin_text = "—" if margin is None else f"{margin:.4f}"
        lines.append(
            f"| {scenario['name']} | {scenario['category']} | "
            f"{summary['attempts']} | {summary['encoding_errors']} | "
            f"{accuracy_text} | {margin_text} |"
        )
    best = report["threshold_diagnostic"]["best_grid_point"]
    lines.extend(
        [
            "",
            "## 拒识诊断（仅限当前 2 只宠物小样本）",
            "",
            f"- 默认闭集未知宠物误接纳率：{report['aggregates']['open_set']['unknown_false_accept_rate_default']:.1%}",
            f"- 网格最佳点：score ≥ {best['match_threshold']:.2f}，margin ≥ {best['minimum_margin']:.2f}",
            f"- 此点干净样本接纳率：{best['clean_true_accept_rate']:.1%}",
            f"- 此点未知样本误接纳率：{best['unknown_false_accept_rate']:.1%}",
            "- 这不是生产阈值；只有两个身份，且未知样本由另一只本地宠物模拟。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/local_pet_gallery_v1/dataset_manifest.json"),
    )
    parser.add_argument(
        "--gallery-model",
        type=Path,
        default=Path("models/local_pet_gallery_joint800_onnx_v1/gallery_model.json"),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("models/dogfacenet_joint800_v1/config.yaml"),
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("models/dogfacenet_joint800_v1/onnx/pet_embedding.onnx"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1,4,8")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logs/pet_matching_challenge_v1")
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("models/pet_api_gallery_challenge_v1"),
    )
    args = parser.parse_args()

    started_at = utc_now()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = output_dir / "fixtures"
    storage_dir = args.storage_dir.expanduser().resolve()

    model, references, validation = load_inputs(
        args.manifest.expanduser().resolve(), args.gallery_model.expanduser().resolve()
    )
    if set(model["identities"]) != {"local-1", "local-2"}:
        raise ValueError("this challenge matrix expects local-1 and local-2")

    samples = list(validation.values())
    for original in list(validation.values()):
        for variant in VARIANTS:
            destination = fixture_dir / variant / original["identity"] / original["path"].name
            create_variant(original["path"], destination, variant)
            samples.append(
                {
                    "key": f"{original['key']}::{variant}",
                    "identity": original["identity"],
                    "path": destination,
                    "variant": variant,
                    "source_key": original["key"],
                }
            )

    hardest = {}
    for identity in model["identities"]:
        identity_samples = [row for row in validation.values() if row["identity"] == identity]
        hardest[identity] = identity_samples[0]
    composite_specs = (
        ("balanced", 600, None),
        ("local1_large", 840, "local-1"),
        ("local2_large", 360, "local-2"),
    )
    composites = []
    for name, left_width, expected in composite_specs:
        path = fixture_dir / "composites" / f"{name}.jpg"
        create_composite(
            hardest["local-1"]["path"],
            hardest["local-2"]["path"],
            path,
            left_width,
        )
        sample = {
            "key": f"composite/{name}",
            "identity": None,
            "path": path,
            "variant": "two_pet_composite",
            "source_key": None,
            "expected_panel_identity": expected,
        }
        composites.append(sample)
        samples.append(sample)

    pipeline = build_pipeline(
        args.config_file.expanduser().resolve(),
        None,
        args.device,
        backend="onnx",
        onnx_model=args.onnx_model.expanduser().resolve(),
        onnx_provider=args.onnx_provider,
        onnx_warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
    )
    backend = pipeline.identity_model.backend_info()
    try:
        for index, sample in enumerate(samples, 1):
            encode_sample(pipeline, sample, index, len(samples))
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for reference in references.values():
        reference["feature"] = reference["fused"]
        reference["variant"] = "original_gallery"
        reference["latency_ms"] = None
        reference["error"] = None

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_variant[sample["variant"]].append(sample)

    store = PetGalleryStore(storage_dir)
    store.bind_model(backend["model_sha256"], backend)
    clear_gallery(store)
    clean = {
        "local-1": ["local-1/g1", "local-1/g2"],
        "local-2": ["local-2/g1", "local-2/g2"],
    }
    original_queries = [{"sample": row} for row in validation.values()]
    scenarios = [
        evaluate_scenario(
            store, references, "clean_multi_ref", "baseline", clean, original_queries
        )
    ]

    for left_ref in ("g1", "g2"):
        for right_ref in ("g1", "g2"):
            assignments = {
                "local-1": [f"local-1/{left_ref}"],
                "local-2": [f"local-2/{right_ref}"],
            }
            held_out = [
                references[f"local-1/{'g2' if left_ref == 'g1' else 'g1'}"],
                references[f"local-2/{'g2' if right_ref == 'g1' else 'g1'}"],
            ]
            queries = [{"sample": row} for row in held_out]
            queries.extend(original_queries)
            scenarios.append(
                evaluate_scenario(
                    store,
                    references,
                    f"single_ref_{left_ref}_{right_ref}",
                    "single_reference",
                    assignments,
                    queries,
                )
            )

    pollution_specs = (
        (
            "polluted_local1",
            {"local-1": ["local-1/g1", "local-2/g1"], "local-2": ["local-2/g2"]},
        ),
        (
            "polluted_local2",
            {"local-1": ["local-1/g2"], "local-2": ["local-2/g1", "local-1/g1"]},
        ),
        (
            "cross_contamination_50",
            {
                "local-1": ["local-1/g1", "local-2/g1"],
                "local-2": ["local-1/g2", "local-2/g2"],
            },
        ),
    )
    for name, assignments in pollution_specs:
        scenarios.append(
            evaluate_scenario(
                store,
                references,
                name,
                "gallery_pollution",
                assignments,
                original_queries,
            )
        )

    for identity, unknown_identity in (("local-1", "local-2"), ("local-2", "local-1")):
        unknown_samples = [
            row
            for row in [*references.values(), *validation.values()]
            if row["identity"] == unknown_identity
        ]
        scenarios.append(
            evaluate_scenario(
                store,
                references,
                f"open_set_{unknown_identity}_against_{identity}",
                "open_set",
                {identity: [f"{identity}/g1", f"{identity}/g2"]},
                [
                    {"sample": row, "expected_identity": None, "unknown": True}
                    for row in unknown_samples
                ],
            )
        )

    degraded_queries = [
        {"sample": row}
        for variant in VARIANTS
        for row in by_variant[variant]
    ]
    scenarios.append(
        evaluate_scenario(
            store,
            references,
            "clean_gallery_degraded_queries",
            "degraded_query",
            clean,
            degraded_queries,
        )
    )
    hard_queries = [{"sample": row} for row in by_variant["hard_combo"]]
    scenarios.append(
        evaluate_scenario(
            store,
            references,
            "polluted_gallery_plus_hard_query",
            "combined_fault",
            pollution_specs[-1][1],
            hard_queries,
        )
    )
    scenarios.append(
        evaluate_scenario(
            store,
            references,
            "two_pet_composites",
            "multi_pet",
            clean,
            [
                {
                    "sample": row,
                    "expected_identity": row["expected_panel_identity"],
                }
                for row in composites
            ],
        )
    )

    baseline = scenarios[0]["summary"]
    aggregates = {
        category: pooled_summary(scenarios, category)
        for category in (
            "baseline",
            "single_reference",
            "gallery_pollution",
            "open_set",
            "degraded_query",
            "combined_fault",
            "multi_pet",
        )
    }
    degraded_rows = scenarios[-3]["rows"]
    degraded_by_variant = {
        variant: summarize_rows([row for row in degraded_rows if row["variant"] == variant])
        for variant in VARIANTS
    }
    clean_rows = scenarios[0]["rows"]
    unknown_rows = [
        row
        for scenario in scenarios
        if scenario["category"] == "open_set"
        for row in scenario["rows"]
    ]
    grid = threshold_grid(clean_rows, unknown_rows)
    latencies = [
        sample["latency_ms"]
        for sample in samples
        if sample.get("latency_ms") is not None
    ]
    report = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": utc_now(),
        "purpose": "diagnostic stress test of the production cosine-prototype matcher",
        "dataset_manifest": str(args.manifest.expanduser().resolve()),
        "gallery_model": str(args.gallery_model.expanduser().resolve()),
        "backend": backend,
        "temporary_gallery": store.summary(),
        "fixture_directory": str(fixture_dir),
        "scenarios": scenarios,
        "aggregates": aggregates,
        "degraded_by_variant": degraded_by_variant,
        "threshold_diagnostic": {
            "best_grid_point": grid[0],
            "top_grid_points": grid[:10],
            "warning": "diagnostic only; two identities are insufficient to calibrate a production threshold",
        },
        "encoding_latency_ms": {
            "count": len(latencies),
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "max": max(latencies),
        },
        "checks": {
            "clean_8_of_8": baseline["known_correct"] == 8
            and baseline["encoding_errors"] == 0,
            "single_reference_rank1_at_least_90pct": (
                aggregates["single_reference"]["end_to_end_accuracy"] >= 0.90
            ),
            "hard_combo_rank1_at_least_50pct": (
                degraded_by_variant["hard_combo"]["end_to_end_accuracy"] >= 0.50
            ),
            "open_set_default_false_accept_zero": (
                aggregates["open_set"]["unknown_false_accept_rate_default"] == 0.0
            ),
        },
        "limitations": [
            "only two local identities are available",
            "capture conditions may be correlated within each identity",
            "synthetic degradations and composites diagnose robustness but do not replace real field data",
            "threshold results are exploratory and must not be treated as production calibration",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = output_dir / "results.csv"
    write_csv(csv_path, scenarios)
    summary_path = output_dir / "summary.md"
    write_summary(summary_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                "temporary_gallery": store.summary(),
                "aggregates": aggregates,
                "degraded_by_variant": degraded_by_variant,
                "threshold_diagnostic": report["threshold_diagnostic"],
                "checks": report["checks"],
                "encoding_latency_ms": report["encoding_latency_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
