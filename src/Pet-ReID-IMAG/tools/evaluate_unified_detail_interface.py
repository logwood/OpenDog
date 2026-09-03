#!/usr/bin/env python3
"""Audit every nose-to-fusion interface on a locked validation set."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_highres import build_highres_from_checkpoint  # noqa: E402
from pet_id.release_compatibility import locked_protocol_paths  # noqa: E402
from train_unified_nose_detail import (  # noqa: E402
    LockedDetailDataset,
    read_json,
    sha256_file,
    workspace_path,
)


ENDPOINT_ORDER = (
    "nose_global_pre_bn_2048",
    "nose_global_post_bn_2048",
    "nose_global_adapter_512",
    "face_global_512",
    "semantic_fused_512",
    "parent_refined_512",
    "nose_detail_pre_bn_2048",
    "nose_detail_post_bn_2048",
    "nose_detail_adapter_512",
    "face_detail_512",
    "detail_final_512",
)

TRANSITIONS = (
    ("nose_global_pre_bn_2048", "nose_global_post_bn_2048"),
    ("nose_global_post_bn_2048", "nose_global_adapter_512"),
    ("face_global_512", "semantic_fused_512"),
    ("semantic_fused_512", "parent_refined_512"),
    ("nose_detail_pre_bn_2048", "nose_detail_post_bn_2048"),
    ("nose_detail_post_bn_2048", "nose_detail_adapter_512"),
    ("parent_refined_512", "detail_final_512"),
)


@dataclass
class EndpointEvaluation:
    metrics: dict[str, float | int]
    scores: torch.Tensor
    truth_columns: torch.Tensor
    order: torch.Tensor
    query_indices: list[int]
    column_targets: list[int]


def atomic_text_dump(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    atomic_text_dump(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for multiple checkpoints; the first is the comparison baseline.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Unsafe checkpoint name: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate checkpoint name: {name}")
        path = workspace_path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append((name, path))
        seen.add(name)
    return specs


def verify_locked_validation(
    config: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], int]:
    lock_path, _, manifest_path = locked_protocol_paths(
        ROOT.parents[1], config["protocol"]
    )
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("The validation protocol is not LOCKED_UNSCORED")
    for key in (
        "identity_disjoint",
        "exact_image_disjoint",
        "validation_training_forbidden",
    ):
        if not bool(lock.get("policy", {}).get(key)):
            raise RuntimeError(f"Locked protocol policy requires {key}=true")
    expected_manifest_hash = lock["splits"]["validation"]["sha256"]
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise RuntimeError("Locked validation manifest hash mismatch")

    manifest = read_json(manifest_path)
    verified_bytes = 0
    for row in manifest["records"]:
        source = workspace_path(row["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256_file(source) != str(row["source_sha256"]):
            raise RuntimeError(f"Locked validation source hash mismatch: {source}")
        verified_bytes += source.stat().st_size
    return lock_path, manifest_path, lock, manifest, verified_bytes


def normalized_row(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and value.shape[-2:] == (1, 1):
        value = value[..., 0, 0]
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"Expected one descriptor row, got {tuple(value.shape)}")
    return F.normalize(value.detach().float().cpu(), dim=1)[0]


def combine_descriptor_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = normalized_row(left)
    right = normalized_row(right)
    return F.normalize(left + right, dim=0)


def append_values(
    destination: dict[str, list[float]],
    name: str,
    value: torch.Tensor,
) -> None:
    destination[name].extend(value.detach().float().reshape(-1).cpu().tolist())


def summary_statistics(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {"count": 0}
    return {
        "count": tensor.numel(),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "minimum": float(tensor.min()),
        "p05": float(torch.quantile(tensor, 0.05)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "maximum": float(tensor.max()),
    }


def evaluate_endpoint(
    values: list[torch.Tensor],
    dataset: LockedDetailDataset,
    *,
    gallery_images: int,
) -> EndpointEvaluation:
    by_target: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for index, (target, value) in enumerate(zip(dataset.targets, values, strict=True)):
        by_target[int(target)].append((index, F.normalize(value.float(), dim=0)))

    column_targets = sorted(by_target)
    target_to_column = {target: column for column, target in enumerate(column_targets)}
    prototypes: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    query_indices: list[int] = []
    truth_columns: list[int] = []
    for target in column_targets:
        rows = by_target[target]
        prototypes.append(
            F.normalize(
                torch.stack([value for _, value in rows[:gallery_images]]).mean(dim=0),
                dim=0,
            )
        )
        for index, value in rows[gallery_images:]:
            queries.append(value)
            query_indices.append(index)
            truth_columns.append(target_to_column[target])

    gallery = torch.stack(prototypes)
    query = torch.stack(queries)
    scores = query @ gallery.T
    truth = torch.tensor(truth_columns, dtype=torch.long)
    order = scores.argsort(dim=1, descending=True)
    top_k = min(5, scores.shape[1])
    top1 = int(order[:, 0].eq(truth).sum())
    top5 = int((order[:, :top_k] == truth[:, None]).any(dim=1).sum())
    labels = torch.zeros_like(scores, dtype=torch.int64)
    labels[torch.arange(len(truth)), truth] = 1
    ranks = (order == truth[:, None]).nonzero()[:, 1] + 1
    same_scores = scores[torch.arange(len(truth)), truth]
    different_mask = ~labels.bool()
    different_scores = scores[different_mask]
    same_mean = float(same_scores.mean())
    different_mean = float(different_scores.mean())
    return EndpointEvaluation(
        metrics={
            "queries": len(truth),
            "gallery_identities": len(gallery),
            "descriptor_dim": int(query.shape[1]),
            "top1_correct": top1,
            "top1_accuracy": top1 / len(truth),
            "top5_correct": top5,
            "top5_accuracy": top5 / len(truth),
            "mrr": float((1.0 / ranks.float()).mean()),
            "auc": float(
                roc_auc_score(labels.flatten().numpy(), scores.flatten().numpy())
            ),
            "same_score_mean": same_mean,
            "different_score_mean": different_mean,
            "same_different_gap": same_mean - different_mean,
        },
        scores=scores,
        truth_columns=truth,
        order=order,
        query_indices=query_indices,
        column_targets=column_targets,
    )


def score_matrix_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.double().flatten()
    right_flat = right.double().flatten()
    left_flat -= left_flat.mean()
    right_flat -= right_flat.mean()
    denominator = left_flat.norm() * right_flat.norm()
    if not bool(denominator > 0):
        return 0.0
    return float((left_flat @ right_flat) / denominator)


def transition_report(
    before_name: str,
    after_name: str,
    evaluations: dict[str, EndpointEvaluation],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    before = evaluations[before_name]
    after = evaluations[after_name]
    if before.query_indices != after.query_indices:
        raise RuntimeError("Endpoint query ordering changed inside one audit")
    if not torch.equal(before.truth_columns, after.truth_columns):
        raise RuntimeError("Endpoint truth columns changed inside one audit")
    before_correct = before.order[:, 0].eq(before.truth_columns)
    after_correct = after.order[:, 0].eq(after.truth_columns)
    rescued = (~before_correct & after_correct).nonzero(as_tuple=False).flatten()
    harmed = (before_correct & ~after_correct).nonzero(as_tuple=False).flatten()

    def examples(indices: torch.Tensor) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query_row in indices[:20].tolist():
            source_index = before.query_indices[query_row]
            record = manifest["records"][source_index]
            before_top = before.column_targets[int(before.order[query_row, 0])]
            after_top = after.column_targets[int(after.order[query_row, 0])]
            rows.append(
                {
                    "identity": str(record["identity"]),
                    "source_path": str(record["source_path"]),
                    "before_top1_target": before_top,
                    "after_top1_target": after_top,
                }
            )
        return rows

    return {
        "before": before_name,
        "after": after_name,
        "top1_correct_delta": int(after_correct.sum() - before_correct.sum()),
        "rescued_queries": int(rescued.numel()),
        "harmed_queries": int(harmed.numel()),
        "score_matrix_correlation": score_matrix_correlation(
            before.scores,
            after.scores,
        ),
        "rescued_examples": examples(rescued),
        "harmed_examples": examples(harmed),
    }


@torch.inference_mode()
def audit_checkpoint(
    name: str,
    checkpoint: Path,
    dataset: LockedDetailDataset,
    manifest: dict[str, Any],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    gallery_images: int,
) -> dict[str, Any]:
    model, model_payload = build_highres_from_checkpoint(
        checkpoint,
        device=device,
        verify_sources=True,
    )
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    endpoints: dict[str, list[torch.Tensor]] = {
        endpoint: [] for endpoint in ENDPOINT_ORDER
    }
    auxiliary: dict[str, list[float]] = defaultdict(list)
    pool_layer = model.parent_model.base_model.nose_encoder.model.heads.pool_layer
    started = time.monotonic()
    for index in range(len(dataset.records)):
        captured_pool: list[torch.Tensor] = []

        def capture_pool(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured_pool.append(output.detach())

        handle = pool_layer.register_forward_hook(capture_pool)
        try:
            row = dataset.load(index)
            rgb = row["high"].unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(rgb, return_aux=True)
        finally:
            handle.remove()
        if len(captured_pool) != 4:
            raise RuntimeError(
                "Expected four nose-pool calls per spatial-detail image, "
                f"got {len(captured_pool)}"
            )

        global_pre_bn = combine_descriptor_pair(captured_pool[0], captured_pool[1])
        detail_pre_bn = combine_descriptor_pair(captured_pool[2], captured_pool[3])
        endpoint_values = {
            "nose_global_pre_bn_2048": global_pre_bn,
            "nose_global_post_bn_2048": normalized_row(
                output["raw_nose_descriptor"]
            ),
            "nose_global_adapter_512": normalized_row(
                output["adapted_nose_descriptor"]
            ),
            "face_global_512": normalized_row(output["face_descriptor"]),
            "semantic_fused_512": normalized_row(output["base_embedding"]),
            "parent_refined_512": normalized_row(
                output["highres_parent_embedding"]
            ),
            "nose_detail_pre_bn_2048": detail_pre_bn,
            "nose_detail_post_bn_2048": normalized_row(
                output["raw_detail_nose_descriptor"]
            ),
            "nose_detail_adapter_512": normalized_row(
                output["detail_nose_descriptor"]
            ),
            "face_detail_512": normalized_row(output["detail_face_descriptor"]),
            "detail_final_512": normalized_row(output["embedding"]),
        }
        for endpoint, value in endpoint_values.items():
            endpoints[endpoint].append(value)

        append_values(auxiliary, "geometry_face_confidence", output["geometry_confidence"][:, 0])
        append_values(auxiliary, "semantic_nose_weight", output["nose_weight"])
        append_values(
            auxiliary,
            "parent_refiner_reliability",
            output["refiner_reliability"],
        )
        append_values(
            auxiliary,
            "parent_refiner_residual_weight",
            output["refiner_residual_weight"],
        )
        append_values(
            auxiliary,
            "parent_refiner_global_gain",
            output["refiner_global_gain"],
        )
        append_values(
            auxiliary,
            "parent_refiner_interaction_l2",
            output["refiner_interaction"].float().norm(dim=1),
        )
        append_values(
            auxiliary, "detail_face_weight", output["detail_weights"][:, 0]
        )
        append_values(
            auxiliary, "detail_nose_weight", output["detail_weights"][:, 1]
        )
        append_values(auxiliary, "detail_global_gain", output["detail_global_gain"])
        append_values(
            auxiliary,
            "detail_interaction_l2",
            output["detail_interaction"].float().norm(dim=1),
        )
        append_values(auxiliary, "detail_scale", output["detail_scale"])
        append_values(auxiliary, "detail_availability", output["detail_availability"])
        auxiliary["global_pre_post_bn_cosine"].append(
            float(
                F.cosine_similarity(
                    global_pre_bn[None],
                    endpoint_values["nose_global_post_bn_2048"][None],
                )
            )
        )
        auxiliary["detail_pre_post_bn_cosine"].append(
            float(
                F.cosine_similarity(
                    detail_pre_bn[None],
                    endpoint_values["nose_detail_post_bn_2048"][None],
                )
            )
        )
        auxiliary["global_detail_post_bn_cosine"].append(
            float(
                F.cosine_similarity(
                    endpoint_values["nose_global_post_bn_2048"][None],
                    endpoint_values["nose_detail_post_bn_2048"][None],
                )
            )
        )
        auxiliary["global_detail_adapter_cosine"].append(
            float(
                F.cosine_similarity(
                    endpoint_values["nose_global_adapter_512"][None],
                    endpoint_values["nose_detail_adapter_512"][None],
                )
            )
        )
        auxiliary["global_detail_face_cosine"].append(
            float(
                F.cosine_similarity(
                    endpoint_values["face_global_512"][None],
                    endpoint_values["face_detail_512"][None],
                )
            )
        )
        if index == 0 or (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "checkpoint": name,
                        "images_complete": index + 1,
                        "images_total": len(dataset.records),
                    }
                ),
                flush=True,
            )

    evaluations = {
        endpoint: evaluate_endpoint(
            endpoints[endpoint],
            dataset,
            gallery_images=gallery_images,
        )
        for endpoint in ENDPOINT_ORDER
    }
    transitions = [
        transition_report(before, after, evaluations, manifest)
        for before, after in TRANSITIONS
    ]
    elapsed = time.monotonic() - started
    result = {
        "name": name,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "model_type": model_payload.get("model_type"),
            "training": model_payload.get("training"),
            "selection": model_payload.get("selection"),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "cuda_max_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
        "endpoints": {
            endpoint: evaluations[endpoint].metrics for endpoint in ENDPOINT_ORDER
        },
        "transitions": transitions,
        "auxiliary_statistics": {
            key: summary_statistics(values) for key, values in sorted(auxiliary.items())
        },
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def comparison_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    endpoint_deltas: dict[str, Any] = {}
    for endpoint in ENDPOINT_ORDER:
        before = baseline["endpoints"][endpoint]
        after = candidate["endpoints"][endpoint]
        endpoint_deltas[endpoint] = {
            "top1_correct": int(after["top1_correct"] - before["top1_correct"]),
            "top1_accuracy": float(after["top1_accuracy"] - before["top1_accuracy"]),
            "top5_correct": int(after["top5_correct"] - before["top5_correct"]),
            "mrr": float(after["mrr"] - before["mrr"]),
            "auc": float(after["auc"] - before["auc"]),
            "same_different_gap": float(
                after["same_different_gap"] - before["same_different_gap"]
            ),
        }
    return {
        "baseline": baseline["name"],
        "candidate": candidate["name"],
        "endpoint_deltas": endpoint_deltas,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Unified spatial-detail nose-interface funnel audit",
        "",
        (
            f"Locked validation: {payload['protocol']['identities']} identities, "
            f"{payload['protocol']['verified_images']} images, "
            f"{payload['protocol']['queries']} queries."
        ),
        "",
    ]
    for checkpoint in payload["checkpoints"]:
        lines.extend(
            [
                f"## {checkpoint['name']}",
                "",
                "| Endpoint | Dim | Top-1 | Top-5 | MRR | AUC | Same-different gap |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for endpoint in ENDPOINT_ORDER:
            metric = checkpoint["endpoints"][endpoint]
            lines.append(
                f"| `{endpoint}` | {metric['descriptor_dim']} | "
                f"{metric['top1_correct']}/{metric['queries']} "
                f"({metric['top1_accuracy']:.4f}) | "
                f"{metric['top5_correct']}/{metric['queries']} "
                f"({metric['top5_accuracy']:.4f}) | "
                f"{metric['mrr']:.6f} | {metric['auc']:.6f} | "
                f"{metric['same_different_gap']:.6f} |"
            )
        lines.extend(
            [
                "",
                "### Stage transitions",
                "",
                "| Before → after | Δ Top-1 | Rescued | Harmed | Score correlation |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for transition in checkpoint["transitions"]:
            lines.append(
                f"| `{transition['before']}` → `{transition['after']}` | "
                f"{transition['top1_correct_delta']:+d} | "
                f"{transition['rescued_queries']} | {transition['harmed_queries']} | "
                f"{transition['score_matrix_correlation']:.6f} |"
            )
        lines.append("")
    for comparison in payload["comparisons"]:
        lines.extend(
            [
                f"## {comparison['candidate']} minus {comparison['baseline']}",
                "",
                "| Endpoint | Δ Top-1 correct | Δ Top-1 | Δ MRR | Δ AUC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for endpoint in ENDPOINT_ORDER:
            delta = comparison["endpoint_deltas"][endpoint]
            lines.append(
                f"| `{endpoint}` | {delta['top1_correct']:+d} | "
                f"{delta['top1_accuracy']:+.4f} | {delta['mrr']:+.6f} | "
                f"{delta['auc']:+.6f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = parse_checkpoint_specs(args.checkpoint)
    output = workspace_path(args.output)
    markdown = (
        workspace_path(args.markdown)
        if args.markdown is not None
        else output.with_suffix(".md")
    )
    lock_path, manifest_path, lock, manifest, verified_bytes = (
        verify_locked_validation(config)
    )
    training_size = int(config["model"]["training_size"])
    dataset = LockedDetailDataset(
        manifest_path,
        training_size=training_size,
        degraded_size=int(config["model"]["degraded_detail_size"]),
        training=False,
        horizontal_flip=0.0,
        color_jitter=0.0,
    )
    expected_identities = int(config["protocol"]["validation_identities"])
    if dataset.num_classes != expected_identities:
        raise RuntimeError(
            f"Validation identity mismatch: {dataset.num_classes} != {expected_identities}"
        )
    gallery_images = int(config["protocol"]["gallery_images_per_identity"])
    device = torch.device(args.device)
    amp_name = str(config["training"]["amp"]).casefold()
    use_amp = device.type == "cuda" and amp_name != "float32"
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    checkpoints = [
        audit_checkpoint(
            name,
            checkpoint,
            dataset,
            manifest,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            gallery_images=gallery_images,
        )
        for name, checkpoint in specs
    ]
    comparisons = [
        comparison_report(checkpoints[0], candidate)
        for candidate in checkpoints[1:]
    ]
    payload = {
        "schema_version": 1,
        "evaluation": "unified_spatial_detail_nose_interface_funnel",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "protocol": {
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "status": lock["status"],
            "validation_manifest": str(manifest_path),
            "validation_manifest_sha256": sha256_file(manifest_path),
            "identities": dataset.num_classes,
            "verified_images": len(manifest["records"]),
            "verified_bytes": verified_bytes,
            "gallery_images_per_identity": gallery_images,
            "queries": dataset.num_classes * (4 - gallery_images),
            "training_size": training_size,
            "amp": amp_name,
            "device": str(device),
        },
        "endpoint_order": list(ENDPOINT_ORDER),
        "checkpoints": checkpoints,
        "comparisons": comparisons,
    }
    atomic_json_dump(payload, output)
    atomic_text_dump(render_markdown(payload), markdown)
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown": str(markdown),
                "checkpoints": [checkpoint["name"] for checkpoint in checkpoints],
                "final_metrics": {
                    checkpoint["name"]: checkpoint["endpoints"]["detail_final_512"]
                    for checkpoint in checkpoints
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
