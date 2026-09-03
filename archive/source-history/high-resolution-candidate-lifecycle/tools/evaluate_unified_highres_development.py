#!/usr/bin/env python3
"""Evaluate a V4 high-resolution candidate against the locked V3 parent.

Only the locked V4 ``development`` split is read.  The script reports two V3
references: the parent evaluated on V4's graph-internal global view and the
production V3 preprocessing path (OpenCV centered letterbox).  Promotion is
never inferred from the blind split here; this command is deliberately a
development-only operation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import _read_bgr  # noqa: E402
from pet_id.unified_data import letterbox_rgb  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    build_external_joint_from_checkpoint,
    configure_strict_cuda_precision,
    sha256_file,
)
from pet_id.unified_fresh_protocol import stable_order_key  # noqa: E402
from pet_id.unified_highres import (  # noqa: E402
    MODEL_TYPE,
    build_highres_from_checkpoint,
)
from pet_id.unified_highres_eval_data import (  # noqa: E402
    UnifiedHighResolutionRawDataset,
)
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--maximum-side", type=int)
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--no-cross-resolution", action="store_true")
    parser.add_argument("--include-queries", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lock(lock_path: Path, manifest_path: Path) -> tuple[dict, dict]:
    lock = load_json(lock_path)
    if lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 protocol lock")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 protocol must remain locked and unscored")
    required = (
        "v4_identity_disjoint",
        "exact_image_disjoint",
        "blind_single_candidate_attempt",
        "blind_training_forbidden",
        "blind_model_selection_forbidden",
        "blind_features_must_not_be_persisted",
        "failed_candidate_keeps_v3_default",
    )
    policy = lock.get("policy", {})
    for key in required:
        if policy.get(key) is not True:
            raise RuntimeError(f"V4 protocol policy is missing {key}")
    development = lock["splits"]["development"]
    if manifest_path.resolve() != Path(development["path"]).expanduser().resolve():
        raise RuntimeError("Only the locked V4 development manifest may be evaluated")
    if sha256_file(manifest_path) != development["sha256"]:
        raise RuntimeError("Development manifest differs from the V4 protocol lock")
    return lock, development


def validate_checkpoint(
    checkpoint_path: Path,
    parent_path: Path,
    lock: dict,
) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Checkpoint is not a V4 high-resolution checkpoint")
    training = payload.get("training") or {}
    if training.get("blind_data_used") is not False:
        raise RuntimeError("V4 checkpoint provenance indicates blind data usage")
    expected_parent = payload["sources"]["parent_v3_checkpoint"]
    if Path(expected_parent["path"]).expanduser().resolve() != parent_path.resolve():
        raise RuntimeError("V4 checkpoint does not use the requested V3 parent")
    if sha256_file(parent_path) != expected_parent["sha256"]:
        raise RuntimeError("V3 parent hash differs from the V4 checkpoint source")
    training_manifest = Path(training.get("manifest", "")).expanduser().resolve()
    locked_training = Path(
        lock["splits"]["training_extension"]["path"]
    ).expanduser().resolve()
    if training_manifest != locked_training:
        raise RuntimeError("V4 checkpoint was not trained on the locked extension split")
    if training.get("manifest_sha256") != lock["splits"]["training_extension"]["sha256"]:
        raise RuntimeError("V4 training manifest hash differs from the protocol lock")
    return payload


def normalized(value: torch.Tensor) -> torch.Tensor:
    value = F.normalize(value.float(), dim=1)
    if not torch.isfinite(value).all():
        raise FloatingPointError("Non-finite embedding encountered")
    return value


def production_parent_input(row: dict[str, Any], *, size: int = 1280) -> torch.Tensor:
    image = _read_bgr(Path(row["source_path"]))
    image = __import__("cv2").cvtColor(image, __import__("cv2").COLOR_BGR2RGB)
    boxed, _, _ = letterbox_rgb(
        image,
        size=size,
        fill_value=0,
        allow_upscale=False,
    )
    return torch.from_numpy(boxed.transpose(2, 0, 1).copy()).float()


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=1)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    lock_path = args.protocol_lock.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, parent_path, lock_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if args.gallery_images_per_identity < 1:
        raise ValueError("gallery-images-per-identity must be positive")

    lock, development_lock = validate_lock(lock_path, manifest_path)
    checkpoint_payload = validate_checkpoint(checkpoint_path, parent_path, lock)
    configured_maximum = int(
        checkpoint_payload["model_config"].get("maximum_input_side", 4096)
    )
    maximum_side = configured_maximum if args.maximum_side is None else int(args.maximum_side)
    if maximum_side < 64 or maximum_side > configured_maximum:
        raise ValueError(
            f"maximum-side must be in [64,{configured_maximum}]"
        )

    precision = configure_strict_cuda_precision()
    device = torch.device(args.device)
    candidate, loaded_payload = build_highres_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=True,
    )
    parent, _ = build_external_joint_from_checkpoint(
        parent_path,
        device=device,
        verify_sources=True,
    )
    candidate.eval()
    parent.eval()
    dataset = UnifiedHighResolutionRawDataset(
        manifest_path,
        expected_split="development",
        maximum_side=maximum_side,
        verify_source_hash=True,
    )

    candidate_features: list[torch.Tensor] = []
    internal_parent_features: list[torch.Tensor] = []
    production_parent_features: list[torch.Tensor] = []
    degraded_features: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    dimensions: list[dict[str, int]] = []
    cross_resolution_cosines: list[float] = []
    low_resolution_anchor_errors: list[float] = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            raw = row["rgb"][None].to(device).contiguous()
            high_aux = candidate(raw, return_aux=True)
            high_feature = normalized(high_aux["embedding"])
            internal_parent = normalized(high_aux["highres_parent_embedding"])
            production_input = production_parent_input(row)[None].to(device)
            production_parent = normalized(parent(production_input))

            candidate_features.append(high_feature[0].cpu())
            internal_parent_features.append(internal_parent[0].cpu())
            production_parent_features.append(production_parent[0].cpu())
            identities.append(str(row["identity"]).casefold())
            source_paths.append(str(row["source_path"]))
            source_sha256.append(str(row["source_sha256"]))
            dimensions.append(
                {
                    "original_height": int(row["original_height"]),
                    "original_width": int(row["original_width"]),
                    "fed_height": int(row["fed_height"]),
                    "fed_width": int(row["fed_width"]),
                }
            )

            if not args.no_cross_resolution:
                # Recreate the same source at a 1280 detail cap.  The result
                # is still passed as a raw tensor so the comparison exercises
                # the V4 low-resolution gate, rather than a second model.
                from pet_id.unified_highres_data import degraded_raw_rgb

                degraded = degraded_raw_rgb(
                    row["source_path"],
                    detail_cap=1280,
                    maximum_side=maximum_side,
                )[None].to(device).contiguous()
                degraded_aux = candidate(degraded, return_aux=True)
                degraded_feature = normalized(degraded_aux["embedding"])
                degraded_features.append(degraded_feature[0].cpu())
                cross_resolution_cosines.append(
                    float(cosine_rows(high_feature, degraded_feature)[0].cpu())
                )
                low_resolution_anchor_errors.append(
                    float(
                        (degraded_aux["embedding"] - degraded_aux["highres_parent_embedding"])
                        .abs()
                        .max()
                        .cpu()
                    )
                )
            if index == 0 or (index + 1) % max(args.progress_every, 1) == 0:
                print(f"v4 high-resolution development: {index + 1}/{len(dataset)}", flush=True)

    candidate_tensor = torch.stack(candidate_features)
    internal_parent_tensor = torch.stack(internal_parent_features)
    production_parent_tensor = torch.stack(production_parent_features)
    metrics_candidate = retrieval_metrics(
        candidate_tensor,
        identities,
        source_paths,
        gallery_images_per_identity=args.gallery_images_per_identity,
        include_queries=args.include_queries,
    )
    metrics_internal_parent = retrieval_metrics(
        internal_parent_tensor,
        identities,
        source_paths,
        gallery_images_per_identity=args.gallery_images_per_identity,
        include_queries=args.include_queries,
    )
    metrics_production_parent = retrieval_metrics(
        production_parent_tensor,
        identities,
        source_paths,
        gallery_images_per_identity=args.gallery_images_per_identity,
        include_queries=args.include_queries,
    )
    checks = {
        "candidate_top1_not_below_internal_parent": metrics_candidate["top1_correct"]
        >= metrics_internal_parent["top1_correct"],
        "candidate_top5_not_below_internal_parent": metrics_candidate["top5_correct"]
        >= metrics_internal_parent["top5_correct"],
        "candidate_top1_not_below_production_parent": metrics_candidate["top1_correct"]
        >= metrics_production_parent["top1_correct"],
        "candidate_top5_not_below_production_parent": metrics_candidate["top5_correct"]
        >= metrics_production_parent["top5_correct"],
        "candidate_output_shape": list(candidate_tensor.shape)
        == [len(dataset), 512],
        "candidate_output_finite": bool(torch.isfinite(candidate_tensor).all()),
    }
    cross_resolution = {
        "checked": not args.no_cross_resolution,
        "pair_count": len(cross_resolution_cosines),
        "minimum_cosine": min(cross_resolution_cosines)
        if cross_resolution_cosines
        else None,
        "mean_cosine": float(np.mean(cross_resolution_cosines))
        if cross_resolution_cosines
        else None,
        "maximum_low_resolution_anchor_abs_error": max(low_resolution_anchor_errors)
        if low_resolution_anchor_errors
        else None,
        "low_resolution_anchor_exact": bool(
            low_resolution_anchor_errors
            and max(low_resolution_anchor_errors) == 0.0
        )
        if not args.no_cross_resolution
        else None,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "unified_v4_real_high_resolution_development_comparison",
        "blind_data_used": False,
        "protocol": {
            "name": PROTOCOL_NAME,
            "lock": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "development_manifest": str(manifest_path),
            "development_manifest_sha256": sha256_file(manifest_path),
            "development_lock_record": development_lock,
            "blind_split_read": False,
        },
        "candidate": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_type": loaded_payload["model_type"],
            "model_config": loaded_payload["model_config"],
            "maximum_side": maximum_side,
        },
        "parent_v3": {
            "checkpoint": str(parent_path),
            "checkpoint_sha256": sha256_file(parent_path),
            "production_preprocessing": "centered_black_letterbox_1280_no_upscale",
        },
        "backend": {
            "device": str(device),
            "cuda_precision": precision,
            "candidate": "pytorch_v4_one_graph",
            "parent": "pytorch_v3_one_graph",
        },
        "dataset": {
            "records": len(dataset),
            "identities": dataset.num_classes,
            "images_per_identity": dataset.images_per_identity,
            "gallery_images_per_identity": args.gallery_images_per_identity,
            "maximum_side": maximum_side,
            "source_sha256_verified": True,
            "fed_shapes": dimensions,
        },
        "candidate_metrics": metrics_candidate,
        "parent_internal_global_metrics": metrics_internal_parent,
        "parent_production_metrics": metrics_production_parent,
        "metric_deltas_vs_internal_parent": {
            "top1_correct": metrics_candidate["top1_correct"]
            - metrics_internal_parent["top1_correct"],
            "top5_correct": metrics_candidate["top5_correct"]
            - metrics_internal_parent["top5_correct"],
        },
        "metric_deltas_vs_production_parent": {
            "top1_correct": metrics_candidate["top1_correct"]
            - metrics_production_parent["top1_correct"],
            "top5_correct": metrics_candidate["top5_correct"]
            - metrics_production_parent["top5_correct"],
        },
        "cross_resolution": cross_resolution,
        "noninferiority": {"checks": checks, "passed": all(checks.values())},
        "per_query_results_persisted": bool(args.include_queries),
        "feature_cache": None,
        "passed": all(checks.values()),
        "default_backend_changed": False,
    }
    if args.feature_cache is not None:
        cache_path = args.feature_cache.expanduser().resolve()
        if cache_path.exists():
            raise FileExistsError(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "candidate": candidate_tensor.numpy(),
            "parent_internal_global": internal_parent_tensor.numpy(),
            "parent_production": production_parent_tensor.numpy(),
            "identities": np.asarray(identities),
            "source_paths": np.asarray(source_paths),
            "source_sha256": np.asarray(source_sha256),
        }
        if degraded_features:
            arrays["candidate_degraded"] = torch.stack(degraded_features).numpy()
        np.savez_compressed(cache_path, **arrays)
        report["feature_cache"] = {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "contains_blind_data": False,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del candidate, parent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": sha256_file(output_path),
                "candidate_metrics": metrics_candidate,
                "parent_production_metrics": metrics_production_parent,
                "cross_resolution": cross_resolution,
                "noninferiority": report["noninferiority"],
                "blind_data_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
