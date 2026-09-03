#!/usr/bin/env python3
"""Export and verify semantic plus locked BIFOR fusion as one ONNX graph."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.transforms import functional as TVF


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.bifor_backbone import FrozenBIFORBodyBackbone
from pet_id.bifor_onnx import (
    BIFOR_ONNX_INPUT_NAMES,
    PreCroppedBIFORPetEmbeddingModel,
)
from pet_id.config import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model
from pet_id.onnx_export import ONNX_OUTPUT_NAMES, extract_precropped_onnx_inputs
from pet_id.model_profiles import get_runtime_profile
from pet_id.release_compatibility import historical_run_path
from pet_id.workspace_paths import normalize_runtime_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_diverse_indices(dataset, count: int) -> list[int]:
    selected, seen = [], set()
    for index, record in enumerate(dataset.records):
        identity = record["identity"].casefold()
        if identity in seen:
            continue
        selected.append(index)
        seen.add(identity)
        if len(selected) == count:
            return selected
    for index in range(len(dataset)):
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Manifest contains fewer than {count} records")


def tensor_inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    excluded = {"targets", "identities", "source_paths"}
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key not in excluded and torch.is_tensor(value)
    }


def path_key(value: str) -> str:
    return Path(value).name.casefold()


def body_crops_from_metadata(
    images_0_255: torch.Tensor,
    source_paths: list[str],
    metadata_path: Path,
) -> tuple[torch.Tensor, list[list[int]]]:
    metadata = np.load(metadata_path, allow_pickle=False)
    rows = {
        path_key(path): index
        for index, path in enumerate(metadata["source_paths"].tolist())
    }
    missing = [path for path in source_paths if path_key(path) not in rows]
    if missing:
        raise ValueError(f"Body metadata is missing paths: {missing}")
    boxes = [
        [int(value) for value in metadata["body_boxes_xyxy"][rows[path_key(path)]]]
        for path in source_paths
    ]
    crops = []
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        crop = images_0_255[index, :, y1:y2, x1:x2]
        if crop.numel() == 0:
            raise RuntimeError(
                f"Empty body crop for {source_paths[index]}: {boxes[index]}"
            )
        crops.append(TVF.resize(crop, [224, 224], antialias=True))
    return torch.stack(crops), boxes


def compare_outputs(reference, candidate) -> list[dict]:
    metrics = []
    for name, expected, actual in zip(ONNX_OUTPUT_NAMES, reference, candidate):
        expected = np.asarray(expected, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        difference = np.abs(expected - actual)
        row = {
            "name": name,
            "shape": list(expected.shape),
            "max_abs_error": float(difference.max(initial=0.0)),
            "mean_abs_error": float(difference.mean()),
        }
        if expected.ndim == 2 and expected.shape[1] > 2:
            numerator = (expected * actual).sum(axis=1)
            denominator = np.linalg.norm(expected, axis=1) * np.linalg.norm(
                actual, axis=1
            )
            row["minimum_cosine"] = float(
                np.min(numerator / np.maximum(denominator, 1e-12))
            )
        metrics.append(row)
    return metrics


def validate_runtime(
    model_path: Path,
    wrapper,
    inputs: tuple[torch.Tensor, ...],
    batch_sizes: list[int],
    *,
    max_abs_tolerance: float,
    minimum_cosine: float,
) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if tuple(item.name for item in session.get_inputs()) != BIFOR_ONNX_INPUT_NAMES:
        raise RuntimeError("Exported ONNX input contract is incorrect")
    if tuple(item.name for item in session.get_outputs()) != ONNX_OUTPUT_NAMES:
        raise RuntimeError("Exported ONNX output contract is incorrect")
    runs = []
    for batch_size in batch_sizes:
        selected = tuple(value[:batch_size].contiguous() for value in inputs)
        with torch.inference_mode():
            torch_outputs = tuple(value.cpu().numpy() for value in wrapper(*selected))
        ort_inputs = {
            name: value.cpu().numpy()
            for name, value in zip(BIFOR_ONNX_INPUT_NAMES, selected)
        }
        runtime_outputs = session.run(list(ONNX_OUTPUT_NAMES), ort_inputs)
        metrics = compare_outputs(torch_outputs, runtime_outputs)
        if any(row["max_abs_error"] > max_abs_tolerance for row in metrics):
            raise RuntimeError(
                f"ONNX max absolute error exceeds {max_abs_tolerance}: {metrics}"
            )
        if metrics[0].get("minimum_cosine", -1.0) < minimum_cosine:
            raise RuntimeError(
                f"ONNX embedding cosine is below {minimum_cosine}: {metrics[0]}"
            )
        embedding = runtime_outputs[0]
        norms = np.linalg.norm(embedding, axis=1)
        rank1_consistent = None
        if batch_size >= 2:
            torch_similarity = torch_outputs[0] @ torch_outputs[0].T
            ort_similarity = embedding @ embedding.T
            np.fill_diagonal(torch_similarity, -np.inf)
            np.fill_diagonal(ort_similarity, -np.inf)
            rank1_consistent = bool(
                np.array_equal(
                    torch_similarity.argmax(axis=1),
                    ort_similarity.argmax(axis=1),
                )
            )
            if not rank1_consistent:
                raise RuntimeError("PyTorch and ONNX nearest-neighbor rankings differ")
        runs.append(
            {
                "batch_size": batch_size,
                "outputs": metrics,
                "onnx_embedding_norm_range": [
                    float(norms.min()),
                    float(norms.max()),
                ],
                "nearest_neighbor_top1_consistent": rank1_consistent,
            }
        )
    return {
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "max_abs_tolerance": max_abs_tolerance,
        "minimum_embedding_cosine": minimum_cosine,
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    legacy = get_runtime_profile("legacy-semantic")
    research = get_runtime_profile("research-bifor")
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=legacy.identity_weights,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=legacy.config,
    )
    parser.add_argument(
        "--bifor-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/BIFOR/f2/bifor.pth",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=historical_run_path(WORKSPACE, "bifor-fusion-checkpoint"),
    )
    parser.add_argument(
        "--body-detector-checkpoint",
        type=Path,
        default=research.body_detector,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "joint-validation-manifest"),
    )
    parser.add_argument(
        "--body-metadata",
        type=Path,
        default=historical_run_path(WORKSPACE, "body-validation")
        / "body_semantic_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=research.onnx.parent,
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--export-batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=5e-4)
    parser.add_argument("--minimum-cosine", type=float, default=0.99999)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = parse_args()
    os.chdir(ROOT)
    paths = {
        name: Path(value).resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    missing = [
        str(path)
        for name, path in paths.items()
        if name != "output_dir" and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing export inputs: {missing}")
    if args.max_dynamic_batch < max(args.export_batch_size, args.validation_batch_size):
        raise ValueError("--max-dynamic-batch is smaller than a requested batch")

    output_dir = paths["output_dir"]
    model_path = output_dir / "pet_embedding.onnx"
    temporary_path = output_dir / "pet_embedding.exporting.onnx"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if temporary_path.exists():
        temporary_path.unlink()

    device = torch.device(args.device)
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(paths["config_file"]))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    identity_model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=paths["semantic_checkpoint"],
    ).eval()
    body_encoder = FrozenBIFORBodyBackbone(paths["bifor_checkpoint"]).to(device).eval()
    wrapper = (
        PreCroppedBIFORPetEmbeddingModel(
            identity_model,
            body_encoder,
            paths["fusion_checkpoint"],
        )
        .to(device)
        .eval()
    )

    dataset = PreparedDogFaceNetDataset(paths["manifest"], training=False)
    sample_count = max(args.export_batch_size, args.validation_batch_size)
    selected_indices = select_diverse_indices(dataset, sample_count)
    batch = collate_prepared_dogfacenet([dataset[index] for index in selected_indices])
    full_inputs = tensor_inputs(batch, device)
    with torch.inference_mode():
        semantic_full = identity_model(**full_inputs)["features"]
        semantic_inputs = extract_precropped_onnx_inputs(identity_model, **full_inputs)
        body_crops, body_boxes = body_crops_from_metadata(
            full_inputs["images_0_255"],
            list(batch["source_paths"]),
            paths["body_metadata"],
        )
        crop_inputs = (
            semantic_inputs[0],
            semantic_inputs[1],
            body_crops.to(device),
            *semantic_inputs[2:],
        )
        wrapped_outputs = wrapper(*crop_inputs)
        semantic_wrapped = wrapper.semantic_model(*semantic_inputs)[0]
    semantic_parity = float((semantic_wrapped - semantic_full).abs().max().cpu())
    if semantic_parity > 1e-6:
        raise RuntimeError(f"Semantic wrapper parity failed: {semantic_parity}")

    export_inputs = tuple(
        value[: args.export_batch_size].contiguous() for value in crop_inputs
    )
    batch_dimension = torch.export.Dim("batch", min=1, max=args.max_dynamic_batch)
    dynamic_shapes = tuple({0: batch_dimension} for _ in export_inputs)
    torch.onnx.export(
        wrapper,
        export_inputs,
        temporary_path,
        input_names=list(BIFOR_ONNX_INPUT_NAMES),
        output_names=list(ONNX_OUTPUT_NAMES),
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        dynamic_shapes=dynamic_shapes,
        optimize=True,
    )
    onnx.checker.check_model(str(temporary_path))

    validation_inputs = tuple(value.detach().cpu() for value in crop_inputs)
    wrapper = wrapper.cpu().eval()
    validation = validate_runtime(
        temporary_path,
        wrapper,
        validation_inputs,
        sorted({1, args.validation_batch_size}),
        max_abs_tolerance=args.max_abs_tolerance,
        minimum_cosine=args.minimum_cosine,
    )
    if model_path.exists():
        model_path.unlink()
    shutil.move(str(temporary_path), str(model_path))
    model_hash = sha256_file(model_path)
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 2,
        "created_at": now,
        "model_type": "semantic_body_fusion",
        "model": str(model_path.relative_to(WORKSPACE)),
        "onnx_sha256": model_hash,
        "onnx_bytes": model_path.stat().st_size,
        "source_checkpoint": str(paths["semantic_checkpoint"].relative_to(WORKSPACE)),
        "source_checkpoint_sha256": sha256_file(paths["semantic_checkpoint"]),
        "config_file": str(paths["config_file"].relative_to(WORKSPACE)),
        "config_sha256": sha256_file(paths["config_file"]),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "fusion_mode": "semantic_residual+bifor_lowrank",
        "precision_provenance": {
            "deployment": "float32 ONNX",
            "projector_selection_features": "bfloat16 autocast converted to float32",
            "note": (
                "The locked projector is unchanged. Production FP32 inference may "
                "differ slightly from historical BF16 protocol metrics."
            ),
        },
        "gallery_compatibility": {
            "compatible_with_legacy_semantic": False,
            "reencoding_required": True,
            "reason": "the locked SVD projection defines a new 512-D coordinate space",
        },
        "body_fusion": {
            "body_weight": wrapper.body_weight,
            "semantic_weight": wrapper.semantic_weight,
            "projection_rank": wrapper.projection_rank,
            "output_dim": wrapper.output_dim,
            "horizontal_flip_tta": True,
            "bifor_checkpoint": str(paths["bifor_checkpoint"].relative_to(WORKSPACE)),
            "bifor_checkpoint_sha256": sha256_file(paths["bifor_checkpoint"]),
            "fusion_checkpoint": str(paths["fusion_checkpoint"].relative_to(WORKSPACE)),
            "fusion_checkpoint_sha256": sha256_file(paths["fusion_checkpoint"]),
            "classification_head": None,
        },
        "body_preprocessing": {
            "detector": {
                "name": "torchvision/fasterrcnn_resnet50_fpn_v2",
                "checkpoint": str(
                    paths["body_detector_checkpoint"].relative_to(WORKSPACE)
                ),
                "checkpoint_sha256": sha256_file(paths["body_detector_checkpoint"]),
                "score_threshold": 0.5,
                "crop_expansion": 0.04,
                "inside_onnx": False,
            },
            "target_size": [224, 224],
            "normalization": "ImageNet mean/std",
        },
        "dynamic_batch": {
            "minimum": 1,
            "declared_maximum": args.max_dynamic_batch,
            "validated_batch_sizes": sorted({1, args.validation_batch_size}),
        },
        "inputs": {
            "nose_crop": {
                "shape": ["N", 3, 244, 244],
                "dtype": "float32",
                "range": [0, 255],
            },
            "face_crop": {
                "shape": ["N", 3, 224, 224],
                "dtype": "float32",
                "range": [0, 255],
            },
            "body_crop": {
                "shape": ["N", 3, 224, 224],
                "dtype": "float32",
                "range": [0, 255],
            },
            "nose_mask": {
                "shape": ["N", 1, 244, 244],
                "dtype": "float32",
                "range": [0, 1],
            },
            "quality_signals": {"shape": ["N", 6], "dtype": "float32"},
            "viewpoint_signals": {"shape": ["N", 4], "dtype": "float32"},
            "branch_available": {
                "shape": ["N", 2],
                "dtype": "bool",
                "order": ["nose", "face"],
            },
        },
        "outputs": {
            "embedding": {"shape": ["N", 512], "l2_normalized": True},
            "nose_embedding": {"shape": ["N", 2048], "l2_normalized": True},
            "face_embedding": {"shape": ["N", 512], "l2_normalized": True},
            "fusion_weights": {"shape": ["N", 2]},
            "joint_weights": {"shape": ["N", 2]},
            "viewpoint_frontality": {"shape": ["N"]},
        },
        "excluded": [
            "AnyFace detection",
            "SAM2 segmentation",
            "dog body detection",
            "EXIF handling",
            "rotated ROI extraction",
            "all training-only classifiers",
        ],
    }
    validation.update(
        {
            "schema_version": 2,
            "validated_at": now,
            "onnx_checker": "passed",
            "onnx_sha256": model_hash,
            "semantic_wrapper_vs_full_pytorch_max_abs": semantic_parity,
            "sample_manifest": str(paths["manifest"]),
            "sample_indices": selected_indices,
            "sample_identities": list(batch["identities"]),
            "sample_body_boxes_xyxy": body_boxes,
            "pytorch_embedding_norm_range": [
                float(wrapped_outputs[0].norm(dim=1).min().cpu()),
                float(wrapped_outputs[0].norm(dim=1).max().cpu()),
            ],
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Semantic + BIFOR low-rank ONNX\n\n"
        "The main output is a 512-D L2-normalized descriptor combining 92% "
        "semantic nose/face and 8% frozen headless BIFOR body information. "
        "The BIFOR flip-TTA and locked rank-500 projection are inside ONNX.\n\n"
        "AnyFace, SAM2, and the frozen dog-body detector remain preprocessing. "
        "Use `pet_id.bifor_onnx_runtime` for the raw-image pipeline. Existing "
        "legacy semantic galleries must be re-encoded before retrieval. The projector "
        "was selected on a historical BF16 feature cache; this package intentionally "
        "uses production FP32 ONNX, so secondary metrics can differ slightly.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "sha256": model_hash,
                "bytes": model_path.stat().st_size,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
