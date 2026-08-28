#!/usr/bin/env python3
"""Export and verify the locked multimodal dog embedding network as ONNX."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.config import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model
from pet_id.onnx_export import (
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    PreCroppedPetEmbeddingModel,
    extract_precropped_onnx_inputs,
)


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
    raise RuntimeError(f"Manifest contains fewer than {count} usable records")


def tensor_inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    excluded = {"targets", "identities", "source_paths"}
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key not in excluded and torch.is_tensor(value)
    }


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
                actual,
                axis=1,
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
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if input_names != list(ONNX_INPUT_NAMES):
        raise RuntimeError(f"Unexpected ONNX inputs: {input_names}")
    if output_names != list(ONNX_OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected ONNX outputs: {output_names}")

    runs = []
    for batch_size in batch_sizes:
        selected = tuple(value[:batch_size].contiguous() for value in inputs)
        with torch.inference_mode():
            torch_outputs = tuple(value.cpu().numpy() for value in wrapper(*selected))
        ort_inputs = {
            name: value.cpu().numpy()
            for name, value in zip(ONNX_INPUT_NAMES, selected)
        }
        runtime_outputs = session.run(list(ONNX_OUTPUT_NAMES), ort_inputs)
        metrics = compare_outputs(torch_outputs, runtime_outputs)
        if any(row["max_abs_error"] > max_abs_tolerance for row in metrics):
            raise RuntimeError(
                f"ONNX max absolute error exceeds {max_abs_tolerance}: {metrics}"
            )
        embedding_metric = metrics[0]
        if embedding_metric.get("minimum_cosine", -1.0) < minimum_cosine:
            raise RuntimeError(
                f"ONNX embedding cosine is below {minimum_cosine}: {metrics}"
            )

        torch_embedding = torch_outputs[0]
        runtime_embedding = runtime_outputs[0]
        torch_norms = np.linalg.norm(torch_embedding, axis=1)
        runtime_norms = np.linalg.norm(runtime_embedding, axis=1)
        rank1_consistent = None
        if batch_size >= 2:
            torch_similarity = torch_embedding @ torch_embedding.T
            runtime_similarity = runtime_embedding @ runtime_embedding.T
            np.fill_diagonal(torch_similarity, -np.inf)
            np.fill_diagonal(runtime_similarity, -np.inf)
            rank1_consistent = bool(
                np.array_equal(
                    torch_similarity.argmax(axis=1),
                    runtime_similarity.argmax(axis=1),
                )
            )
            if not rank1_consistent:
                raise RuntimeError("ONNX and PyTorch nearest-neighbor rankings differ")
        runs.append(
            {
                "batch_size": batch_size,
                "outputs": metrics,
                "torch_embedding_norm_range": [
                    float(torch_norms.min()),
                    float(torch_norms.max()),
                ],
                "onnx_embedding_norm_range": [
                    float(runtime_norms.min()),
                    float(runtime_norms.max()),
                ],
                "nearest_neighbor_top1_consistent": rank1_consistent,
            }
        )
    del session
    return {
        "provider": providers[0],
        "onnxruntime_version": ort.__version__,
        "max_abs_tolerance": max_abs_tolerance,
        "minimum_embedding_cosine": minimum_cosine,
        "runs": runs,
    }


def write_readme(output_dir: Path, embedding_dim: int, fusion_mode: str) -> None:
    (output_dir / "README.md").write_text(
        "# DogFaceNet joint800 ONNX embedding\n\n"
        "This package exports the identity network only. AnyFace detection, SAM2 "
        "nose masking, EXIF handling, and rotated crop extraction remain application "
        "preprocessing.\n\n"
        "Inputs are RGB float32 crops in the 0-255 range plus the quality, viewpoint, "
        "and branch-availability signals documented in `metadata.json`. The main "
        f"`embedding` output is an L2-normalized {embedding_dim}-D descriptor intended "
        "for cosine "
        "gallery retrieval. The 800-class training classifier is intentionally not "
        "part of this deployment graph, so new identities can be registered without "
        "re-exporting the model.\n\n"
        f"Fusion mode: `{fusion_mode}`.\n\n"
        "See `validation.json` for ONNX checker and PyTorch/ONNX Runtime parity results.\n",
        encoding="utf-8",
    )


def main() -> None:
    # PyTorch's ONNX exporter may print Unicode status symbols.  Keep the CLI
    # usable on Windows consoles whose active code page cannot encode them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("models/dogfacenet_joint800_v1/model_final.pth"),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("models/dogfacenet_joint800_v1/config.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "logs/dogfacenet_joint800_protocol_v1/blind_test_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/dogfacenet_joint800_v1/onnx"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--export-batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--max-dynamic-batch", type=int, default=32)
    parser.add_argument("--max-abs-tolerance", type=float, default=5e-4)
    parser.add_argument("--minimum-cosine", type=float, default=0.99999)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    checkpoint = args.checkpoint.resolve()
    config_file = args.config_file.resolve()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    model_path = output_dir / "pet_embedding.onnx"
    temporary_model_path = output_dir / "pet_embedding.exporting.onnx"
    metadata_path = output_dir / "metadata.json"
    validation_path = output_dir / "validation.json"
    required = (checkpoint, config_file, manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing export inputs: {missing}")
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing ONNX model: {model_path}")
    if args.export_batch_size < 1 or args.validation_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_dynamic_batch < max(
        args.export_batch_size,
        args.validation_batch_size,
    ):
        raise ValueError("--max-dynamic-batch is smaller than a requested batch")

    output_dir.mkdir(parents=True, exist_ok=True)
    if temporary_model_path.exists():
        temporary_model_path.unlink()

    device = torch.device(args.device)
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.defrost()
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    identity_model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=checkpoint,
    ).eval()
    dataset = PreparedDogFaceNetDataset(manifest, training=False)
    sample_count = max(args.export_batch_size, args.validation_batch_size)
    selected_indices = select_diverse_indices(dataset, sample_count)
    batch = collate_prepared_dogfacenet(
        [dataset[index] for index in selected_indices]
    )
    full_inputs = tensor_inputs(batch, device)
    with torch.inference_mode():
        full_output = identity_model(**full_inputs)
        crop_inputs = extract_precropped_onnx_inputs(
            identity_model,
            **full_inputs,
        )
        wrapper = PreCroppedPetEmbeddingModel(identity_model).to(device).eval()
        wrapped_output = wrapper(*crop_inputs)
    parity_delta = float(
        (wrapped_output[0] - full_output["features"]).abs().max().detach()
    )
    if parity_delta > 1e-6:
        raise RuntimeError(
            f"Pre-cropped wrapper differs from full model: max abs {parity_delta}"
        )

    export_inputs = tuple(
        value[: args.export_batch_size].contiguous()
        for value in crop_inputs
    )
    batch_dimension = torch.export.Dim(
        "batch",
        min=1,
        max=args.max_dynamic_batch,
    )
    dynamic_shapes = tuple({0: batch_dimension} for _ in export_inputs)
    torch.onnx.export(
        wrapper,
        export_inputs,
        temporary_model_path,
        input_names=list(ONNX_INPUT_NAMES),
        output_names=list(ONNX_OUTPUT_NAMES),
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        dynamic_shapes=dynamic_shapes,
        optimize=True,
    )
    onnx.checker.check_model(str(temporary_model_path))

    validation_inputs = tuple(value.detach().cpu() for value in crop_inputs)
    wrapper = wrapper.cpu().eval()
    validation = validate_runtime(
        temporary_model_path,
        wrapper,
        validation_inputs,
        sorted({1, args.validation_batch_size}),
        max_abs_tolerance=args.max_abs_tolerance,
        minimum_cosine=args.minimum_cosine,
    )
    validation.update(
        {
            "schema_version": 1,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "onnx_checker": "passed",
            "wrapper_vs_full_pytorch_max_abs": parity_delta,
            "sample_manifest": str(manifest),
            "sample_indices": selected_indices,
            "sample_identities": [batch["identities"][i] for i in range(sample_count)],
        }
    )

    if model_path.exists():
        model_path.unlink()
    shutil.move(str(temporary_model_path), str(model_path))
    model_hash = sha256_file(model_path)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "onnx_sha256": model_hash,
        "onnx_bytes": model_path.stat().st_size,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "config_file": str(config_file),
        "config_sha256": sha256_file(config_file),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "fusion_mode": identity_model.fusion_mode,
        "dynamic_batch": {
            "minimum": 1,
            "declared_maximum": args.max_dynamic_batch,
            "validated_batch_sizes": sorted({1, args.validation_batch_size}),
        },
        "inputs": {
            "nose_crop": {"shape": ["N", 3, 244, 244], "dtype": "float32", "range": [0, 255]},
            "face_crop": {"shape": ["N", 3, 224, 224], "dtype": "float32", "range": [0, 255]},
            "nose_mask": {"shape": ["N", 1, 244, 244], "dtype": "float32", "range": [0, 1]},
            "quality_signals": {
                "shape": ["N", 6],
                "dtype": "float32",
                "order": [
                    "nose_quality",
                    "face_quality",
                    "detection_confidence",
                    "sam_predicted_iou",
                    "nose_resolution",
                    "face_resolution",
                ],
            },
            "viewpoint_signals": {
                "shape": ["N", 4],
                "dtype": "float32",
                "order": [
                    "nose_horizontal",
                    "mouth_horizontal",
                    "nose_mouth_distance_asymmetry",
                    "nose_vertical",
                ],
            },
            "branch_available": {
                "shape": ["N", 2],
                "dtype": "bool",
                "order": ["nose", "face"],
            },
        },
        "outputs": {
            "embedding": {
                "shape": ["N", identity_model.fused_dim],
                "l2_normalized": True,
            },
            "nose_embedding": {"shape": ["N", 2048], "l2_normalized": True},
            "face_embedding": {"shape": ["N", 512], "l2_normalized": True},
            "fusion_weights": {"shape": ["N", 2]},
            "joint_weights": {"shape": ["N", 2]},
            "viewpoint_frontality": {"shape": ["N"]},
        },
        "excluded": [
            "AnyFace detection",
            "SAM2 segmentation",
            "EXIF handling",
            "rotated ROI extraction",
            "800-class training-only classifier",
        ],
    }
    validation["onnx_sha256"] = model_hash
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(
        output_dir,
        identity_model.fused_dim,
        identity_model.fusion_mode,
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "sha256": model_hash,
                "bytes": model_path.stat().st_size,
                "metadata": str(metadata_path),
                "validation": str(validation_path),
                "wrapper_vs_full_pytorch_max_abs": parity_delta,
                "runtime_runs": validation["runs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
