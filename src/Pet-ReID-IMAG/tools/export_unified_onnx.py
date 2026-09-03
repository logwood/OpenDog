#!/usr/bin/env python3
"""Export and verify one RGB-to-512D UnifiedPetReID ONNX graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified import UnifiedPetReIDExport
from pet_id.unified_data import UnifiedManifestDataset
from pet_id.unified_runtime import (
    UNIFIED_ONNX_INPUT_NAMES,
    UNIFIED_ONNX_OUTPUT_NAMES,
)
from pet_id.unified_training import (
    build_model_from_checkpoint,
    load_acceptance,
    model_configuration,
    sha256_file,
)
from pet_id.release_compatibility import acceptance_path, historical_run_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=acceptance_path(WORKSPACE, "legacy-training"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "shared-fusion-baseline")
        / "dev_validation_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--export-batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=2e-3)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_diverse_indices(dataset, count: int) -> list[int]:
    selected: list[int] = []
    identities: set[str] = set()
    for index, record in enumerate(dataset.records):
        identity = record["identity"].casefold()
        if identity in identities:
            continue
        selected.append(index)
        identities.add(identity)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Manifest has fewer than {count} identities")


def compare_embedding(
    expected: np.ndarray, actual: np.ndarray
) -> dict[str, float | list[int]]:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    difference = np.abs(expected - actual)
    cosine = (expected * actual).sum(axis=1) / np.maximum(
        np.linalg.norm(expected, axis=1)
        * np.linalg.norm(actual, axis=1),
        1e-12,
    )
    return {
        "shape": list(expected.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min()),
    }


def validate_ort(
    model_path: Path,
    wrapper: torch.nn.Module,
    samples: torch.Tensor,
    batch_sizes: list[int],
    *,
    max_abs_tolerance: float,
    minimum_cosine: float,
) -> dict:
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    if (
        tuple(item.name for item in session.get_inputs())
        != UNIFIED_ONNX_INPUT_NAMES
    ):
        raise RuntimeError("Exported ONNX input contract is not ('rgb',)")
    if (
        tuple(item.name for item in session.get_outputs())
        != UNIFIED_ONNX_OUTPUT_NAMES
    ):
        raise RuntimeError(
            "Exported ONNX output contract is not ('embedding',)"
        )
    runs = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            value = samples[:batch_size].contiguous()
            expected = wrapper(value).detach().cpu().numpy()
            actual = session.run(
                list(UNIFIED_ONNX_OUTPUT_NAMES),
                {"rgb": value.cpu().numpy()},
            )[0]
            metrics = compare_embedding(expected, actual)
            if metrics["max_abs_error"] > max_abs_tolerance:
                raise RuntimeError(
                    f"ONNX max error exceeds tolerance: {metrics}"
                )
            if metrics["minimum_cosine"] < minimum_cosine:
                raise RuntimeError(
                    f"ONNX cosine is below tolerance: {metrics}"
                )
            expected_norms = np.linalg.norm(expected, axis=1)
            actual_norms = np.linalg.norm(actual, axis=1)
            runs.append(
                {
                    "batch_size": batch_size,
                    "embedding": metrics,
                    "pytorch_norm_range": [
                        float(expected_norms.min()),
                        float(expected_norms.max()),
                    ],
                    "onnx_norm_range": [
                        float(actual_norms.min()),
                        float(actual_norms.max()),
                    ],
                }
            )
    return {
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "max_abs_tolerance": max_abs_tolerance,
        "minimum_cosine": minimum_cosine,
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    arcface_path = args.arcface_checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (
        checkpoint_path,
        arcface_path,
        acceptance_path,
        manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.export_batch_size, args.validation_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    if args.max_dynamic_batch < max(
        args.export_batch_size, args.validation_batch_size
    ):
        raise ValueError("max dynamic batch is smaller than validation batch")
    acceptance = load_acceptance(acceptance_path)
    expected_arcface = acceptance["source_weight_locks"][
        "dog_arcface_checkpoint"
    ]["sha256"]
    if sha256_file(arcface_path) != expected_arcface:
        raise RuntimeError("ArcFace checkpoint differs from acceptance lock")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "unified_pet_reid.onnx"
    temporary_path = output_dir / "unified_pet_reid.exporting.onnx"
    metadata_path = output_dir / "metadata.json"
    validation_path = output_dir / "validation.json"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(model_path)
    if temporary_path.exists():
        temporary_path.unlink()

    device = torch.device(args.device)
    model, checkpoint = build_model_from_checkpoint(
        checkpoint_path, arcface_path, device=device
    )
    model.eval()
    preprocessing = checkpoint.get("preprocessing", {})
    allow_upscale = bool(
        preprocessing.get("letterbox_allow_upscale", True)
    )
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=allow_upscale,
    )
    sample_count = max(
        args.export_batch_size, args.validation_batch_size
    )
    selected_indices = select_diverse_indices(dataset, sample_count)
    samples = torch.stack(
        [dataset[index]["rgb"] for index in selected_indices]
    )
    wrapper = UnifiedPetReIDExport(model).to(device).eval()
    export_input = samples[: args.export_batch_size].to(device).contiguous()
    with torch.inference_mode():
        output = wrapper(export_input)
    if output.shape != (args.export_batch_size, 512):
        raise RuntimeError(f"Unexpected PyTorch output: {output.shape}")

    batch_dimension = torch.export.Dim(
        "batch", min=1, max=args.max_dynamic_batch
    )
    torch.onnx.export(
        wrapper,
        (export_input,),
        temporary_path,
        input_names=list(UNIFIED_ONNX_INPUT_NAMES),
        output_names=list(UNIFIED_ONNX_OUTPUT_NAMES),
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        dynamic_shapes=({0: batch_dimension},),
        optimize=True,
    )
    onnx.checker.check_model(str(temporary_path))

    wrapper = wrapper.cpu().eval()
    validation = validate_ort(
        temporary_path,
        wrapper,
        samples,
        sorted({1, args.validation_batch_size}),
        max_abs_tolerance=args.max_abs_tolerance,
        minimum_cosine=args.minimum_cosine,
    )
    validation.update(
        {
            "schema_version": 1,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "onnx_checker": "passed",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "sample_indices": selected_indices,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
    )
    if model_path.exists():
        model_path.unlink()
    os.replace(temporary_path, model_path)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_pet_reid",
        "model": str(model_path),
        "onnx_sha256": sha256_file(model_path),
        "onnx_bytes": model_path.stat().st_size,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "arcface_checkpoint_sha256": expected_arcface,
        "model_config": model_configuration(model),
        "preprocessing": {
            "letterbox_allow_upscale": allow_upscale,
            "letterbox_fill": 0,
            "input_range": [0, 255],
        },
        "inputs": {
            "rgb": {
                "shape": ["N", 3, model.input_size, model.input_size],
                "dtype": "float32",
            }
        },
        "outputs": {
            "embedding": {
                "shape": ["N", 512],
                "dtype": "float32",
                "l2_normalized": True,
            }
        },
        "dynamic_batch": {
            "minimum": 1,
            "declared_maximum": args.max_dynamic_batch,
            "validated": sorted({1, args.validation_batch_size}),
        },
        "external_models": [],
        "runtime_forbidden_dependencies": acceptance["policy"][
            "runtime_forbidden_dependencies"
        ],
        "promotion_status": "experimental_until_acceptance_passes",
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
    }
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "metadata": str(metadata_path),
                "validation": str(validation_path),
                "onnx_sha256": metadata["onnx_sha256"],
                "contract": {
                    "inputs": metadata["inputs"],
                    "outputs": metadata["outputs"],
                    "external_models": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
