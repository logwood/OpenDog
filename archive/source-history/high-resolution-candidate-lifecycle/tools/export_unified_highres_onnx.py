#!/usr/bin/env python3
"""Export and validate the dynamic one-graph UnifiedPetReID V4 candidate.

The exporter deliberately uses the legacy ONNX path.  PyTorch 2.11's dynamo
exporter currently specializes the Shape/ GridSample arithmetic used by V4;
the legacy exporter preserves symbolic height and width.  The resulting
artifact is still checked with ONNX Runtime on CPU and CUDA before it is
published.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch


for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import (  # noqa: E402
    configure_strict_cuda_precision,
    sha256_file,
)
from pet_id.unified_highres import (  # noqa: E402
    MODEL_TYPE,
    UnifiedHighResolutionPetReIDExport,
    build_highres_from_checkpoint,
)
from pet_id.unified_highres_data import (  # noqa: E402
    HIGHRES_MIN_INPUT_SIDE,
    load_raw_rgb,
    validate_highres_dimensions,
)
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402
from pet_id.unified_runtime import (  # noqa: E402
    UNIFIED_ONNX_INPUT_NAMES,
    UNIFIED_ONNX_OUTPUT_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=3e-3)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument(
        "--shape",
        dest="shapes",
        action="append",
        default=[],
        metavar="HxW",
        help="Additional dynamic validation shape, for example 4032x3024",
    )
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_shape(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        raise ValueError(f"Shape must be HxW, got {value!r}")
    left, right = text.split("x", 1)
    height, width = int(left), int(right)
    validate_highres_dimensions(
        height,
        width,
        minimum_side=HIGHRES_MIN_INPUT_SIDE,
        maximum_side=10_000_000,
    )
    return height, width


def validate_protocol_lock(path: Path, validation_manifest: Path | None) -> dict | None:
    if path is None:
        return None
    lock = load_json(path)
    if lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 protocol lock")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 protocol must remain locked and unscored during export")
    policy = lock.get("policy", {})
    required = (
        "v4_identity_disjoint",
        "exact_image_disjoint",
        "blind_single_candidate_attempt",
        "blind_training_forbidden",
        "blind_model_selection_forbidden",
        "blind_features_must_not_be_persisted",
        "failed_candidate_keeps_v3_default",
    )
    for key in required:
        if policy.get(key) is not True:
            raise RuntimeError(f"V4 protocol policy is missing {key}")
    if validation_manifest is not None:
        development = lock["splits"]["development"]
        if validation_manifest.resolve() != Path(development["path"]).resolve():
            raise RuntimeError("Validation manifest must be the locked V4 development split")
        if sha256_file(validation_manifest) != development["sha256"]:
            raise RuntimeError("Validation manifest differs from the V4 protocol lock")
    return lock


def validate_checkpoint_provenance(checkpoint: dict, checkpoint_path: Path) -> None:
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("Unexpected V4 checkpoint schema")
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Checkpoint is not a UnifiedHighResolutionPetReID checkpoint")
    training = checkpoint.get("training") or {}
    if training.get("blind_data_used") is not False:
        raise RuntimeError("V4 checkpoint training provenance is not blind-safe")
    source = checkpoint.get("sources", {}).get("parent_v3_checkpoint", {})
    parent_path = Path(source.get("path", "")).expanduser().resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    if sha256_file(parent_path) != source.get("sha256"):
        raise RuntimeError("V3 parent checkpoint hash differs from V4 checkpoint")


def select_manifest_samples(manifest_path: Path, *, maximum_side: int) -> list[torch.Tensor]:
    payload = load_json(manifest_path)
    if payload.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Validation manifest has the wrong protocol")
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("Validation manifest has no records")
    # Pick one record per identity, in manifest order.  The selected tensors
    # are only used for parity checks and are never written to disk.
    selected: list[torch.Tensor] = []
    seen: set[str] = set()
    for record in records:
        identity = str(record["identity"]).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        source = Path(record["source_path"]).expanduser().resolve()
        if sha256_file(source) != str(record["source_sha256"]).casefold():
            raise RuntimeError(f"Validation source hash differs: {source}")
        tensor, _ = load_raw_rgb(source, maximum_side=maximum_side)
        selected.append(tensor)
        if len(selected) >= 2:
            break
    if not selected:
        raise RuntimeError("Could not select a validation sample")
    return selected


def synthetic_samples(
    shapes: Iterable[tuple[int, int]],
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260901)
    return [
        torch.rand((3, height, width), generator=generator, dtype=torch.float32)
        .mul_(255.0)
        for height, width in shapes
    ]


def compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if expected.shape != actual.shape:
        raise RuntimeError(f"Output shape mismatch: {expected.shape} != {actual.shape}")
    difference = np.abs(expected - actual)
    cosine = (expected * actual).sum(axis=1) / np.maximum(
        np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1),
        1e-12,
    )
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
    }


def validate_graph_contract(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [
        item.name for item in model.graph.input if item.name not in initializer_names
    ]
    graph_outputs = [item.name for item in model.graph.output]
    if graph_inputs != list(UNIFIED_ONNX_INPUT_NAMES):
        raise RuntimeError(f"Unexpected V4 graph inputs: {graph_inputs}")
    if graph_outputs != list(UNIFIED_ONNX_OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected V4 graph outputs: {graph_outputs}")
    input_value = model.graph.input[0]
    input_shape = input_value.type.tensor_type.shape.dim
    if len(input_shape) != 4 or input_shape[1].dim_value != 3:
        raise RuntimeError("V4 graph input must be [N,3,H,W]")
    if input_shape[2].dim_value or input_shape[3].dim_value:
        raise RuntimeError("V4 graph height and width must remain dynamic")
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_shape) != 2 or output_shape[1].dim_value != 512:
        raise RuntimeError(
            "V4 graph output second dimension must be static 512"
        )
    if output_shape[1].dim_param:
        raise RuntimeError("V4 graph output embedding dimension must not be symbolic")
    external_tensors = [
        item.name
        for item in model.graph.initializer
        if item.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external_tensors:
        raise RuntimeError("V4 ONNX unexpectedly uses external tensor files")
    return {
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "input_shape": [
            dim.dim_param or int(dim.dim_value) for dim in input_shape
        ],
        "output_shape": [
            dim.dim_param or int(dim.dim_value) for dim in output_shape
        ],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "external_tensor_files": [],
    }


def provider_list(provider: str) -> list:
    if provider == "CPUExecutionProvider":
        return [provider]
    return [
        (provider, {"use_tf32": "0"}),
        "CPUExecutionProvider",
    ]
def freeze_output_embedding_dimension(path: Path) -> None:
    """Make the invariant 512-D output explicit in the ONNX type."""

    model = onnx.load(str(path), load_external_data=False)
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    if len(dimensions) != 2:
        raise RuntimeError("V4 graph output is not rank two")
    dimension = dimensions[1]
    if dimension.dim_value not in (0, 512):
        raise RuntimeError(
            f"Unexpected exported embedding dimension: {dimension.dim_value}"
        )
    dimension.ClearField("dim_param")
    dimension.dim_value = 512
    onnx.save(model, str(path), save_as_external_data=False)




def validate_provider(
    model_path: Path,
    samples: list[torch.Tensor],
    expected: dict[tuple[int, int], np.ndarray],
    *,
    provider: str,
    max_abs_tolerance: float,
    minimum_cosine: float,
) -> dict:
    session = ort.InferenceSession(
        str(model_path),
        providers=provider_list(provider),
    )
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Refusing ONNX provider fallback: {session.get_providers()}")
    if tuple(item.name for item in session.get_inputs()) != UNIFIED_ONNX_INPUT_NAMES:
        raise RuntimeError("V4 ONNX input contract is not exactly ('rgb',)")
    if tuple(item.name for item in session.get_outputs()) != UNIFIED_ONNX_OUTPUT_NAMES:
        raise RuntimeError("V4 ONNX output contract is not exactly ('embedding',)")
    runs = []
    for sample_index, sample in enumerate(samples):
        array = sample.numpy().astype(np.float32, copy=False)[None]
        # Run both batch=1 and (for the first shape) batch=2 to exercise the
        # symbolic batch dimension without creating another large sample.
        batches = [array]
        if sample_index == 0:
            batches.append(np.concatenate((array, array), axis=0))
        for batch in batches:
            actual = session.run(
                list(UNIFIED_ONNX_OUTPUT_NAMES),
                {"rgb": batch},
            )[0]
            key = (sample_index, batch.shape[0])
            if key not in expected:
                raise RuntimeError(f"Missing PyTorch parity sample for {key}")
            parity = compare(expected[key], actual)
            if parity["max_abs_error"] > max_abs_tolerance:
                raise RuntimeError(f"{provider} max error exceeds tolerance: {parity}")
            if parity["minimum_cosine"] < minimum_cosine:
                raise RuntimeError(f"{provider} cosine is below tolerance: {parity}")
            norms = np.linalg.norm(actual, axis=1)
            if not np.allclose(norms, 1.0, atol=3e-4, rtol=3e-4):
                raise RuntimeError(f"{provider} output is not L2 normalized: {norms}")
            runs.append(
                {
                    "batch_size": int(batch.shape[0]),
                    "height": int(batch.shape[2]),
                    "width": int(batch.shape[3]),
                    "embedding": parity,
                    "norm_range": [float(norms.min()), float(norms.max())],
                }
            )
    active = session.get_providers()
    del session
    gc.collect()
    return {"provider": provider, "provider_chain": active, "runs": runs}


def main() -> None:
    args = parse_args()
    if args.validation_batch_size < 1:
        raise ValueError("validation-batch-size must be positive")
    if args.max_dynamic_batch < args.validation_batch_size:
        raise ValueError("max-dynamic-batch is smaller than validation-batch-size")
    if args.opset < 17:
        raise ValueError("V4 export requires an ONNX opset of at least 17")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "unified_pet_reid_v4.onnx"
    temporary_path = output_dir / "unified_pet_reid_v4.exporting.onnx"
    metadata_path = output_dir / "metadata.json"
    validation_path = output_dir / "validation.json"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(model_path)
    if temporary_path.exists():
        temporary_path.unlink()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_provenance(checkpoint, checkpoint_path)
    validation_manifest = (
        args.validation_manifest.expanduser().resolve()
        if args.validation_manifest is not None
        else None
    )
    lock = validate_protocol_lock(
        args.protocol_lock.expanduser().resolve() if args.protocol_lock else None,
        validation_manifest,
    )

    precision = configure_strict_cuda_precision()
    device = torch.device(args.device)
    model, loaded_checkpoint = build_highres_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=True,
    )
    if loaded_checkpoint.get("training", {}).get("blind_data_used") is not False:
        raise RuntimeError("Loaded V4 checkpoint is not blind-safe")
    model.eval()
    wrapper = UnifiedHighResolutionPetReIDExport(model).to(device).eval()
    maximum_side = int(
        loaded_checkpoint["model_config"].get("maximum_input_side", 4096)
    )

    samples: list[torch.Tensor] = []
    sample_source = None
    if validation_manifest is not None:
        samples.extend(
            select_manifest_samples(validation_manifest, maximum_side=maximum_side)
        )
        sample_source = str(validation_manifest)
    shapes = [parse_shape(value) for value in args.shapes]
    if not shapes:
        shapes = [(64, 64), (208, 126), (800, 600), (4032, 3024)]
    for tensor in synthetic_samples(shapes):
        # Avoid an input larger than the declared deployment cap in a formal
        # validation run; the explicit shape flag remains useful for testing
        # an independently chosen cap.
        if max(tensor.shape[-2:]) <= maximum_side:
            samples.append(tensor)
    if not samples:
        raise RuntimeError("No validation samples were available")

    # The first sample is the trace input.  Its spatial dimensions are
    # intentionally non-square when a manifest is supplied, proving that the
    # legacy exporter did not specialize H/W.
    export_input = samples[0][None].to(device).contiguous()
    expected: dict[tuple[int, int], np.ndarray] = {}
    with torch.inference_mode():
        for sample_index, sample in enumerate(samples):
            one = sample[None].to(device).contiguous()
            value = wrapper(one).float().cpu().numpy()
            expected[(sample_index, 1)] = value
            if sample_index == 0:
                pair = torch.cat((one, one), dim=0)
                pair_value = wrapper(pair).float().cpu().numpy()
                expected[(sample_index, 2)] = pair_value
    if expected[(0, 1)].shape != (1, 512):
        raise RuntimeError("Unexpected PyTorch V4 output contract")

    # Legacy exporter uses dynamic_axes rather than dynamic_shapes.  Keep the
    # embedding dimension static: runtime consumers can reject malformed
    # graphs before allocating any gallery state.
    torch.onnx.export(
        wrapper,
        (export_input,),
        temporary_path,
        input_names=list(UNIFIED_ONNX_INPUT_NAMES),
        output_names=list(UNIFIED_ONNX_OUTPUT_NAMES),
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        external_data=False,
        dynamic_axes={
            "rgb": {0: "batch", 2: "height", 3: "width"},
            "embedding": {0: "batch"},
        },
    )
    freeze_output_embedding_dimension(temporary_path)
    graph_contract = validate_graph_contract(temporary_path)
    del wrapper, model, export_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    providers: list[str] = []
    if not args.skip_cpu:
        providers.append("CPUExecutionProvider")
    if not args.skip_cuda:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA ONNX Runtime provider is unavailable")
        providers.append("CUDAExecutionProvider")
    # CPU first gives a deterministic fallback report and releases its graph
    # before the CUDA session is created on memory-constrained machines.
    providers = [
        item for item in ("CPUExecutionProvider", "CUDAExecutionProvider")
        if item in providers
    ]
    provider_reports = [
        validate_provider(
            temporary_path,
            samples,
            expected,
            provider=provider,
            max_abs_tolerance=args.max_abs_tolerance,
            minimum_cosine=args.minimum_cosine,
        )
        for provider in providers
    ]
    if not provider_reports:
        raise RuntimeError("At least one ONNX Runtime provider must be validated")

    if model_path.exists():
        model_path.unlink()
    os.replace(temporary_path, model_path)
    model_hash = sha256_file(model_path)
    model_bytes = model_path.stat().st_size
    runtime_contract = dict(loaded_checkpoint["runtime_contract"])
    runtime_contract["outputs"] = {
        "embedding": {
            "dtype": "float32",
            "shape": ["N", 512],
            "l2_normalized": True,
        }
    }
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": MODEL_TYPE,
        "model": str(model_path),
        "onnx_sha256": model_hash,
        "onnx_bytes": model_bytes,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_schema_version": loaded_checkpoint["schema_version"],
        "model_config": loaded_checkpoint["model_config"],
        "preprocessing": loaded_checkpoint["preprocessing"],
        "runtime_contract": runtime_contract,
        "inputs": runtime_contract["inputs"],
        "outputs": runtime_contract["outputs"],
        "external_models": [],
        "runtime_forbidden_dependencies": [
            "AnyFace",
            "SAM2",
            "Faster R-CNN",
            "MegaDescriptor",
            "any independent localization or identity ONNX session",
        ],
        "dynamic_input": {
            "batch": {"minimum": 1, "declared_maximum": args.max_dynamic_batch},
            "height": {"minimum": HIGHRES_MIN_INPUT_SIDE, "maximum": maximum_side},
            "width": {"minimum": HIGHRES_MIN_INPUT_SIDE, "maximum": maximum_side},
            "validated_shapes": [
                [int(sample.shape[-2]), int(sample.shape[-1])] for sample in samples
            ],
        },
        "protocol_lock": (
            {
                "path": str(args.protocol_lock.expanduser().resolve()),
                "sha256": sha256_file(args.protocol_lock.expanduser().resolve()),
                "protocol_name": lock["protocol_name"],
            }
            if args.protocol_lock is not None
            else None
        ),
        "validation_manifest": (
            {
                "path": str(validation_manifest),
                "sha256": sha256_file(validation_manifest),
            }
            if validation_manifest is not None
            else None
        ),
        "promotion_status": "experimental_until_v4_development_and_blind_pass",
        "default_backend_changed": False,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "exporter": "torch.onnx.legacy",
        "pytorch_cuda_precision": precision,
    }
    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "unified_v4_dynamic_high_resolution_onnx_export_validation",
        "blind_data_used": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "protocol_lock": metadata["protocol_lock"],
        "validation_manifest": metadata["validation_manifest"],
        "sample_source": sample_source,
        "sample_shapes": [
            [int(sample.shape[-2]), int(sample.shape[-1])] for sample in samples
        ],
        "thresholds": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "minimum_cosine": args.minimum_cosine,
        },
        "onnx_checker": "passed",
        "single_graph_contract": graph_contract,
        "providers": provider_reports,
        "onnx": {
            "path": str(model_path),
            "sha256": model_hash,
            "bytes": model_bytes,
        },
        "output_shape_static_512": True,
        "passed": True,
        "default_backend_changed": False,
        "pytorch_cuda_precision": precision,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "metadata": str(metadata_path),
                "validation": str(validation_path),
                "onnx_sha256": model_hash,
                "onnx_bytes": model_bytes,
                "contract": graph_contract,
                "providers": provider_reports,
                "blind_data_used": False,
                "default_backend_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
