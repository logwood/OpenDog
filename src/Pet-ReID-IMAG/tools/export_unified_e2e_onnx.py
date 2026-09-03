#!/usr/bin/env python3
"""Export the production UnifiedPetReID checkpoint as a raw-RGB end-to-end ONNX graph.

The historical fixed-input artifact accepts a 1280x1280 tensor that has already been
letterboxed by Python. This exporter keeps that artifact immutable and emits
the replacement contract instead: one float32 RGB pixel tensor with dynamic
batch/height/width, graph-internal letterboxing, the complete learned model,
and one L2-normalized 512-D embedding. Image decoding/EXIF handling remains
at the transport boundary and is intentionally not represented as an ONNX
dependency.

Only a development manifest may be used. The script refuses paths or
manifest declarations that look like a blind split and records the resulting
provenance explicitly so a deployment can audit that no blind data entered the
export or its validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
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

from pet_id.unified_data import letterbox_rgb  # noqa: E402
from pet_id.unified_e2e import UnifiedEndToEndPetReIDExport  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    build_external_joint_from_checkpoint,
    configure_strict_cuda_precision,
)
from pet_id.unified_runtime import (  # noqa: E402
    UNIFIED_ONNX_INPUT_NAMES,
    UNIFIED_ONNX_OUTPUT_NAMES,
)
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.release_compatibility import (  # noqa: E402
    acceptance_path,
    acceptance_protocol_name,
    historical_run_path,
)


ACCEPTANCE_PROTOCOL = acceptance_protocol_name("external-development")
PRODUCTION_PROFILE = get_runtime_profile("production")
DEFAULT_PACKAGE = PRODUCTION_PROFILE.onnx.parent.parent.parent
DEFAULT_CHECKPOINT = DEFAULT_PACKAGE / "model_final.pth"
DEFAULT_ACCEPTANCE = acceptance_path(WORKSPACE, "external-development")
DEFAULT_MANIFEST = historical_run_path(WORKSPACE, "external-development-manifest")
DEFAULT_OUTPUT = DEFAULT_PACKAGE / "onnx/e2e"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--max-input-side", type=int, default=10000)
    parser.add_argument("--minimum-input-side", type=int, default=2)
    parser.add_argument("--max-abs-tolerance", type=float, default=3e-3)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument(
        "--minimum-legacy-cosine",
        type=float,
        default=0.995,
        help="minimum cosine against Python's legacy cv2 letterbox semantic",
    )
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _assert_development_only(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Reject blind data before opening any image in the manifest."""

    path_text = str(manifest_path).casefold()
    split_text = str(manifest.get("protocol_split", "")).casefold()
    if "blind" in path_text or "blind" in split_text:
        raise RuntimeError(
            "The E2E exporter only accepts a development manifest; blind data is forbidden"
        )
    if split_text and split_text not in {"development", "dev", "validation"}:
        raise RuntimeError(
            f"Unsupported manifest split for E2E export: {manifest.get('protocol_split')!r}"
        )
    if manifest.get("blind_data_used") is True:
        raise RuntimeError("Manifest is marked as using blind data")


def _read_rgb(record: dict[str, Any]) -> np.ndarray:
    source = Path(str(record.get("source_path", ""))).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_hash = str(record.get("source_sha256", ""))
    if expected_hash and sha256_file(source).casefold() != expected_hash.casefold():
        raise RuntimeError(f"Development image hash mismatch: {source}")
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode development image: {source}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Development image is not RGB: {source}")
    if min(int(rgb.shape[0]), int(rgb.shape[1])) < 2:
        raise RuntimeError(f"Development image is too small: {source}")
    return np.ascontiguousarray(rgb)


def _select_records(manifest: dict[str, Any], count: int) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Development manifest has no records")
    selected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = str(record.get("identity", "")).casefold()
        if not identity or identity in identities:
            continue
        selected.append(record)
        identities.add(identity)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Development manifest has fewer than {count} identities")
    return selected


def _compare(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if expected.shape != actual.shape:
        raise RuntimeError(f"Embedding shape mismatch: {expected.shape} vs {actual.shape}")
    difference = np.abs(expected - actual)
    expected_norm = np.linalg.norm(expected, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    cosine = (expected * actual).sum(axis=1) / np.maximum(
        expected_norm * actual_norm, 1e-12
    )
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min(initial=1.0)),
        "mean_cosine": float(cosine.mean()),
        "expected_norm_range": [float(expected_norm.min()), float(expected_norm.max())],
        "actual_norm_range": [float(actual_norm.min()), float(actual_norm.max())],
    }


def _raw_batch(images: list[np.ndarray]) -> np.ndarray:
    shapes = {tuple(image.shape[:2]) for image in images}
    if len(shapes) != 1:
        raise ValueError("All images in one ONNX batch must have the same H/W")
    return np.ascontiguousarray(
        np.stack([image.transpose(2, 0, 1) for image in images]).astype(
            np.float32, copy=False
        )
    )


def _legacy_batch(images: list[np.ndarray], input_size: int) -> np.ndarray:
    boxed = [
        letterbox_rgb(
            image,
            size=input_size,
            fill_value=0,
            allow_upscale=False,
        )[0]
        for image in images
    ]
    return np.ascontiguousarray(
        np.stack([image.transpose(2, 0, 1) for image in boxed]).astype(
            np.float32, copy=False
        )
    )


def _graph_contract(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [
        item for item in model.graph.input if item.name not in initializer_names
    ]
    graph_outputs = list(model.graph.output)
    input_names = [item.name for item in graph_inputs]
    output_names = [item.name for item in graph_outputs]
    if input_names != list(UNIFIED_ONNX_INPUT_NAMES):
        raise RuntimeError(f"E2E graph must have exactly one rgb input: {input_names}")
    if output_names != list(UNIFIED_ONNX_OUTPUT_NAMES):
        raise RuntimeError(
            f"E2E graph must have exactly one embedding output: {output_names}"
        )
    input_type = graph_inputs[0].type.tensor_type
    output_type = graph_outputs[0].type.tensor_type
    if input_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("E2E rgb input is not float32")
    if output_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("E2E embedding output is not float32")
    input_dims = list(input_type.shape.dim)
    output_dims = list(output_type.shape.dim)
    if len(input_dims) != 4 or input_dims[1].dim_value != 3:
        raise RuntimeError("E2E input must be [N,3,H,W]")
    if not input_dims[0].dim_param or not input_dims[2].dim_param or not input_dims[3].dim_param:
        raise RuntimeError("E2E batch/height/width dimensions must remain dynamic")
    if len(output_dims) != 2 or output_dims[1].dim_value != 512:
        raise RuntimeError("E2E output must be [N,512]")
    external_tensors = [
        item.name
        for item in model.graph.initializer
        if item.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external_tensors:
        raise RuntimeError(f"E2E graph unexpectedly uses external tensors: {external_tensors[:5]}")
    sidecars = sorted(path.parent.glob(path.name + ".data*"))
    if sidecars:
        raise RuntimeError(f"E2E export left external-data sidecars: {sidecars}")
    op_types = sorted({node.op_type for node in model.graph.node})
    return {
        "inputs": input_names,
        "outputs": output_names,
        "input_shape": [
            dim.dim_param if dim.dim_param else int(dim.dim_value)
            for dim in input_dims
        ],
        "output_shape": [
            dim.dim_param if dim.dim_param else int(dim.dim_value)
            for dim in output_dims
        ],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "op_types": op_types,
        "external_tensor_files": [],
        "onnx_checker": "passed",
    }


def _canonicalize_embedding_shape(path: Path) -> None:
    """Make the statically known 512-wide output explicit in graph metadata.

    The legacy exporter can leave the second output dimension as a generated
    symbolic name even though the final linear layer is statically 512-wide.
    Shape inference verifies that fact; we then write only the harmless output
    shape annotation back, keeping dynamic batch/H/W input dimensions intact.
    """

    model = onnx.load(str(path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    inferred_dims = inferred.graph.output[0].type.tensor_type.shape.dim
    if len(inferred_dims) != 2 or inferred_dims[1].dim_value != 512:
        raise RuntimeError("Shape inference did not prove a [N,512] embedding")
    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    output_dims[1].ClearField("dim_param")
    output_dims[1].dim_value = 512
    onnx.save(model, str(path))


def _providers(provider: str) -> list[Any]:
    if provider == "CPUExecutionProvider":
        return [provider]
    return [
        (
            "CUDAExecutionProvider",
            {
                "use_tf32": "0",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


def _validate_ort(
    model_path: Path,
    cases: list[tuple[str, np.ndarray]],
    expected: dict[str, np.ndarray],
    *,
    provider: str,
    max_abs_tolerance: float,
    minimum_cosine: float,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(model_path), providers=_providers(provider))
    active = session.get_providers()
    if not active or active[0] != provider:
        raise RuntimeError(f"Refusing ONNX provider fallback: {active}")
    if tuple(item.name for item in session.get_inputs()) != UNIFIED_ONNX_INPUT_NAMES:
        raise RuntimeError("ORT input contract is not ('rgb',)")
    if tuple(item.name for item in session.get_outputs()) != UNIFIED_ONNX_OUTPUT_NAMES:
        raise RuntimeError("ORT output contract is not ('embedding',)")
    reports = []
    for name, value in cases:
        actual = session.run(
            list(UNIFIED_ONNX_OUTPUT_NAMES),
            {"rgb": np.ascontiguousarray(value, dtype=np.float32)},
        )[0]
        parity = _compare(expected[name], actual)
        if parity["max_abs_error"] > max_abs_tolerance:
            raise RuntimeError(f"{provider} max error exceeds tolerance for {name}: {parity}")
        if parity["minimum_cosine"] < minimum_cosine:
            raise RuntimeError(f"{provider} cosine is below tolerance for {name}: {parity}")
        if not np.allclose(parity["actual_norm_range"], [1.0, 1.0], atol=3e-3, rtol=3e-3):
            raise RuntimeError(f"{provider} output is not L2-normalized for {name}: {parity}")
        reports.append({"name": name, "input_shape": list(value.shape), **parity})
    del session
    gc.collect()
    return {"provider": provider, "provider_chain": active, "runs": reports}


def _pytorch_expected(
    wrapper: torch.nn.Module,
    cases: list[tuple[str, np.ndarray]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    expected: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, value in cases:
            tensor = torch.from_numpy(value).to(device).contiguous()
            output = wrapper(tensor).detach().float().cpu().numpy()
            if output.shape != (value.shape[0], 512):
                raise RuntimeError(f"Unexpected PyTorch E2E output for {name}: {output.shape}")
            if not np.allclose(np.linalg.norm(output, axis=1), 1.0, atol=3e-3, rtol=3e-3):
                raise RuntimeError(f"PyTorch E2E output is not normalized for {name}")
            expected[name] = output
    return expected


def main() -> None:
    args = parse_args()
    if args.validation_batch_size < 2:
        raise ValueError("validation batch size must be at least two")
    if args.max_dynamic_batch < args.validation_batch_size:
        raise ValueError("max dynamic batch is smaller than validation batch")
    if args.minimum_input_side < 2:
        raise ValueError("minimum input side must be at least two")
    if args.max_input_side < args.minimum_input_side:
        raise ValueError("maximum input side must be >= minimum input side")
    if args.opset < 16:
        raise ValueError("opset must support AffineGrid and GridSample")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (checkpoint_path, acceptance_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    acceptance = _load_json(acceptance_path)
    if acceptance.get("protocol_name") != ACCEPTANCE_PROTOCOL:
        raise RuntimeError("Acceptance file is not the locked production protocol")
    manifest = _load_json(manifest_path)
    _assert_development_only(manifest_path, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "unified_pet_reid.onnx"
    temporary_path = output_dir / "unified_pet_reid.exporting.onnx"
    metadata_path = output_dir / "metadata.json"
    validation_path = output_dir / "validation.json"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(model_path)
    if temporary_path.exists():
        temporary_path.unlink()

    precision = configure_strict_cuda_precision()
    device = torch.device(args.device)
    model, checkpoint = build_external_joint_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=True,
    )
    if checkpoint.get("training", {}).get("blind_data_used") is not False:
        raise RuntimeError("Checkpoint provenance is not blind-safe")
    if checkpoint.get("model_type") != "unified_external_joint_pet_reid":
        raise RuntimeError("Checkpoint is not the selected production model")
    input_size = int(model.input_size)
    records = _select_records(manifest, max(args.validation_batch_size, 3))
    images = [_read_rgb(record) for record in records]

    cases: list[tuple[str, np.ndarray]] = []
    for index, image in enumerate(images):
        cases.append((f"native_{index}", _raw_batch([image])))
    # The first two development records may have different dimensions. A
    # repeated batch still proves dynamic N without introducing graph-external
    # resizing; native cases above prove dynamic H/W.
    cases.append(("batch", _raw_batch([images[0]] * args.validation_batch_size)))

    wrapper = UnifiedEndToEndPetReIDExport(model).to(device).eval()
    export_input = torch.from_numpy(cases[-1][1]).to(device).contiguous()
    with torch.inference_mode():
        export_output = wrapper(export_input)
    if tuple(export_output.shape) != (export_input.shape[0], 512):
        raise RuntimeError(f"Unexpected export input/output shape: {export_output.shape}")

    # The legacy semantic is a development reference only. It verifies that
    # moving geometry into the graph did not silently change the model meaning.
    native_cases = [(name, value) for name, value in cases if name.startswith("native_")]
    legacy_cases = [
        (name, _legacy_batch([images[int(name.rsplit("_", 1)[-1])]], input_size))
        for name, _ in native_cases
    ]
    legacy_expected = _pytorch_expected(model, legacy_cases, device)
    raw_expected = _pytorch_expected(wrapper, native_cases, device)
    legacy_reports = []
    for (name, _), (raw_name, _) in zip(legacy_cases, native_cases):
        parity = _compare(legacy_expected[name], raw_expected[raw_name])
        if parity["minimum_cosine"] < args.minimum_legacy_cosine:
            raise RuntimeError(
                f"Graph letterbox changed development semantics for {name}: {parity}"
            )
        legacy_reports.append({"name": name, **parity})

    # The legacy exporter preserves Shape->AffineGrid->GridSample as symbolic
    # H/W nodes on the installed PyTorch/ONNX versions.
    torch.onnx.export(
        wrapper,
        (export_input,),
        temporary_path,
        input_names=list(UNIFIED_ONNX_INPUT_NAMES),
        output_names=list(UNIFIED_ONNX_OUTPUT_NAMES),
        opset_version=args.opset,
        dynamo=False,
        dynamic_axes={
            "rgb": {0: "batch", 2: "height", 3: "width"},
            "embedding": {0: "batch"},
        },
        external_data=False,
        do_constant_folding=True,
    )
    _canonicalize_embedding_shape(temporary_path)
    graph_contract = _graph_contract(temporary_path)
    del wrapper, model, export_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # CPU parity uses a CPU PyTorch reference. CUDA parity, when requested,
    # uses a CUDA reference so provider-specific drift is visible.
    cpu_model, _ = build_external_joint_from_checkpoint(
        checkpoint_path,
        device="cpu",
        verify_sources=True,
    )
    cpu_wrapper = UnifiedEndToEndPetReIDExport(cpu_model).eval()
    cpu_expected = _pytorch_expected(cpu_wrapper, cases, torch.device("cpu"))
    provider_reports = [
        _validate_ort(
            temporary_path,
            cases,
            cpu_expected,
            provider="CPUExecutionProvider",
            max_abs_tolerance=args.max_abs_tolerance,
            minimum_cosine=args.minimum_cosine,
        )
    ]
    if not args.skip_cuda:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA ONNX Runtime provider is unavailable; use --skip-cuda explicitly")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA provider is available but torch CUDA is unavailable")
        cuda_model, _ = build_external_joint_from_checkpoint(
            checkpoint_path,
            device="cuda",
            verify_sources=True,
        )
        cuda_wrapper = UnifiedEndToEndPetReIDExport(cuda_model).cuda().eval()
        cuda_expected = _pytorch_expected(cuda_wrapper, cases, torch.device("cuda"))
        provider_reports.append(
            _validate_ort(
                temporary_path,
                cases,
                cuda_expected,
                provider="CUDAExecutionProvider",
                max_abs_tolerance=args.max_abs_tolerance,
                minimum_cosine=args.minimum_cosine,
            )
        )
        del cuda_wrapper, cuda_model
        torch.cuda.empty_cache()

    del cpu_wrapper, cpu_model
    gc.collect()

    if model_path.exists():
        model_path.unlink()
    os.replace(temporary_path, model_path)

    model_hash = sha256_file(model_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_hash = sha256_file(manifest_path)
    graph_preprocessing = [
        "raw_float32_rgb_pixels_0_255",
        "centered_black_letterbox_inside_onnx",
        "imagenet_pixel_normalization_inside_onnx",
        "learned_geometry_and_rotated_crops_inside_onnx",
        "identity_and_fusion_inside_onnx",
        "l2_normalization_inside_onnx",
    ]
    input_contract = {
        "dtype": "float32",
        "shape": ["N", 3, "H", "W"],
        "raw_pixels": True,
        "value_range": [0, 255],
        "height_width_minimum": args.minimum_input_side,
        "height_width_maximum": args.max_input_side,
    }
    output_contract = {
        "dtype": "float32",
        "shape": ["N", 512],
        "l2_normalized": True,
    }
    metadata = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_external_joint_pet_reid",
        "model": str(model_path),
        "onnx_sha256": model_hash,
        "onnx_bytes": model_path.stat().st_size,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_hash,
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "model_config": checkpoint.get("model_config", {}),
        "raw_spatial_input": True,
        "graph_preprocessing": graph_preprocessing,
        "preprocessing": {
            "input_range": [0, 255],
            "color_order": "RGB",
            "letterbox": "centered_black",
            "letterbox_allow_upscale": False,
            "raw_spatial_input": True,
            "graph_internal": graph_preprocessing,
            "transport_boundary": [
                "JPEG/PNG decode",
                "EXIF orientation",
                "BGR_to_RGB conversion",
            ],
        },
        "inputs": {"rgb": input_contract},
        "outputs": {"embedding": output_contract},
        "runtime_contract": {
            "inputs": {"rgb": input_contract},
            "outputs": {"embedding": output_contract},
            "external_models": [],
        },
        "external_models": [],
        "runtime_forbidden_dependencies": [
            "AnyFace",
            "SAM2",
            "Faster R-CNN",
            "MegaDescriptor",
            "any independent localization or identity ONNX session",
        ],
        "dynamic_batch": {
            "minimum": 1,
            "declared_maximum": args.max_dynamic_batch,
            "validated": sorted({int(value.shape[0]) for _, value in cases}),
        },
        "dynamic_spatial": {
            "height_symbol": "height",
            "width_symbol": "width",
            "validated_shapes": [list(value.shape[-2:]) for _, value in cases],
        },
        "provenance": {
            "protocol_name": ACCEPTANCE_PROTOCOL,
            "acceptance": {
                "path": str(acceptance_path),
                "sha256": sha256_file(acceptance_path),
            },
            "development_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_hash,
                "protocol_split": manifest.get("protocol_split", "development"),
                "records_considered": len(manifest.get("records", [])),
            },
            "blind_data_used": False,
            "blind_data_paths": [],
            "export_script": str(Path(__file__).resolve()),
        },
        "blind_data_used": False,
        "promotion_status": "validated_e2e_development",
        "default_backend_changed": True,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "pytorch_cuda_precision": precision,
    }
    validation = {
        "schema_version": 2,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "blind_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "development_manifest": str(manifest_path),
        "development_manifest_sha256": manifest_hash,
        "sample_identities": [str(record.get("identity", "")) for record in records],
        "single_graph_contract": graph_contract,
        "legacy_semantic_parity": {
            "minimum_cosine_threshold": args.minimum_legacy_cosine,
            "runs": legacy_reports,
        },
        "providers": provider_reports,
        "thresholds": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "minimum_cosine": args.minimum_cosine,
        },
        "onnx": {
            "path": str(model_path),
            "sha256": model_hash,
            "bytes": model_path.stat().st_size,
        },
        "pytorch_cuda_precision": precision,
    }
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "metadata": str(metadata_path),
                "validation": str(validation_path),
                "onnx_sha256": model_hash,
                "onnx_bytes": model_path.stat().st_size,
                "contract": graph_contract,
                "providers": provider_reports,
                "blind_data_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
