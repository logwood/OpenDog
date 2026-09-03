#!/usr/bin/env python3
"""Export and validate one external-joint RGB-to-512D UnifiedPetReID graph."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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

from pet_id.unified_external_data import UnifiedRawManifestDataset  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    UnifiedExternalJointPetReIDExport,
    build_external_joint_from_checkpoint,
    configure_strict_cuda_precision,
    sha256_file,
)
from pet_id.unified_runtime import (  # noqa: E402
    UNIFIED_ONNX_INPUT_NAMES,
    UNIFIED_ONNX_OUTPUT_NAMES,
)
from pet_id.release_compatibility import (  # noqa: E402
    acceptance_protocol_name,
    historical_purpose,
)


ACCEPTANCE_PROTOCOL = acceptance_protocol_name("external-development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--legacy-development-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=2e-3)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def locked_file(record: dict, name: str) -> Path:
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{name} hash differs from acceptance")
    return path


def validate_acceptance(path: Path) -> tuple[dict, Path]:
    acceptance = load_json(path)
    if acceptance.get("schema_version") != 3:
        raise RuntimeError("Unexpected external acceptance schema")
    if acceptance.get("protocol_name") != ACCEPTANCE_PROTOCOL:
        raise RuntimeError("Unexpected external acceptance protocol")
    locked_file(acceptance["protocol_lock"], "Protocol lock")
    locked_file(acceptance["baseline_lock"], "Baseline lock")
    manifest = locked_file(acceptance["development"], "Development manifest")
    return acceptance, manifest


def validate_development_evidence(
    report_path: Path,
    legacy_path: Path,
    *,
    checkpoint_path: Path,
    acceptance_path: Path,
    acceptance: dict,
) -> dict:
    checkpoint_hash = sha256_file(checkpoint_path)
    acceptance_hash = sha256_file(acceptance_path)
    report = load_json(report_path)
    if report.get("purpose") != "external_joint_development":
        raise RuntimeError("External development report has the wrong purpose")
    if report.get("blind_data_used") is not False or report.get("passed") is not True:
        raise RuntimeError("External development evidence did not pass")
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("External development checkpoint hash mismatch")
    if report.get("manifest", {}).get("sha256") != acceptance["development"]["sha256"]:
        raise RuntimeError("External development manifest hash mismatch")
    if report.get("acceptance", {}).get("sha256") != acceptance_hash:
        raise RuntimeError("External development acceptance hash mismatch")
    candidate = report.get("candidate", {})
    parent = report.get("parent_base", {})
    checks = {
        "semantic_top1": int(candidate.get("top1_correct", -1))
        >= int(acceptance["development"]["minimum_top1_correct"]),
        "semantic_top5": int(candidate.get("top5_correct", -1))
        >= int(acceptance["development"]["minimum_top5_correct"]),
        "parent_top1": int(candidate.get("top1_correct", -1))
        >= int(parent.get("top1_correct", 10**9)),
        "parent_top5": int(candidate.get("top5_correct", -1))
        >= int(parent.get("top5_correct", 10**9)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"External development noninferiority failed: {checks}")
    if not (
        int(candidate["top1_correct"]) > int(parent["top1_correct"])
        or int(candidate["top5_correct"]) > int(parent["top5_correct"])
    ):
        raise RuntimeError("Candidate has no real external development rank gain")

    legacy = load_json(legacy_path)
    if legacy.get("purpose") != historical_purpose("legacy-external-joint-guard"):
        raise RuntimeError("Legacy development report has the wrong purpose")
    if legacy.get("blind_data_used") is not False or legacy.get("passed") is not True:
        raise RuntimeError("Legacy development evidence did not pass")
    if legacy.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Legacy development checkpoint hash mismatch")
    if legacy.get("parent_noninferiority", {}).get("passed") is not True:
        raise RuntimeError("Legacy parent noninferiority did not pass")
    return {
        "external": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "candidate": {
                "top1_correct": int(candidate["top1_correct"]),
                "top5_correct": int(candidate["top5_correct"]),
            },
            "parent": {
                "top1_correct": int(parent["top1_correct"]),
                "top5_correct": int(parent["top5_correct"]),
            },
        },
        "legacy": {
            "path": str(legacy_path),
            "sha256": sha256_file(legacy_path),
            "candidate": legacy["candidate"],
            "parent": legacy["parent"],
        },
    }


def select_diverse_indices(dataset: UnifiedRawManifestDataset, count: int) -> list[int]:
    selected: list[int] = []
    identities: set[str] = set()
    for index, record in enumerate(dataset.records):
        identity = str(record["identity"]).casefold()
        if identity in identities:
            continue
        selected.append(index)
        identities.add(identity)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Manifest has fewer than {count} identities")


def compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
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


def validate_session(
    model_path: Path,
    samples: np.ndarray,
    expected: dict[int, np.ndarray],
    batch_sizes: list[int],
    *,
    provider: str,
    max_abs_tolerance: float,
    minimum_cosine: float,
) -> dict:
    providers = (
        ["CPUExecutionProvider"]
        if provider == "CPUExecutionProvider"
        else [
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
    )
    session = ort.InferenceSession(str(model_path), providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Refusing ONNX provider fallback: {session.get_providers()}")
    if tuple(item.name for item in session.get_inputs()) != UNIFIED_ONNX_INPUT_NAMES:
        raise RuntimeError("ONNX input contract is not exactly ('rgb',)")
    if tuple(item.name for item in session.get_outputs()) != UNIFIED_ONNX_OUTPUT_NAMES:
        raise RuntimeError("ONNX output contract is not exactly ('embedding',)")
    runs = []
    for batch_size in batch_sizes:
        actual = session.run(
            list(UNIFIED_ONNX_OUTPUT_NAMES),
            {"rgb": samples[:batch_size]},
        )[0]
        parity = compare(expected[batch_size], actual)
        if parity["max_abs_error"] > max_abs_tolerance:
            raise RuntimeError(f"{provider} max error exceeds tolerance: {parity}")
        if parity["minimum_cosine"] < minimum_cosine:
            raise RuntimeError(f"{provider} cosine is below tolerance: {parity}")
        norms = np.linalg.norm(actual, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
            raise RuntimeError(f"{provider} output is not L2 normalized: {norms}")
        runs.append(
            {
                "batch_size": batch_size,
                "embedding": parity,
                "norm_range": [float(norms.min()), float(norms.max())],
            }
        )
    active = session.get_providers()
    del session
    gc.collect()
    return {"provider": provider, "provider_chain": active, "runs": runs}


def validate_graph_contract(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [
        item.name for item in model.graph.input if item.name not in initializer_names
    ]
    graph_outputs = [item.name for item in model.graph.output]
    if graph_inputs != list(UNIFIED_ONNX_INPUT_NAMES):
        raise RuntimeError(f"Unexpected graph inputs: {graph_inputs}")
    if graph_outputs != list(UNIFIED_ONNX_OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected graph outputs: {graph_outputs}")
    external_tensors = [
        item.name
        for item in model.graph.initializer
        if item.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external_tensors:
        raise RuntimeError("Unified ONNX unexpectedly uses external tensor files")
    return {
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "external_tensor_files": [],
    }


def main() -> None:
    args = parse_args()
    precision = configure_strict_cuda_precision()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    development_path = args.development_report.expanduser().resolve()
    legacy_path = args.legacy_development_report.expanduser().resolve()
    for path in (
        checkpoint_path,
        acceptance_path,
        development_path,
        legacy_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.validation_batch_size < 2:
        raise ValueError("validation batch size must be at least two")
    if args.max_dynamic_batch < args.validation_batch_size:
        raise ValueError("max dynamic batch is smaller than validation batch")
    acceptance, manifest_path = validate_acceptance(acceptance_path)
    evidence = validate_development_evidence(
        development_path,
        legacy_path,
        checkpoint_path=checkpoint_path,
        acceptance_path=acceptance_path,
        acceptance=acceptance,
    )

    output_dir = args.output_dir.expanduser().resolve()
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
    model, checkpoint = build_external_joint_from_checkpoint(
        checkpoint_path, device=device, verify_sources=True
    )
    if checkpoint.get("training", {}).get("blind_data_used") is not False:
        raise RuntimeError("Checkpoint training provenance is not blind-safe")
    if checkpoint.get("training", {}).get("acceptance_sha256") != sha256_file(
        acceptance_path
    ):
        raise RuntimeError("Checkpoint was not trained under this acceptance")
    dataset = UnifiedRawManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=False,
    )
    selected_indices = select_diverse_indices(dataset, args.validation_batch_size)
    samples = torch.stack([dataset[index]["rgb"] for index in selected_indices])
    wrapper = UnifiedExternalJointPetReIDExport(model).to(device).eval()
    batch_sizes = sorted({1, args.validation_batch_size})
    expected: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for batch_size in batch_sizes:
            expected[batch_size] = (
                wrapper(samples[:batch_size].to(device).contiguous())
                .float()
                .cpu()
                .numpy()
            )
    if expected[args.validation_batch_size].shape != (
        args.validation_batch_size,
        512,
    ):
        raise RuntimeError("Unexpected PyTorch output contract")

    export_input = samples[: args.validation_batch_size].to(device).contiguous()
    batch_dimension = torch.export.Dim("batch", min=1, max=args.max_dynamic_batch)
    torch.onnx.export(
        wrapper,
        (export_input,),
        temporary_path,
        input_names=list(UNIFIED_ONNX_INPUT_NAMES),
        output_names=list(UNIFIED_ONNX_OUTPUT_NAMES),
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"rgb": {0: batch_dimension}},
        optimize=True,
    )
    graph_contract = validate_graph_contract(temporary_path)
    del wrapper, model, export_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sample_array = samples.numpy().astype(np.float32, copy=False)
    providers = []
    if not args.skip_cpu:
        providers.append("CPUExecutionProvider")
    if not args.skip_cuda:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA ONNX Runtime provider is unavailable")
        providers.append("CUDAExecutionProvider")
    provider_reports = [
        validate_session(
            temporary_path,
            sample_array,
            expected,
            batch_sizes,
            provider=provider,
            max_abs_tolerance=args.max_abs_tolerance,
            minimum_cosine=args.minimum_cosine,
        )
        for provider in providers
    ]
    if model_path.exists():
        model_path.unlink()
    os.replace(temporary_path, model_path)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_external_joint_pet_reid",
        "model": str(model_path),
        "onnx_sha256": sha256_file(model_path),
        "onnx_bytes": model_path.stat().st_size,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_schema_version": checkpoint["schema_version"],
        "model_config": checkpoint["model_config"],
        "preprocessing": checkpoint["preprocessing"],
        "inputs": checkpoint["runtime_contract"]["inputs"],
        "outputs": checkpoint["runtime_contract"]["outputs"],
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
            "validated": batch_sizes,
        },
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
            "protocol_name": acceptance["protocol_name"],
        },
        "development_reports": evidence,
        "promotion_status": "experimental_until_blind_passes",
        "default_backend_changed": False,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "pytorch_cuda_precision": precision,
    }
    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "onnx_checker": "passed",
        "single_graph_contract": graph_contract,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_indices": selected_indices,
        "batch_sizes": batch_sizes,
        "thresholds": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "minimum_cosine": args.minimum_cosine,
        },
        "providers": provider_reports,
        "onnx": {
            "path": str(model_path),
            "sha256": metadata["onnx_sha256"],
            "bytes": metadata["onnx_bytes"],
        },
        "passed": True,
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
                "onnx_sha256": metadata["onnx_sha256"],
                "onnx_bytes": metadata["onnx_bytes"],
                "contract": graph_contract,
                "providers": provider_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
