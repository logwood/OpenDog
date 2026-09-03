#!/usr/bin/env python3
"""Export and validate one RGB-to-512D UnifiedSemanticPetReID ONNX graph."""

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

# PyTorch's dynamo exporter prints Unicode status glyphs. Windows terminals
# may otherwise inherit a legacy GBK code page and fail after graph capture.
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_runtime import (  # noqa: E402
    UNIFIED_ONNX_INPUT_NAMES,
    UNIFIED_ONNX_OUTPUT_NAMES,
)
from pet_id.unified_semantic import UnifiedSemanticPetReIDExport  # noqa: E402
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)
from pet_id.unified_training import load_acceptance, sha256_file  # noqa: E402
from pet_id.release_compatibility import (  # noqa: E402
    acceptance_path,
    acceptance_protocol_name,
    historical_purpose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=acceptance_path(WORKSPACE, "baseline-training"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to the development manifest locked by --acceptance.",
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
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


def validate_development_report(
    report: dict,
    *,
    report_path: Path,
    acceptance: dict,
    acceptance_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
) -> dict:
    if report.get("purpose") != historical_purpose("semantic-development-selection"):
        raise RuntimeError("Development report is not the canonical selection evaluation")
    if report.get("blind_data_used") is not False:
        raise RuntimeError("Development report must explicitly exclude blind data")
    if report.get("passed") is not True:
        raise RuntimeError("Development report did not pass all gates")
    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_hash = sha256_file(manifest_path)
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Development report/checkpoint hash mismatch")
    if report.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Development report/manifest hash mismatch")
    if manifest_hash != acceptance["development"]["sha256"]:
        raise RuntimeError("Export manifest differs from the locked acceptance")
    report_acceptance = report.get("acceptance", {})
    if report_acceptance.get("sha256") != sha256_file(acceptance_path):
        raise RuntimeError("Development report/acceptance hash mismatch")
    baseline_path = Path(acceptance["baseline_lock"]["path"]).resolve()
    if sha256_file(baseline_path) != acceptance["baseline_lock"]["sha256"]:
        raise RuntimeError("Baseline lock differs from the locked acceptance")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    thresholds = baseline["reports"]["development"]["metrics"]
    clean = report.get("clean", {})
    checks = {
        "top1": int(clean.get("top1_correct", -1))
        >= int(thresholds["top1_correct"]),
        "top5": int(clean.get("top5_correct", -1))
        >= int(thresholds["top5_correct"]),
    }
    if report.get("checks") != checks or not all(checks.values()):
        raise RuntimeError(f"Development noninferiority failed: {checks}")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "thresholds": {
            "minimum_top1_correct": int(thresholds["top1_correct"]),
            "minimum_top5_correct": int(thresholds["top5_correct"]),
        },
        "actual": {
            "top1_correct": int(clean["top1_correct"]),
            "top5_correct": int(clean["top5_correct"]),
        },
        "passed": True,
    }


def select_diverse_indices(dataset, count: int) -> list[int]:
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
            (
                "CUDAExecutionProvider",
                {
                    "use_tf32": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
    )
    session = ort.InferenceSession(str(model_path), providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(
            f"Refusing ONNX provider fallback: {session.get_providers()}"
        )
    if tuple(item.name for item in session.get_inputs()) != (UNIFIED_ONNX_INPUT_NAMES):
        raise RuntimeError("ONNX input contract is not exactly ('rgb',)")
    if tuple(item.name for item in session.get_outputs()) != (
        UNIFIED_ONNX_OUTPUT_NAMES
    ):
        raise RuntimeError("ONNX output contract is not exactly ('embedding',)")
    runs = []
    for batch_size in batch_sizes:
        actual = session.run(
            list(UNIFIED_ONNX_OUTPUT_NAMES),
            {"rgb": samples[:batch_size]},
        )[0]
        metrics = compare(expected[batch_size], actual)
        if metrics["max_abs_error"] > max_abs_tolerance:
            raise RuntimeError(f"{provider} max error exceeds tolerance: {metrics}")
        if metrics["minimum_cosine"] < minimum_cosine:
            raise RuntimeError(f"{provider} cosine is below tolerance: {metrics}")
        norms = np.linalg.norm(actual, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
            raise RuntimeError(f"{provider} output is not L2 normalized: {norms}")
        runs.append(
            {
                "batch_size": batch_size,
                "embedding": metrics,
                "norm_range": [float(norms.min()), float(norms.max())],
            }
        )
    active = session.get_providers()
    del session
    gc.collect()
    return {
        "provider": provider,
        "provider_chain": active,
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    development_path = args.development_report.expanduser().resolve()
    for path in (checkpoint_path, acceptance_path, development_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    acceptance = load_acceptance(
        acceptance_path,
        expected_protocol=acceptance_protocol_name("baseline-training"),
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else Path(acceptance["development"]["path"]).expanduser().resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256_file(manifest_path) != acceptance["development"]["sha256"]:
        raise RuntimeError("Export requires the locked protected development manifest")
    output_dir = args.output_dir.expanduser().resolve()
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_evidence = validate_development_report(
        development,
        report_path=development_path,
        acceptance=acceptance,
        acceptance_path=acceptance_path,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )
    if args.validation_batch_size < 2:
        raise ValueError("validation batch size must be at least two")
    if args.max_dynamic_batch < args.validation_batch_size:
        raise ValueError("max dynamic batch is smaller than validation batch")

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
    model, checkpoint = build_unified_semantic_from_checkpoint(
        checkpoint_path,
        device=device,
        verify_sources=True,
    )
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=False,
    )
    selected_indices = select_diverse_indices(dataset, args.validation_batch_size)
    samples = torch.stack([dataset[index]["rgb"] for index in selected_indices])
    wrapper = UnifiedSemanticPetReIDExport(model).to(device).eval()
    batch_sizes = sorted({1, args.validation_batch_size})
    expected = {}
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
    onnx.checker.check_model(str(temporary_path))
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
    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "onnx_checker": "passed",
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
        "passed": True,
    }
    if model_path.exists():
        model_path.unlink()
    os.replace(temporary_path, model_path)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "unified_semantic_pet_reid",
        "model": str(model_path),
        "onnx_sha256": sha256_file(model_path),
        "onnx_bytes": model_path.stat().st_size,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
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
        "development_report": development_evidence,
        "promotion_status": "experimental_until_all_acceptance_gates_pass",
        "default_backend_changed": False,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
    }
    validation["onnx"] = {
        "path": str(model_path),
        "sha256": metadata["onnx_sha256"],
        "bytes": metadata["onnx_bytes"],
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
                "onnx_bytes": metadata["onnx_bytes"],
                "contract": {
                    "inputs": metadata["inputs"],
                    "outputs": metadata["outputs"],
                    "external_models": [],
                },
                "providers": provider_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
