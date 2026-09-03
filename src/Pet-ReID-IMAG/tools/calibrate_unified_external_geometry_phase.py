#!/usr/bin/env python3
"""Collect external-dev PyTorch/CPU/CUDA raw geometry for stable quantization."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader


for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_data import UnifiedRawManifestDataset  # noqa: E402
from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)
from pet_id.release_compatibility import acceptance_protocol_name  # noqa: E402

_CALIBRATION_PATH = ROOT / "tools/calibrate_unified_semantic_geometry_phase.py"
_CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "_unified_geometry_phase_tool", _CALIBRATION_PATH
)
if _CALIBRATION_SPEC is None or _CALIBRATION_SPEC.loader is None:
    raise ImportError(_CALIBRATION_PATH)
_CALIBRATION_MODULE = importlib.util.module_from_spec(_CALIBRATION_SPEC)
_CALIBRATION_SPEC.loader.exec_module(_CALIBRATION_MODULE)
RawGeometryExport = _CALIBRATION_MODULE.RawGeometryExport
make_session = _CALIBRATION_MODULE.make_session


ACCEPTANCE_PROTOCOL = acceptance_protocol_name("external-development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--disable-pytorch-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.disable_pytorch_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = args.checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    if not checkpoint_path.is_file() or not acceptance_path.is_file():
        raise FileNotFoundError("checkpoint or acceptance is missing")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if (
        acceptance.get("schema_version") != 3
        or acceptance.get("protocol_name") != ACCEPTANCE_PROTOCOL
    ):
        raise RuntimeError("Unexpected external acceptance contract")
    manifest_path = Path(acceptance["development"]["path"]).expanduser().resolve()
    if sha256_file(manifest_path) != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest differs from acceptance")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "raw_geometry.onnx"
    temporary_path = output_dir / "raw_geometry.exporting.onnx"
    arrays_path = output_dir / "raw_geometry_backends.npz"
    report_path = output_dir / "geometry_phase_report.json"
    for path in (debug_path, arrays_path, report_path):
        if path.exists():
            raise FileExistsError(path)
    if temporary_path.exists():
        temporary_path.unlink()

    device = torch.device(args.device)
    model, checkpoint = build_unified_semantic_from_checkpoint(
        checkpoint_path, device=device, verify_sources=True
    )
    dataset = UnifiedRawManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    wrapper = RawGeometryExport(model).to(device).eval()
    samples = torch.stack([dataset[0]["rgb"], dataset[4]["rgb"]])
    batch_dimension = torch.export.Dim(
        "batch", min=1, max=args.max_dynamic_batch
    )
    torch.onnx.export(
        wrapper,
        (samples.to(device).contiguous(),),
        temporary_path,
        input_names=["rgb"],
        output_names=["boxes_cxcywh", "angle_radians", "confidence"],
        opset_version=20,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"rgb": {0: batch_dimension}},
        optimize=True,
    )
    onnx.checker.check_model(str(temporary_path))
    os.replace(temporary_path, debug_path)
    sessions = {
        name: make_session(debug_path, name)
        for name in ("CPUExecutionProvider", "CUDAExecutionProvider")
    }
    pytorch_outputs: dict[str, list[np.ndarray]] = {
        "boxes": [],
        "angles": [],
        "confidence": [],
    }
    backend_outputs = {
        provider: {"boxes": [], "angles": [], "confidence": []}
        for provider in sessions
    }
    source_sha256: list[str] = []
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            rgb_cpu = batch["rgb"].numpy().astype(np.float32, copy=False)
            expected = wrapper(
                batch["rgb"].to(device, non_blocking=True).contiguous()
            )
            for name, value in zip(
                ("boxes", "angles", "confidence"), expected
            ):
                pytorch_outputs[name].append(value.float().cpu().numpy())
            for provider, session in sessions.items():
                actual = session.run(None, {"rgb": rgb_cpu})
                for name, value in zip(
                    ("boxes", "angles", "confidence"), actual
                ):
                    backend_outputs[provider][name].append(
                        np.asarray(value, dtype=np.float32)
                    )
            source_sha256.extend(batch["source_sha256"])
            processed += int(rgb_cpu.shape[0])
            if processed == args.batch_size or processed % 64 == 0:
                print(
                    f"external raw geometry parity: {processed}/{len(dataset)}",
                    flush=True,
                )
    pytorch_values = {
        name: np.concatenate(values)
        for name, values in pytorch_outputs.items()
    }
    backend_values = {
        provider: {
            name: np.concatenate(values)
            for name, values in outputs.items()
        }
        for provider, outputs in backend_outputs.items()
    }
    np.savez_compressed(
        arrays_path,
        pytorch_boxes=pytorch_values["boxes"],
        pytorch_angles=pytorch_values["angles"],
        pytorch_confidence=pytorch_values["confidence"],
        cpu_boxes=backend_values["CPUExecutionProvider"]["boxes"],
        cpu_angles=backend_values["CPUExecutionProvider"]["angles"],
        cpu_confidence=backend_values["CPUExecutionProvider"]["confidence"],
        cuda_boxes=backend_values["CUDAExecutionProvider"]["boxes"],
        cuda_angles=backend_values["CUDAExecutionProvider"]["angles"],
        cuda_confidence=backend_values["CUDAExecutionProvider"]["confidence"],
        source_sha256=np.asarray(source_sha256),
    )
    providers = {}
    for provider, values in backend_values.items():
        providers[provider] = {
            "provider_chain": sessions[provider].get_providers(),
            "raw_box_max_abs_error": float(
                np.abs(pytorch_values["boxes"] - values["boxes"]).max()
            ),
            "raw_angle_max_abs_error": float(
                np.abs(pytorch_values["angles"] - values["angles"]).max()
            ),
            "confidence_max_abs_error": float(
                np.abs(
                    pytorch_values["confidence"] - values["confidence"]
                ).max()
            ),
        }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "external_development_backend_geometry_capture",
        "blind_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "geometry_checkpoint_sha256": checkpoint["sources"][
            "geometry_checkpoint"
        ]["sha256"],
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(dataset),
        "debug_onnx": str(debug_path),
        "debug_onnx_sha256": sha256_file(debug_path),
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "providers": providers,
        "pytorch_precision": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "passed": (
            len(dataset) == acceptance["development"]["records"]
            and all(
                row["provider_chain"][0] == provider
                for provider, row in providers.items()
            )
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    del sessions, wrapper, model
    gc.collect()


if __name__ == "__main__":
    main()
