#!/usr/bin/env python3
"""Calibrate stable box phases from development PyTorch/ONNX geometry only."""

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
from torch import nn
from torch.utils.data import DataLoader

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_geometry_stability import (  # noqa: E402
    DEFAULT_GEOMETRY_ANGLE_OFFSET,
    LOCKED_GEOMETRY_BOX_STEP,
    choose_backend_stable_offset,
)
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)
from pet_id.unified_training import sha256_file  # noqa: E402
from pet_id.release_compatibility import historical_run_path  # noqa: E402


DEV_MANIFEST_SHA256 = (
    "e522215a91ae76481b97856f771c19ff6fa06702cad92db82e1ef65f4da5b9c7"
)
BOX_STEP_CANDIDATES = (
    1.0 / 300.0,
    1.0 / 250.0,
    1.0 / 200.0,
    1.0 / 160.0,
    1.0 / 150.0,
    1.0 / 125.0,
    1.0 / 100.0,
    1.0 / 80.0,
    1.0 / 64.0,
    1.0 / 50.0,
)
ANGLE_STEP_CANDIDATES = (
    1.0 / 512.0,
    1.0 / 400.0,
    1.0 / 320.0,
    1.0 / 256.0,
    1.0 / 200.0,
    1.0 / 160.0,
    1.0 / 128.0,
    1.0 / 104.0,
    1.0 / 80.0,
    1.0 / 64.0,
)
MINIMUM_BOUNDARY_MARGIN = 2e-6


class RawGeometryExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.geometry = model.geometry_frontend

    def forward(self, rgb: torch.Tensor):
        rgb = self.geometry._validate_rgb(rgb)
        prediction, _ = self.geometry._localize(rgb)
        return (
            prediction.boxes_cxcywh,
            prediction.angle_radians,
            prediction.confidence,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "fresh-baseline")
        / "prepared/development/manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-dynamic-batch", type=int, default=8)
    parser.add_argument("--reuse-debug-onnx", action="store_true")
    return parser.parse_args()


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    providers = (
        ["CPUExecutionProvider"]
        if provider == "CPUExecutionProvider"
        else [
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
    )
    session = ort.InferenceSession(str(path), providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Refusing provider fallback: {session.get_providers()}")
    return session


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("checkpoint or manifest is missing")
    if sha256_file(manifest_path) != DEV_MANIFEST_SHA256:
        raise RuntimeError("Geometry phase calibration is restricted to locked dev")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "raw_geometry.onnx"
    temporary_path = output_dir / "raw_geometry.exporting.onnx"
    arrays_path = output_dir / "raw_geometry_backends.npz"
    report_path = output_dir / "geometry_phase_report.json"
    for path in (arrays_path, report_path):
        if path.exists():
            raise FileExistsError(path)
    if debug_path.exists() and not args.reuse_debug_onnx:
        raise FileExistsError(debug_path)
    if args.reuse_debug_onnx and not debug_path.is_file():
        raise FileNotFoundError(debug_path)
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
    if not args.reuse_debug_onnx:
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
    else:
        onnx.checker.check_model(str(debug_path))
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
                ("boxes", "angles", "confidence"),
                expected,
            ):
                pytorch_outputs[name].append(value.float().cpu().numpy())
            for provider, session in sessions.items():
                actual = session.run(None, {"rgb": rgb_cpu})
                for name, value in zip(
                    ("boxes", "angles", "confidence"),
                    actual,
                ):
                    backend_outputs[provider][name].append(
                        np.asarray(value, dtype=np.float32)
                    )
            source_sha256.extend(batch["source_sha256"])
            processed += int(rgb_cpu.shape[0])
            if processed == args.batch_size or processed % 25 == 0:
                print(f"raw geometry parity: {processed}/{len(dataset)}", flush=True)

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
    step_reports: list[dict] = []
    selected_step: float | None = None
    selected_offsets: np.ndarray | None = None
    selected_coordinates: list[dict] | None = None
    for step in BOX_STEP_CANDIDATES:
        offsets = np.zeros((2, 4), dtype=np.float32)
        coordinate_reports: list[dict] = []
        for part in range(2):
            for coordinate in range(4):
                selected = choose_backend_stable_offset(
                    pytorch_values["boxes"][:, part, coordinate],
                    [
                        backend_values[provider]["boxes"][:, part, coordinate]
                        for provider in sessions
                    ],
                    step=step,
                    require_all=False,
                )
                offsets[part, coordinate] = np.float32(selected["offset"])
                coordinate_reports.append(
                    {
                        "part": part,
                        "coordinate": coordinate,
                        **selected,
                    }
                )
        passed = (
            all(item["all_match"] for item in coordinate_reports)
            and min(
                item["minimum_boundary_margin"] for item in coordinate_reports
            ) >= MINIMUM_BOUNDARY_MARGIN
        )
        step_reports.append(
            {
                "box_step": step,
                "box_offsets": offsets.tolist(),
                "coordinates": coordinate_reports,
                "passed": passed,
            }
        )
        if passed:
            selected_step = step
            selected_offsets = offsets
            selected_coordinates = coordinate_reports
            break
    angle_reports: list[dict] = []
    selected_angle_step: float | None = None
    selected_angle_offset: float | None = None
    selected_angle: dict | None = None
    for step in ANGLE_STEP_CANDIDATES:
        candidate = choose_backend_stable_offset(
            pytorch_values["angles"],
            [
                backend_values[provider]["angles"]
                for provider in sessions
            ],
            step=step,
            require_all=False,
        )
        passed = (
            candidate["all_match"]
            and candidate["minimum_boundary_margin"]
            >= MINIMUM_BOUNDARY_MARGIN
        )
        row = {"angle_step": step, **candidate, "passed": passed}
        angle_reports.append(row)
        if passed:
            selected_angle_step = step
            selected_angle_offset = float(candidate["offset"])
            selected_angle = row
            break
    provider_reports = {}
    for provider, values in backend_values.items():
        box_error = np.abs(pytorch_values["boxes"] - values["boxes"])
        angle_error = np.abs(pytorch_values["angles"] - values["angles"])
        confidence_error = np.abs(
            pytorch_values["confidence"] - values["confidence"]
        )
        provider_reports[provider] = {
            "provider_chain": sessions[provider].get_providers(),
            "raw_box_max_abs_error": float(box_error.max()),
            "raw_box_mean_abs_error": float(box_error.mean()),
            "raw_angle_max_abs_error": float(angle_error.max()),
            "raw_angle_mean_abs_error": float(angle_error.mean()),
            "confidence_max_abs_error": float(confidence_error.max()),
            "confidence_mean_abs_error": float(confidence_error.mean()),
        }
    checks = {
        "stable_step_found": selected_step is not None,
        "all_box_bins_match": (
            selected_coordinates is not None
            and all(item["all_match"] for item in selected_coordinates)
        ),
        "cpu_provider_active": (
            sessions["CPUExecutionProvider"].get_providers()[0]
            == "CPUExecutionProvider"
        ),
        "cuda_provider_active": (
            sessions["CUDAExecutionProvider"].get_providers()[0]
            == "CUDAExecutionProvider"
        ),
        "stable_angle_step_found": selected_angle_step is not None,
        "all_angle_bins_match": (
            selected_angle is not None and selected_angle["all_match"]
        ),
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development_only_backend_geometry_phase_calibration",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "geometry_checkpoint_sha256": checkpoint["sources"][
            "geometry_checkpoint"
        ]["sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(dataset),
        "debug_onnx": str(debug_path),
        "debug_onnx_sha256": sha256_file(debug_path),
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "box_step_candidates": list(BOX_STEP_CANDIDATES),
        "step_search": step_reports,
        "box_step": selected_step,
        "box_offsets": (
            selected_offsets.tolist() if selected_offsets is not None else None
        ),
        "box_piecewise": None,
        "box_minimum_boundary_margin": (
            min(item["minimum_boundary_margin"] for item in selected_coordinates)
            if selected_coordinates is not None
            else None
        ),
        "angle_step_candidates": list(ANGLE_STEP_CANDIDATES),
        "angle_search": angle_reports,
        "angle_step": selected_angle_step,
        "angle_offset": (
            selected_angle_offset
            if selected_angle_offset is not None
            else DEFAULT_GEOMETRY_ANGLE_OFFSET
        ),
        "angle_minimum_boundary_margin": (
            selected_angle["minimum_boundary_margin"]
            if selected_angle is not None
            else None
        ),
        "offset_semantics": "dimensionless additive phase in round(value / step + offset)",
        "coordinates": selected_coordinates,
        "providers": provider_reports,
        "checks": checks,
        "passed": all(checks.values()),
        "blind_data_used": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"Geometry phase calibration failed: {checks}")
    del sessions, wrapper, model
    gc.collect()


if __name__ == "__main__":
    main()
