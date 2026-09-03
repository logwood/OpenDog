#!/usr/bin/env python3
"""Embed one locked development calibration into a unified checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_training import (
    atomic_torch_save,
    build_model_from_checkpoint,
    load_acceptance,
    model_configuration,
    sha256_file,
)
from pet_id.release_compatibility import acceptance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--calibration-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        "--allow-compatible-descendant",
        action="store_true",
        help=(
            "Allow applying a calibration selected on the locked development "
            "checkpoint to a separately fixed-epoch final-fit descendant."
        ),
    )
    return parser.parse_args()


def find_calibration(report: dict, name: str) -> dict:
    matches = [
        row for row in report.get("results", [])
        if row.get("calibration", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one calibration named {name!r}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    report_path = args.calibration_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    arcface_path = args.arcface_checkpoint.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    if output_path == checkpoint_path:
        raise RuntimeError("Refusing to overwrite the source checkpoint")

    acceptance = load_acceptance(acceptance_path)
    expected_arcface_hash = acceptance["source_weight_locks"][
        "dog_arcface_checkpoint"
    ]["sha256"]
    if sha256_file(arcface_path) != expected_arcface_hash:
        raise RuntimeError("ArcFace checkpoint differs from the acceptance lock")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol_guard") != "locked_development_validation_only":
        raise RuntimeError("Calibration report is not development-guarded")
    expected_manifest_hash = acceptance["development"]["validation_manifest"][
        "sha256"
    ]
    if report.get("manifest_sha256") != expected_manifest_hash:
        raise RuntimeError("Calibration report manifest differs from the lock")
    checkpoint_hash = sha256_file(checkpoint_path)
    report_checkpoint_hash = report.get("checkpoint_sha256")
    if (
        not args.allow_compatible_descendant
        and checkpoint_hash != report_checkpoint_hash
    ):
        raise RuntimeError(
            "Calibration report was produced from a different checkpoint"
        )

    selected = find_calibration(report, args.calibration_name)
    calibration = selected["calibration"]
    if float(calibration.get("angle_scale", 1.0)) != 1.0 or float(
        calibration.get("angle_offset_radians", 0.0)
    ) != 0.0:
        raise RuntimeError(
            "The graph calibration module currently accepts box adjustments only"
        )

    model, payload = build_model_from_checkpoint(
        checkpoint_path, arcface_path, device="cpu"
    )
    report_configuration = report.get("model_config")
    if report_configuration is not None and (
        model_configuration(model) != report_configuration
    ):
        raise RuntimeError("Calibration report architecture differs from checkpoint")
    model.geometry_calibration.set_part(
        "face",
        center_offset=(
            float(calibration.get("center_x_in_widths", 0.0)),
            float(calibration.get("center_y_in_heights", 0.0)),
        ),
        size_scale=(
            float(calibration.get("width_scale", 1.0)),
            float(calibration.get("height_scale", 1.0)),
        ),
    )
    model.eval()
    calibrated = dict(payload)
    calibrated.update(
        {
            "stage": "geometry_calibration",
            "model_config": model_configuration(model),
            "model": model.state_dict(),
            "optimizer": None,
            "selection_key": [
                selected["metrics"]["top1_correct"],
                selected["metrics"]["top5_correct"],
                selected["metrics"]["mean_reciprocal_rank"],
            ],
            "parent_checkpoint": str(checkpoint_path),
            "parent_checkpoint_sha256": checkpoint_hash,
            "geometry_calibration": {
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "source_report": str(report_path),
                "source_report_sha256": sha256_file(report_path),
                "source_checkpoint_sha256": report_checkpoint_hash,
                "compatible_descendant": bool(args.allow_compatible_descendant),
                "selected": selected,
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(calibrated, output_path)
    restored, restored_payload = build_model_from_checkpoint(
        output_path, arcface_path, device="cpu"
    )
    if not isinstance(restored, nn.Module):
        raise RuntimeError("Calibrated checkpoint reload failed")
    result = {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "stage": restored_payload["stage"],
        "calibration": selected["calibration"],
        "development_metrics": {
            key: selected["metrics"][key]
            for key in (
                "top1_correct",
                "top1_accuracy",
                "top5_correct",
                "top5_accuracy",
                "mean_reciprocal_rank",
            )
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
