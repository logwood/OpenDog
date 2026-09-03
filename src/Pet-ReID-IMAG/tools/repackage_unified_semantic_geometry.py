#!/usr/bin/env python3
"""Repackage a unified semantic parent with a locked geometry stability map."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_geometry_stability import StableGeometryDiscretizer  # noqa: E402
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
    semantic_model_configuration,
)
from pet_id.unified_training import atomic_torch_save, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stability-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    stability_path = args.stability_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, stability_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    if stability.get("passed") is not True:
        raise RuntimeError("Geometry stability report did not pass")
    if stability.get("blind_data_used") is not False:
        raise RuntimeError("Geometry stability report used blind data")
    if stability.get("records") != 656:
        raise RuntimeError("Geometry stability record count changed")
    model, payload = build_unified_semantic_from_checkpoint(
        checkpoint_path, device="cpu", verify_sources=True
    )
    geometry_hash = payload["sources"]["geometry_checkpoint"]["sha256"]
    if stability.get("geometry_checkpoint_sha256") != geometry_hash:
        raise RuntimeError("Geometry stability weights differ from checkpoint")
    original_configuration = model.geometry_discretizer.configuration()
    discretizer = StableGeometryDiscretizer(
        box_step=stability["box_step"],
        box_offsets=stability["box_offsets"],
        angle_step=stability["angle_step"],
        angle_offset=stability["angle_offset"],
        box_piecewise=stability["box_piecewise"],
    ).to("cpu")
    model.geometry_discretizer = discretizer
    model.eval()
    repackaged = dict(payload)
    repackaged.update(
        {
            "model_config": semantic_model_configuration(model),
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "parent_checkpoint": str(checkpoint_path),
            "parent_checkpoint_sha256": sha256_file(checkpoint_path),
            "external_geometry_stability_evidence": {
                "path": str(stability_path),
                "sha256": sha256_file(stability_path),
                "records": stability["records"],
                "legacy_manifest_sha256": stability["captures"]["legacy"][
                    "manifest_sha256"
                ],
                "external_manifest_sha256": stability["captures"]["external"][
                    "manifest_sha256"
                ],
            },
            "repackaged_at": datetime.now(timezone.utc).isoformat(),
            "promotion_status": "development_validation_required",
            "default_backend_changed": False,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(repackaged, output_path)
    restored, restored_payload = build_unified_semantic_from_checkpoint(
        output_path, device="cpu", verify_sources=True
    )
    restored_configuration = restored.geometry_discretizer.configuration()
    expected_configuration = discretizer.configuration()
    if restored_configuration != expected_configuration:
        raise RuntimeError("Repackaged geometry configuration did not reload")
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": sha256_file(checkpoint_path),
        "stability_report": str(stability_path),
        "stability_report_sha256": sha256_file(stability_path),
        "original_geometry_discretization": original_configuration,
        "new_geometry_discretization": restored_configuration,
        "source_weights_unchanged": (
            restored_payload["sources"] == payload["sources"]
        ),
        "passed": True,
        "default_backend_changed": False,
    }
    report_path = output_path.with_suffix(".repackage.json")
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": result["output"],
                "output_sha256": result["output_sha256"],
                "report": str(report_path),
                "stability_report_sha256": result[
                    "stability_report_sha256"
                ],
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
