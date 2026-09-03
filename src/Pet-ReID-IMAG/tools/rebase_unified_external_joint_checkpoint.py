#!/usr/bin/env python3
"""Rebase trained joint weights onto a geometry-stable semantic parent."""

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

from pet_id.unified_external_model import (  # noqa: E402
    UnifiedExternalJointPetReID,
    build_external_joint_from_checkpoint,
    create_external_joint_checkpoint,
    save_external_joint_checkpoint,
    sha256_file,
)
from pet_id.unified_semantic_checkpoint import (  # noqa: E402
    build_unified_semantic_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--new-base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.checkpoint.expanduser().resolve()
    base_path = args.new_base_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (source_path, base_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    source, payload = build_external_joint_from_checkpoint(
        source_path, device="cpu", verify_sources=True
    )
    base, base_payload = build_unified_semantic_from_checkpoint(
        base_path, device="cpu", verify_sources=True
    )
    if base_payload.get("external_geometry_stability_evidence") is None:
        raise RuntimeError("New base lacks external geometry stability evidence")
    refiner = source.refiner.configuration()
    rebased = UnifiedExternalJointPetReID(
        base,
        hidden_dim=int(refiner["hidden_dim"]),
        maximum_residual_weight=float(refiner["maximum_residual_weight"]),
        maximum_interaction_norm=float(refiner["maximum_interaction_norm"]),
        interaction_scale_mode=str(refiner["interaction_scale_mode"]),
    )
    incompatible = rebased.load_state_dict(source.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Joint rebase state mismatch: {incompatible}")
    # The geometry discretizer uses non-persistent buffers. Loading the trained
    # state must not replace the new base checkpoint's stability map.
    rebased.base_model.geometry_discretizer = base.geometry_discretizer
    rebased.configure_trainable(nose_adapter=False, refiner=False)
    rebased.eval()
    training = dict(payload.get("training") or {})
    training["rebase"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_joint_checkpoint": str(source_path),
        "source_joint_checkpoint_sha256": sha256_file(source_path),
        "new_base_checkpoint": str(base_path),
        "new_base_checkpoint_sha256": sha256_file(base_path),
        "geometry_stability_evidence": base_payload[
            "external_geometry_stability_evidence"
        ],
        "additional_gradient_steps": 0,
        "blind_data_used": False,
    }
    selection = dict(payload.get("selection") or {})
    selection["rebase_rule"] = (
        "weights unchanged; candidate must repeat external and legacy "
        "development gates after geometry stabilization"
    )
    rebased_payload = create_external_joint_checkpoint(
        rebased,
        base_checkpoint=base_path,
        training=training,
        selection=selection,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_external_joint_checkpoint(rebased_payload, output_path)
    restored, restored_payload = build_external_joint_from_checkpoint(
        output_path, device="cpu", verify_sources=True
    )
    if (
        restored.base_model.geometry_discretizer.configuration()
        != base.geometry_discretizer.configuration()
    ):
        raise RuntimeError("Rebased joint geometry configuration did not reload")
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "source_joint_checkpoint": str(source_path),
        "source_joint_checkpoint_sha256": sha256_file(source_path),
        "new_base_checkpoint": str(base_path),
        "new_base_checkpoint_sha256": sha256_file(base_path),
        "refiner": restored.refiner.configuration(),
        "geometry_discretization": (
            restored.base_model.geometry_discretizer.configuration()
        ),
        "blind_data_used": restored_payload["training"]["blind_data_used"],
        "passed": True,
        "default_backend_changed": False,
    }
    report_path = output_path.with_suffix(".rebase.json")
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
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
