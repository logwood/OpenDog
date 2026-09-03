#!/usr/bin/env python3
"""Derive one boundary map stable on both legacy and external development."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import sha256_file  # noqa: E402

_DERIVATION_PATH = ROOT / "tools/derive_unified_geometry_stability.py"
_DERIVATION_SPEC = importlib.util.spec_from_file_location(
    "_unified_geometry_stability_tool", _DERIVATION_PATH
)
if _DERIVATION_SPEC is None or _DERIVATION_SPEC.loader is None:
    raise ImportError(_DERIVATION_PATH)
_DERIVATION_MODULE = importlib.util.module_from_spec(_DERIVATION_SPEC)
_DERIVATION_SPEC.loader.exec_module(_DERIVATION_MODULE)
build_shifted_boundary_box = _DERIVATION_MODULE.build_shifted_boundary_box


REQUIRED_ARRAYS = (
    "pytorch_boxes",
    "pytorch_angles",
    "cpu_boxes",
    "cpu_angles",
    "cuda_boxes",
    "cuda_angles",
    "source_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-arrays", type=Path, required=True)
    parser.add_argument("--legacy-report", type=Path, required=True)
    parser.add_argument("--external-arrays", type=Path, required=True)
    parser.add_argument("--external-report", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_shifted_existing_piecewise(
    values: dict[str, np.ndarray],
    existing_configuration: list[dict],
    *,
    minimum_boundary_margin: float = 2e-6,
) -> dict:
    indexed = {
        (int(record["part"]), int(record["coordinate"])): record
        for record in existing_configuration
    }
    if set(indexed) != {
        (part, coordinate)
        for part in range(2)
        for coordinate in range(4)
    }:
        raise RuntimeError("Existing piecewise configuration is incomplete")
    configuration = []
    coordinates = []
    total_changed = 0
    total_shifted = 0
    total_removed = 0
    construction_margin = 2.0 * minimum_boundary_margin
    for part in range(2):
        for coordinate in range(4):
            record = indexed[(part, coordinate)]
            thresholds = np.asarray(record["thresholds"], dtype=np.float32)
            existing_levels = np.asarray(record["levels"], dtype=np.float32)
            if existing_levels.size != thresholds.size + 1:
                raise RuntimeError("Existing piecewise level count is invalid")
            if thresholds.size > 1 and not np.all(
                thresholds[1:] > thresholds[:-1]
            ):
                raise RuntimeError("Existing piecewise thresholds are not sorted")
            reference = values["pytorch_boxes"][:, part, coordinate]
            variants = np.stack(
                (
                    reference,
                    values["cpu_boxes"][:, part, coordinate],
                    values["cuda_boxes"][:, part, coordinate],
                ),
                axis=1,
            )
            original_reference_bins = np.sum(
                reference[:, None] >= thresholds[None, :], axis=1
            )
            keep = np.ones(thresholds.size, dtype=bool)
            shifted_thresholds = thresholds.copy()
            shift_records = []
            removed_records = []
            for index, threshold in enumerate(thresholds):
                reference_below = reference < threshold
                reference_above = ~reference_below
                lower_max = float(
                    np.max(variants[reference_below], initial=0.0)
                )
                upper_min = float(
                    np.min(variants[reference_above], initial=1.0)
                )
                separation = upper_min - lower_max
                if separation < 2.0 * construction_margin:
                    keep[index] = False
                    removed_records.append(
                        {
                            "index": index,
                            "original": float(threshold),
                            "lower_max": lower_max,
                            "upper_min": upper_min,
                            "separation": separation,
                        }
                    )
                    continue
                safe_lower = lower_max + construction_margin
                safe_upper = upper_min - construction_margin
                selected = np.float32(
                    np.clip(float(threshold), safe_lower, safe_upper)
                )
                if (
                    float(selected) - lower_max < minimum_boundary_margin
                    or upper_min - float(selected) < minimum_boundary_margin
                ):
                    raise RuntimeError("Float32 boundary lost backend margin")
                shifted_thresholds[index] = selected
                if selected != threshold:
                    shift_records.append(
                        {
                            "index": index,
                            "original": float(threshold),
                            "selected": float(selected),
                            "absolute_shift": abs(
                                float(selected) - float(threshold)
                            ),
                        }
                    )
            retained_thresholds = shifted_thresholds[keep]
            if retained_thresholds.size > 1 and not np.all(
                retained_thresholds[1:] > retained_thresholds[:-1]
            ):
                raise RuntimeError("Shifted existing boundaries are not sorted")
            group_for_original_bin = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(keep.astype(np.int64)),
                )
            )
            expected_groups = group_for_original_bin[
                original_reference_bins
            ]
            actual_groups = np.sum(
                variants[:, :, None]
                >= retained_thresholds[None, None, :],
                axis=2,
            )
            if not np.all(actual_groups == expected_groups[:, None]):
                raise RuntimeError(
                    "Existing-boundary shifts did not stabilize backend groups"
                )
            group_count = int(group_for_original_bin[-1]) + 1
            levels = np.empty(group_count, dtype=np.float32)
            for group in range(group_count):
                original_bins = np.flatnonzero(
                    group_for_original_bin == group
                )
                observed_bins = original_reference_bins[
                    expected_groups == group
                ]
                if observed_bins.size:
                    observed_levels = existing_levels[observed_bins]
                    unique, counts = np.unique(
                        observed_levels, return_counts=True
                    )
                    levels[group] = unique[np.argmax(counts)]
                else:
                    levels[group] = existing_levels[
                        original_bins[len(original_bins) // 2]
                    ]
            original_output = existing_levels[original_reference_bins]
            stable_output = levels[expected_groups]
            displacement = stable_output - original_output
            changed_count = int(np.count_nonzero(displacement))
            total_changed += changed_count
            total_shifted += len(shift_records)
            total_removed += len(removed_records)
            configuration.append(
                {
                    "part": part,
                    "coordinate": coordinate,
                    "thresholds": retained_thresholds.tolist(),
                    "levels": levels.tolist(),
                }
            )
            coordinates.append(
                {
                    "part": part,
                    "coordinate": coordinate,
                    "original_thresholds": int(thresholds.size),
                    "retained_thresholds": int(keep.sum()),
                    "shifted_thresholds": len(shift_records),
                    "removed_thresholds": len(removed_records),
                    "changed_reference_records": changed_count,
                    "mean_abs_change_from_existing": float(
                        np.mean(np.abs(displacement), dtype=np.float64)
                    ),
                    "max_abs_change_from_existing": float(
                        np.max(np.abs(displacement), initial=0.0)
                    ),
                    "shift_records": shift_records,
                    "removed_records": removed_records,
                    "all_backend_groups_match": True,
                }
            )
    return {
        "step": None,
        "level_mode": "existing_boundary_shift_mode",
        "piecewise": configuration,
        "coordinates": coordinates,
        "shifted_thresholds": total_shifted,
        "removed_thresholds": total_removed,
        "changed_reference_coordinates": total_changed,
        "minimum_boundary_margin": minimum_boundary_margin,
        "passed": True,
    }


def load_capture(arrays_path: Path, report_path: Path) -> tuple[dict, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("blind_data_used") is not False:
        raise RuntimeError("Geometry capture must explicitly exclude blind data")
    if report.get("arrays_sha256") != sha256_file(arrays_path):
        raise RuntimeError("Geometry capture/array hash mismatch")
    payload = np.load(arrays_path, allow_pickle=False)
    missing = [name for name in REQUIRED_ARRAYS if name not in payload]
    if missing:
        raise RuntimeError(f"Raw geometry arrays are incomplete: {missing}")
    return {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}, report


def main() -> None:
    args = parse_args()
    paths = {
        "legacy_arrays": args.legacy_arrays.expanduser().resolve(),
        "legacy_report": args.legacy_report.expanduser().resolve(),
        "external_arrays": args.external_arrays.expanduser().resolve(),
        "external_report": args.external_report.expanduser().resolve(),
        "base_checkpoint": args.base_checkpoint.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    legacy, legacy_report = load_capture(
        paths["legacy_arrays"], paths["legacy_report"]
    )
    external, external_report = load_capture(
        paths["external_arrays"], paths["external_report"]
    )
    if legacy_report["geometry_checkpoint_sha256"] != external_report[
        "geometry_checkpoint_sha256"
    ]:
        raise RuntimeError("Legacy/external captures use different geometry weights")
    legacy_hashes = set(legacy["source_sha256"].astype(str).tolist())
    external_hashes = set(external["source_sha256"].astype(str).tolist())
    if legacy_hashes & external_hashes:
        raise RuntimeError("Legacy/external geometry captures overlap")
    base_payload = torch.load(
        paths["base_checkpoint"], map_location="cpu", weights_only=False
    )
    if base_payload.get("model_type") != "unified_semantic_pet_reid":
        raise RuntimeError("Base checkpoint is not unified semantic")
    if base_payload["sources"]["geometry_checkpoint"]["sha256"] != legacy_report[
        "geometry_checkpoint_sha256"
    ]:
        raise RuntimeError("Base checkpoint uses different geometry weights")
    existing_piecewise = base_payload["model_config"][
        "geometry_discretization"
    ].get("box_piecewise")
    if not existing_piecewise:
        raise RuntimeError("Base checkpoint lacks existing piecewise geometry")
    values = {
        name: np.concatenate((legacy[name], external[name]), axis=0)
        for name in REQUIRED_ARRAYS
        if name != "source_sha256"
    }
    selected = build_shifted_existing_piecewise(
        values, existing_piecewise
    )
    selected["offsets"] = base_payload["model_config"][
        "geometry_discretization"
    ]["box_offsets"]
    search = [selected]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "legacy_and_external_development_geometry_stability",
        "blind_data_used": False,
        "geometry_checkpoint_sha256": legacy_report[
            "geometry_checkpoint_sha256"
        ],
        "base_checkpoint": {
            "path": str(paths["base_checkpoint"]),
            "sha256": sha256_file(paths["base_checkpoint"]),
        },
        "captures": {
            "legacy": {
                "report": str(paths["legacy_report"]),
                "report_sha256": sha256_file(paths["legacy_report"]),
                "arrays": str(paths["legacy_arrays"]),
                "arrays_sha256": sha256_file(paths["legacy_arrays"]),
                "manifest_sha256": legacy_report["manifest_sha256"],
                "records": int(legacy["pytorch_boxes"].shape[0]),
            },
            "external": {
                "report": str(paths["external_report"]),
                "report_sha256": sha256_file(paths["external_report"]),
                "arrays": str(paths["external_arrays"]),
                "arrays_sha256": sha256_file(paths["external_arrays"]),
                "manifest_sha256": external_report["manifest_sha256"],
                "records": int(external["pytorch_boxes"].shape[0]),
            },
        },
        "records": int(values["pytorch_boxes"].shape[0]),
        "box_step": selected["step"],
        "box_offsets": selected["offsets"],
        "box_piecewise": selected["piecewise"],
        "angle_step": None,
        "angle_offset": 0.0,
        "selection": {
            "mode": "boundary_shift",
            "shifted_thresholds": selected["shifted_thresholds"],
            "removed_thresholds": selected["removed_thresholds"],
            "changed_reference_coordinates": selected[
                "changed_reference_coordinates"
            ],
            "minimum_boundary_margin": selected[
                "minimum_boundary_margin"
            ],
            "coordinates": selected["coordinates"],
            "search": search,
        },
        "checks": {
            "all_backend_groups_match": selected["passed"],
            "source_hashes_disjoint": True,
            "blind_data_used": False,
        },
        "passed": selected["passed"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "records": report["records"],
                "selection": {
                    key: report["selection"][key]
                    for key in (
                        "shifted_thresholds",
                        "removed_thresholds",
                        "changed_reference_coordinates",
                        "minimum_boundary_margin",
                    )
                },
                "passed": report["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
