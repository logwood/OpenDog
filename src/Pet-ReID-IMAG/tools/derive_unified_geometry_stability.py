#!/usr/bin/env python3
"""Derive robust box/angle discretization from saved dev backend geometry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_geometry_stability import (  # noqa: E402
    DEFAULT_GEOMETRY_BOX_OFFSETS,
    LOCKED_GEOMETRY_BOX_STEP,
    choose_backend_stable_offset,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-arrays", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "fresh-baseline")
        / "prepared/development/manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--angle-mode",
        choices=("stable", "continuous"),
        default="stable",
    )
    parser.add_argument(
        "--box-mode",
        choices=(
            "global",
            "sparse",
            "piecewise",
            "boundary_shift",
            "continuous",
        ),
        default="global",
    )
    parser.add_argument(
        "--piecewise-level-mode",
        choices=("mode", "mean"),
        default="mode",
        help=(
            "Output level for merged piecewise bins: 'mode' preserves the "
            "most records exactly; 'mean' minimizes squared displacement "
            "from the locked grid."
        ),
    )
    return parser.parse_args()


def search_box(values: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    reports = []
    selected = None
    for step in BOX_STEP_CANDIDATES:
        offsets = np.zeros((2, 4), dtype=np.float32)
        coordinates = []
        for part in range(2):
            for coordinate in range(4):
                result = choose_backend_stable_offset(
                    values["pytorch_boxes"][:, part, coordinate],
                    [
                        values["cpu_boxes"][:, part, coordinate],
                        values["cuda_boxes"][:, part, coordinate],
                    ],
                    step=step,
                    require_all=False,
                )
                offsets[part, coordinate] = result["offset"]
                coordinates.append(
                    {"part": part, "coordinate": coordinate, **result}
                )
        minimum_margin = min(
            item["minimum_boundary_margin"] for item in coordinates
        )
        passed = (
            all(item["all_match"] for item in coordinates)
            and minimum_margin >= MINIMUM_BOUNDARY_MARGIN
        )
        row = {
            "step": step,
            "offsets": offsets.tolist(),
            "minimum_boundary_margin": minimum_margin,
            "coordinates": coordinates,
            "passed": passed,
        }
        reports.append(row)
        if selected is None and passed:
            selected = row
    if selected is None:
        raise RuntimeError("No robust box discretization was found")
    return selected, reports


def search_angle(values: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    reports = []
    selected = None
    for step in ANGLE_STEP_CANDIDATES:
        result = choose_backend_stable_offset(
            values["pytorch_angles"],
            [values["cpu_angles"], values["cuda_angles"]],
            step=step,
            require_all=False,
        )
        passed = (
            result["all_match"]
            and result["minimum_boundary_margin"] >= MINIMUM_BOUNDARY_MARGIN
        )
        row = {"step": step, **result, "passed": passed}
        reports.append(row)
        if selected is None and passed:
            selected = row
    if selected is None:
        raise RuntimeError("No robust angle discretization was found")
    return selected, reports


def fixed_offset_result(
    reference: np.ndarray,
    backends: list[np.ndarray],
    *,
    step: float,
    offset: float,
) -> dict:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    backend_values = [
        np.asarray(value, dtype=np.float32).reshape(-1) for value in backends
    ]
    step32 = np.float32(step)
    offset32 = np.float32(offset)
    reference_bins = np.rint(reference / step32 + offset32).astype(np.int64)
    matching_pairs = sum(
        int(
            np.count_nonzero(
                np.rint(value / step32 + offset32).astype(np.int64)
                == reference_bins
            )
        )
        for value in backend_values
    )
    all_values = np.concatenate((reference, *backend_values))
    residues = np.mod(
        all_values.astype(np.float64) / float(step32),
        1.0,
    )
    boundary = float(np.mod(0.5 - float(offset32), 1.0))
    distance = np.abs(np.mod(residues - boundary + 0.5, 1.0) - 0.5)
    margin = float(distance.min(initial=0.5) * float(step32))
    total_pairs = int(reference.size * len(backend_values))
    return {
        "offset": float(offset32),
        "matching_pairs": matching_pairs,
        "total_pairs": total_pairs,
        "all_match": matching_pairs == total_pairs,
        "minimum_boundary_margin": margin,
    }


def search_sparse_box(values: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    steps = np.full((2, 4), LOCKED_GEOMETRY_BOX_STEP, dtype=np.float32)
    offsets = np.asarray(DEFAULT_GEOMETRY_BOX_OFFSETS, dtype=np.float32)
    coordinates = []
    for part in range(2):
        for coordinate in range(4):
            reference = values["pytorch_boxes"][:, part, coordinate]
            backends = [
                values["cpu_boxes"][:, part, coordinate],
                values["cuda_boxes"][:, part, coordinate],
            ]
            original = fixed_offset_result(
                reference,
                backends,
                step=LOCKED_GEOMETRY_BOX_STEP,
                offset=float(offsets[part, coordinate]),
            )
            selected = None
            selection = "locked_original"
            if original["all_match"]:
                selected = {
                    "step": LOCKED_GEOMETRY_BOX_STEP,
                    **original,
                }
            else:
                selection = "minimum_robust_change"
                for step in BOX_STEP_CANDIDATES:
                    candidate = choose_backend_stable_offset(
                        reference,
                        backends,
                        step=step,
                        require_all=False,
                        preferred_step=LOCKED_GEOMETRY_BOX_STEP,
                        preferred_offset=float(
                            DEFAULT_GEOMETRY_BOX_OFFSETS[part][coordinate]
                        ),
                        minimum_boundary_margin=MINIMUM_BOUNDARY_MARGIN,
                    )
                    if (
                        candidate["all_match"]
                        and candidate["minimum_boundary_margin"]
                        >= MINIMUM_BOUNDARY_MARGIN
                    ):
                        selected = {"step": step, **candidate}
                        break
            if selected is None:
                raise RuntimeError(
                    f"No sparse box solution for part={part}, coordinate={coordinate}"
                )
            steps[part, coordinate] = selected["step"]
            offsets[part, coordinate] = selected["offset"]
            coordinates.append(
                {
                    "part": part,
                    "coordinate": coordinate,
                    "selection": selection,
                    "original": original,
                    "selected": selected,
                }
            )
    result = {
        "step": steps.tolist(),
        "offsets": offsets.tolist(),
        "minimum_boundary_margin": min(
            row["selected"]["minimum_boundary_margin"] for row in coordinates
        ),
        "coordinates": coordinates,
        "changed_coordinates": sum(
            row["selection"] != "locked_original" for row in coordinates
        ),
        "passed": True,
    }
    return result, [result]


def build_piecewise_box(
    values: dict[str, np.ndarray],
    *,
    level_mode: str,
) -> tuple[dict, list[dict]]:
    if level_mode not in {"mode", "mean"}:
        raise ValueError(f"Unsupported piecewise level mode: {level_mode}")
    configuration = []
    coordinates = []
    total_changed = 0
    for part in range(2):
        for coordinate in range(4):
            step = np.float32(LOCKED_GEOMETRY_BOX_STEP)
            offset = np.float32(
                DEFAULT_GEOMETRY_BOX_OFFSETS[part][coordinate]
            )
            k_values = np.arange(-4, 310, dtype=np.int64)
            all_thresholds = (
                (k_values.astype(np.float64) + 0.5 - float(offset))
                * float(step)
            )
            thresholds = all_thresholds[
                (all_thresholds > 0.0) & (all_thresholds < 1.0)
            ].astype(np.float32)
            probes = np.empty(thresholds.size + 1, dtype=np.float32)
            probes[0] = np.float32(0.0)
            probes[-1] = np.float32(1.0)
            if thresholds.size > 1:
                probes[1:-1] = 0.5 * (
                    thresholds[:-1] + thresholds[1:]
                )
            elif thresholds.size == 1:
                probes = np.asarray([0.0, 1.0], dtype=np.float32)
            uniform_levels = (
                np.rint(probes / step + offset) - offset
            ) * step
            uniform_levels = np.clip(uniform_levels, 0.0, 1.0).astype(
                np.float32
            )
            reference = values["pytorch_boxes"][:, part, coordinate]
            backend_rows = [
                values["cpu_boxes"][:, part, coordinate],
                values["cuda_boxes"][:, part, coordinate],
            ]
            variants = np.stack((reference, *backend_rows), axis=1)
            variant_bins = np.sum(
                variants[:, :, None] >= thresholds[None, None, :],
                axis=2,
            )
            keep = np.ones(thresholds.size, dtype=bool)
            for row in variant_bins:
                lower = int(row.min())
                upper = int(row.max())
                if upper > lower:
                    keep[lower:upper] = False
            group_for_bin = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(keep.astype(np.int64)),
                )
            )
            group_count = int(group_for_bin[-1]) + 1
            reference_bins = variant_bins[:, 0]
            reference_groups = group_for_bin[reference_bins]
            backend_groups = [
                group_for_bin[variant_bins[:, index]]
                for index in range(1, variant_bins.shape[1])
            ]
            if any(
                not np.array_equal(reference_groups, group)
                for group in backend_groups
            ):
                raise RuntimeError("Piecewise merge did not stabilize backend bins")
            levels = np.empty(group_count, dtype=np.float32)
            changed = np.zeros(reference.shape[0], dtype=bool)
            original_output = uniform_levels[reference_bins]
            for group in range(group_count):
                original_bins = np.flatnonzero(group_for_bin == group)
                observed = reference_bins[reference_groups == group]
                if observed.size:
                    if level_mode == "mean":
                        levels[group] = np.float32(
                            np.mean(
                                uniform_levels[observed],
                                dtype=np.float64,
                            )
                        )
                    else:
                        unique, counts = np.unique(observed, return_counts=True)
                        selected_bin = int(unique[np.argmax(counts)])
                        levels[group] = uniform_levels[selected_bin]
                else:
                    selected_bin = int(original_bins[len(original_bins) // 2])
                    levels[group] = uniform_levels[selected_bin]
                changed[reference_groups == group] = (
                    original_output[reference_groups == group]
                    != levels[group]
                )
            changed_count = int(changed.sum())
            total_changed += changed_count
            piecewise_output = levels[reference_groups]
            displacement = piecewise_output - original_output
            record = {
                "part": part,
                "coordinate": coordinate,
                "thresholds": thresholds[keep].tolist(),
                "levels": levels.tolist(),
            }
            configuration.append(record)
            coordinates.append(
                {
                    "part": part,
                    "coordinate": coordinate,
                    "original_thresholds": int(thresholds.size),
                    "retained_thresholds": int(keep.sum()),
                    "merged_thresholds": int((~keep).sum()),
                    "changed_reference_records": changed_count,
                    "mean_abs_change_from_locked_grid": float(
                        np.mean(np.abs(displacement), dtype=np.float64)
                    ),
                    "max_abs_change_from_locked_grid": float(
                        np.max(np.abs(displacement), initial=0.0)
                    ),
                    "mean_squared_change_from_locked_grid": float(
                        np.mean(displacement.astype(np.float64) ** 2)
                    ),
                    "backend_records": int(reference.size * len(backend_rows)),
                    "all_backend_groups_match": True,
                }
            )
    result = {
        "step": None,
        "level_mode": level_mode,
        "offsets": [
            list(row) for row in DEFAULT_GEOMETRY_BOX_OFFSETS
        ],
        "piecewise": configuration,
        "coordinates": coordinates,
        "changed_reference_coordinates": total_changed,
        "passed": True,
    }
    return result, [result]


def build_continuous_box() -> tuple[dict, list[dict]]:
    """Keep predicted boxes continuous and defer stability to full-graph parity."""

    result = {
        "step": None,
        "level_mode": None,
        "offsets": [
            list(row) for row in DEFAULT_GEOMETRY_BOX_OFFSETS
        ],
        "piecewise": None,
        "minimum_boundary_margin": None,
        "coordinates": [],
        "passed": True,
    }
    return result, [result]


def build_shifted_boundary_box(
    values: dict[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    """Move separable grid boundaries and merge only contradictory ones.

    Every retained boundary is moved by the minimum amount needed to put the
    PyTorch, CPU ONNX, and CUDA ONNX values on the PyTorch reference side with
    a small margin. This preserves the locked 1/300 output level for every
    reference record except groups whose backend labels are not separable in
    one dimension; only those contradictory boundaries are merged.
    """

    configuration = []
    coordinates = []
    total_changed = 0
    total_shifted = 0
    total_removed = 0
    for part in range(2):
        for coordinate in range(4):
            step = np.float32(LOCKED_GEOMETRY_BOX_STEP)
            offset = np.float32(
                DEFAULT_GEOMETRY_BOX_OFFSETS[part][coordinate]
            )
            k_values = np.arange(-4, 310, dtype=np.int64)
            all_thresholds = (
                (k_values.astype(np.float64) + 0.5 - float(offset))
                * float(step)
            )
            thresholds = all_thresholds[
                (all_thresholds > 0.0) & (all_thresholds < 1.0)
            ].astype(np.float32)
            probes = np.empty(thresholds.size + 1, dtype=np.float32)
            probes[0] = np.float32(0.0)
            probes[-1] = np.float32(1.0)
            probes[1:-1] = 0.5 * (
                thresholds[:-1] + thresholds[1:]
            )
            uniform_levels = (
                np.rint(probes / step + offset) - offset
            ) * step
            uniform_levels = np.clip(
                uniform_levels, 0.0, 1.0
            ).astype(np.float32)

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
                reference[:, None] >= thresholds[None, :],
                axis=1,
            )
            keep = np.ones(thresholds.size, dtype=bool)
            shifted_thresholds = thresholds.copy()
            shift_records = []
            removed_records = []
            construction_margin = 2.0 * MINIMUM_BOUNDARY_MARGIN
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
                actual_lower_margin = float(selected) - lower_max
                actual_upper_margin = upper_min - float(selected)
                if (
                    actual_lower_margin < MINIMUM_BOUNDARY_MARGIN
                    or actual_upper_margin < MINIMUM_BOUNDARY_MARGIN
                ):
                    raise RuntimeError(
                        "Float32 boundary lost the required backend margin"
                    )
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
                            "lower_margin": actual_lower_margin,
                            "upper_margin": actual_upper_margin,
                        }
                    )

            retained_thresholds = shifted_thresholds[keep]
            if retained_thresholds.size > 1 and not np.all(
                retained_thresholds[1:] > retained_thresholds[:-1]
            ):
                raise RuntimeError("Shifted box boundaries are not sorted")
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
                    "Shifted boundaries did not stabilize backend groups"
                )

            group_count = int(group_for_original_bin[-1]) + 1
            levels = np.empty(group_count, dtype=np.float32)
            for group in range(group_count):
                original_bins = np.flatnonzero(
                    group_for_original_bin == group
                )
                observed = original_reference_bins[
                    expected_groups == group
                ]
                if observed.size:
                    unique, counts = np.unique(
                        observed, return_counts=True
                    )
                    selected_bin = int(unique[np.argmax(counts)])
                else:
                    selected_bin = int(
                        original_bins[len(original_bins) // 2]
                    )
                levels[group] = uniform_levels[selected_bin]
            original_output = uniform_levels[original_reference_bins]
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
                    "mean_abs_change_from_locked_grid": float(
                        np.mean(np.abs(displacement), dtype=np.float64)
                    ),
                    "max_abs_change_from_locked_grid": float(
                        np.max(np.abs(displacement), initial=0.0)
                    ),
                    "shift_records": shift_records,
                    "removed_records": removed_records,
                    "all_backend_groups_match": True,
                }
            )
    result = {
        "step": None,
        "level_mode": "boundary_shift_mode",
        "offsets": [
            list(row) for row in DEFAULT_GEOMETRY_BOX_OFFSETS
        ],
        "piecewise": configuration,
        "coordinates": coordinates,
        "shifted_thresholds": total_shifted,
        "removed_thresholds": total_removed,
        "changed_reference_coordinates": total_changed,
        "minimum_boundary_margin": MINIMUM_BOUNDARY_MARGIN,
        "passed": True,
    }
    return result, [result]


def main() -> None:
    args = parse_args()
    arrays_path = args.raw_arrays.expanduser().resolve()
    calibration_path = args.calibration_report.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (arrays_path, calibration_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if sha256_file(manifest_path) != DEV_MANIFEST_SHA256:
        raise RuntimeError("Stability derivation is restricted to locked dev")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("manifest_sha256") != DEV_MANIFEST_SHA256:
        raise RuntimeError("Calibration report did not use locked dev")
    if calibration.get("arrays_sha256") != sha256_file(arrays_path):
        raise RuntimeError("Raw geometry arrays changed after calibration")
    payload = np.load(arrays_path, allow_pickle=False)
    required = (
        "pytorch_boxes",
        "pytorch_angles",
        "cpu_boxes",
        "cpu_angles",
        "cuda_boxes",
        "cuda_angles",
        "source_sha256",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise RuntimeError(f"Raw geometry arrays are incomplete: {missing}")
    values = {name: np.asarray(payload[name]) for name in required}
    records = int(values["source_sha256"].reshape(-1).shape[0])
    if records != 144:
        raise RuntimeError(f"Expected 144 protected development records, got {records}")
    if values["pytorch_boxes"].shape != (records, 2, 4):
        raise RuntimeError(f"Expected {records} [2,4] PyTorch boxes")
    if values["pytorch_angles"].reshape(-1).shape != (records,):
        raise RuntimeError(f"Expected {records} PyTorch angles")
    for name in ("cpu_boxes", "cuda_boxes"):
        if values[name].shape != values["pytorch_boxes"].shape:
            raise RuntimeError(f"{name} shape mismatch")
    for name in ("pytorch_angles", "cpu_angles", "cuda_angles"):
        values[name] = values[name].reshape(-1)
    if len(set(values["source_sha256"].astype(str).tolist())) != records:
        raise RuntimeError(f"Expected {records} unique development source hashes")

    if args.box_mode == "sparse":
        selected_box, box_search = search_sparse_box(values)
    elif args.box_mode == "piecewise":
        selected_box, box_search = build_piecewise_box(
            values,
            level_mode=args.piecewise_level_mode,
        )
    elif args.box_mode == "boundary_shift":
        selected_box, box_search = build_shifted_boundary_box(values)
    elif args.box_mode == "continuous":
        selected_box, box_search = build_continuous_box()
    else:
        selected_box, box_search = search_box(values)
    selected_angle, angle_search = search_angle(values)
    use_stable_angle = args.angle_mode == "stable"
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development_only_backend_geometry_discretization",
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": sha256_file(calibration_path),
        "raw_arrays": str(arrays_path),
        "raw_arrays_sha256": sha256_file(arrays_path),
        "geometry_checkpoint_sha256": calibration[
            "geometry_checkpoint_sha256"
        ],
        "manifest": str(manifest_path),
        "manifest_sha256": DEV_MANIFEST_SHA256,
        "records": records,
        "minimum_boundary_margin": MINIMUM_BOUNDARY_MARGIN,
        "box_mode": args.box_mode,
        "piecewise_level_mode": selected_box.get("level_mode"),
        "box_step": selected_box["step"],
        "box_offsets": selected_box["offsets"],
        "box_piecewise": selected_box.get("piecewise"),
        "box_minimum_boundary_margin": selected_box.get(
            "minimum_boundary_margin"
        ),
        "angle_mode": args.angle_mode,
        "angle_step": selected_angle["step"] if use_stable_angle else None,
        "angle_offset": selected_angle["offset"] if use_stable_angle else 0.0,
        "angle_minimum_boundary_margin": (
            selected_angle["minimum_boundary_margin"]
            if use_stable_angle
            else None
        ),
        "box_search": box_search,
        "angle_search": angle_search,
        "checks": {
            "box_all_backend_bins_match": (
                None if args.box_mode == "continuous" else True
            ),
            "continuous_box_requires_full_graph_parity": (
                args.box_mode == "continuous"
            ),
            "angle_all_backend_bins_match": (
                True if use_stable_angle else None
            ),
            "continuous_angle_requires_full_graph_parity": (
                not use_stable_angle
            ),
            "changed_coordinates_boundary_margin_passed": (
                args.box_mode != "sparse"
                or all(
                    row["selection"] == "locked_original"
                    or row["selected"]["minimum_boundary_margin"]
                    >= MINIMUM_BOUNDARY_MARGIN
                    for row in selected_box["coordinates"]
                )
            ),
            "stable_original_coordinates_preserved": (
                args.box_mode != "sparse"
                or all(
                    row["selection"] != "locked_original"
                    or (
                        row["selected"]["step"]
                        == LOCKED_GEOMETRY_BOX_STEP
                        and row["selected"]["offset"]
                        == row["original"]["offset"]
                    )
                    for row in selected_box["coordinates"]
                )
            ),
            "piecewise_backend_groups_match": (
                True if args.box_mode == "piecewise" else None
            ),
            "boundary_shift_backend_groups_match": (
                True if args.box_mode == "boundary_shift" else None
            ),
            "development_manifest_locked": True,
        },
        "passed": True,
        "blind_data_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
