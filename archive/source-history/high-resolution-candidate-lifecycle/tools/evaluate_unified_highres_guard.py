#!/usr/bin/env python3
"""Guard a trained V4 candidate on locked V3/legacy development protocols.

V4 is exactly equal to V3 for any input whose longest side is at most 1280.
This evaluator nevertheless recomputes both candidates on the same source
images and checks aggregate retrieval counts, per-image anchor equality and a
synthetic nose-conflict path.  Blind manifests are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import _read_bgr  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    configure_strict_cuda_precision,
    sha256_file,
)
from pet_id.unified_highres import build_highres_from_checkpoint  # noqa: E402
from pet_id.unified_highres_data import load_raw_rgb  # noqa: E402
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v3-acceptance", type=Path, required=True)
    parser.add_argument("--v2-acceptance", type=Path, required=True)
    parser.add_argument("--v3-baseline-report", type=Path, required=True)
    parser.add_argument("--legacy-baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=32)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def donor_indices(identities: list[str], gallery_count: int = 2) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    names = sorted(grouped)
    donor = torch.arange(len(identities))
    for identity_index, identity in enumerate(names):
        source = grouped[identity][gallery_count:]
        replacement = grouped[names[(identity_index + 1) % len(names)]][gallery_count:]
        for source_index, replacement_index in zip(source, replacement):
            donor[source_index] = replacement_index
    return donor


def query_mask(identities: list[str], gallery_count: int = 2) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    mask = torch.zeros(len(identities), dtype=torch.bool)
    for indices in grouped.values():
        mask[indices[gallery_count:]] = True
    return mask


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "queries"}


def validate_acceptances(
    v3_path: Path,
    v2_path: Path,
) -> tuple[dict, dict, Path, Path]:
    v3 = load_json(v3_path)
    v2 = load_json(v2_path)
    if v3.get("protocol_name") != "unified_pet_reid_v3_external_strict_noninferiority":
        raise RuntimeError("Unexpected V3 acceptance")
    v3_manifest = Path(v3["development"]["path"]).expanduser().resolve()
    if sha256_file(v3_manifest) != v3["development"]["sha256"]:
        raise RuntimeError("V3 development manifest differs from acceptance")
    v2_manifest = Path(v2["development"]["path"]).expanduser().resolve()
    if sha256_file(v2_manifest) != v2["development"]["sha256"]:
        raise RuntimeError("V2 development manifest differs from acceptance")
    return v3, v2, v3_manifest, v2_manifest


def validate_baseline_report(
    report_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    purpose: str,
) -> dict:
    report = load_json(report_path)
    if report.get("purpose") != purpose:
        raise RuntimeError(f"Baseline report has wrong purpose: {report_path}")
    if report.get("blind_data_used") is not False or report.get("passed") is not True:
        raise RuntimeError(f"Baseline report is not valid: {report_path}")
    if report.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise RuntimeError(f"Baseline checkpoint hash mismatch: {report_path}")
    manifest_record = report.get("manifest")
    digest = (
        manifest_record.get("sha256")
        if isinstance(manifest_record, dict)
        else report.get("manifest_sha256")
    )
    if digest != expected_manifest_sha256:
        raise RuntimeError(f"Baseline manifest hash mismatch: {report_path}")
    return report


def evaluate_manifest(
    model,
    manifest_path: Path,
    *,
    device: torch.device,
    progress_every: int,
    label: str,
    conflict: bool,
) -> dict[str, Any]:
    payload = load_json(manifest_path)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError(f"Manifest has no records: {manifest_path}")
    identities = [str(record["identity"]).casefold() for record in records]
    source_paths = [str(Path(record["source_path"]).expanduser().resolve()) for record in records]
    candidate_rows: list[torch.Tensor] = []
    parent_rows: list[torch.Tensor] = []
    face_rows: list[torch.Tensor] = []
    nose_rows: list[torch.Tensor] = []
    confidence_rows: list[torch.Tensor] = []
    energy_face_rows: list[torch.Tensor] = []
    energy_nose_rows: list[torch.Tensor] = []
    detail_scale_rows: list[torch.Tensor] = []
    availability_rows: list[torch.Tensor] = []
    maximum_anchor_error = 0.0
    long_side_maximum = 0
    resized_oversize_records = 0

    with torch.inference_mode():
        for index, record in enumerate(records):
            source = Path(record["source_path"]).expanduser().resolve()
            if sha256_file(source) != str(record["source_sha256"]).casefold():
                raise RuntimeError(f"Source hash differs from manifest: {source}")
            raw, dimensions = load_raw_rgb(source, maximum_side=model.maximum_input_side)
            long_side_maximum = max(
                long_side_maximum,
                int(dimensions["original_height"]),
                int(dimensions["original_width"]),
            )
            resized_oversize_records += int(
                dimensions["original_height"] != dimensions["fed_height"]
                or dimensions["original_width"] != dimensions["fed_width"]
            )
            output = model(raw[None].to(device).contiguous(), return_aux=True)
            candidate = F.normalize(output["embedding"].float(), dim=1)
            parent = F.normalize(output["highres_parent_embedding"].float(), dim=1)
            anchor_error = float((candidate - parent).abs().max().cpu())
            maximum_anchor_error = max(maximum_anchor_error, anchor_error)
            candidate_rows.append(candidate[0].cpu())
            parent_rows.append(parent[0].cpu())
            if conflict:
                face_rows.append(output["detail_face_descriptor"][0].float().cpu())
                nose_rows.append(output["detail_nose_descriptor"][0].float().cpu())
                confidence_rows.append(output["geometry_confidence"][0, 0].float().cpu())
                energy_face_rows.append(output["detail_signals"][0, 8].float().cpu())
                energy_nose_rows.append(output["detail_signals"][0, 9].float().cpu())
                detail_scale_rows.append(output["detail_scale"][0].float().cpu())
                availability_rows.append(output["detail_availability"][0].float().cpu())
            if index == 0 or (index + 1) % max(progress_every, 1) == 0:
                print(f"{label}: {index + 1}/{len(records)}", flush=True)

    candidate_tensor = torch.stack(candidate_rows)
    parent_tensor = torch.stack(parent_rows)
    clean = retrieval_metrics(candidate_tensor, identities, source_paths)
    parent_clean = retrieval_metrics(parent_tensor, identities, source_paths)
    result: dict[str, Any] = {
        "records": len(records),
        "identities": len(set(identities)),
        "maximum_source_side": long_side_maximum,
        "resized_oversize_records": resized_oversize_records,
        "candidate_clean": compact(clean),
        "parent_clean": compact(parent_clean),
        "maximum_candidate_parent_abs_error": maximum_anchor_error,
        "candidate_parent_exact": maximum_anchor_error == 0.0,
    }
    if not conflict:
        return result

    donors = donor_indices(identities)
    mask = query_mask(identities)
    face = torch.stack(face_rows).to(device)
    nose = torch.stack(nose_rows)
    confidence = torch.stack(confidence_rows).to(device)
    detail_scale = torch.stack(detail_scale_rows).to(device)
    availability = torch.stack(availability_rows).to(device)
    face_energy = torch.stack(energy_face_rows).to(device)
    nose_energy = torch.stack(energy_nose_rows)
    with torch.inference_mode():
        corrupt = model.refiner(
            parent_tensor.to(device),
            face,
            nose.index_select(0, donors).to(device),
            confidence,
            detail_scale,
            availability,
            face_energy,
            nose_energy.index_select(0, donors).to(device),
        ).float().cpu()
    conflict_candidate = candidate_tensor.clone()
    conflict_candidate[mask] = corrupt[mask]
    conflict_parent = parent_tensor.clone()
    # On this locked legacy set availability is zero, so the V4 refiner must
    # preserve the parent even when nose evidence is deliberately mismatched.
    conflict_parent[mask] = parent_tensor[mask]
    result["candidate_conflict"] = compact(
        retrieval_metrics(conflict_candidate, identities, source_paths)
    )
    result["parent_conflict"] = compact(
        retrieval_metrics(conflict_parent, identities, source_paths)
    )
    result["maximum_conflict_parent_abs_error"] = float(
        (conflict_candidate[mask] - conflict_parent[mask]).abs().max()
    )
    result["conflict_parent_exact"] = (
        result["maximum_conflict_parent_abs_error"] == 0.0
    )
    return result


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    v3_path = args.v3_acceptance.expanduser().resolve()
    v2_path = args.v2_acceptance.expanduser().resolve()
    v3_report_path = args.v3_baseline_report.expanduser().resolve()
    legacy_report_path = args.legacy_baseline_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (
        checkpoint_path,
        v3_path,
        v2_path,
        v3_report_path,
        legacy_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)

    v3, v2, v3_manifest, legacy_manifest = validate_acceptances(v3_path, v2_path)
    model, checkpoint = build_highres_from_checkpoint(
        checkpoint_path,
        device=args.device,
        verify_sources=True,
    )
    if checkpoint.get("training", {}).get("blind_data_used") is not False:
        raise RuntimeError("V4 checkpoint provenance is not blind-safe")
    parent_hash = checkpoint["sources"]["parent_v3_checkpoint"]["sha256"]
    v3_baseline = validate_baseline_report(
        v3_report_path,
        expected_checkpoint_sha256=parent_hash,
        expected_manifest_sha256=sha256_file(v3_manifest),
        purpose="unified_v3_external_joint_development",
    )
    legacy_baseline = validate_baseline_report(
        legacy_report_path,
        expected_checkpoint_sha256=parent_hash,
        expected_manifest_sha256=sha256_file(legacy_manifest),
        purpose="external_joint_legacy_v2_development_guard",
    )
    precision = configure_strict_cuda_precision()
    device = torch.device(args.device)
    model.eval()
    v3_result = evaluate_manifest(
        model,
        v3_manifest,
        device=device,
        progress_every=args.progress_every,
        label="v4 locked V3 development guard",
        conflict=False,
    )
    legacy_result = evaluate_manifest(
        model,
        legacy_manifest,
        device=device,
        progress_every=args.progress_every,
        label="v4 legacy clean/conflict guard",
        conflict=True,
    )
    checks = {
        "v3_top1_not_below_production_v3": v3_result["candidate_clean"]["top1_correct"]
        >= v3_baseline["candidate"]["top1_correct"],
        "v3_top5_not_below_production_v3": v3_result["candidate_clean"]["top5_correct"]
        >= v3_baseline["candidate"]["top5_correct"],
        "v3_parent_top1_noninferior": v3_result["candidate_clean"]["top1_correct"]
        >= v3_result["parent_clean"]["top1_correct"],
        "v3_parent_top5_noninferior": v3_result["candidate_clean"]["top5_correct"]
        >= v3_result["parent_clean"]["top5_correct"],
        "legacy_clean_top1_not_below_production_v3": legacy_result["candidate_clean"]["top1_correct"]
        >= legacy_baseline["candidate"]["clean"]["top1_correct"],
        "legacy_clean_top5_not_below_production_v3": legacy_result["candidate_clean"]["top5_correct"]
        >= legacy_baseline["candidate"]["clean"]["top5_correct"],
        "legacy_conflict_top1_not_below_production_v3": legacy_result["candidate_conflict"]["top1_correct"]
        >= legacy_baseline["candidate"]["conflict"]["top1_correct"],
        "legacy_conflict_top5_not_below_production_v3": legacy_result["candidate_conflict"]["top5_correct"]
        >= legacy_baseline["candidate"]["conflict"]["top5_correct"],
        "legacy_exact_parent_anchor": legacy_result["candidate_parent_exact"],
        "legacy_conflict_exact_parent_anchor": legacy_result["conflict_parent_exact"],
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "unified_v4_locked_v3_and_legacy_development_guard",
        "blind_data_used": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "parent_v3_sha256": parent_hash,
        },
        "acceptances": {
            "v3": {"path": str(v3_path), "sha256": sha256_file(v3_path)},
            "v2": {"path": str(v2_path), "sha256": sha256_file(v2_path)},
        },
        "baseline_reports": {
            "v3": {"path": str(v3_report_path), "sha256": sha256_file(v3_report_path)},
            "legacy": {"path": str(legacy_report_path), "sha256": sha256_file(legacy_report_path)},
        },
        "manifests": {
            "v3_development": {"path": str(v3_manifest), "sha256": sha256_file(v3_manifest)},
            "legacy_development": {"path": str(legacy_manifest), "sha256": sha256_file(legacy_manifest)},
        },
        "cuda_precision": precision,
        "v3_development": v3_result,
        "legacy_clean_conflict": legacy_result,
        "noninferiority": {"checks": checks, "passed": all(checks.values())},
        "passed": all(checks.values()),
        "default_backend_changed": False,
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
                "sha256": sha256_file(output_path),
                "v3_development": v3_result,
                "legacy_clean_conflict": legacy_result,
                "noninferiority": report["noninferiority"],
                "blind_data_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
