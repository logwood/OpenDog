#!/usr/bin/env python3
"""Guard a new external-joint candidate on the locked legacy v2 development set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_external_model import (  # noqa: E402
    build_external_joint_from_checkpoint,
    configure_strict_cuda_precision,
    sha256_file,
)
from pet_id.unified_training import retrieval_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--v2-acceptance", type=Path, required=True)
    parser.add_argument("--v3-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


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


def compact(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "queries"}


def main() -> None:
    args = parse_args()
    precision = configure_strict_cuda_precision()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    v2_path = args.v2_acceptance.expanduser().resolve()
    v3_path = args.v3_acceptance.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, v2_path, v3_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    manifest_path = Path(v2["development"]["path"]).resolve()
    if sha256_file(manifest_path) != v2["development"]["sha256"]:
        raise RuntimeError("Legacy development manifest differs from v2 acceptance")
    baseline_path = Path(v2["baseline_lock"]["path"]).resolve()
    if sha256_file(baseline_path) != v2["baseline_lock"]["sha256"]:
        raise RuntimeError("Legacy baseline lock differs from v2 acceptance")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))[
        "reports"
    ]["development"]["metrics"]
    device = torch.device(args.device)
    model, payload = build_external_joint_from_checkpoint(
        checkpoint_path, device=device, verify_sources=True
    )
    if payload.get("training", {}).get("acceptance_sha256") != sha256_file(v3_path):
        raise RuntimeError("Candidate was not trained under this v3 acceptance")
    dataset = UnifiedManifestDataset(
        manifest_path,
        input_size=model.input_size,
        training=False,
        allow_letterbox_upscale=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    candidate_rows = []
    base_rows = []
    face_rows = []
    nose_rows = []
    confidence_rows = []
    identities: list[str] = []
    source_paths: list[str] = []
    records = 0
    with torch.inference_mode():
        for raw in loader:
            output = model(raw["rgb"].to(device), return_aux=True)
            candidate_rows.append(output["embedding"].float().cpu())
            base_rows.append(output["base_embedding"].float().cpu())
            face_rows.append(output["face_descriptor"].float().cpu())
            nose_rows.append(output["adapted_nose_descriptor"].float().cpu())
            confidence_rows.append(output["geometry_confidence"][:, 0].float().cpu())
            identities.extend(raw["identity"])
            source_paths.extend(raw["source_path"])
            records += int(raw["rgb"].shape[0])
            if records == len(raw["rgb"]) or records % 36 == 0:
                print(f"legacy guard: {records}/{len(dataset)}", flush=True)
    candidate = torch.cat(candidate_rows)
    parent = torch.cat(base_rows)
    face = torch.cat(face_rows)
    nose = torch.cat(nose_rows)
    confidence = torch.cat(confidence_rows)
    clean = retrieval_metrics(candidate, identities, source_paths)
    parent_clean = retrieval_metrics(parent, identities, source_paths)

    donors = donor_indices(identities)
    with torch.inference_mode():
        face_device = face.to(device)
        corrupt_nose = nose.index_select(0, donors).to(device)
        confidence_device = confidence.to(device)
        corrupt_base = model.base_model.fusion(
            face_device,
            corrupt_nose,
            confidence_device,
        )
        corrupt_candidate = model.refiner(
            corrupt_base,
            face_device,
            corrupt_nose,
            confidence_device,
        ).float().cpu()
    query_mask = torch.zeros(len(identities), dtype=torch.bool)
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    for indices in grouped.values():
        query_mask[indices[2:]] = True
    conflict_candidate = candidate.clone()
    conflict_candidate[query_mask] = corrupt_candidate[query_mask]
    conflict_parent = parent.clone()
    conflict_parent[query_mask] = corrupt_base.float().cpu()[query_mask]
    conflict = retrieval_metrics(conflict_candidate, identities, source_paths)
    parent_conflict = retrieval_metrics(conflict_parent, identities, source_paths)
    baseline_checks = {
        "clean_top1": clean["top1_correct"] >= baseline["top1_correct"],
        "clean_top5": clean["top5_correct"] >= baseline["top5_correct"],
    }
    parent_checks = {
        "clean_top1": clean["top1_correct"] >= parent_clean["top1_correct"],
        "clean_top5": clean["top5_correct"] >= parent_clean["top5_correct"],
        "conflict_top1": conflict["top1_correct"] >= parent_conflict["top1_correct"],
        "conflict_top5": conflict["top5_correct"] >= parent_conflict["top5_correct"],
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "external_joint_legacy_v2_development_guard",
        "blind_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cuda_precision": precision,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate": {"clean": compact(clean), "conflict": compact(conflict)},
        "parent": {
            "clean": compact(parent_clean),
            "conflict": compact(parent_conflict),
        },
        "semantic_v3_baseline": compact(baseline),
        "semantic_v3_noninferiority": {
            "checks": baseline_checks,
            "passed": all(baseline_checks.values()),
        },
        "parent_noninferiority": {
            "checks": parent_checks,
            "passed": all(parent_checks.values()),
        },
        "passed": all(baseline_checks.values()) and all(parent_checks.values()),
        "default_backend_changed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
