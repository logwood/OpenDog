#!/usr/bin/env python3
"""Build the external YT-BB-Dog UnifiedPetReID v3 protocol once."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_protocol import (  # noqa: E402
    PROTOCOL_SPLITS,
    build_external_protocol,
    collect_identity_images,
    hash_image_roots,
    sha256_file,
    validate_raw_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bifor-root",
        type=Path,
        default=WORKSPACE / "data/BIFOR/YT-BB-dog",
    )
    parser.add_argument(
        "--historical-root",
        action="append",
        type=Path,
        default=None,
        help="Image root known to have influenced existing models; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/unified/v3/external_bifor_protocol_20260831",
    )
    parser.add_argument("--training-identities", type=int, default=1024)
    parser.add_argument("--development-identities", type=int, default=128)
    parser.add_argument("--blind-identities", type=int, default=128)
    parser.add_argument("--training-images-per-identity", type=int, default=4)
    parser.add_argument("--evaluation-images-per-identity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Protocol output is immutable and already exists: {output_dir}"
        )
    bifor_root = args.bifor_root.expanduser().resolve()
    historical_roots = args.historical_root or [
        WORKSPACE / "data/processed/pet-reid-imag/dir_train_fusai",
        WORKSPACE / "data/raw/DogFaceNet_alignment",
    ]
    historical_roots = [path.expanduser().resolve() for path in historical_roots]

    print("hashing historical model-training image roots", flush=True)
    historical = hash_image_roots(historical_roots)
    print(
        f"historical unique images: {len(historical['unique_sha256'])}",
        flush=True,
    )
    print("hashing YT-BB-Dog official training split", flush=True)
    training_groups, training_audit = collect_identity_images(
        bifor_root / "train",
        dataset_namespace="bifor:yt-bb-dog",
        source_split="official_train",
    )
    print("hashing YT-BB-Dog official test split", flush=True)
    evaluation_groups, evaluation_audit = collect_identity_images(
        bifor_root / "test",
        dataset_namespace="bifor:yt-bb-dog",
        source_split="official_test",
    )
    manifests, audit = build_external_protocol(
        training_groups=training_groups,
        evaluation_groups=evaluation_groups,
        historical_sha256=historical["unique_sha256"],
        training_identities=args.training_identities,
        development_identities=args.development_identities,
        blind_identities=args.blind_identities,
        training_images_per_identity=args.training_images_per_identity,
        evaluation_images_per_identity=args.evaluation_images_per_identity,
        seed=args.seed,
    )
    audit["sources"] = {
        "historical": {
            "roots": historical["roots"],
            "duplicate_sha256": historical["duplicate_sha256"],
        },
        "bifor_training": training_audit,
        "bifor_evaluation": evaluation_audit,
    }

    output_dir.mkdir(parents=True)
    manifest_records = {}
    for split in PROTOCOL_SPLITS:
        path = output_dir / f"{split}.manifest.json"
        validate_raw_manifest(manifests[split], expected_split=split)
        write_json(path, manifests[split])
        summary = validate_raw_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            expected_split=split,
        )
        manifest_records[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            **summary,
            "images_per_identity": manifests[split]["images_per_identity"],
        }
    audit_path = output_dir / "audit.json"
    write_json(audit_path, audit)
    module_path = ROOT / "pet_id/unified_external_protocol.py"
    tool_path = Path(__file__).resolve()
    lock = {
        "schema_version": 1,
        "protocol_name": "unified_pet_reid_external_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "YT-BB-Dog",
            "root": str(bifor_root),
            "training_source": "official train identities",
            "evaluation_source": "official test identities",
        },
        "manifests": manifest_records,
        "audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "code_sha256": {
            str(module_path.relative_to(WORKSPACE)).replace("\\", "/"): sha256_file(
                module_path
            ),
            str(tool_path.relative_to(WORKSPACE)).replace("\\", "/"): sha256_file(
                tool_path
            ),
        },
        "policy": {
            "identity_disjoint": True,
            "exact_source_sha256_disjoint": True,
            "blind_candidate_attempts": 1,
            "blind_results_must_not_change_candidate": True,
            "failed_candidate_keeps_existing_default": True,
            "promotion_rule": (
                "candidate Top-1 and Top-5 counts must both match or exceed "
                "the locked semantic-v3 baseline"
            ),
        },
        "status": "LOCKED_AWAITING_BASELINE",
    }
    lock_path = output_dir / "protocol_lock.json"
    write_json(lock_path, lock)
    print(
        json.dumps(
            {
                "protocol_lock": str(lock_path),
                "protocol_lock_sha256": sha256_file(lock_path),
                "manifests": manifest_records,
                "historical_unique_sha256": len(historical["unique_sha256"]),
                "historical_overlap": audit["historical_overlap"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
