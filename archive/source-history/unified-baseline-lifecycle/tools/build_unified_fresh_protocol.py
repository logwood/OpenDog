#!/usr/bin/env python3
"""Lock a never-before-used DogFaceNet protocol for UnifiedPetReID v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import build_alignment_index  # noqa: E402
from pet_id.unified_fresh_protocol import (  # noqa: E402
    SPLIT_NAMES,
    build_fresh_protocol,
    collect_historical_identities,
    discover_history_manifests,
    protocol_manifest_summary,
    sha256_file,
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=WORKSPACE / "DogFaceNet_alignment"
    )
    parser.add_argument(
        "--history-root", type=Path, default=WORKSPACE / "artifacts/runs"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-identities", type=int, default=80)
    parser.add_argument("--development-identities", type=int, default=36)
    parser.add_argument("--blind-identities", type=int, default=36)
    parser.add_argument("--minimum-images-per-identity", type=int, default=4)
    parser.add_argument("--evaluation-images-per-identity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    history_root = args.history_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing protocol: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    history_paths = discover_history_manifests(history_root)
    history_identities, history_evidence = collect_historical_identities(
        history_paths
    )
    records, index_report = build_alignment_index(dataset_root)
    manifests, audit = build_fresh_protocol(
        records,
        historical_identities=history_identities,
        training_identities=args.training_identities,
        development_identities=args.development_identities,
        blind_identities=args.blind_identities,
        minimum_images_per_identity=args.minimum_images_per_identity,
        evaluation_images_per_identity=args.evaluation_images_per_identity,
        seed=args.seed,
    )
    audit.update(
        {
            "dataset_root": str(dataset_root),
            "labels_csv": str((dataset_root / "labels.csv").resolve()),
            "labels_csv_sha256": sha256_file(dataset_root / "labels.csv"),
            "alignment_index": index_report,
            "history_root": str(history_root),
            "history_manifest_files": len(history_evidence),
            "history_evidence": history_evidence,
            "blind_results_used": False,
            "blind_per_sample_results_used": False,
        }
    )

    output_paths: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        path = output_dir / f"{split_name}_raw_manifest.json"
        atomic_json(path, manifests[split_name])
        output_paths[split_name] = path
    audit_path = output_dir / "split_audit.json"
    atomic_json(audit_path, audit)

    lock = {
        "schema_version": 1,
        "protocol_name": "dogfacenet_unified_fresh_v2",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_UNSCORED",
        "policy": {
            "identity_disjoint": True,
            "exact_image_disjoint": True,
            "blind_single_candidate_attempt": True,
            "blind_training_forbidden": True,
            "blind_model_selection_forbidden": True,
            "failed_candidate_keeps_existing_default": True,
        },
        "inputs": {
            "labels_csv": str((dataset_root / "labels.csv").resolve()),
            "labels_csv_sha256": audit["labels_csv_sha256"],
            "history_root": str(history_root),
            "history_evidence_sha256": sha256_file(audit_path),
        },
        "splits": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                **protocol_manifest_summary(manifests[name]),
            }
            for name, path in output_paths.items()
        },
        "split_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "blind_attempt_marker": str(
            output_dir / "blind_test.attempt.json"
        ),
    }
    lock_path = output_dir / "protocol_lock.json"
    atomic_json(lock_path, lock)
    print(
        json.dumps(
            {
                "protocol_lock": str(lock_path),
                "protocol_lock_sha256": sha256_file(lock_path),
                "eligible_identities": audit["eligible_identities"],
                "splits": lock["splits"],
                "status": lock["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
