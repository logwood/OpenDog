#!/usr/bin/env python3
"""Lock a real-high-resolution train/dev/blind protocol for V4."""

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
from pet_id.unified_fresh_protocol import sha256_file  # noqa: E402
from pet_id.unified_highres_protocol import (  # noqa: E402
    PROTOCOL_NAME,
    SPLIT_NAMES,
    build_highres_protocol,
    manifest_summary,
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE / "data/raw/DogFaceNet_alignment",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-identities", type=int, default=20)
    parser.add_argument("--development-identities", type=int, default=8)
    parser.add_argument("--blind-identities", type=int, default=8)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--minimum-max-side", type=int, default=1280)
    parser.add_argument("--minimum-eye-distance", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite protocol: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, index_report = build_alignment_index(dataset_root)
    manifests, audit = build_highres_protocol(
        records,
        training_identities=args.training_identities,
        development_identities=args.development_identities,
        blind_identities=args.blind_identities,
        images_per_identity=args.images_per_identity,
        minimum_max_side=args.minimum_max_side,
        minimum_eye_distance=args.minimum_eye_distance,
        seed=args.seed,
    )
    audit.update(
        {
            "dataset_root": str(dataset_root),
            "labels_csv": str((dataset_root / "labels.csv").resolve()),
            "labels_csv_sha256": sha256_file(dataset_root / "labels.csv"),
            "alignment_index": index_report,
            "v3_blind_manifest_used": False,
            "previous_blind_features_used": False,
            "previous_blind_results_used": False,
        }
    )
    manifest_paths: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        path = output_dir / f"{split}.manifest.json"
        atomic_json(path, manifests[split])
        manifest_paths[split] = path
    audit_path = output_dir / "split_audit.json"
    atomic_json(audit_path, audit)
    lock = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_UNSCORED",
        "policy": {
            "v4_identity_disjoint": True,
            "exact_image_disjoint": True,
            "blind_single_candidate_attempt": True,
            "blind_training_forbidden": True,
            "blind_model_selection_forbidden": True,
            "blind_features_must_not_be_persisted": True,
            "failed_candidate_keeps_v3_default": True,
        },
        "source": {
            "labels_csv": str((dataset_root / "labels.csv").resolve()),
            "labels_csv_sha256": audit["labels_csv_sha256"],
            "split_audit": str(audit_path),
            "split_audit_sha256": sha256_file(audit_path),
        },
        "criteria": audit["criteria"],
        "splits": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                **manifest_summary(manifests[split]),
            }
            for split, path in manifest_paths.items()
        },
        "blind_attempt_marker": str(output_dir / "blind_test.attempt.json"),
        "default_backend_changed": False,
    }
    lock_path = output_dir / "protocol_lock.json"
    atomic_json(lock_path, lock)
    print(
        json.dumps(
            {
                "protocol_lock": str(lock_path),
                "protocol_lock_sha256": sha256_file(lock_path),
                "eligible_identities": audit["eligible_identities"],
                "reserve_identities": len(audit["reserve_identities"]),
                "splits": lock["splits"],
                "status": lock["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
