#!/usr/bin/env python3
"""Prepare frozen AnyFace/SAM2 geometry for one locked UnifiedPetReID split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg  # noqa: E402

from pet_id import add_retri_config  # noqa: E402
from pet_id.dogfacenet_alignment import (  # noqa: E402
    AlignmentIndexRecord,
    dogfacenet_identity_from_filename,
    geometry_cache_namespace,
    prepare_alignment_record,
)
from pet_id.localization import AnyFaceDetector, SAM2NoseSegmenter  # noqa: E402
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.unified_fresh_protocol import sha256_file  # noqa: E402
from pet_id.workspace_paths import normalize_runtime_config  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    identity_profile = get_runtime_profile("legacy-semantic")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_manifest", type=Path)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=identity_profile.config,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-long-side", type=int, default=1280)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args()


def alignment_record(row: dict) -> AlignmentIndexRecord:
    identity = str(row["identity"])
    canonical = str(row["canonical_filename"])
    parsed = dogfacenet_identity_from_filename(canonical)
    if parsed.casefold() != identity.casefold():
        raise RuntimeError(
            f"Canonical identity mismatch: {canonical!r} -> {parsed!r}, "
            f"manifest says {identity!r}"
        )
    return AlignmentIndexRecord(
        source_path=Path(row["source_path"]).expanduser().resolve(),
        canonical_filename=canonical,
        identity=identity,
        left_eye=tuple(float(value) for value in row["left_eye"]),
        right_eye=tuple(float(value) for value in row["right_eye"]),
        nose=tuple(float(value) for value in row["nose"]),
    )


def main() -> None:
    args = parse_args()
    raw_path = args.raw_manifest.expanduser().resolve()
    lock_path = args.protocol_lock.expanduser().resolve()
    config_path = args.config_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (raw_path, lock_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    split = str(raw.get("protocol_split", ""))
    if split not in lock["splits"]:
        raise RuntimeError(f"Split {split!r} is absent from the protocol lock")
    locked = lock["splits"][split]
    if sha256_file(raw_path) != locked["sha256"]:
        raise RuntimeError("Raw manifest hash differs from the protocol lock")
    rows = list(raw.get("records", ()))
    if len(rows) != int(locked["records"]):
        raise RuntimeError("Raw manifest record count differs from the protocol lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    options = cfg.MULTIMODAL
    namespace = geometry_cache_namespace(
        (options.ANYFACE_WEIGHTS, options.SAM2_CHECKPOINT),
        {
            "geometry_schema": 3,
            "anyface_image_size": options.ANYFACE_IMAGE_SIZE,
            "anyface_confidence": options.ANYFACE_CONFIDENCE,
            "sam2_config": options.SAM2_CONFIG,
            "max_long_side": args.max_long_side,
        },
    )
    detector = AnyFaceDetector(
        options.ANYFACE_WEIGHTS,
        repository_root=options.ANYFACE_ROOT,
        device=args.device,
        image_size=options.ANYFACE_IMAGE_SIZE,
        confidence_threshold=options.ANYFACE_CONFIDENCE,
    )
    segmenter = SAM2NoseSegmenter(
        options.SAM2_CHECKPOINT,
        config=options.SAM2_CONFIG,
        device=args.device,
    )

    prepared: list[dict] = []
    failures: list[dict] = []

    def checkpoint() -> None:
        identity_counts = Counter(
            str(record["identity"]).casefold() for record in prepared
        )
        payload = {
            **{key: value for key, value in raw.items() if key != "records"},
            "schema_version": 2,
            "raw_manifest": str(raw_path),
            "raw_manifest_sha256": locked["sha256"],
            "protocol_lock": str(lock_path),
            "protocol_lock_sha256": sha256_file(lock_path),
            "geometry_config": str(config_path),
            "geometry_config_sha256": sha256_file(config_path),
            "geometry_namespace": namespace,
            "max_long_side": int(args.max_long_side),
            "records": prepared,
            "failures": failures,
            "prepared_records": len(prepared),
            "prepared_identities": len(identity_counts),
            "identity_record_counts": dict(sorted(identity_counts.items())),
            "status": "COMPLETE" if len(prepared) + len(failures) == len(rows) else "RUNNING",
        }
        atomic_json(manifest_path, payload)

    for index, row in enumerate(rows, 1):
        source_path = Path(row["source_path"]).expanduser().resolve()
        if sha256_file(source_path) != row["source_sha256"]:
            raise RuntimeError(f"Locked source image hash changed: {source_path}")
        try:
            record = prepare_alignment_record(
                alignment_record(row),
                detector,
                segmenter,
                output_root=output_dir,
                namespace=namespace,
                max_long_side=args.max_long_side,
                allow_raw_nose_fallback=options.ALLOW_RAW_NOSE_FALLBACK,
            )
            record["source_sha256"] = row["source_sha256"]
            prepared.append(record)
        except Exception as error:
            failures.append(
                {
                    "record_index": index - 1,
                    "source_sha256": row["source_sha256"],
                    "identity": row["identity"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if index == 1 or index % max(int(args.checkpoint_interval), 1) == 0:
            checkpoint()
            print(
                json.dumps(
                    {
                        "split": split,
                        "processed": index,
                        "total": len(rows),
                        "prepared": len(prepared),
                        "failed": len(failures),
                    }
                ),
                flush=True,
            )
    checkpoint()
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    final["manifest_sha256_before_completion_record"] = sha256_file(manifest_path)
    final["completion"] = {
        "processed": len(rows),
        "prepared": len(prepared),
        "failed": len(failures),
        "all_records_prepared": not failures,
        "minimum_identity_records": min(final["identity_record_counts"].values(), default=0),
    }
    atomic_json(manifest_path, final)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "split": split,
                **final["completion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
