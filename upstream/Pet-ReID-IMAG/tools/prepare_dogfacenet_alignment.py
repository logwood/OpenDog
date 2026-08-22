#!/usr/bin/env python3
"""Precompute frozen AnyFace/SAM geometry for DogFaceNet identity training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.dogfacenet_alignment import (
    build_alignment_index,
    geometry_cache_namespace,
    prepare_alignment_record,
)
from pet_id.localization import AnyFaceDetector, SAM2NoseSegmenter


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="DogFaceNet_alignment directory")
    parser.add_argument(
        "--archive",
        default=None,
        help="optional original ZIP, used to repair mojibake filenames by CRC",
    )
    parser.add_argument("--config-file", default="configs/multimodal_dogfacenet_train.yaml")
    parser.add_argument("--output-dir", default="logs/dogfacenet_alignment_geometry")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-long-side", type=int, default=1280)
    parser.add_argument(
        "--min-eye-distance",
        type=float,
        default=128.0,
        help="minimum annotated eye spacing in source pixels (128 selects the high-detail tier)",
    )
    parser.add_argument("--min-images-per-identity", type=int, default=2)
    parser.add_argument("--max-identities", type=int, default=0)
    parser.add_argument("--max-images-per-identity", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-crc-repair", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    archive = Path(args.archive).resolve() if args.archive else None
    if archive is None:
        candidate = dataset_root.parent / f"{dataset_root.name}.zip"
        archive = candidate if candidate.is_file() else None
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    index, index_report = build_alignment_index(
        dataset_root,
        archive_path=archive,
        repair_crc=not args.no_crc_repair,
    )
    filtered = [item for item in index if item.eye_distance >= args.min_eye_distance]
    groups = defaultdict(list)
    for item in filtered:
        groups[item.identity.casefold()].append(item)
    groups = {
        identity: sorted(items, key=lambda item: item.eye_distance, reverse=True)
        for identity, items in groups.items()
        if len(items) >= args.min_images_per_identity
    }
    ranked_identities = sorted(
        groups,
        key=lambda identity: (
            -sum(item.eye_distance for item in groups[identity]) / len(groups[identity]),
            identity,
        ),
    )
    if args.max_identities > 0:
        ranked_identities = ranked_identities[: args.max_identities]
    selected = []
    for identity in ranked_identities:
        items = groups[identity]
        if args.max_images_per_identity > 0:
            items = items[: args.max_images_per_identity]
        selected.extend(items)
    if args.limit > 0:
        selected = selected[: args.limit]

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
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

    prepared, failures = [], []
    manifest_path = output_root / "manifest.json"
    for position, item in enumerate(selected, 1):
        try:
            prepared.append(
                prepare_alignment_record(
                    item,
                    detector,
                    segmenter,
                    output_root=output_root,
                    namespace=namespace,
                    max_long_side=args.max_long_side,
                    allow_raw_nose_fallback=options.ALLOW_RAW_NOSE_FALLBACK,
                )
            )
            status = "ok"
        except Exception as error:
            failures.append(
                {
                    "canonical_filename": item.canonical_filename,
                    "source_path": str(item.source_path.resolve()),
                    "identity": item.identity,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            status = "failed"
        print(f"[{position}/{len(selected)}] {status}: {item.canonical_filename}", flush=True)
        if position % 25 == 0:
            _write_manifest(
                manifest_path,
                {
                    "schema_version": 1,
                    "namespace": namespace,
                    "dataset_root": str(dataset_root),
                    "index_report": index_report,
                    "selection_count": len(selected),
                    "records": prepared,
                    "failures": failures,
                },
            )

    identity_counts = Counter(record["identity"].casefold() for record in prepared)
    manifest = {
        "schema_version": 1,
        "namespace": namespace,
        "dataset_root": str(dataset_root),
        "index_report": index_report,
        "selection": {
            "min_eye_distance": args.min_eye_distance,
            "min_images_per_identity": args.min_images_per_identity,
            "max_identities": args.max_identities,
            "max_images_per_identity": args.max_images_per_identity,
            "max_long_side": args.max_long_side,
        },
        "selection_count": len(selected),
        "records": prepared,
        "failures": failures,
        "prepared_identities": len(identity_counts),
        "trainable_identities": sum(
            count >= args.min_images_per_identity for count in identity_counts.values()
        ),
    }
    _write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "index": index_report,
                "selected": len(selected),
                "prepared": len(prepared),
                "failed": len(failures),
                "prepared_identities": len(identity_counts),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
