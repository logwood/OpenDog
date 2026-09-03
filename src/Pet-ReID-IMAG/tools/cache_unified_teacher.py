#!/usr/bin/env python3
"""Cache legacy-semantic descriptors for unified development training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg  # noqa: E402

from pet_id import add_retri_config  # noqa: E402
from pet_id.dogfacenet_alignment import (  # noqa: E402
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model  # noqa: E402
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.release_compatibility import acceptance_path  # noqa: E402
from pet_id.workspace_paths import normalize_runtime_config  # noqa: E402


FORBIDDEN_TRAINING_SPLIT_TOKENS = (
    "blind",
    "test",
    "spent",
    "fresh",
    "development",
    "validation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    profile = get_runtime_profile("legacy-semantic")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=profile.identity_weights,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=profile.config,
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=acceptance_path(WORKSPACE, "legacy-training"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-evaluation-cache",
        action="store_true",
        help="Permit non-training splits for evaluation only; never pass such a cache to training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config_file.expanduser().resolve()
    acceptance_path = args.acceptance.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("protocol_split", "")).casefold()
    if (
        any(token in split for token in FORBIDDEN_TRAINING_SPLIT_TOKENS)
        and not args.allow_evaluation_cache
    ):
        raise ValueError(
            f"Refusing to build a training teacher cache from protected split {split!r}"
        )
    for path in (manifest_path, checkpoint_path, config_path, acceptance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint_path)
    config_hash = sha256_file(config_path)
    teacher_name = None
    for name, source in acceptance.get("teacher_sources", {}).items():
        if (
            source["checkpoint"]["sha256"] == checkpoint_hash
            and source["config"]["sha256"] == config_hash
        ):
            teacher_name = name
            break
    if teacher_name is None:
        raise RuntimeError(
            "Teacher checkpoint/config pair is not locked by the acceptance protocol"
        )

    device = torch.device(args.device)
    dataset = PreparedDogFaceNetDataset(manifest_path, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prepared_dogfacenet,
    )
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=str(checkpoint_path),
    ).eval()

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    manifest_hash = sha256_file(manifest_path)
    model_hash = checkpoint_hash
    cache_key = hashlib.sha256(
        f"{manifest_hash}:{model_hash}:{config_hash}:{amp_dtype}".encode()
    ).hexdigest()[:16]
    shard_dir = output_path.parent / f"{output_path.stem}_shards_{cache_key}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    path_to_record = {
        str(Path(record["source_path"]).resolve()).casefold(): record
        for record in dataset.records
    }
    for batch_index, batch in enumerate(loader):
        shard_path = shard_dir / f"batch_{batch_index:06d}.npz"
        if shard_path.is_file():
            with np.load(shard_path, allow_pickle=False) as shard:
                if (
                    str(shard["cache_key"].item()) == cache_key
                    and len(shard["source_sha256"]) == len(batch["source_paths"])
                ):
                    print(
                        f"teacher cache hit: {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}",
                        flush=True,
                    )
                    continue
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "targets"
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = model(**inputs)
        records = [
            path_to_record[str(Path(path).resolve()).casefold()]
            for path in batch["source_paths"]
        ]
        temporary = shard_path.with_suffix(".exporting.npz")
        np.savez_compressed(
            temporary,
            cache_key=np.asarray(cache_key),
            source_sha256=np.asarray(
                [record["source_sha256"] for record in records]
            ),
            source_paths=np.asarray(
                [str(Path(path).resolve()) for path in batch["source_paths"]]
            ),
            identities=np.asarray(
                [identity.casefold() for identity in batch["identities"]]
            ),
            embedding=output["features"].detach().float().cpu().numpy(),
            face_embedding=output["face_features"].detach().float().cpu().numpy(),
        )
        os.replace(temporary, shard_path)
        print(
            f"teacher features: {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}",
            flush=True,
        )

    arrays: dict[str, list[np.ndarray]] = {
        "source_sha256": [],
        "source_paths": [],
        "identities": [],
        "embedding": [],
        "face_embedding": [],
    }
    shard_paths = sorted(shard_dir.glob("batch_*.npz"))
    expected_shards = (len(dataset) + args.batch_size - 1) // args.batch_size
    if len(shard_paths) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} shards, found {len(shard_paths)}"
        )
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            if str(shard["cache_key"].item()) != cache_key:
                raise RuntimeError(f"Stale teacher shard: {shard_path}")
            for name in arrays:
                arrays[name].append(np.asarray(shard[name]))
    combined = {
        name: np.concatenate(rows, axis=0)
        for name, rows in arrays.items()
    }
    if len(combined["source_sha256"]) != len(dataset):
        raise RuntimeError("Combined teacher cache row count is incorrect")
    if (
        combined["embedding"].ndim != 2
        or combined["embedding"].shape[0] != len(dataset)
    ):
        raise RuntimeError(
            "Expected [records,descriptor_dim] teacher embedding, "
            f"got {combined['embedding'].shape}"
        )
    if combined["face_embedding"].shape != (len(dataset), 512):
        raise RuntimeError("Expected [records,512] teacher face_embedding")
    if not np.isfinite(combined["embedding"]).all():
        raise FloatingPointError("Teacher cache contains non-finite embeddings")
    norms = np.linalg.norm(combined["embedding"], axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError(
            f"Teacher embeddings are not normalized: {norms.min()}..{norms.max()}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".exporting.npz")
    np.savez_compressed(temporary, **combined)
    os.replace(temporary, output_path)
    metadata = {
        "schema_version": 1,
        "purpose": "unified_pet_reid_development_distillation",
        "training_eligible": not any(
            token in split for token in FORBIDDEN_TRAINING_SPLIT_TOKENS
        ),
        "protocol_split": split,
        "teacher_name": teacher_name,
        "acceptance_sha256": sha256_file(acceptance_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": model_hash,
        "config_file": str(config_path),
        "config_sha256": config_hash,
        "records": len(dataset),
        "identities": dataset.num_classes,
        "amp_dtype": str(amp_dtype).removeprefix("torch.")
        if use_amp
        else "float32",
        "cache_key": cache_key,
        "archive": str(output_path),
        "archive_sha256": sha256_file(output_path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in combined.items()
        },
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

