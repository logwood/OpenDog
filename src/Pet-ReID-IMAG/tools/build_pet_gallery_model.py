#!/usr/bin/env python3
"""Build a safe prototype gallery from labeled reference images."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.gallery import (
    build_pipeline,
    encode_primary,
    normalized_array,
    normalized_prototypes,
    sha256_file,
)
from pet_id.workspace_paths import SELECTED_MODELS_ROOT


def encode_references(
    records: list[dict],
    config: Path,
    checkpoint: Path | None,
    device: str,
    *,
    backend: str = "pytorch",
    onnx_model: Path | None = None,
    onnx_provider: str = "cuda",
    onnx_warmup_batches: tuple[int, ...] = (),
):
    pipeline = build_pipeline(
        config,
        checkpoint,
        device,
        backend=backend,
        onnx_model=onnx_model,
        onnx_provider=onnx_provider,
        onnx_warmup_batches=onnx_warmup_batches,
        verify_onnx_source_checkpoint=backend == "onnx",
    )
    backend_info = (
        pipeline.identity_model.backend_info()
        if hasattr(pipeline.identity_model, "backend_info")
        else {"backend": "pytorch", "device": str(pipeline.device)}
    )
    encoded = []
    try:
        for index, record in enumerate(records, 1):
            path = Path(record["library_path"])
            descriptor, inference = encode_primary(pipeline, path)
            encoded.append(
                {
                    "path": str(path.resolve()),
                    "fused": normalized_array(descriptor.fused_feature),
                    "nose": normalized_array(descriptor.nose_feature),
                    "face": normalized_array(descriptor.face_feature),
                    "inference": inference,
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "selected" if checkpoint or backend == "onnx" else "frozen",
                        "backend": backend_info,
                        "reference": index,
                        "total": len(records),
                        "path": str(path),
                        "detections": inference["detections"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return encoded, backend_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config-file", type=Path, default=Path("configs/multimodal_inference.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("pytorch", "onnx"), default="pytorch")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_joint800_v1" / "onnx" / "pet_embedding.onnx",
    )
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1,4,8")
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="omit frozen PyTorch ablation descriptors from the gallery package",
    )
    args = parser.parse_args()
    from pet_id.onnx_runtime import parse_warmup_batches

    warmup_batches = parse_warmup_batches(args.onnx_warmup_batches)

    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config_file.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gallery_records = [row for row in manifest["records"] if row["split"] == "gallery"]
    identities = list(manifest["identities"])
    if len(identities) < 2:
        raise ValueError("a gallery model needs at least two identities")
    counts = {identity: 0 for identity in identities}
    for row in gallery_records:
        counts[row["identity"]] += 1
    if not all(counts.values()):
        raise ValueError(f"every identity needs a gallery reference: {counts}")

    selected, selected_backend = encode_references(
        gallery_records,
        config_path,
        checkpoint_path,
        args.device,
        backend=args.backend,
        onnx_model=args.onnx_model.resolve(),
        onnx_provider=args.onnx_provider,
        onnx_warmup_batches=warmup_batches,
    )
    frozen = None
    frozen_backend = None
    if not args.production_only:
        frozen, frozen_backend = encode_references(
            gallery_records,
            config_path,
            None,
            args.device,
            backend="pytorch",
        )
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    reference_identity_indices = np.asarray(
        [identity_to_index[row["identity"]] for row in gallery_records], dtype=np.int64
    )
    arrays: dict[str, np.ndarray] = {
        "reference_identity_indices": reference_identity_indices,
    }
    variant_sources = {
        "selected_fused": (selected, "fused"),
        "selected_nose": (selected, "nose"),
        "selected_face": (selected, "face"),
    }
    if frozen is not None:
        variant_sources.update(
            frozen_fused=(frozen, "fused"),
            frozen_nose=(frozen, "nose"),
            frozen_face=(frozen, "face"),
        )
    for name, (rows, feature_name) in variant_sources.items():
        references = np.stack([row[feature_name] for row in rows]).astype(np.float32)
        arrays[f"{name}_references"] = references
        arrays[f"{name}_prototypes"] = normalized_prototypes(
            references, reference_identity_indices, len(identities)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "gallery_features.npz"
    np.savez_compressed(features_path, **arrays)
    model_path = output_dir / "gallery_model.json"
    variant_descriptions = {
        "selected_fused": "locked residual-view joint checkpoint",
        "frozen_fused": "frozen pretrained encoders with quality-gated concatenation",
        "selected_nose": "nose branch from the locked selected model",
        "selected_face": "face ArcFace branch from the locked selected model",
        "frozen_nose": "frozen nose checkpoint",
        "frozen_face": "frozen dog.pt ArcFace checkpoint",
    }
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "locked_multimodal_checkpoint_plus_l2_prototype_gallery",
        "identity_model_policy": (
            "the neural checkpoint is locked; only gallery reference prototypes are added"
        ),
        "prototype_policy": "mean reference descriptors per identity, then L2 normalize",
        "score": "cosine similarity",
        "identities": identities,
        "identity_to_index": identity_to_index,
        "references": [
            ({
                "identity": record["identity"],
                "path": record["library_path"],
                "sha256": record["sha256"],
                "selected_inference": selected[index]["inference"],
            } | (
                {"frozen_inference": frozen[index]["inference"]}
                if frozen is not None
                else {}
            ))
            for index, record in enumerate(gallery_records)
        ],
        "variants": {
            name: variant_descriptions[name] for name in variant_sources
        },
        "selected_backend": selected_backend,
        "frozen_backend": frozen_backend,
        "production_only": args.production_only,
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "features_file": features_path.name,
        "features_sha256": sha256_file(features_path),
    }
    model_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "gallery_model": str(model_path),
                "features": str(features_path),
                "identities": identities,
                "references": len(gallery_records),
                "variants": list(variant_sources),
                "selected_backend": selected_backend,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
