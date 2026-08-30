#!/usr/bin/env python3
"""Serve incremental pet enrollment and identification over HTTP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.api import create_app  # noqa: E402
from pet_id.gallery import build_pipeline, sha256_file  # noqa: E402
from pet_id.gallery_service import (  # noqa: E402
    MultimodalPipelineEncoder,
    PetGalleryStore,
    PetIdentificationService,
)
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.recognition_agent import (  # noqa: E402
    AgentFeatureEncoder,
    MegaDescriptorEncoder,
)
from pet_id.workspace_paths import (  # noqa: E402
    GALLERY_STORE_ROOT,
    SELECTED_MODELS_ROOT,
    resolve_legacy_path,
)


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the semantic-v3 pet enrollment and identification API"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=GALLERY_STORE_ROOT / "pet_api_gallery_semantic_v3_v1",
    )
    parser.add_argument(
        "--seed-gallery-model",
        type=Path,
        help="idempotently import an existing gallery_model.json before serving",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "config.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=("pytorch", "onnx", "onnx-bifor"),
        default="onnx",
    )
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "model_final.pth",
        help="PyTorch weights, or the source checkpoint used to verify ONNX metadata",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=(
            SELECTED_MODELS_ROOT
            / "dogfacenet_semantic_v3_v1"
            / "onnx"
            / "pet_embedding.onnx"
        ),
    )
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=(
            SELECTED_MODELS_ROOT.parent
            / "pretrained"
            / "body_detection"
            / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        ),
        help="frozen target-dog detector used only by --backend onnx-bifor",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="enable score-level evidence fusion with frozen independent experts",
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        default=(
            SELECTED_MODELS_ROOT.parent
            / "pretrained"
            / "megadescriptor"
            / "MegaDescriptor-B-224"
            / "pytorch_model.bin"
        ),
    )
    parser.add_argument(
        "--megadescriptor-device",
        help="defaults to --device; use cpu to keep the expert off GPU",
    )
    parser.add_argument("--onnx-warmup-batches", default="1,4,8")
    parser.add_argument("--maximum-upload-mb", type=float, default=15.0)
    parser.add_argument("--maximum-image-megapixels", type=float, default=25.0)
    parser.add_argument("--maximum-images-per-request", type=int, default=8)
    parser.add_argument(
        "--allow-multiple-pets-for-enrollment",
        action="store_true",
        help="allow enrollment images with multiple detected pets (primary detection wins)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        help="optional calibrated cosine threshold; omitted means closed-set Top-1",
    )
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PET_REID_API_KEY"),
        help="X-API-Key value; prefer setting PET_REID_API_KEY",
    )
    parser.add_argument(
        "--allow-unauthenticated-remote",
        action="store_true",
        help="explicitly permit a non-loopback bind without an API key",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.port < 1 or args.port > 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.maximum_upload_mb <= 0 or args.maximum_image_megapixels <= 0:
        raise ValueError("upload size and image pixel limits must be positive")
    if args.maximum_images_per_request < 1:
        raise ValueError("--maximum-images-per-request must be positive")
    if args.match_threshold is not None and not -1.0 <= args.match_threshold <= 1.0:
        raise ValueError("--match-threshold must be between -1 and 1")
    if not 0.0 <= args.minimum_margin <= 2.0:
        raise ValueError("--minimum-margin must be between 0 and 2")
    if (
        args.host.casefold() not in LOOPBACK_HOSTS
        and not args.api_key
        and not args.allow_unauthenticated_remote
    ):
        raise ValueError(
            "a non-loopback bind requires PET_REID_API_KEY/--api-key, or the explicit "
            "--allow-unauthenticated-remote override"
        )

    config_file = resolve_legacy_path(args.config_file)
    onnx_model = resolve_legacy_path(args.onnx_model)
    identity_weights = resolve_legacy_path(args.identity_weights)
    storage_dir = resolve_legacy_path(args.storage_dir)
    pipeline = build_pipeline(
        config_file,
        identity_weights,
        args.device,
        backend=args.backend,
        onnx_model=(onnx_model if args.backend in {"onnx", "onnx-bifor"} else None),
        onnx_provider=args.onnx_provider,
        onnx_warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
        verify_onnx_source_checkpoint=args.backend in {"onnx", "onnx-bifor"},
        body_detector=args.body_detector if args.backend == "onnx-bifor" else None,
    )
    encoder = MultimodalPipelineEncoder(pipeline)
    if args.agent:
        if args.backend != "onnx-bifor":
            raise ValueError("--agent currently requires --backend onnx-bifor")
        encoder = AgentFeatureEncoder(
            encoder,
            [
                MegaDescriptorEncoder(
                    resolve_legacy_path(args.megadescriptor_checkpoint),
                    device=args.megadescriptor_device or args.device,
                )
            ],
        )
    backend_info = encoder.backend_info()
    model_fingerprint = backend_info.get("model_sha256")
    if not model_fingerprint and args.backend == "pytorch":
        model_fingerprint = sha256_file(identity_weights)
        backend_info["source_checkpoint_sha256"] = model_fingerprint
        encoder.backend_info = lambda: dict(backend_info)

    store = PetGalleryStore(storage_dir)
    service = PetIdentificationService(
        store,
        encoder,
        model_fingerprint=model_fingerprint,
        maximum_upload_bytes=int(args.maximum_upload_mb * 1024 * 1024),
        maximum_image_pixels=int(args.maximum_image_megapixels * 1_000_000),
        maximum_images_per_request=args.maximum_images_per_request,
        require_single_pet_for_enrollment=not args.allow_multiple_pets_for_enrollment,
        default_match_threshold=args.match_threshold,
        default_minimum_margin=args.minimum_margin,
    )
    imported = None
    if args.seed_gallery_model is not None:
        imported = service.import_gallery_model(
            resolve_legacy_path(args.seed_gallery_model)
        )

    print(
        json.dumps(
            {
                "event": "pet_api_ready",
                "url": f"http://{args.host}:{args.port}",
                "docs": f"http://{args.host}:{args.port}/docs",
                "authenticated": bool(args.api_key),
                "backend": service.backend_info(),
                "gallery": store.summary(),
                "seed_import": imported,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    import uvicorn

    app = create_app(service, api_key=args.api_key)
    uvicorn.run(
        app, host=args.host, port=args.port, log_level=args.log_level, workers=1
    )


if __name__ == "__main__":
    main()
