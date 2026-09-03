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
from pet_id.model_profiles import (  # noqa: E402
    ModelProfile,
    get_runtime_profile,
    profile_for_backend,
    runtime_profile_names,
)
from pet_id.reference_scoring import (  # noqa: E402
    CENTROID_SCORING,
    DEFAULT_REFERENCE_SCORE_WEIGHT,
    DEFAULT_REFERENCE_TOP_K,
    LEARNED_REFERENCE_SET_SCORING,
    MAX_REFERENCE_TOP_K,
    REFERENCE_SET_SCORING,
    validate_reference_score_weight,
    validate_reference_top_k,
)
from pet_id.reference_set_model import ReferenceSetMatcherRuntime  # noqa: E402
from pet_id.reference_set_onnx_runtime import ReferenceSetONNXRuntime  # noqa: E402
from pet_id.recognition_agent import (  # noqa: E402
    AgentFeatureEncoder,
    MegaDescriptorEncoder,
)
from pet_id.unified_highres_runtime import UnifiedHighResolutionONNXRuntimePipeline  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.workspace_paths import resolve_legacy_path  # noqa: E402


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _coerce_profile(selection: str | ModelProfile) -> ModelProfile:
    if isinstance(selection, ModelProfile):
        return selection
    try:
        return get_runtime_profile(selection)
    except ValueError:
        return profile_for_backend(selection)


def default_onnx_model(profile: str | ModelProfile = "production") -> Path:
    return _coerce_profile(profile).onnx


def default_storage_dir(profile: str | ModelProfile = "production") -> Path:
    return _coerce_profile(profile).persistent_gallery


def resolve_runtime_profile(args: argparse.Namespace) -> ModelProfile:
    requested = getattr(args, "profile", None)
    backend = getattr(args, "backend", None)
    agent = bool(getattr(args, "agent", False))
    profile = (
        get_runtime_profile(requested)
        if requested
        else profile_for_backend(backend, agent=agent)
        if backend
        else get_runtime_profile("production")
    )
    if backend and backend != profile.backend:
        raise ValueError(
            f"--backend {backend!r} conflicts with --profile {profile.name!r}"
        )
    if agent and not profile.agent_mode:
        raise ValueError("--agent requires the research-agent profile")
    return profile


def _required_path(value: Path | None, *, field: str) -> Path:
    if value is None:
        raise ValueError(f"The selected runtime profile does not provide {field}")
    return resolve_legacy_path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the UnifiedPetReID enrollment and identification API"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--profile",
        choices=runtime_profile_names(),
        help="deployment role; defaults to production",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        help=(
            "defaults to an independent production-baseline or "
            "current-development Gallery for the selected backend"
        ),
    )
    parser.add_argument(
        "--seed-gallery-model",
        type=Path,
        help="idempotently import an existing gallery_model.json before serving",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="compatibility override for profiles that use a FastReID config",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=(
            "pytorch",
            "onnx",
            "onnx-bifor",
            "unified-onnx",
            "onnx-highres",
        ),
        help="compatibility override; prefer --profile",
    )
    parser.add_argument(
        "--identity-weights",
        type=Path,
        help=("PyTorch/legacy ONNX checkpoint; ignored by both unified backends"),
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        help=(
            "defaults to the packaged production-baseline or "
            "current-development ONNX for the selected backend"
        ),
    )
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        help="compatibility override for body-fusion research profiles",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="enable score-level evidence fusion with frozen independent experts",
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        help="compatibility override for the multi-expert research profile",
    )
    parser.add_argument(
        "--megadescriptor-device",
        help="defaults to --device; use cpu to keep the expert off GPU",
    )
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument("--maximum-upload-mb", type=float, default=15.0)
    parser.add_argument("--maximum-image-megapixels", type=float, default=25.0)
    parser.add_argument("--maximum-images-per-request", type=int, default=8)
    parser.add_argument(
        "--allow-multiple-pets-for-enrollment",
        action="store_true",
        help=(
            "allow multiple detected pets for legacy detector backends; the unified "
            "single-graph backend expects callers to provide one-primary-pet images"
        ),
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        help="optional calibrated cosine threshold; omitted means closed-set Top-1",
    )
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    parser.add_argument(
        "--scoring-mode",
        choices=(
            CENTROID_SCORING,
            REFERENCE_SET_SCORING,
            LEARNED_REFERENCE_SET_SCORING,
        ),
        default=CENTROID_SCORING,
        help=(
            "gallery scoring strategy; reference_set is the deterministic blend, "
            "learned_reference_set uses the trained query-conditioned matcher"
        ),
    )
    parser.add_argument(
        "--reference-top-k",
        type=int,
        default=DEFAULT_REFERENCE_TOP_K,
        help=(
            f"number of strongest reference similarities averaged by "
            f"reference_set (1-{MAX_REFERENCE_TOP_K})"
        ),
    )
    parser.add_argument(
        "--reference-score-weight",
        type=float,
        default=DEFAULT_REFERENCE_SCORE_WEIGHT,
        help=(
            "weight of the per-reference score in reference_set mode "
            "(0-1)"
        ),
    )
    parser.add_argument(
        "--reference-matcher-checkpoint",
        "--reference-matcher",
        dest="reference_matcher_checkpoint",
        type=Path,
        help=(
            "optional trained query-conditioned reference matcher checkpoint; "
            "use with learned_reference_set or select that mode per request"
        ),
    )
    parser.add_argument(
        "--reference-matcher-device",
        help="PyTorch matcher device; defaults to --device",
    )
    parser.add_argument(
        "--reference-matcher-provider",
        choices=("auto", "cuda", "cpu"),
        help="ONNX matcher provider; defaults to --onnx-provider",
    )
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


def build_runtime_encoder(args: argparse.Namespace):
    """Build only the model artifacts required by the selected backend."""

    profile = resolve_runtime_profile(args)
    backend = profile.backend
    onnx_model = resolve_legacy_path(
        args.onnx_model or default_onnx_model(profile)
    )
    identity_weights: Path | None = None
    if backend == "onnx-highres":
        pipeline = UnifiedHighResolutionONNXRuntimePipeline(
            onnx_model,
            provider=args.onnx_provider,
            device=args.device,
            profile=profile,
            warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
        )
    elif backend == "unified-onnx":
        pipeline = UnifiedONNXRuntimePipeline(
            onnx_model,
            provider=args.onnx_provider,
            device=args.device,
            profile=profile,
            warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
        )
    else:
        config_file = _required_path(
            args.config_file or profile.config, field="config"
        )
        identity_weights = _required_path(
            args.identity_weights or profile.identity_weights,
            field="identity weights",
        )
        pipeline = build_pipeline(
            config_file,
            identity_weights,
            args.device,
            backend=backend,
            onnx_model=(onnx_model if backend in {"onnx", "onnx-bifor"} else None),
            onnx_provider=args.onnx_provider,
            onnx_warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
            verify_onnx_source_checkpoint=backend in {"onnx", "onnx-bifor"},
            body_detector=(
                _required_path(
                    args.body_detector or profile.body_detector,
                    field="body detector",
                )
                if backend == "onnx-bifor"
                else None
            ),
        )
    encoder = MultimodalPipelineEncoder(pipeline)
    encoder.profile_info = profile.public_metadata()
    if profile.agent_mode:
        if backend != "onnx-bifor":
            raise ValueError("The multi-expert profile requires body-fusion inference")
        encoder = AgentFeatureEncoder(
            encoder,
            [
                MegaDescriptorEncoder(
                    _required_path(
                        args.megadescriptor_checkpoint or profile.expert_checkpoint,
                        field="expert checkpoint",
                    ),
                    device=args.megadescriptor_device or args.device,
                )
            ],
        )
    return encoder, identity_weights


def build_reference_matcher(args: argparse.Namespace):
    """Load the optional learned reference-set matcher selected by the CLI."""

    value = getattr(args, "reference_matcher_checkpoint", None)
    if value is None:
        return None
    path = resolve_legacy_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"reference matcher checkpoint is missing: {path}")
    if path.suffix.casefold() == ".onnx":
        return ReferenceSetONNXRuntime(
            path,
            provider=(
                getattr(args, "reference_matcher_provider", None)
                or getattr(args, "onnx_provider", "cpu")
            ),
        )
    return ReferenceSetMatcherRuntime.from_checkpoint(
        path,
        device=(
            getattr(args, "reference_matcher_device", None)
            or getattr(args, "device", "cpu")
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    profile = resolve_runtime_profile(args)
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
    try:
        reference_top_k = validate_reference_top_k(args.reference_top_k)
        reference_score_weight = validate_reference_score_weight(
            args.reference_score_weight
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    if (
        args.scoring_mode == LEARNED_REFERENCE_SET_SCORING
        and args.reference_matcher_checkpoint is None
    ):
        raise ValueError(
            "--scoring-mode learned_reference_set requires "
            "--reference-matcher-checkpoint"
        )
    if (
        args.host.casefold() not in LOOPBACK_HOSTS
        and not args.api_key
        and not args.allow_unauthenticated_remote
    ):
        raise ValueError(
            "a non-loopback bind requires PET_REID_API_KEY/--api-key, or the explicit "
            "--allow-unauthenticated-remote override"
        )

    storage_arg = args.storage_dir or default_storage_dir(profile)
    storage_dir = resolve_legacy_path(storage_arg)
    encoder, identity_weights = build_runtime_encoder(args)
    reference_matcher = build_reference_matcher(args)
    backend_info = encoder.backend_info()
    model_fingerprint = backend_info.get("model_sha256")
    if not model_fingerprint and profile.backend == "pytorch":
        if identity_weights is None:
            raise RuntimeError("PyTorch backend requires identity weights")
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
        default_scoring_mode=args.scoring_mode,
        reference_top_k=reference_top_k,
        reference_score_weight=reference_score_weight,
        reference_matcher=reference_matcher,
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
