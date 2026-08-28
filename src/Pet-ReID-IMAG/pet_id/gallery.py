"""Prototype-gallery helpers for adding identities without retraining the encoders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from fastreid.config import get_cfg

from .config import add_retri_config
from .multimodal import PetDescriptor, build_multimodal_pipeline
from .workspace_paths import normalize_runtime_config, resolve_legacy_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exif_oriented_bgr(path: Path) -> np.ndarray:
    """Read a phone image in its displayed orientation and return BGR."""

    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def collect_images(values: Iterable[str | Path]) -> list[Path]:
    images: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            images.extend(
                item
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(
                f"image input does not exist or is unsupported: {path}"
            )
    if not images:
        raise RuntimeError("no input images found")
    return images


def build_pipeline(
    config_file: Path,
    checkpoint: Path | None,
    device: str,
    *,
    backend: str = "pytorch",
    onnx_model: Path | None = None,
    onnx_provider: str = "cuda",
    onnx_warmup_batches: tuple[int, ...] = (),
    verify_onnx_source_checkpoint: bool = False,
    body_detector: Path | None = None,
):
    config_file = resolve_legacy_path(config_file)
    checkpoint = resolve_legacy_path(checkpoint) if checkpoint else None
    onnx_model = resolve_legacy_path(onnx_model) if onnx_model else None
    body_detector = resolve_legacy_path(body_detector) if body_detector else None
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    cfg.MULTIMODAL.IDENTITY_WEIGHTS = (
        str(checkpoint) if checkpoint and backend == "pytorch" else ""
    )
    normalize_runtime_config(cfg)
    cfg.freeze()
    if backend == "pytorch":
        return build_multimodal_pipeline(cfg, device=device)
    if backend not in {"onnx", "onnx-bifor"}:
        raise ValueError("identity backend must be 'pytorch', 'onnx', or 'onnx-bifor'")
    if onnx_model is None:
        raise ValueError("onnx_model is required for the ONNX identity backend")
    if backend == "onnx-bifor":
        if body_detector is None:
            raise ValueError("body_detector is required for the BIFOR ONNX backend")
        from .bifor_onnx_runtime import build_bifor_onnx_multimodal_pipeline

        return build_bifor_onnx_multimodal_pipeline(
            cfg,
            model_path=onnx_model,
            body_detector_checkpoint=body_detector,
            provider=onnx_provider,
            source_checkpoint=(checkpoint if verify_onnx_source_checkpoint else None),
            device=device,
            warmup_batches=onnx_warmup_batches,
        )
    from .onnx_runtime import build_onnx_multimodal_pipeline

    return build_onnx_multimodal_pipeline(
        cfg,
        model_path=onnx_model,
        provider=onnx_provider,
        source_checkpoint=(checkpoint if verify_onnx_source_checkpoint else None),
        device=device,
        warmup_batches=onnx_warmup_batches,
    )


def descriptor_priority(descriptor: PetDescriptor) -> tuple[float, float]:
    detection = descriptor.detection
    if detection is None:
        return (0.0, descriptor.branch_quality[0])
    x1, y1, x2, y2 = detection.bbox_xyxy
    return (max(x2 - x1, 0.0) * max(y2 - y1, 0.0), detection.confidence)


def encode_primary(pipeline, path: Path) -> tuple[PetDescriptor, dict]:
    descriptors = pipeline.encode_image(load_exif_oriented_bgr(path))
    if not descriptors:
        raise RuntimeError(f"no dog descriptor produced for {path}")
    selected_index = max(
        range(len(descriptors)),
        key=lambda index: descriptor_priority(descriptors[index]),
    )
    descriptor = descriptors[selected_index]
    return descriptor, {
        "detections": len(descriptors),
        "selected_detection": selected_index,
        "descriptor": descriptor.metadata_dict(),
    }


def normalized_array(feature: torch.Tensor) -> np.ndarray:
    value = feature.detach().float().cpu().numpy().astype(np.float32, copy=False)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("descriptor has an invalid norm")
    return value / norm


def normalized_prototypes(
    reference_features: np.ndarray,
    reference_identity_indices: np.ndarray,
    identity_count: int,
) -> np.ndarray:
    rows = []
    for identity_index in range(identity_count):
        selected = reference_features[reference_identity_indices == identity_index]
        if not len(selected):
            raise ValueError(f"identity {identity_index} has no gallery references")
        prototype = selected.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"identity {identity_index} produced an invalid prototype")
        rows.append((prototype / norm).astype(np.float32))
    return np.stack(rows)


def load_gallery_model(model_json: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads(model_json.read_text(encoding="utf-8"))
    feature_path = (model_json.parent / metadata["features_file"]).resolve()
    expected_hash = metadata["features_sha256"]
    actual_hash = sha256_file(feature_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"gallery features hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    with np.load(feature_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays
