"""Frozen independent experts and score-level evidence fusion for Pet ReID.

The primary BIFOR descriptor and MegaDescriptor never share an embedding space.
Each expert is stored and compared in its own namespace; only monotonic cosine
evidence and reliability weights are combined at decision time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .gallery_service import EncodedPetImage, PetFeatureEncoder, normalize_feature


MEGADESCRIPTOR_EXPERT_ID = "megadescriptor_b224"
MEGADESCRIPTOR_ARCHITECTURE = "swin_base_patch4_window7_224"
MEGADESCRIPTOR_INPUT_SIZE = (224, 224)
MEGADESCRIPTOR_MEAN = (0.485, 0.456, 0.406)
MEGADESCRIPTOR_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert_legacy_swin_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the checkpoint's timm 0.x Swin layout to timm 1.x.

    Old timm stored patch merging on the stage being left. Current timm stores
    the same tensors on the stage being entered. Position indices and attention
    masks are deterministic buffers in current timm and must not be restored.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith("relative_position_index") or key.endswith("attn_mask"):
            continue
        converted_key = re.sub(
            r"^layers\.(\d+)\.downsample\.",
            lambda match: f"layers.{int(match.group(1)) + 1}.downsample.",
            key,
        )
        converted[converted_key] = value
    return converted


@dataclass(frozen=True)
class MegaDescriptorResult:
    feature: np.ndarray
    metadata: dict[str, Any]


class MegaDescriptorEncoder:
    """Offline-only frozen MegaDescriptor-B-224 feature extractor."""

    expert_id = MEGADESCRIPTOR_EXPERT_ID

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
    ):
        import timm

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MegaDescriptor checkpoint is missing: {self.checkpoint_path}"
            )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MegaDescriptor requested CUDA but CUDA is unavailable")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise RuntimeError("MegaDescriptor checkpoint has no model state dictionary")
        model = timm.create_model(
            MEGADESCRIPTOR_ARCHITECTURE,
            pretrained=False,
            num_classes=0,
        )
        model.load_state_dict(
            _convert_legacy_swin_state_dict(state_dict),
            strict=True,
        )
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.model_sha256 = sha256_file(self.checkpoint_path)

    @staticmethod
    def _body_crop(
        image: Image.Image,
        primary_metadata: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        width, height = image.size
        descriptor = primary_metadata.get("descriptor")
        descriptor = descriptor if isinstance(descriptor, dict) else {}
        diagnostics = descriptor.get("runtime_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        body = diagnostics.get("body")
        body = body if isinstance(body, dict) else {}
        detected = bool(body.get("detected"))
        bbox = body.get("bbox_xyxy")
        inference_size = descriptor.get("inference_size")
        valid_bbox = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and isinstance(inference_size, list)
            and len(inference_size) == 2
            and float(inference_size[0]) > 0
            and float(inference_size[1]) > 0
        )
        if not detected or not valid_bbox:
            return image, {
                "body_detected": False,
                "body_detection_score": float(body.get("score") or 0.0),
                "crop_mode": "full_image_fallback",
                "bbox_original_xyxy": [0.0, 0.0, float(width), float(height)],
                "crop_coverage": 1.0,
            }
        scale_x = width / float(inference_size[0])
        scale_y = height / float(inference_size[1])
        x1, y1, x2, y2 = (
            float(bbox[0]) * scale_x,
            float(bbox[1]) * scale_y,
            float(bbox[2]) * scale_x,
            float(bbox[3]) * scale_y,
        )
        pad_x = max(x2 - x1, 0.0) * 0.04
        pad_y = max(y2 - y1, 0.0) * 0.04
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(width), x2 + pad_x)
        y2 = min(float(height), y2 + pad_y)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return image, {
                "body_detected": False,
                "body_detection_score": float(body.get("score") or 0.0),
                "crop_mode": "invalid_box_fallback",
                "bbox_original_xyxy": [0.0, 0.0, float(width), float(height)],
                "crop_coverage": 1.0,
            }
        integer_box = (
            int(np.floor(x1)),
            int(np.floor(y1)),
            int(np.ceil(x2)),
            int(np.ceil(y2)),
        )
        coverage = ((x2 - x1) * (y2 - y1)) / max(width * height, 1)
        return image.crop(integer_box), {
            "body_detected": True,
            "body_detection_score": float(body.get("score") or 0.0),
            "crop_mode": "bifor_body_box",
            "bbox_original_xyxy": [x1, y1, x2, y2],
            "crop_coverage": float(np.clip(coverage, 0.0, 1.0)),
        }

    @staticmethod
    def _quality(image: Image.Image) -> dict[str, float]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        luminance_mean = float(gray.mean()) / 255.0
        luminance_std = float(gray.std()) / 255.0
        dark_fraction = float(np.mean(gray <= 20))
        bright_fraction = float(np.mean(gray >= 235))
        sharpness = float(np.clip(laplacian_variance / 400.0, 0.0, 1.0))
        exposure = float(
            np.clip(
                1.0 - max(dark_fraction, bright_fraction) * 2.0,
                0.0,
                1.0,
            )
        )
        return {
            "laplacian_variance": laplacian_variance,
            "sharpness": sharpness,
            "luminance_mean": luminance_mean,
            "luminance_std": luminance_std,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
            "exposure": exposure,
        }

    @staticmethod
    def _prepare(image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            MEGADESCRIPTOR_INPUT_SIZE,
            Image.Resampling.BICUBIC,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        mean = torch.tensor(MEGADESCRIPTOR_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(MEGADESCRIPTOR_STD, dtype=torch.float32)[:, None, None]
        return (tensor - mean) / std

    def encode_file(
        self,
        path: str | Path,
        primary_metadata: dict[str, Any],
    ) -> MegaDescriptorResult:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        crop, crop_metadata = self._body_crop(image, primary_metadata)
        quality = self._quality(crop)
        tensor = self._prepare(crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(tensor)
            else:
                output = self.model(tensor)
        feature = normalize_feature(
            output[0].detach().float().cpu().numpy(),
            self.expert_id,
        )
        return MegaDescriptorResult(
            feature=feature,
            metadata={
                **crop_metadata,
                "quality": quality,
                "input_size": list(MEGADESCRIPTOR_INPUT_SIZE),
                "preprocessing": "resize_224_imagenet_normalize_v1",
            },
        )

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend": "pytorch-frozen",
            "architecture": MEGADESCRIPTOR_ARCHITECTURE,
            "model_sha256": self.model_sha256,
            "feature_dim": 1024,
            "input_size": list(MEGADESCRIPTOR_INPUT_SIZE),
            "preprocessing": "resize_224_imagenet_normalize_v1",
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "license": "CC-BY-NC-4.0",
            "commercial_use": False,
        }


class AgentFeatureEncoder:
    """Compose the primary encoder with frozen, independent feature experts."""

    def __init__(
        self,
        primary: PetFeatureEncoder,
        experts: Sequence[MegaDescriptorEncoder],
    ):
        self.primary = primary
        self.experts = tuple(experts)
        if len({expert.expert_id for expert in self.experts}) != len(self.experts):
            raise ValueError("expert ids must be unique")

    def encode_file(self, path: Path) -> EncodedPetImage:
        primary = self.primary.encode_file(path)
        features: dict[str, np.ndarray] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for expert in self.experts:
            result = expert.encode_file(path, primary.metadata)
            features[expert.expert_id] = result.feature
            metadata[expert.expert_id] = result.metadata
        return EncodedPetImage(
            fused=primary.fused,
            nose=primary.nose,
            face=primary.face,
            metadata=primary.metadata,
            expert_features=features,
            expert_metadata=metadata,
        )

    def backend_info(self) -> dict[str, Any]:
        primary = dict(self.primary.backend_info())
        primary["agent"] = {
            "version": "multi_expert_evidence_v1",
            "fusion_level": "score",
            "calibration": "zero_shot_monotonic_v1",
            "trained": False,
        }
        primary["experts"] = {
            expert.expert_id: expert.backend_info() for expert in self.experts
        }
        return primary


def cosine_to_evidence(score: np.ndarray | float) -> np.ndarray | float:
    """Monotonic common scale for fusion; deliberately not a probability."""

    return np.clip((np.asarray(score) + 1.0) * 0.5, 0.0, 1.0)


def expert_reliabilities(
    encoded: EncodedPetImage,
    expert_ids: Sequence[str],
) -> dict[str, float]:
    descriptor = encoded.metadata.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, dict) else {}
    available = descriptor.get("branch_available")
    available = (
        [bool(value) for value in available[:2]]
        if isinstance(available, list) and len(available) >= 2
        else [True, True]
    )
    qualities = descriptor.get("branch_quality")
    qualities = (
        [float(value) for value in qualities[:2]]
        if isinstance(qualities, list) and len(qualities) >= 2
        else [0.5, 0.5]
    )
    usable = [qualities[index] for index in range(2) if available[index]]
    branch_quality = float(np.mean(usable)) if usable else 0.0
    bifor = float(
        np.clip(
            0.35 + 0.45 * branch_quality + 0.20 * (1.0 if all(available) else 0.0),
            0.10,
            1.0,
        )
    )
    reliabilities = {"bifor": bifor}
    for expert_id in expert_ids:
        metadata = encoded.expert_metadata.get(expert_id, {})
        quality = metadata.get("quality")
        quality = quality if isinstance(quality, dict) else {}
        detected = bool(metadata.get("body_detected"))
        detector_score = float(metadata.get("body_detection_score") or 0.0)
        sharpness = float(quality.get("sharpness", 0.5))
        exposure = float(quality.get("exposure", 0.5))
        localization = (
            0.55 + 0.45 * float(np.clip(detector_score, 0.0, 1.0))
            if detected
            else 0.38
        )
        reliabilities[expert_id] = float(
            np.clip(
                localization * (0.55 + 0.25 * sharpness + 0.20 * exposure),
                0.10,
                1.0,
            )
        )
    total = sum(reliabilities.values())
    return {key: value / total for key, value in reliabilities.items()}


def build_agent_decision(
    *,
    prototypes: Sequence[dict[str, Any]],
    encoded: EncodedPetImage,
    bifor_scores: np.ndarray,
    expert_scores: dict[str, np.ndarray],
    top_k: int,
    requested_threshold: float | None,
    requested_margin: float,
) -> dict[str, Any]:
    """Fuse independent rankings and return candidates plus an Agent rationale."""

    weights = expert_reliabilities(encoded, sorted(expert_scores))
    evidence = weights["bifor"] * cosine_to_evidence(bifor_scores)
    for expert_id, scores in expert_scores.items():
        evidence = evidence + weights[expert_id] * cosine_to_evidence(scores)
    fused_scores = np.asarray(evidence * 2.0 - 1.0, dtype=np.float32)
    order = np.argsort(-fused_scores)
    candidates: list[dict[str, Any]] = []
    for index in order[: min(top_k, len(order))]:
        position = int(index)
        candidates.append(
            {
                "pet_id": prototypes[position]["pet_id"],
                "display_name": prototypes[position]["display_name"],
                "score": float(fused_scores[position]),
                "reference_count": prototypes[position]["reference_count"],
                "expert_scores": {
                    "bifor": float(bifor_scores[position]),
                    **{
                        expert_id: float(scores[position])
                        for expert_id, scores in expert_scores.items()
                    },
                },
            }
        )
    best = candidates[0]
    runner_up = candidates[1]["score"] if len(candidates) > 1 else None
    margin = None if runner_up is None else float(best["score"] - runner_up)

    expert_arrays = {"bifor": bifor_scores, **expert_scores}
    expert_results: dict[str, dict[str, Any]] = {}
    top_ids: dict[str, str] = {}
    for expert_id, scores in expert_arrays.items():
        expert_order = np.argsort(-scores)
        best_index = int(expert_order[0])
        second_score = float(scores[int(expert_order[1])]) if len(scores) > 1 else None
        top_id = str(prototypes[best_index]["pet_id"])
        top_ids[expert_id] = top_id
        expert_results[expert_id] = {
            "pet_id": top_id,
            "display_name": prototypes[best_index]["display_name"],
            "score": float(scores[best_index]),
            "evidence": float(cosine_to_evidence(scores[best_index])),
            "margin": (
                None
                if second_score is None
                else float(scores[best_index] - second_score)
            ),
        }
    expert_agreement = len(set(top_ids.values())) == 1
    descriptor = encoded.metadata.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, dict) else {}
    available = descriptor.get("branch_available")
    available = (
        [bool(value) for value in available[:2]]
        if isinstance(available, list) and len(available) >= 2
        else [True, True]
    )
    qualities = descriptor.get("branch_quality")
    qualities = (
        [float(value) for value in qualities[:2]]
        if isinstance(qualities, list) and len(qualities) >= 2
        else None
    )
    mega_id = next(iter(sorted(expert_scores)), None)
    mega_metadata = encoded.expert_metadata.get(mega_id, {}) if mega_id else {}
    image_quality = mega_metadata.get("quality")
    image_quality = image_quality if isinstance(image_quality, dict) else {}
    low_blur = float(image_quality.get("sharpness", 1.0)) < 0.25
    exposure_bad = float(image_quality.get("exposure", 1.0)) < 0.55
    body_missing = bool(mega_id) and not bool(mega_metadata.get("body_detected"))
    branch_low = bool(
        qualities
        and any(available[index] and qualities[index] < 0.35 for index in range(2))
    )

    zero_shot_threshold = 0.35
    zero_shot_margin = 0.08
    threshold = (
        zero_shot_threshold
        if requested_threshold is None
        else float(requested_threshold)
    )
    required_margin = max(float(requested_margin), zero_shot_margin)
    reasons: list[str] = []
    recommendations: list[str] = []
    if best["score"] < threshold:
        reasons.append("insufficient_identity_evidence")
    if margin is not None and margin < required_margin:
        reasons.append("low_fused_margin")
    reliable_experts = [
        expert_id
        for expert_id, weight in weights.items()
        if weight >= 0.25
    ]
    reliable_top_ids = {top_ids[expert_id] for expert_id in reliable_experts}
    reliable_conflict = len(reliable_top_ids) > 1
    if reliable_conflict:
        reasons.append("expert_conflict")
        recommendations.append("补拍一张正面脸和鼻部，以及左右侧身各一张。")
    if not all(available) or branch_low:
        reasons.append("nose_face_quality_limited")
        recommendations.append("在均匀光线下补拍清晰的正面脸和鼻部。")
    if body_missing:
        reasons.append("body_not_detected")
        recommendations.append("让整只狗进入画面，补拍完整的左侧身或右侧身。")
    if low_blur:
        reasons.append("motion_or_focus_blur")
        recommendations.append("稳定相机并提高快门，重新拍一张清晰照片。")
    if exposure_bad:
        reasons.append("uneven_or_extreme_lighting")
        recommendations.append("避开背光和强反光，换到均匀柔和的光照下拍摄。")

    weak_evidence = best["score"] < threshold
    low_margin = margin is not None and margin < required_margin
    quality_reason_count = sum(
        reason
        in {
            "nose_face_quality_limited",
            "body_not_detected",
            "motion_or_focus_blur",
            "uneven_or_extreme_lighting",
        }
        for reason in reasons
    )
    if weak_evidence and quality_reason_count == 0:
        decision = "possible_unknown"
    elif (
        weak_evidence
        or low_margin
        or reliable_conflict
        or quality_reason_count >= 2
    ):
        decision = "needs_more_evidence"
    else:
        decision = "matched"
    accepted = decision == "matched"
    if not accepted and not recommendations:
        recommendations.append("补拍 2–3 张不同角度的清晰照片后再比对。")

    return {
        "candidates": candidates,
        "best": best,
        "margin": margin,
        "accepted": accepted,
        "agent": {
            "decision": decision,
            "expert_agreement": expert_agreement,
            "calibration": "zero_shot_monotonic_v1",
            "score_semantics": "weighted_cosine_not_probability",
            "expert_weights": weights,
            "expert_results": expert_results,
            "quality": {
                "branch_available": available,
                "branch_quality": qualities,
                "body_detected": (
                    bool(mega_metadata.get("body_detected")) if mega_id else None
                ),
                "body_detection_score": (
                    float(mega_metadata.get("body_detection_score") or 0.0)
                    if mega_id
                    else None
                ),
                **image_quality,
            },
            "thresholds": {
                "match_score": threshold,
                "minimum_margin": required_margin,
                "source": (
                    "request_override"
                    if requested_threshold is not None
                    else "zero_shot_heuristic_v1"
                ),
            },
            "reasons": list(dict.fromkeys(reasons)),
            "capture_recommendations": list(dict.fromkeys(recommendations)),
        },
    }
