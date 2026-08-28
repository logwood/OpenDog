"""ONNX Runtime adapter for the semantic-v3 + BIFOR deployment package."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision.transforms import functional as TVF

from .body_detection import FrozenDogBodyDetector
from .bifor_onnx import BIFOR_ONNX_INPUT_NAMES
from .localization import AnyFaceDetector, SAM2NoseSegmenter, VIEWPOINT_DIM
from .multimodal import MultimodalPetIDPipeline, QUALITY_DIM
from .onnx_export import ONNX_OUTPUT_NAMES
from .onnx_runtime import (
    ONNXRuntimeIdentityModel,
    parse_warmup_batches,
    resolve_execution_provider,
    sha256_file,
)


_EXPECTED_INPUT_TYPES = (
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(bool)",
)
_DEFAULT_OUTPUT_WIDTHS = (512, 2048, 512, 2, 2, None)


class BIFORONNXRuntimeIdentityModel(ONNXRuntimeIdentityModel):
    """Drop-in identity backend that obtains and encodes a target body crop."""

    requires_body_crop = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        body_detector_checkpoint: str | Path,
        provider: str = "cuda",
        device: str | torch.device | None = None,
        metadata_path: str | Path | None = None,
        source_checkpoint: str | Path | None = None,
        verify_hash: bool = True,
        warmup_batches: Sequence[int] = (),
    ) -> None:
        self.body_size = (224, 224)
        super().__init__(
            model_path,
            provider=provider,
            device=device,
            metadata_path=metadata_path,
            source_checkpoint=source_checkpoint,
            verify_hash=verify_hash,
            warmup_batches=(),
        )
        self.body_size = self._metadata_image_size("body_crop", (224, 224))
        detector_config = self.metadata.get("body_preprocessing", {}).get(
            "detector", {}
        )
        detector_checkpoint = Path(body_detector_checkpoint).expanduser().resolve()
        expected_hash = detector_config.get("checkpoint_sha256")
        if expected_hash:
            actual_hash = sha256_file(detector_checkpoint)
            if actual_hash.casefold() != str(expected_hash).casefold():
                raise ValueError(
                    "Body detector checkpoint hash mismatch: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        self.body_detector = FrozenDogBodyDetector(
            detector_checkpoint,
            device=self.crop_device,
            score_threshold=float(detector_config.get("score_threshold", 0.5)),
            crop_expansion=float(detector_config.get("crop_expansion", 0.04)),
        )
        self.warmup(warmup_batches)

    def _validate_session_contract(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        names = tuple(item.name for item in inputs)
        if names != BIFOR_ONNX_INPUT_NAMES:
            raise RuntimeError(f"Unexpected BIFOR ONNX inputs: {names}")
        output_names = tuple(item.name for item in outputs)
        if output_names != ONNX_OUTPUT_NAMES:
            raise RuntimeError(f"Unexpected ONNX outputs: {output_names}")
        types = tuple(item.type for item in inputs)
        if types != _EXPECTED_INPUT_TYPES:
            raise RuntimeError(f"Unexpected BIFOR ONNX input dtypes: {types}")
        expected_input_shapes = (
            (3, *self.nose_size),
            (3, *self.face_size),
            (3, *self.body_size),
            (1, *self.nose_size),
            (QUALITY_DIM,),
            (VIEWPOINT_DIM,),
            (2,),
        )
        for item, expected_shape in zip(inputs, expected_input_shapes):
            if tuple(item.shape[1:]) != expected_shape:
                raise RuntimeError(
                    f"Unexpected shape for {item.name}: {item.shape}; "
                    f"expected [N, {', '.join(map(str, expected_shape))}]"
                )
        expected_output_widths = (
            self._metadata_output_width("embedding", _DEFAULT_OUTPUT_WIDTHS[0]),
            *_DEFAULT_OUTPUT_WIDTHS[1:],
        )
        for item, expected_width in zip(outputs, expected_output_widths):
            expected_shape = () if expected_width is None else (expected_width,)
            if tuple(item.shape[1:]) != expected_shape:
                raise RuntimeError(f"Unexpected shape for {item.name}: {item.shape}")

    def warmup(self, batch_sizes: Sequence[int]) -> None:
        for batch_size in parse_warmup_batches(batch_sizes):
            inputs = (
                torch.zeros(
                    batch_size,
                    3,
                    *self.nose_size,
                    dtype=torch.float32,
                    device=self.crop_device,
                ),
                torch.zeros(
                    batch_size,
                    3,
                    *self.face_size,
                    dtype=torch.float32,
                    device=self.crop_device,
                ),
                torch.zeros(
                    batch_size,
                    3,
                    *self.body_size,
                    dtype=torch.float32,
                    device=self.crop_device,
                ),
                torch.ones(
                    batch_size,
                    1,
                    *self.nose_size,
                    dtype=torch.float32,
                    device=self.crop_device,
                ),
                torch.ones(batch_size, QUALITY_DIM, device=self.crop_device),
                torch.zeros(batch_size, VIEWPOINT_DIM, device=self.crop_device),
                torch.ones(batch_size, 2, dtype=torch.bool, device=self.crop_device),
            )
            self._run_precropped(inputs)

    def _run_precropped(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        prepared_rows = []
        for index, value in enumerate(inputs):
            expected_dtype = torch.bool if index == 6 else torch.float32
            prepared_rows.append(
                value.detach()
                .to(device=self.crop_device, dtype=expected_dtype)
                .contiguous()
            )
        prepared = tuple(prepared_rows)
        if self.provider == "cuda":
            binding = self.session.io_binding()
            device_index = self.crop_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            for name, value in zip(BIFOR_ONNX_INPUT_NAMES, prepared):
                element_type = np.bool_ if value.dtype == torch.bool else np.float32
                binding.bind_input(
                    name,
                    "cuda",
                    int(device_index),
                    element_type,
                    tuple(value.shape),
                    value.data_ptr(),
                )
            for name in ONNX_OUTPUT_NAMES:
                binding.bind_output(name, "cpu")
            self.session.run_with_iobinding(binding)
            arrays = binding.copy_outputs_to_cpu()
        else:
            feeds = {
                name: value.cpu().numpy()
                for name, value in zip(BIFOR_ONNX_INPUT_NAMES, prepared)
            }
            arrays = self.session.run(list(ONNX_OUTPUT_NAMES), feeds)
        return tuple(torch.from_numpy(np.asarray(value)) for value in arrays)

    @staticmethod
    def _crop_body_rois(
        images: torch.Tensor,
        body_rois: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Match the locked experiment's integer slice then antialias resize."""

        rows = []
        height, width = images.shape[-2:]
        for roi in body_rois.detach().cpu().tolist():
            image_index = int(roi[0])
            x1 = max(int(round(roi[1])), 0)
            y1 = max(int(round(roi[2])), 0)
            x2 = min(int(round(roi[3])), width)
            y2 = min(int(round(roi[4])), height)
            crop = images[image_index, :, y1:y2, x1:x2]
            if crop.numel() == 0:
                raise RuntimeError(f"Empty body crop for ROI {roi}")
            rows.append(TVF.resize(crop, list(output_size), antialias=True))
        return torch.stack(rows)

    def forward(
        self,
        images_0_255: torch.Tensor,
        *,
        face_rois: torch.Tensor,
        nose_rois: torch.Tensor,
        roll_angles_radians: torch.Tensor,
        nose_masks: torch.Tensor,
        quality_signals: torch.Tensor,
        branch_available: torch.Tensor,
        viewpoint_signals: torch.Tensor | None = None,
        body_rois: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict:
        if targets is not None:
            raise RuntimeError("The BIFOR ONNX identity backend is inference-only")
        images = images_0_255.to(self.crop_device, dtype=torch.float32)
        dtype = images.dtype
        face_rois = face_rois.to(self.crop_device, dtype=dtype)
        nose_rois = nose_rois.to(self.crop_device, dtype=dtype)
        angles = roll_angles_radians.to(self.crop_device, dtype=dtype)
        quality = quality_signals.to(self.crop_device, dtype=dtype)
        available = branch_available.to(self.crop_device, dtype=torch.bool)
        batch_size = quality.shape[0]
        if quality.shape != (batch_size, QUALITY_DIM):
            raise ValueError(f"quality_signals must have shape [N, {QUALITY_DIM}]")
        if batch_size < 1:
            raise ValueError("The BIFOR ONNX backend requires at least one ROI")
        if available.shape != (batch_size, 2):
            raise ValueError("branch_available must have shape [N, 2]")
        if not bool(available.any(dim=1).all()):
            raise ValueError("Every sample needs at least one nose/face branch")
        viewpoints = (
            torch.zeros(
                batch_size,
                VIEWPOINT_DIM,
                device=self.crop_device,
                dtype=dtype,
            )
            if viewpoint_signals is None
            else viewpoint_signals.to(self.crop_device, dtype=dtype)
        )
        if viewpoints.shape != (batch_size, VIEWPOINT_DIM):
            raise ValueError(f"viewpoint_signals must have shape [N, {VIEWPOINT_DIM}]")
        if face_rois.shape != (batch_size, 5) or nose_rois.shape != (batch_size, 5):
            raise ValueError("face_rois and nose_rois must both have shape [N, 5]")
        if angles.shape != (batch_size,):
            raise ValueError("roll_angles_radians must have shape [N]")
        if nose_masks.shape[0] != batch_size:
            raise ValueError("nose_masks must contain one mask per ROI")

        if body_rois is None:
            body_rois, body_detected, body_scores = self.body_detector.locate(
                images,
                face_rois,
            )
        else:
            body_rois = body_rois.to(self.crop_device, dtype=dtype)
            if body_rois.shape != (batch_size, 5):
                raise ValueError("body_rois must have shape [N, 5]")
            body_detected = torch.ones(
                batch_size, dtype=torch.bool, device=self.crop_device
            )
            body_scores = torch.ones(batch_size, dtype=dtype, device=self.crop_device)

        face_crops = self.cropper(images, face_rois, angles, self.face_size)
        nose_crops = self.cropper(images, nose_rois, angles, self.nose_size)
        body_crops = self._crop_body_rois(
            images,
            body_rois,
            self.body_size,
        )
        mask_rois = nose_rois.clone()
        mask_rois[:, 0] = torch.arange(
            batch_size,
            device=self.crop_device,
            dtype=dtype,
        )
        mask_crops = self.cropper(
            nose_masks.to(self.crop_device, dtype=dtype),
            mask_rois,
            angles,
            self.nose_size,
        ).clamp(0, 1)
        outputs = self._run_precropped(
            (
                nose_crops,
                face_crops,
                body_crops,
                mask_crops,
                quality,
                viewpoints,
                available,
            )
        )
        return {
            "features": outputs[0],
            "nose_features": outputs[1],
            "face_features": outputs[2],
            "fusion_weights": outputs[3],
            "joint_weights": outputs[4],
            "viewpoint_frontality": outputs[5],
            "effective_branch_available": available.detach().cpu(),
            "body_detected": body_detected.detach().cpu(),
            "body_detection_scores": body_scores.detach().cpu(),
            "body_rois": body_rois.detach().cpu(),
            "backend": "onnxruntime-bifor",
        }

    def backend_info(self) -> dict:
        info = super().backend_info()
        info.update(
            {
                "backend": "onnxruntime-bifor",
                "body_size": list(self.body_size),
                "body_detector": str(self.body_detector.checkpoint_path),
                "body_weight": self.metadata.get("body_fusion", {}).get("body_weight"),
            }
        )
        return info


def build_bifor_onnx_multimodal_pipeline(
    cfg,
    *,
    model_path: str | Path,
    body_detector_checkpoint: str | Path,
    provider: str = "cuda",
    device: str | torch.device | None = None,
    source_checkpoint: str | Path | None = None,
    warmup_batches: Sequence[int] = (),
) -> MultimodalPetIDPipeline:
    """Construct the existing image API around the BIFOR ONNX identity graph."""

    options = cfg.MULTIMODAL
    requested_device = torch.device(device or cfg.MODEL.DEVICE)
    resolved_provider = resolve_execution_provider(
        provider,
        __import__("onnxruntime").get_available_providers(),
        torch_cuda_available=torch.cuda.is_available(),
    )
    crop_device = requested_device
    if resolved_provider == "cpu":
        crop_device = torch.device("cpu")
    identity_model = BIFORONNXRuntimeIdentityModel(
        model_path,
        body_detector_checkpoint=body_detector_checkpoint,
        provider=resolved_provider,
        device=crop_device,
        source_checkpoint=source_checkpoint,
        warmup_batches=(),
    )
    detector = AnyFaceDetector(
        options.ANYFACE_WEIGHTS,
        repository_root=options.ANYFACE_ROOT,
        device=requested_device,
        image_size=options.ANYFACE_IMAGE_SIZE,
        confidence_threshold=options.ANYFACE_CONFIDENCE,
    )
    segmenter = SAM2NoseSegmenter(
        options.SAM2_CHECKPOINT,
        config=options.SAM2_CONFIG,
        device=requested_device,
    )
    pipeline = MultimodalPetIDPipeline(
        detector,
        segmenter,
        identity_model,
        allow_raw_nose_fallback=options.ALLOW_RAW_NOSE_FALLBACK,
        max_long_side=options.MAX_LONG_SIDE,
    )
    identity_model.warmup(warmup_batches)
    return pipeline
