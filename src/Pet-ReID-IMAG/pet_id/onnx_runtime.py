"""ONNX Runtime identity backend for the multimodal pet pipeline.

AnyFace, SAM 2, and rotated ROI extraction remain application preprocessing.
This module mirrors the callable contract of ``LocalEndToEndPetIDModel`` while
running the fixed-size crop-to-embedding network with ONNX Runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .localization import AnyFaceDetector, SAM2NoseSegmenter, VIEWPOINT_DIM
from .multimodal import DifferentiableROICropper, MultimodalPetIDPipeline, QUALITY_DIM
from .onnx_export import ONNX_INPUT_NAMES, ONNX_OUTPUT_NAMES


_EXPECTED_INPUT_TYPES = (
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(float)",
    "tensor(bool)",
)
_DEFAULT_OUTPUT_WIDTHS = (3072, 2048, 512, 2, 2, None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_execution_provider(
    requested: str,
    available_providers: Iterable[str],
    *,
    torch_cuda_available: bool,
) -> str:
    """Resolve ``auto`` without ever silently downgrading an explicit request."""

    requested = str(requested).casefold()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("ONNX provider must be one of: auto, cuda, cpu")
    available = set(available_providers)
    cuda_ready = "CUDAExecutionProvider" in available and torch_cuda_available
    if requested == "auto":
        return "cuda" if cuda_ready else "cpu"
    if requested == "cuda" and not cuda_ready:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but is unavailable. Install the "
            "CUDA-compatible requirements-onnx-gpu.txt package set and verify CUDA DLLs."
        )
    if requested == "cpu" and "CPUExecutionProvider" not in available:
        raise RuntimeError("CPUExecutionProvider is unavailable in this ONNX Runtime build")
    return requested


def parse_warmup_batches(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        batches = tuple(int(item) for item in items)
    else:
        batches = tuple(int(item) for item in value)
    if any(batch < 1 for batch in batches):
        raise ValueError("ONNX warmup batch sizes must be positive")
    return tuple(dict.fromkeys(batches))


class ONNXRuntimeIdentityModel(nn.Module):
    """ROI-to-descriptor adapter backed by an ONNX Runtime session.

    CUDA inputs use I/O binding against the active PyTorch CUDA stream. Outputs
    are copied to CPU because ``PetDescriptor`` and gallery retrieval already
    consume CPU descriptor tensors.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "cuda",
        device: str | torch.device | None = None,
        metadata_path: str | Path | None = None,
        source_checkpoint: str | Path | None = None,
        verify_hash: bool = True,
        warmup_batches: Sequence[int] = (),
    ):
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install requirements-onnx-gpu.txt "
                "for CUDA or requirements-onnx.txt for CPU."
            ) from error

        self._ort = ort
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX identity model not found: {self.model_path}")
        candidate_metadata = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        self.metadata_path = candidate_metadata if candidate_metadata.is_file() else None
        self.metadata = self._load_and_validate_metadata(verify_hash=verify_hash)
        self.source_checkpoint = (
            Path(source_checkpoint).expanduser().resolve()
            if source_checkpoint is not None
            else None
        )
        self.source_checkpoint_sha256 = None
        if self.source_checkpoint is not None:
            if not self.source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"ONNX source checkpoint not found: {self.source_checkpoint}"
                )
            self.source_checkpoint_sha256 = sha256_file(self.source_checkpoint)
            expected_checkpoint_hash = self.metadata.get("source_checkpoint_sha256")
            if expected_checkpoint_hash and self.source_checkpoint_sha256.casefold() != str(
                expected_checkpoint_hash
            ).casefold():
                raise ValueError(
                    "ONNX source checkpoint hash does not match deployment metadata: "
                    f"expected {expected_checkpoint_hash}, got {self.source_checkpoint_sha256}"
                )

        available = ort.get_available_providers()
        self.provider = resolve_execution_provider(
            provider,
            available,
            torch_cuda_available=torch.cuda.is_available(),
        )
        requested_device = torch.device(
            device or ("cuda" if self.provider == "cuda" else "cpu")
        )
        if self.provider == "cuda" and requested_device.type != "cuda":
            raise ValueError("CUDAExecutionProvider requires a CUDA preprocessing device")
        if self.provider == "cpu" and requested_device.type != "cpu":
            raise ValueError("CPUExecutionProvider requires a CPU preprocessing device")
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device(
                "cuda", torch.cuda.current_device()
            )
        self.crop_device = requested_device
        if self.crop_device.type == "cuda":
            torch.empty(1, device=self.crop_device)
        self._device_anchor = nn.Parameter(
            torch.empty(0, device=self.crop_device),
            requires_grad=False,
        )
        self.cropper = DifferentiableROICropper()
        self.nose_size = self._metadata_image_size("nose_crop", (244, 244))
        self.face_size = self._metadata_image_size("face_crop", (224, 224))
        self.label_to_identity: dict[int, str] = {}
        self.identity_to_label: dict[str, int] = {}
        self.num_classes = 0

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3
        if self.provider == "cuda":
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            device_index = self.crop_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            stream = torch.cuda.current_stream(self.crop_device)
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": int(device_index),
                        "user_compute_stream": str(stream.cuda_stream),
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )
        active = self.session.get_providers()
        expected = (
            "CUDAExecutionProvider" if self.provider == "cuda" else "CPUExecutionProvider"
        )
        if not active or active[0] != expected:
            raise RuntimeError(
                f"Requested {expected}, but ONNX Runtime activated {active}. "
                "Refusing silent provider fallback."
            )
        self._validate_session_contract()
        self.eval()
        self.warmup(warmup_batches)

    @property
    def device(self) -> torch.device:
        return self.crop_device

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("ONNXRuntimeIdentityModel is inference-only")
        return super().train(False)

    def _load_and_validate_metadata(self, *, verify_hash: bool) -> dict:
        metadata = (
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata_path is not None
            else {}
        )
        expected_hash = metadata.get("onnx_sha256")
        actual_hash = sha256_file(self.model_path) if verify_hash else None
        if actual_hash and expected_hash:
            if actual_hash.casefold() != str(expected_hash).casefold():
                raise ValueError(
                    f"ONNX model hash mismatch: expected {expected_hash}, got {actual_hash}"
                )
        self.model_sha256 = actual_hash or expected_hash
        return metadata

    def _metadata_image_size(
        self,
        input_name: str,
        default: tuple[int, int],
    ) -> tuple[int, int]:
        shape = self.metadata.get("inputs", {}).get(input_name, {}).get("shape")
        if shape and len(shape) == 4:
            return int(shape[2]), int(shape[3])
        return default

    def _metadata_output_width(self, name: str, default: int) -> int:
        shape = self.metadata.get("outputs", {}).get(name, {}).get("shape")
        if shape and len(shape) == 2 and isinstance(shape[1], int):
            return int(shape[1])
        return int(default)

    def _validate_session_contract(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        names = tuple(item.name for item in inputs)
        if names != ONNX_INPUT_NAMES:
            raise RuntimeError(f"Unexpected ONNX inputs: {names}")
        output_names = tuple(item.name for item in outputs)
        if output_names != ONNX_OUTPUT_NAMES:
            raise RuntimeError(f"Unexpected ONNX outputs: {output_names}")
        types = tuple(item.type for item in inputs)
        if types != _EXPECTED_INPUT_TYPES:
            raise RuntimeError(f"Unexpected ONNX input dtypes: {types}")
        expected_input_shapes = (
            (3, *self.nose_size),
            (3, *self.face_size),
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

    def backend_info(self) -> dict:
        return {
            "backend": "onnxruntime",
            "model": str(self.model_path),
            "metadata": str(self.metadata_path) if self.metadata_path else None,
            "model_sha256": self.model_sha256,
            "source_checkpoint": str(self.source_checkpoint) if self.source_checkpoint else None,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "provider": self.session.get_providers()[0],
            "provider_chain": self.session.get_providers(),
            "onnxruntime_version": self._ort.__version__,
            "crop_device": str(self.crop_device),
            "nose_size": list(self.nose_size),
            "face_size": list(self.face_size),
            "embedding_dim": self._metadata_output_width(
                "embedding", _DEFAULT_OUTPUT_WIDTHS[0]
            ),
            "fusion_mode": self.metadata.get("fusion_mode", "legacy_concat"),
        }

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
            expected_dtype = torch.bool if index == 5 else torch.float32
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
            for name, value in zip(ONNX_INPUT_NAMES, prepared):
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
                for name, value in zip(ONNX_INPUT_NAMES, prepared)
            }
            arrays = self.session.run(list(ONNX_OUTPUT_NAMES), feeds)
        return tuple(torch.from_numpy(np.asarray(value)) for value in arrays)

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
        targets: torch.Tensor | None = None,
    ) -> dict:
        if targets is not None:
            raise RuntimeError("The ONNX identity backend does not contain training heads")
        images = images_0_255.to(self.crop_device, dtype=torch.float32)
        dtype = images.dtype
        face_rois = face_rois.to(self.crop_device, dtype=dtype)
        nose_rois = nose_rois.to(self.crop_device, dtype=dtype)
        angles = roll_angles_radians.to(self.crop_device, dtype=dtype)
        quality = quality_signals.to(self.crop_device, dtype=dtype)
        available = branch_available.to(self.crop_device, dtype=torch.bool)
        if quality.ndim != 2 or quality.shape[1] != QUALITY_DIM:
            raise ValueError(f"quality_signals must have shape [N, {QUALITY_DIM}]")
        batch_size = quality.shape[0]
        if batch_size < 1:
            raise ValueError("The ONNX identity backend requires at least one ROI")
        if available.shape != (batch_size, 2):
            raise ValueError("branch_available must have shape [N, 2]")
        if not bool(available.any(dim=1).all()):
            raise ValueError("Every sample needs at least one identity branch")
        if viewpoint_signals is None:
            viewpoints = torch.zeros(
                batch_size,
                VIEWPOINT_DIM,
                device=self.crop_device,
                dtype=dtype,
            )
        else:
            viewpoints = viewpoint_signals.to(self.crop_device, dtype=dtype)
        if viewpoints.shape != (batch_size, VIEWPOINT_DIM):
            raise ValueError(
                f"viewpoint_signals must have shape [N, {VIEWPOINT_DIM}]"
            )
        if face_rois.shape != (batch_size, 5) or nose_rois.shape != (batch_size, 5):
            raise ValueError("face_rois and nose_rois must both have shape [N, 5]")
        if angles.shape != (batch_size,):
            raise ValueError("roll_angles_radians must have shape [N]")
        if nose_masks.shape[0] != batch_size:
            raise ValueError("nose_masks must contain one mask per ROI")

        face_crops = self.cropper(images, face_rois, angles, self.face_size)
        nose_crops = self.cropper(images, nose_rois, angles, self.nose_size)
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
            (nose_crops, face_crops, mask_crops, quality, viewpoints, available)
        )
        return {
            "features": outputs[0],
            "nose_features": outputs[1],
            "face_features": outputs[2],
            "fusion_weights": outputs[3],
            "joint_weights": outputs[4],
            "viewpoint_frontality": outputs[5],
            "effective_branch_available": available.detach().cpu(),
            "backend": "onnxruntime",
        }


def build_onnx_multimodal_pipeline(
    cfg,
    *,
    model_path: str | Path,
    provider: str = "cuda",
    device: str | torch.device | None = None,
    source_checkpoint: str | Path | None = None,
    warmup_batches: Sequence[int] = (),
) -> MultimodalPetIDPipeline:
    """Construct frozen geometry providers with the ONNX identity backend."""

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
    identity_model = ONNXRuntimeIdentityModel(
        model_path,
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
