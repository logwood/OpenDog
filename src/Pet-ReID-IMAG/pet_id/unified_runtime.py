"""One-model ONNX Runtime pipelines for UnifiedPetReID.

The current end-to-end graph consumes raw RGB pixel tensors and performs the
letterbox/normalization path inside ONNX before emitting one 512-D descriptor.
Older fixed-square exports remain readable for rollback, but are explicitly
marked as ``raw_spatial_input = false`` by their metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from .multimodal import PetDescriptor
from .model_profiles import ModelProfile, profile_for_model_path
from .unified_data import letterbox_rgb


UNIFIED_ONNX_INPUT_NAMES = ("rgb",)
UNIFIED_ONNX_OUTPUT_NAMES = ("embedding",)
RAW_RGB_MINIMUM = 0.0
RAW_RGB_MAXIMUM = 255.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_raw_rgb_input(rgb: np.ndarray) -> np.ndarray:
    """Return one contiguous float32 raw-RGB batch without changing pixels.

    Raw UnifiedPetReID graphs own all image preprocessing, so the runtime
    boundary must reject malformed pixels instead of clipping, scaling, or
    otherwise repairing them outside ONNX.
    """

    value = np.ascontiguousarray(rgb, dtype=np.float32)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("rgb must have shape [N,3,H,W]")
    if value.shape[0] < 1:
        raise ValueError("rgb batch must contain at least one image")
    if value.shape[2] < 1 or value.shape[3] < 1:
        raise ValueError("rgb height and width must be positive")
    if not np.isfinite(value).all():
        raise ValueError("raw RGB pixels must be finite")
    minimum = float(value.min())
    maximum = float(value.max())
    if minimum < RAW_RGB_MINIMUM or maximum > RAW_RGB_MAXIMUM:
        raise ValueError(
            "raw RGB pixels must remain in the declared 0..255 range; "
            f"observed [{minimum}, {maximum}]"
        )
    return value


def resolve_unified_provider(
    requested: str,
    available_providers: Iterable[str],
    *,
    torch_cuda_available: bool,
) -> str:
    requested = str(requested).casefold()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("ONNX provider must be one of: auto, cuda, cpu")
    available = set(available_providers)
    cuda_ready = "CUDAExecutionProvider" in available and torch_cuda_available
    if requested == "auto":
        return "cuda" if cuda_ready else "cpu"
    if requested == "cuda" and not cuda_ready:
        raise RuntimeError("CUDAExecutionProvider was requested but unavailable")
    if requested == "cpu" and "CPUExecutionProvider" not in available:
        raise RuntimeError("CPUExecutionProvider is unavailable")
    return requested


class UnifiedONNXRuntimePipeline:
    """Adapt strict unified ONNX inference to the existing gallery API."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "cuda",
        device: str | torch.device | None = None,
        metadata_path: str | Path | None = None,
        source_checkpoint: str | Path | None = None,
        profile: ModelProfile | None = None,
        verify_hash: bool = True,
        warmup_batches: Sequence[int] = (),
    ):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is not installed; install its CPU or CUDA package"
            ) from error
        self._ort = ort
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.profile = profile or profile_for_model_path(self.model_path)
        candidate_metadata = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        self.metadata_path = (
            candidate_metadata if candidate_metadata.is_file() else None
        )
        self.metadata = (
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata_path is not None
            else {}
        )
        self.model_sha256 = sha256_file(self.model_path)
        expected_model_hash = self.metadata.get("onnx_sha256")
        if (
            verify_hash
            and expected_model_hash
            and self.model_sha256.casefold() != str(expected_model_hash).casefold()
        ):
            raise ValueError("Unified ONNX hash differs from deployment metadata")
        self.source_checkpoint = (
            Path(source_checkpoint).expanduser().resolve()
            if source_checkpoint is not None
            else None
        )
        self.source_checkpoint_sha256 = None
        if self.source_checkpoint is not None:
            if not self.source_checkpoint.is_file():
                raise FileNotFoundError(self.source_checkpoint)
            self.source_checkpoint_sha256 = sha256_file(self.source_checkpoint)
            expected_checkpoint_hash = self.metadata.get("source_checkpoint_sha256")
            if (
                expected_checkpoint_hash
                and self.source_checkpoint_sha256.casefold()
                != str(expected_checkpoint_hash).casefold()
            ):
                raise ValueError("Unified source checkpoint hash differs from metadata")

        self.provider = resolve_unified_provider(
            provider,
            ort.get_available_providers(),
            torch_cuda_available=torch.cuda.is_available(),
        )
        requested_device = torch.device(
            device or ("cuda" if self.provider == "cuda" else "cpu")
        )
        if requested_device.type != self.provider:
            raise ValueError(
                f"{self.provider} provider requires a {self.provider} device"
            )
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.tensor_device = requested_device

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if self.provider == "cuda":
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            device_index = int(self.tensor_device.index or 0)
            stream = torch.cuda.current_stream(self.tensor_device)
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_index,
                        "user_compute_stream": str(stream.cuda_stream),
                        "use_tf32": "0",
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
            sess_options=options,
            providers=providers,
        )
        expected_provider = (
            "CUDAExecutionProvider"
            if self.provider == "cuda"
            else "CPUExecutionProvider"
        )
        active = self.session.get_providers()
        if not active or active[0] != expected_provider:
            raise RuntimeError(
                f"Refusing provider fallback: expected {expected_provider}, "
                f"activated {active}"
            )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if tuple(item.name for item in inputs) != UNIFIED_ONNX_INPUT_NAMES:
            raise RuntimeError("Unified ONNX must have exactly one 'rgb' input")
        if tuple(item.name for item in outputs) != UNIFIED_ONNX_OUTPUT_NAMES:
            raise RuntimeError("Unified ONNX must have exactly one 'embedding' output")
        if inputs[0].type != "tensor(float)":
            raise RuntimeError("Unified rgb input must be float32")
        if outputs[0].type != "tensor(float)":
            raise RuntimeError("Unified embedding output must be float32")
        input_shape = tuple(inputs[0].shape)
        output_shape = tuple(outputs[0].shape)
        if len(input_shape) != 4 or input_shape[1] != 3:
            raise RuntimeError(f"Unexpected unified input shape: {input_shape}")
        if len(output_shape) != 2 or output_shape[1] != 512:
            raise RuntimeError(f"Unexpected unified output shape: {output_shape}")
        preprocessing = self.metadata.get("preprocessing", {})
        contract = self.metadata.get("runtime_contract", {})
        input_contract = contract.get("inputs", {}).get("rgb", {})
        declared_raw = bool(
            self.metadata.get("raw_spatial_input", False)
            or preprocessing.get("raw_spatial_input", False)
            or input_contract.get("raw_pixels", False)
        )
        dynamic_spatial = not (
            isinstance(input_shape[2], int) and isinstance(input_shape[3], int)
        )
        if dynamic_spatial and not declared_raw:
            raise RuntimeError(
                "Dynamic unified ONNX inputs require metadata that explicitly "
                "declares raw RGB pixels"
            )
        self.raw_spatial_input = declared_raw
        self.letterbox_allow_upscale = bool(
            preprocessing.get("letterbox_allow_upscale", False)
        )
        self.minimum_input_side = int(input_contract.get("height_width_minimum", 2))
        self.maximum_input_side = int(
            input_contract.get("height_width_maximum", 10000)
        )
        if self.minimum_input_side < 2:
            raise RuntimeError("Unified input minimum side must be at least two")
        if self.maximum_input_side < self.minimum_input_side:
            raise RuntimeError("Unified input maximum side is below its minimum")
        if self.raw_spatial_input:
            if input_contract.get("raw_pixels") is not True:
                raise RuntimeError("Unified raw input metadata must set raw_pixels=true")
            if input_contract.get("value_range") != [0, 255]:
                raise RuntimeError(
                    "Unified raw input metadata must declare value_range [0,255]"
                )
            output_contract = contract.get("outputs", {}).get("embedding", {})
            if output_contract.get("l2_normalized") is not True:
                raise RuntimeError(
                    "Unified metadata must declare graph-internal L2 normalization"
                )
            if contract.get("external_models") != []:
                raise RuntimeError(
                    "Unified metadata must declare an empty external_models list"
                )
            if input_shape[2] is not None and isinstance(input_shape[2], int):
                if input_shape[2] < self.minimum_input_side:
                    raise RuntimeError("Unified input height is below its minimum")
            if input_shape[3] is not None and isinstance(input_shape[3], int):
                if input_shape[3] < self.minimum_input_side:
                    raise RuntimeError("Unified input width is below its minimum")
            self.input_size = None
        else:
            if input_shape[2] != input_shape[3] or not isinstance(input_shape[2], int):
                raise RuntimeError("Unified input must use one fixed square size")
            self.input_size = int(input_shape[2])
        self.input_shape = input_shape
        self.identity_model = self
        self.warmup(warmup_batches)

    @property
    def device(self) -> torch.device:
        return self.tensor_device

    def _run(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = validate_raw_rgb_input(rgb)
        if self.raw_spatial_input:
            height, width = (int(rgb.shape[2]), int(rgb.shape[3]))
            if min(height, width) < self.minimum_input_side:
                raise ValueError(
                    "raw RGB height and width must both be at least "
                    f"{self.minimum_input_side}"
                )
            if max(height, width) > self.maximum_input_side:
                raise ValueError(
                    "raw RGB maximum side is "
                    f"{self.maximum_input_side}; graph input was not resized externally"
                )
        else:
            expected = (3, self.input_size, self.input_size)
            if tuple(rgb.shape[1:]) != expected:
                raise ValueError(f"rgb must have shape [N,{expected}]")
        if self.provider == "cuda":
            value = torch.from_numpy(rgb).to(self.tensor_device).contiguous()
            binding = self.session.io_binding()
            binding.bind_input(
                "rgb",
                "cuda",
                int(self.tensor_device.index or 0),
                np.float32,
                tuple(value.shape),
                value.data_ptr(),
            )
            binding.bind_output("embedding", "cpu")
            self.session.run_with_iobinding(binding)
            output = binding.copy_outputs_to_cpu()[0]
        else:
            output = self.session.run(list(UNIFIED_ONNX_OUTPUT_NAMES), {"rgb": rgb})[0]
        embedding = torch.from_numpy(np.asarray(output, dtype=np.float32)).float()
        if embedding.shape != (rgb.shape[0], 512):
            raise RuntimeError(f"Unified ONNX returned {tuple(embedding.shape)}")
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("Unified ONNX returned non-finite values")
        norms = embedding.norm(dim=1, keepdim=True)
        if (norms <= 0).any():
            raise FloatingPointError("Unified ONNX returned a zero descriptor")
        # L2 normalization is part of the exported graph.  Do not normalize a
        # second time in Python: doing so would hide a malformed deployment
        # graph and make the API a non-end-to-end post-processing stage.
        if not torch.allclose(norms, torch.ones_like(norms), atol=3e-3, rtol=3e-3):
            raise FloatingPointError(
                "Unified ONNX graph returned a non-normalized descriptor"
            )
        return embedding

    def warmup(self, batch_sizes: Sequence[int]) -> None:
        batches = tuple(dict.fromkeys(int(item) for item in batch_sizes))
        if any(item < 1 for item in batches):
            raise ValueError("warmup batch sizes must be positive")
        if self.raw_spatial_input:
            side = min(1280, self.maximum_input_side)
            side = max(side, self.minimum_input_side)
            shape = (side, side)
        else:
            shape = (self.input_size, self.input_size)
        for batch_size in batches:
            self._run(
                np.zeros(
                    (batch_size, 3, shape[0], shape[1]),
                    dtype=np.float32,
                )
            )

    def _descriptor(
        self,
        embedding: torch.Tensor,
        *,
        width: int,
        height: int,
        input_shape: tuple[int, ...],
    ) -> PetDescriptor:
        diagnostics = {
            "unified": {
                "single_graph": True,
                "external_models": [],
                "provider": self.session.get_providers()[0],
                "raw_spatial_input": bool(self.raw_spatial_input),
                "graph_preprocessing": self.metadata.get(
                    "graph_preprocessing",
                    self.metadata.get("preprocessing", {}).get("graph_internal", []),
                ),
                "input_shape": [int(value) for value in input_shape],
            }
        }
        return PetDescriptor(
            fused_feature=embedding,
            nose_feature=None,
            face_feature=None,
            fusion_weights=None,
            branch_quality=None,
            branch_available=None,
            detection=None,
            inference_size=(width, height),
            runtime_diagnostics=diagnostics,
        )

    def encode_rgb(self, image_rgb: np.ndarray) -> list[PetDescriptor]:
        """Encode one raw HWC RGB pixel array without graph-external transforms."""

        rgb = np.asarray(image_rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("image_rgb must be one HWC RGB image with three channels")
        height, width = (int(rgb.shape[0]), int(rgb.shape[1]))
        if min(height, width) < 2:
            raise ValueError("image is too small for unified inference")
        if self.raw_spatial_input:
            batch = rgb.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
        else:
            boxed, _, _ = letterbox_rgb(
                rgb,
                size=self.input_size,
                fill_value=0,
                allow_upscale=self.letterbox_allow_upscale,
            )
            batch = boxed.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
        embedding = self._run(batch)[0]
        return [
            self._descriptor(
                embedding,
                width=width,
                height=height,
                input_shape=tuple(int(value) for value in batch.shape),
            )
        ]

    def encode_image(self, image) -> list[PetDescriptor]:
        """Encode a BGR image/path for the legacy gallery API.

        The BGR-to-RGB conversion is a transport adapter.  For a raw-input
        graph all geometric resizing and normalization happen after this call,
        inside ONNX.
        """

        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image}")
        else:
            image_bgr = np.asarray(image)
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image must be one BGR image with three channels")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.encode_rgb(rgb)

    def encode_embedding(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Return the graph output directly for tensor-oriented integrations."""

        return self.encode_rgb(image_rgb)[0].fused_feature

    def backend_info(self) -> dict:
        profile_info = (
            self.profile.public_metadata()
            if self.profile is not None
            else {
                "deployment_profile": "custom",
                "deployment_role": "custom",
                "release_role": "custom",
                "display_name": "统一识别 · 自定义模型",
                "summary": "固定输入联合模型 · RGB → 512D",
                "capability": "unified-embedding",
                "model_package": None,
            }
        )
        return {
            "backend": "onnxruntime-unified",
            **profile_info,
            "model": str(self.model_path),
            "metadata": (str(self.metadata_path) if self.metadata_path else None),
            "model_sha256": self.model_sha256,
            "source_checkpoint": (
                str(self.source_checkpoint)
                if self.source_checkpoint is not None
                else None
            ),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "provider": self.session.get_providers()[0],
            "provider_chain": self.session.get_providers(),
            "onnxruntime_version": self._ort.__version__,
            "device": str(self.tensor_device),
            "embedding_dim": 512,
            "input_size": self.input_size,
            "input_shape": list(self.input_shape),
            "raw_spatial_input": bool(self.raw_spatial_input),
            "minimum_input_side": self.minimum_input_side,
            "maximum_input_side": self.maximum_input_side,
            "letterbox_allow_upscale": self.letterbox_allow_upscale,
            "graph_preprocessing": self.metadata.get(
                "graph_preprocessing",
                self.metadata.get("preprocessing", {}).get("graph_internal", []),
            ),
            "external_models": [],
            "single_graph": True,
        }
