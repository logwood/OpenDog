"""Runtime adapter for the dynamic spatial-detail unified model.

The production runtime remains in :mod:`unified_runtime` and expects a fixed
1280 square.  This adapter has a different capability contract: callers provide
a raw RGB image tensor with dynamic ``H``/``W``, and the graph performs the
global letterbox plus high-resolution local sampling internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from .multimodal import PetDescriptor
from .model_profiles import ModelProfile, profile_for_model_path
from .unified_highres_data import (
    HIGHRES_MIN_INPUT_SIDE,
    validate_highres_dimensions,
)
from .unified_runtime import (
    UNIFIED_ONNX_INPUT_NAMES,
    UNIFIED_ONNX_OUTPUT_NAMES,
    resolve_unified_provider,
    sha256_file,
    validate_raw_rgb_input,
)


class UnifiedHighResolutionONNXRuntimePipeline:
    """Adapt a raw-spatial ONNX graph to the gallery API.

    ``encode_image`` accepts either a BGR numpy image or a path and returns
    one :class:`~pet_id.multimodal.PetDescriptor`, just like the fixed-shape
    adapter. No spatial resize, detector, segmenter, or second identity model
    is instantiated outside the graph; oversized inputs are rejected against
    the declared contract instead of being silently rewritten.
    """

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
        warmup_shapes: Sequence[tuple[int, int]] = (),
        minimum_input_side: int | None = None,
        maximum_input_side: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment dependent
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
            raise ValueError("ONNX hash differs from deployment metadata")

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
            expected_checkpoint_hash = self.metadata.get(
                "source_checkpoint_sha256"
            )
            if (
                expected_checkpoint_hash
                and self.source_checkpoint_sha256.casefold()
                != str(expected_checkpoint_hash).casefold()
            ):
                raise ValueError("Source checkpoint hash differs from metadata")

        contract = self.metadata.get("runtime_contract", {})
        input_contract = contract.get("inputs", {}).get("rgb", {})
        output_contract = contract.get("outputs", {}).get("embedding", {})
        metadata_minimum = int(
            input_contract.get("height_width_minimum", HIGHRES_MIN_INPUT_SIDE)
        )
        metadata_maximum = int(
            input_contract.get("height_width_maximum", 4096)
        )
        self.minimum_input_side = int(
            metadata_minimum if minimum_input_side is None else minimum_input_side
        )
        self.maximum_input_side = int(
            metadata_maximum if maximum_input_side is None else maximum_input_side
        )
        if self.minimum_input_side < 2:
            raise ValueError("minimum_input_side must be at least two")
        if self.maximum_input_side < self.minimum_input_side:
            raise ValueError("maximum_input_side must be at least minimum_input_side")
        declared_output_shape = output_contract.get("shape")
        if declared_output_shape and (
            len(declared_output_shape) != 2
            or int(declared_output_shape[1]) != 512
        ):
            raise RuntimeError("Metadata must declare a [N,512] output")

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
                f"Refusing provider fallback: expected {expected_provider}, activated {active}"
            )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if tuple(item.name for item in inputs) != UNIFIED_ONNX_INPUT_NAMES:
            raise RuntimeError("ONNX must have exactly one 'rgb' input")
        if tuple(item.name for item in outputs) != UNIFIED_ONNX_OUTPUT_NAMES:
            raise RuntimeError("ONNX must have exactly one 'embedding' output")
        if inputs[0].type != "tensor(float)":
            raise RuntimeError("RGB input must be float32")
        if outputs[0].type != "tensor(float)":
            raise RuntimeError("Embedding output must be float32")
        input_shape = tuple(inputs[0].shape)
        output_shape = tuple(outputs[0].shape)
        if len(input_shape) != 4 or input_shape[1] != 3:
            raise RuntimeError(f"Unexpected input shape: {input_shape}")
        if len(output_shape) != 2 or output_shape[1] != 512:
            raise RuntimeError(
                "ONNX output second dimension must be the static value 512; "
                f"got {output_shape}"
            )
        if isinstance(input_shape[2], int) or isinstance(input_shape[3], int):
            raise RuntimeError(
                "ONNX must expose dynamic height and width dimensions"
            )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_size = None  # compatibility marker; this graph is not square-fixed
        self.identity_model = self

        requested_shapes = tuple(warmup_shapes)
        if not requested_shapes and warmup_batches:
            requested_shapes = tuple((1280, 1280) for _ in warmup_batches)
        if len(requested_shapes) not in {0, len(tuple(warmup_batches))}:
            raise ValueError("warmup_shapes must match warmup_batches")
        for batch_size, shape in zip(warmup_batches, requested_shapes):
            height, width = validate_highres_dimensions(
                shape[0],
                shape[1],
                minimum_side=self.minimum_input_side,
                maximum_side=self.maximum_input_side,
            )
            self._run(
                np.zeros(
                    (int(batch_size), 3, height, width),
                    dtype=np.float32,
                )
            )

    @property
    def device(self) -> torch.device:
        return self.tensor_device

    def _run(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = validate_raw_rgb_input(rgb)
        validate_highres_dimensions(
            rgb.shape[2],
            rgb.shape[3],
            minimum_side=self.minimum_input_side,
            maximum_side=self.maximum_input_side,
        )
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
            output = self.session.run(
                list(UNIFIED_ONNX_OUTPUT_NAMES),
                {"rgb": rgb},
            )[0]
        embedding = torch.from_numpy(np.asarray(output, dtype=np.float32)).float()
        expected_shape = (rgb.shape[0], 512)
        if tuple(embedding.shape) != expected_shape:
            raise RuntimeError(
                f"ONNX returned {tuple(embedding.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("ONNX returned non-finite values")
        norms = embedding.norm(dim=1, keepdim=True)
        if (norms <= 0).any():
            raise FloatingPointError("ONNX returned a zero descriptor")
        if not torch.allclose(norms, torch.ones_like(norms), atol=3e-3, rtol=3e-3):
            raise FloatingPointError(
                "ONNX graph returned a non-normalized descriptor"
            )
        # L2 normalization is part of the graph contract; do not add a
        # second Python post-processing stage.
        return embedding

    def warmup(self, batch_sizes: Sequence[int], *, shape: tuple[int, int]) -> None:
        height, width = validate_highres_dimensions(
            shape[0],
            shape[1],
            minimum_side=self.minimum_input_side,
            maximum_side=self.maximum_input_side,
        )
        for batch_size in tuple(dict.fromkeys(int(item) for item in batch_sizes)):
            if batch_size < 1:
                raise ValueError("warmup batch sizes must be positive")
            self._run(
                np.zeros((batch_size, 3, height, width), dtype=np.float32)
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
                "model_type": "unified_high_resolution_pet_reid",
                "raw_spatial_input": True,
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
        """Encode raw HWC RGB pixels; all spatial work stays in ONNX."""

        rgb = np.asarray(image_rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("image_rgb must be one HWC RGB image with three channels")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if min(height, width) < 2:
            raise ValueError("image is too small for spatial-detail inference")
        batch = rgb.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
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
        """Encode a BGR image/path without external resize or quality tails."""

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

    def backend_info(self) -> dict:
        profile_info = (
            self.profile.public_metadata()
            if self.profile is not None
            else {
                "deployment_profile": "custom",
                "deployment_role": "custom",
                "release_role": "custom",
                "display_name": "高分辨率统一识别 · 自定义模型",
                "summary": "动态空间细节模型 · RGB → 512D",
                "capability": "spatial-detail-embedding",
                "model_package": None,
            }
        )
        return {
            "backend": "onnxruntime-unified-highres",
            **profile_info,
            "model": str(self.model_path),
            "metadata": str(self.metadata_path) if self.metadata_path else None,
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
            "input_shape": list(self.input_shape),
            "minimum_input_side": self.minimum_input_side,
            "maximum_input_side": self.maximum_input_side,
            "raw_spatial_input": True,
            "external_models": [],
            "single_graph": True,
        }


def build_highres_onnx_pipeline(
    model_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    source_checkpoint: str | Path | None = None,
    provider: str = "cuda",
    device: str | torch.device | None = None,
    verify_hash: bool = True,
    warmup_batches: Sequence[int] = (),
    warmup_shapes: Sequence[tuple[int, int]] = (),
) -> UnifiedHighResolutionONNXRuntimePipeline:
    """Construct the spatial-detail adapter with an explicit dynamic-input contract."""

    return UnifiedHighResolutionONNXRuntimePipeline(
        model_path,
        metadata_path=metadata_path,
        source_checkpoint=source_checkpoint,
        provider=provider,
        device=device,
        verify_hash=verify_hash,
        warmup_batches=warmup_batches,
        warmup_shapes=warmup_shapes,
    )
