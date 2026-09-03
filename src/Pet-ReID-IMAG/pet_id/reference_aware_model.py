"""End-to-end image-set matching built on the shared pet descriptor encoder.

The regular identity model still maps one image to one descriptor.  This
module adds the *model-level* set contract needed when an identity has several
reference views: a query image and a padded set of reference images are passed
through the same encoder, and a query-conditioned matcher scores the resulting
descriptors.  Keeping this wrapper separate makes the existing single-image
checkpoint and ONNX graph backwards compatible while allowing the encoder and
set head to be trained jointly in an experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .reference_set_model import (
    QueryConditionedReferenceMatcher,
    ReferenceSetMatcherRuntime,
)


MODEL_FORMAT = "reference-aware-pet-reid"


def _embedding_from_encoder_output(output: Any) -> torch.Tensor:
    """Extract a descriptor from the common encoder return conventions."""

    if isinstance(output, Mapping):
        for key in ("embedding", "descriptor", "output"):
            candidate = output.get(key)
            if isinstance(candidate, torch.Tensor):
                output = candidate
                break
        else:
            raise ValueError(
                "image encoder mapping output must contain an embedding tensor"
            )
    elif isinstance(output, (tuple, list)):
        candidates = [item for item in output if isinstance(item, torch.Tensor)]
        if not candidates:
            raise ValueError("image encoder output contains no tensor descriptor")
        output = candidates[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(
            "image encoder must return a tensor or a mapping containing one"
        )
    if output.ndim != 2:
        raise ValueError("image encoder descriptor must have shape [batch, dimension]")
    if not bool(torch.isfinite(output.float()).all()):
        raise ValueError("image encoder descriptor contains non-finite values")
    return output


class ReferenceAwarePetReID(nn.Module):
    """Jointly apply an image encoder and a query-conditioned set matcher.

    ``query_rgb`` has shape ``[B, C, H, W]`` and ``reference_rgb`` has shape
    ``[B, K, C, H, W]``.  ``reference_mask`` marks real reference rows, which
    lets a fixed-width batch represent identities with different numbers of
    views.  Gradients flow through both the matcher and the shared image
    encoder; callers can freeze the encoder for a cheap head-only experiment.

    The descriptor-only ``score_descriptors`` method is intentionally exposed
    as well.  It lets a serving layer use the same learned set head with a
    descriptor gallery, avoiding a second image encode for every candidate.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        matcher: QueryConditionedReferenceMatcher,
        *,
        max_references: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(matcher, QueryConditionedReferenceMatcher):
            raise TypeError("matcher must be a QueryConditionedReferenceMatcher")
        encoder_dim = getattr(image_encoder, "descriptor_dim", None)
        if encoder_dim is not None and int(encoder_dim) != matcher.descriptor_dim:
            raise ValueError(
                "image encoder and matcher descriptor dimensions differ: "
                f"{encoder_dim} != {matcher.descriptor_dim}"
            )
        resolved_max = (
            matcher.max_references if max_references is None else int(max_references)
        )
        if resolved_max < 1 or resolved_max > matcher.max_references:
            raise ValueError(
                "max_references must be between 1 and matcher.max_references"
            )
        self.image_encoder = image_encoder
        self.matcher = matcher
        self.max_references = resolved_max
        self.descriptor_dim = matcher.descriptor_dim

    @property
    def input_size(self) -> int | None:
        value = getattr(self.image_encoder, "input_size", None)
        return int(value) if value is not None else None

    def configuration(self) -> dict[str, Any]:
        """Return a serializable model-level configuration."""

        encoder_configuration = getattr(self.image_encoder, "configuration", None)
        if callable(encoder_configuration):
            value = encoder_configuration()
            if not isinstance(value, Mapping):
                raise ValueError("image encoder configuration must be a mapping")
            encoder_config: dict[str, Any] = dict(value)
        else:
            encoder_config = {
                "type": type(self.image_encoder).__name__,
                "input_size": self.input_size,
                "descriptor_dim": self.descriptor_dim,
            }
        return {
            "format": MODEL_FORMAT,
            "descriptor_dim": self.descriptor_dim,
            "max_references": self.max_references,
            "encoder": encoder_config,
            "matcher": self.matcher.configuration(),
        }

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a 4-D image batch and enforce the matcher descriptor contract."""

        if images.ndim != 4 or images.shape[1] < 1:
            raise ValueError("images must have shape [batch, channels, height, width]")
        encoded = _embedding_from_encoder_output(self.image_encoder(images))
        if encoded.shape[0] != images.shape[0]:
            raise ValueError("image encoder changed the batch dimension")
        if encoded.shape[1] != self.descriptor_dim:
            raise ValueError(
                "image encoder descriptor width does not match matcher: "
                f"{encoded.shape[1]} != {self.descriptor_dim}"
            )
        return F.normalize(encoded.float(), dim=1, eps=1e-12)

    def encode_reference_images(self, reference_rgb: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,K,C,H,W]`` references without scoring them yet."""

        if reference_rgb.ndim != 5:
            raise ValueError(
                "reference_rgb must have shape [batch, references, channels, height, width]"
            )
        batch, count = reference_rgb.shape[:2]
        if count < 1 or count > self.max_references:
            raise ValueError(
                f"reference_rgb must contain between 1 and {self.max_references} rows"
            )
        flattened = reference_rgb.reshape(batch * count, *reference_rgb.shape[2:])
        descriptors = self.encode_images(flattened)
        return descriptors.reshape(batch, count, self.descriptor_dim)

    def _validate_set_inputs(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if query_rgb.ndim != 4:
            raise ValueError(
                "query_rgb must have shape [batch, channels, height, width]"
            )
        if reference_rgb.ndim != 5:
            raise ValueError(
                "reference_rgb must have shape [batch, references, channels, height, width]"
            )
        if query_rgb.shape[0] != reference_rgb.shape[0]:
            raise ValueError("query and reference batch dimensions must match")
        if query_rgb.shape[1] != reference_rgb.shape[2]:
            raise ValueError("query and reference channel dimensions must match")
        if reference_rgb.shape[1] < 1 or reference_rgb.shape[1] > self.max_references:
            raise ValueError(
                f"reference_rgb must contain between 1 and {self.max_references} rows"
            )
        if reference_mask is not None:
            if tuple(reference_mask.shape) != tuple(reference_rgb.shape[:2]):
                raise ValueError("reference_mask must have shape [batch, references]")
            if (
                not torch.is_floating_point(reference_mask)
                and reference_mask.dtype != torch.bool
            ):
                reference_mask = reference_mask != 0
            elif not bool(torch.isfinite(reference_mask.float()).all()):
                raise ValueError("reference_mask must contain finite values")
            reference_mask = reference_mask.to(
                device=reference_rgb.device,
                dtype=torch.bool,
            )
            if not bool(reference_mask.any(dim=1).all()):
                raise ValueError("each query must have at least one reference")
        return reference_mask

    def forward_encoded(
        self,
        query_descriptor: torch.Tensor,
        reference_descriptors: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Score already encoded query/reference sets with the same model head."""

        return self.matcher(
            query_descriptor,
            reference_descriptors,
            reference_mask,
            return_aux=return_aux,
        )

    def _encode_images_export(self, images: torch.Tensor) -> torch.Tensor:
        """Encode without eager-only Python validation for ONNX tracing."""

        encoded = self.image_encoder(images)
        if isinstance(encoded, Mapping):
            encoded = encoded.get(
                "embedding",
                encoded.get("descriptor", encoded.get("output")),
            )
        elif isinstance(encoded, (tuple, list)):
            encoded = next(item for item in encoded if isinstance(item, torch.Tensor))
        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError("image encoder export path returned no tensor")
        return F.normalize(encoded.float(), dim=1, eps=1e-12)

    def _encode_reference_images_export(
        self, reference_rgb: torch.Tensor
    ) -> torch.Tensor:
        flattened = reference_rgb.flatten(0, 1)
        encoded = self._encode_images_export(flattened)
        return encoded.reshape(
            reference_rgb.shape[0], reference_rgb.shape[1], self.descriptor_dim
        )

    def forward_export(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-only image-set path used by the ONNX export wrapper."""

        query_descriptor = self._encode_images_export(query_rgb)
        reference_descriptors = self._encode_reference_images_export(reference_rgb)
        return self.matcher.forward_export(
            query_descriptor,
            reference_descriptors,
            reference_mask,
        )

    def score_descriptors(
        self,
        query_descriptor: torch.Tensor,
        reference_descriptors: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Alias used by descriptor-gallery integrations."""

        return self.forward_encoded(
            query_descriptor,
            reference_descriptors,
            reference_mask,
            return_aux=return_aux,
        )

    def forward(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Encode and score one batch of query/reference image sets."""

        reference_mask = self._validate_set_inputs(
            query_rgb, reference_rgb, reference_mask
        )
        query_descriptor = self.encode_images(query_rgb)
        reference_descriptors = self.encode_reference_images(reference_rgb)
        output = self.forward_encoded(
            query_descriptor,
            reference_descriptors,
            reference_mask,
            return_aux=return_aux,
        )
        if not return_aux:
            return output
        if not isinstance(output, dict):
            raise RuntimeError("matcher returned no auxiliary output")
        return {
            **output,
            "query_descriptor": query_descriptor,
            "reference_descriptors": reference_descriptors,
        }

    def freeze_encoder(self) -> None:
        """Freeze the shared image encoder for head-only warm-up training."""

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Make every encoder parameter trainable."""

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(True)

    def descriptor_scorer(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ReferenceAwareDescriptorScorer":
        """Return the gallery-service adapter for this model's set head."""

        return ReferenceAwareDescriptorScorer(self, device=device)


def build_reference_aware_encoder_from_checkpoint(
    checkpoint_path: str | Path,
    arcface_checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    verify_sources: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a supported single-image encoder for the set wrapper.

    The workspace has several single-image checkpoint envelopes.  The
    reference-aware head only needs their common ``RGB -> descriptor``
    contract, so dispatch on the envelope's model type instead of assuming
    every checkpoint is the original ``UnifiedPetReID`` shape.  In particular,
    external-joint and high-resolution packages carry nested source models and
    must be restored by their own verified loaders.

    ``arcface_checkpoint`` is used only by the legacy/base loader.  The
    packaged external-joint and high-resolution checkpoints verify their
    embedded source chain themselves.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"single-image checkpoint must be a mapping: {checkpoint}")
    model_type = payload.get("model_type")

    if model_type == "unified_high_resolution_pet_reid":
        from .unified_highres import build_highres_from_checkpoint

        return build_highres_from_checkpoint(
            checkpoint,
            device=device,
            verify_sources=verify_sources,
        )
    if model_type == "unified_external_joint_pet_reid":
        from .unified_external_model import build_external_joint_from_checkpoint

        return build_external_joint_from_checkpoint(
            checkpoint,
            device=device,
            verify_sources=verify_sources,
        )
    if model_type == "unified_semantic_pet_reid":
        from .unified_semantic_checkpoint import build_unified_semantic_from_checkpoint

        return build_unified_semantic_from_checkpoint(
            checkpoint,
            device=device,
            verify_sources=verify_sources,
        )

    if arcface_checkpoint is None:
        raise ValueError(
            "This checkpoint does not contain a supported packaged model type; "
            "an arcface checkpoint is required for the legacy base loader"
        )
    from .unified_training import build_model_from_checkpoint

    return build_model_from_checkpoint(
        checkpoint,
        arcface_checkpoint,
        device=device,
    )


class ReferenceAwarePetReIDExport(nn.Module):
    """Tensor-only export boundary for a fixed-width image reference set."""

    def __init__(self, model: ReferenceAwarePetReID) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        query_rgb: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_export(query_rgb, reference_rgb, reference_mask)


class ReferenceAwareDescriptorScorer:
    """NumPy gallery scorer backed by the set head inside the joint model.

    Online galleries already cache one descriptor per enrolled image.  This
    adapter reuses those descriptors and the exact matcher weights from the
    joint model, while the full image-set ``forward`` remains available for
    training and offline end-to-end evaluation.
    """

    def __init__(
        self,
        model: ReferenceAwarePetReID,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            parameter = next(model.parameters(), None)
            target = parameter.device if parameter is not None else torch.device("cpu")
        else:
            target = torch.device(device)
        self.model = model.to(target).eval()
        self.runtime = ReferenceSetMatcherRuntime(self.model.matcher, device=target)

    def backend_info(self) -> dict[str, Any]:
        return {
            "type": MODEL_FORMAT,
            "descriptor_dim": self.model.descriptor_dim,
            "max_references": self.model.max_references,
            "model_config": self.model.configuration(),
            "encoder_fingerprint": None,
        }

    def score(self, query, references):
        return self.runtime.score(query, references)

    def score_many(self, query, reference_sets):
        return self.runtime.score_many(query, reference_sets)

    def score_gallery(self, query, prototypes):
        return self.runtime.score_gallery(query, prototypes)


def create_reference_aware_checkpoint(
    model: ReferenceAwarePetReID,
    *,
    base_encoder_checkpoint: str | Path | None = None,
    encoder_fingerprint: str | None = None,
    training: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained checkpoint payload for joint experiments."""

    return {
        "format": MODEL_FORMAT,
        "model_config": model.configuration(),
        "model": model.state_dict(),
        "base_encoder_checkpoint": (
            str(Path(base_encoder_checkpoint).expanduser().resolve())
            if base_encoder_checkpoint is not None
            else None
        ),
        "encoder_fingerprint": encoder_fingerprint,
        "training": dict(training or {}),
        "optimizer": dict(optimizer_state or {}),
    }


def save_reference_aware_model(
    model: ReferenceAwarePetReID,
    path: str | Path,
    *,
    base_encoder_checkpoint: str | Path | None = None,
    encoder_fingerprint: str | None = None,
    training: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a joint model checkpoint."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".writing")
    torch.save(
        create_reference_aware_checkpoint(
            model,
            base_encoder_checkpoint=base_encoder_checkpoint,
            encoder_fingerprint=encoder_fingerprint,
            training=training,
            optimizer_state=optimizer_state,
        ),
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def build_reference_aware_model_from_checkpoint(
    path: str | Path,
    image_encoder: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ReferenceAwarePetReID, dict[str, Any]]:
    """Restore a joint checkpoint around a caller-provided encoder.

    The encoder factory is deliberately supplied by the caller because the
    project has several compatible single-image backends.  The checkpoint
    still validates the descriptor and matcher architecture before loading.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"Unexpected reference-aware model format: {checkpoint_path}")
    config = payload.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("reference-aware checkpoint has no model_config")
    matcher_config = config.get("matcher")
    if not isinstance(matcher_config, Mapping):
        raise ValueError("reference-aware checkpoint has no matcher configuration")
    constructor_keys = {
        "descriptor_dim",
        "hidden_dim",
        "max_references",
        "reference_top_k",
        "reference_score_weight",
        "attention_temperature",
        "maximum_residual",
    }
    matcher = QueryConditionedReferenceMatcher(
        **{
            key: matcher_config[key]
            for key in constructor_keys
            if key in matcher_config
        }
    )
    model = ReferenceAwarePetReID(
        image_encoder,
        matcher,
        max_references=int(config.get("max_references", matcher.max_references)),
    )
    if int(config.get("descriptor_dim", model.descriptor_dim)) != model.descriptor_dim:
        raise ValueError("reference-aware checkpoint descriptor dimension mismatch")
    incompatible = model.load_state_dict(payload.get("model", {}), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"reference-aware checkpoint mismatch: {incompatible}")
    return model.to(device), dict(payload)


__all__ = [
    "MODEL_FORMAT",
    "ReferenceAwareDescriptorScorer",
    "ReferenceAwarePetReID",
    "ReferenceAwarePetReIDExport",
    "build_reference_aware_encoder_from_checkpoint",
    "build_reference_aware_model_from_checkpoint",
    "create_reference_aware_checkpoint",
    "save_reference_aware_model",
]
