"""Opt-in local visibility research model; no additional identity encoder.

The two visibility logits use existing localization feature maps. They are
neither attention concentration nor calibrated correctness probabilities.
The released geometry/encoders/adapter/fusion parameters/refiner stay frozen.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .unified_local_path import LocalPathUnifiedPetReID, build_released_v3_safely
from .unified_training import atomic_torch_save, sha256_file
from .workspace_paths import resolve_legacy_path

MODEL_TYPE = "unified_local_part_visibility_v1"
SCHEMA_VERSION = 1


class LocalPartVisibilityHead(nn.Module):
    """Independent face/nose usability logits from spatial features, not IDs.

    Existing spatial attention only pools features; its entropy/confidence is
    never a reliability target or gate input. Context is reduced to two logits,
    never to a separate global identity vector.
    """

    def __init__(self, input_channels: int, hidden_channels: int = 32):
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Visibility dimensions must be positive")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, 1), nn.GELU()
        )
        self.classifiers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * self.hidden_channels),
                nn.Linear(2 * self.hidden_channels, 1),
            )
            for _ in range(2)
        )
        # Conservative engineering initialization. BCE supervision still has
        # non-zero gradients; this is not a trained or calibrated model.
        for classifier in self.classifiers:
            nn.init.zeros_(classifier[-1].weight)
            nn.init.constant_(classifier[-1].bias, -2.0)

    def forward(self, features, attention):
        if not torch.onnx.is_in_onnx_export():
            if features.ndim != 4 or features.shape[1] != self.input_channels:
                raise ValueError("Visibility features must be [N,C,H,W]")
            if attention.shape != (features.shape[0], 2, *features.shape[-2:]):
                raise ValueError("Visibility attention must be [N,2,H,W]")
        reduced = self.reduction(features)
        spatial = reduced.flatten(2).transpose(1, 2)
        part_features = attention.flatten(2).to(spatial.dtype) @ spatial
        context = reduced.mean(dim=(2, 3))
        return torch.cat(
            [
                classifier(torch.cat((part_features[:, index], context), dim=1))
                for index, classifier in enumerate(self.classifiers)
            ],
            dim=1,
        )


def visibility_admission(logits):
    """Zero for logits <= 0; continuous admission above the binary boundary.

    This fixed mapping is a graph design, not a development-set threshold sweep.
    Unknown labels are masked in the loss, never substituted for negative logits.
    """
    return (2.0 * logits.sigmoid() - 1.0).clamp(0.0, 1.0)


def fuse_with_visibility(face, nose, logits, maximum_nose_weight, refiner):
    """Gate *both* nose routes and retain bitwise local-face fallback.

    For any finite nose vector and refiner parameters, either unusable part
    disables nose fusion AND the complete refiner correction. No second
    normalization of the face is allowed on the disabled route.
    """
    probabilities = logits.sigmoid()
    admission = visibility_admission(logits)
    joint_admission = admission[:, 0] * admission[:, 1]
    active = joint_admission[:, None] > 0
    weight = maximum_nose_weight.to(face) * joint_admission
    mixed = F.normalize(
        F.normalize(face, dim=1) * (1.0 - weight[:, None])
        + F.normalize(nose, dim=1) * weight[:, None],
        dim=1,
    )
    fused = torch.where(active, mixed, face)
    # The old refiner's fourth signal is now the independently supervised face
    # score. Its complete output is gated below, including its interaction term.
    refined = refiner(fused, face, nose, probabilities[:, 0])
    corrected = F.normalize(
        fused + joint_admission[:, None] * (refined - fused), dim=1
    )
    embedding = torch.where(active, corrected, face)
    return {
        "embedding": embedding,
        "base_embedding": fused,
        "visibility_logits": logits,
        "visibility_probabilities": probabilities,
        "part_admission": admission,
        "joint_admission": joint_admission,
        "nose_weight": weight,
        "refiner_admission": joint_admission,
        "exact_local_face_fallback": ~active[:, 0],
    }


class LocalVisibilityUnifiedPetReID(LocalPathUnifiedPetReID):
    """Single raw RGB -> local geometry/face/nose/visibility/fusion -> 512D."""

    def __init__(self, model, *, hidden_channels: int = 32):
        super().__init__(model)  # Legacy options are deliberately not exposed.
        channels = model.base_model.geometry_frontend.geometry.reduction[0].in_channels
        self.visibility_head = LocalPartVisibilityHead(channels, hidden_channels)
        self.configure_visibility_training()

    def configure_visibility_training(self):
        self.model.requires_grad_(False).eval()
        self.visibility_head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen backbone's BN running buffers must not drift in head training.
        self.model.eval()
        return self

    def configuration(self):
        local = super().configuration()
        return {
            "model_type": MODEL_TYPE,
            "local_path": local,
            "visibility": {
                "input_channels": self.visibility_head.input_channels,
                "hidden_channels": self.visibility_head.hidden_channels,
                "parts": ["face", "nose"],
                "features": "geometry_spatial_part_pool_and_context",
                "gate": "product_of_rectified_sigmoid_admissions",
                "nose_weight_bound": "released_fusion_maximum_nose_weight",
                "refiner_signal": "predicted_face_usability",
                "refiner_correction_gated": True,
                "disabled_output": "bitwise_local_face_descriptor",
                "calibrated": False,
            },
            "training_scope": "visibility_head_only_frozen_base_in_eval_mode",
            "global_identity_branch": False,
            "runtime_external_models": [],
        }

    def _fuse_local_descriptors(
        self, face, nose, prediction, nose_valid_fraction, geometry_features
    ):
        logits = self.visibility_head(geometry_features, prediction.attention)
        return fuse_with_visibility(
            face,
            nose,
            logits,
            self.model.base_model.fusion.maximum_nose_weight,
            self.model.refiner,
        )


def build_local_visibility_candidate(source_v3_checkpoint, *, device="cpu", **kwargs):
    base, _ = build_released_v3_safely(source_v3_checkpoint, device=device)
    return LocalVisibilityUnifiedPetReID(base, **kwargs).to(device).eval()


def save_untrained_visibility_checkpoint(model, path, *, source_v3_checkpoint):
    """Engineering snapshot only; no API to invent training/calibration evidence."""
    path, source = Path(path).resolve(), Path(source_v3_checkpoint).resolve()
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_config": model.configuration(),
        "source_v3_checkpoint": {"path": str(source), "sha256": sha256_file(source)},
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "status": "untrained_engineering_only_not_for_deployment",
        "production_changed": False,
    }
    atomic_torch_save(payload, path)
    return payload


def load_local_visibility_checkpoint(path, *, device="cpu", for_training=False):
    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=True, mmap=True)
    expected_keys = {
        "schema_version", "model_type", "model_config", "source_v3_checkpoint",
        "model", "status", "production_changed",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("model_type") != MODEL_TYPE
        or payload.get("status") != "untrained_engineering_only_not_for_deployment"
        or payload.get("production_changed") is not False
    ):
        raise ValueError("Unsupported visibility checkpoint envelope")
    source = payload["source_v3_checkpoint"]
    if set(source) != {"path", "sha256"}:
        raise ValueError("Invalid visibility checkpoint source")
    source_path = resolve_legacy_path(source["path"])
    if sha256_file(source_path) != source["sha256"]:
        raise ValueError("Visibility source V3 hash mismatch")
    config = payload["model_config"]
    model = build_local_visibility_candidate(
        source_path, device=device, hidden_channels=config["visibility"]["hidden_channels"]
    )
    if config != model.configuration():
        raise ValueError("Visibility configuration differs from graph definition")
    # Head-only scope is enforceable: even a syntactically valid full checkpoint
    # must not smuggle a modified encoder, fusion weight or BN buffer.
    for key, original in model.model.state_dict().items():
        saved = payload["model"].get("model." + key)
        if saved is None or not torch.equal(saved.cpu(), original.detach().cpu()):
            raise ValueError(f"Frozen V3 state changed in visibility checkpoint: {key}")
    model.load_state_dict(payload["model"], strict=True)
    if for_training:
        model.configure_visibility_training()
        model.train()
    else:
        model.eval().requires_grad_(False)
    return model, payload
