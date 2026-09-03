"""Compatibility adapters for immutable artifacts created by older releases.

Version-shaped strings are intentionally isolated here. New implementation code
must use capability names; this boundary only translates frozen external schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_LEGACY_DETAIL_SOURCE_KEY = "v4_checkpoint"
_LEGACY_DETAIL_STATE_PREFIX = "v4_model."
_DETAIL_SOURCE_KEY = "detail_checkpoint"
_DETAIL_STATE_PREFIX = "detail_model."
_BASELINE_CONTROLLER_PROTOCOL = Path(
    "artifacts/runs/agent_v2/learned_controller_v1/protocol.json"
)
_SHARED_CONTROLLER_FEATURE_CACHE = Path(
    "artifacts/runs/agent_v1/formal_joint100_20_v1/feature_cache.sqlite3"
)
_HISTORICAL_MODEL_TRAINING_MANIFESTS = (
    Path("artifacts/runs/legacy/dogfacenet_joint100_protocol_v1/train_manifest.json"),
    Path("artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/train_manifest.json"),
)
_CONTROLLER_PROTOCOL_SEARCH_GLOB = "agent_v*/**/protocol.json"
_FULL_RESOLUTION_SEMANTIC_BASE = Path(
    "artifacts/runs/unified/v2/semantic_base/model.pth"
)
_FULL_RESOLUTION_GEOMETRY = Path(
    "artifacts/runs/unified/v2/geometry/model_best.pth"
)
_DETAIL_FINAL_ENDPOINT = "detail_final_512"
_LEGACY_DETAIL_FINAL_ENDPOINT = "v4_final_512"
_PARENT_SOURCE_KEY = "parent_checkpoint"
_LEGACY_PARENT_SOURCE_KEY = "parent_v3_checkpoint"
_HIGH_RESOLUTION_PROTOCOL_NAME = "unified_pet_reid_real_high_resolution"
_LEGACY_HIGH_RESOLUTION_PROTOCOL_NAME = (
    "unified_pet_reid_v4_real_high_resolution"
)
_FUSION_MODE_ALIASES = {
    "legacy_concat": "legacy_concat",
    "shared_projection": "shared_projection",
    "semantic_residual": "semantic_residual",
    "shared_space_v2": "shared_projection",
    "semantic_residual_v3": "semantic_residual",
}
_FUSION_SIGNATURE_ALIASES = {
    **_FUSION_MODE_ALIASES,
    "semantic_residual+bifor_lowrank": "semantic_residual+bifor_lowrank",
    "semantic_residual_v3+bifor_lowrank_v1": (
        "semantic_residual+bifor_lowrank"
    ),
}
_LOWRANK_BODY_FUSION_ARCHITECTURES = {
    "lowrank_semantic_body_fusion",
    "lowrank_semantic_body_fusion_v1",
}
_ACCEPTANCE_SCHEMA_BY_PROTOCOL = {
    "unified_pet_reid_v1_noninferiority": 1,
    "unified_pet_reid_v2_strict_noninferiority": 2,
    "unified_pet_reid_v3_external_strict_noninferiority": 3,
}
_ACCEPTANCE_PROTOCOLS = {
    "baseline-training": "unified_pet_reid_v2_strict_noninferiority",
    "external-development": "unified_pet_reid_v3_external_strict_noninferiority",
    "external-runtime": "unified_pet_reid_external_v3",
}
_ACCEPTANCE_PATHS = {
    "legacy-training": Path("models/acceptance/unified_pet_reid_v1.json"),
    "baseline-training": Path("models/acceptance/unified_pet_reid_v2.json"),
    "external-development": Path(
        "models/acceptance/unified_pet_reid_v3.json"
    ),
}
_SOURCE_LOCK_ALIASES = {
    "arcface-checkpoint": (
        "dog_arcface_checkpoint",
    ),
    "semantic-checkpoint": (
        "semantic_checkpoint",
        "semantic_v3_checkpoint",
    ),
    "semantic-config": (
        "semantic_config",
        "semantic_v3_config",
    ),
    "semantic-onnx": (
        "semantic_onnx",
        "semantic_v3_onnx",
    ),
}
_HISTORICAL_RUN_PATHS = {
    "fresh-baseline": Path(
        "artifacts/runs/legacy/dogfacenet_unified_fresh_v2_protocol_20260831"
    ),
    "shared-fusion-baseline": Path(
        "artifacts/runs/legacy/dogfacenet_shared_v3_protocol_v1"
    ),
    "body-validation": Path(
        "artifacts/runs/legacy/dogfacenet_body_swinv2b_joint100_validation_v1"
    ),
    "geometry-round1": Path("artifacts/runs/unified/v2/geometry_round1"),
    "semantic-teacher-training": Path(
        "artifacts/runs/unified/v2/teacher_training_semantic_v3.npz"
    ),
    "semantic-teacher-development": Path(
        "artifacts/runs/unified/v2/teacher_development_semantic_v3.npz"
    ),
    "external-development-manifest": Path(
        "artifacts/runs/unified/v3/external_bifor_protocol_20260831/"
        "development.manifest.json"
    ),
    "semantic-prototype-validation": Path(
        "artifacts/runs/unified/v1/dev1280_fpn_semantic_prototype_v1/"
        "dev_validation_features_constant_predicted.npz"
    ),
    "semantic-policy-sensitivity-summary": Path(
        "artifacts/runs/unified/v1/dev1280_fpn_semantic_prototype_v1/"
        "confidence_conflict_sensitivity/summary.json"
    ),
    "bifor-fusion-checkpoint": Path(
        "artifacts/runs/bifor/lowrank_joint100_v1/model_final.pth"
    ),
    "joint-validation-manifest": Path(
        "artifacts/runs/legacy/dogfacenet_joint100_protocol_v1/"
        "validation_manifest.json"
    ),
    "bifor-evaluation": Path(
        "artifacts/runs/bifor/lowrank_joint100_v1/evaluation.json"
    ),
    "bifor-runtime-validation": Path(
        "artifacts/runs/bifor/onnx_raw_runtime_validation_v1/evaluation.json"
    ),
    "bifor-onnx-validation": Path(
        "artifacts/runs/bifor/onnx_protocol_validation_v1/evaluation.json"
    ),
    "joint-protocol": Path(
        "artifacts/runs/legacy/dogfacenet_joint100_protocol_v1"
    ),
    "joint-rollback-protocol": Path(
        "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1"
    ),
    "agent-formal-evaluation": Path(
        "artifacts/runs/agent_v1/formal_joint100_20_v1"
    ),
    "full-resolution-standard-protocol": Path(
        "artifacts/protocols/unified_v4_full_standard35"
    ),
    "nose-author-reproduction-checkpoint": Path(
        "artifacts/runs/unified/v4/nose_author_repro_direct/"
        "model_nose_author_repro.pth"
    ),
}
_HISTORICAL_ARTIFACT_PATHS = {
    "joint-rollback-package": Path(
        "models/selected/dogfacenet_joint800_v1"
    ),
    "joint-rollback-gallery": Path(
        "models/selected/local_pet_gallery_joint800_onnx_v1"
    ),
    "legacy-local-gallery-manifest": Path(
        "data/processed/pet-reid-imag/local_pet_gallery_v1/"
        "dataset_manifest.json"
    ),
}
_HISTORICAL_PURPOSES = {
    "semantic-development-selection": "unified_v2_development_only_model_selection",
    "legacy-external-joint-guard": "external_joint_legacy_v2_development_guard",
    "external-joint-development": "unified_v3_external_joint_development",
}


def acceptance_protocol_name(role: str) -> str:
    """Return the frozen protocol identifier for a stable role name."""

    try:
        return _ACCEPTANCE_PROTOCOLS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown acceptance role {role!r}") from exc


def acceptance_path(workspace: str | Path, role: str) -> Path:
    """Resolve a frozen acceptance file without exposing its release label."""

    try:
        relative = _ACCEPTANCE_PATHS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown acceptance role {role!r}") from exc
    return Path(workspace) / relative


def source_weight_lock(
    acceptance: Mapping[str, Any], capability: str
) -> Mapping[str, Any]:
    """Read a source lock through its capability name and legacy aliases."""

    try:
        aliases = _SOURCE_LOCK_ALIASES[capability]
    except KeyError as exc:
        raise ValueError(f"Unknown source-lock capability {capability!r}") from exc
    locks = acceptance.get("source_weight_locks")
    if not isinstance(locks, Mapping):
        raise ValueError("Acceptance record has no source_weight_locks object")
    for key in aliases:
        value = locks.get(key)
        if isinstance(value, Mapping):
            return value
    raise ValueError(
        f"Acceptance record has no source lock for capability {capability!r}"
    )


def historical_run_path(workspace: str | Path, role: str) -> Path:
    """Resolve an immutable historical run directory by stable role."""

    try:
        relative = _HISTORICAL_RUN_PATHS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown historical run role {role!r}") from exc
    return Path(workspace) / relative


def historical_artifact_path(workspace: str | Path, role: str) -> Path:
    """Resolve a frozen package or data artifact by stable role."""

    try:
        relative = _HISTORICAL_ARTIFACT_PATHS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown historical artifact role {role!r}") from exc
    return Path(workspace) / relative


def locked_protocol_paths(
    workspace: str | Path,
    protocol: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    """Resolve a locked protocol from a stable role or an older explicit config."""

    root = Path(workspace)
    role = protocol.get("role")
    if role:
        protocol_root = historical_run_path(root, str(role))
        return (
            protocol_root / "protocol_lock.json",
            protocol_root / "train.manifest.json",
            protocol_root / "validation.manifest.json",
        )

    def resolve(field: str) -> Path:
        value = protocol.get(field)
        if not isinstance(value, (str, Path)):
            raise ValueError(f"Protocol config requires {field!r}")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    return (
        resolve("lock"),
        resolve("train_manifest"),
        resolve("validation_manifest"),
    )


def historical_purpose(role: str) -> str:
    """Return an immutable report purpose through a stable role name."""

    try:
        return _HISTORICAL_PURPOSES[role]
    except KeyError as exc:
        raise ValueError(f"Unknown historical purpose role {role!r}") from exc


def detail_checkpoint_source(sources: Mapping[str, Any]) -> Mapping[str, Any]:
    source = sources.get(_DETAIL_SOURCE_KEY)
    if not isinstance(source, Mapping):
        source = sources.get(_LEGACY_DETAIL_SOURCE_KEY)
    if not isinstance(source, Mapping):
        raise ValueError("Structural checkpoint has no spatial-detail source")
    return source


def parent_checkpoint_source(sources: Mapping[str, Any]) -> Mapping[str, Any]:
    source = sources.get(_PARENT_SOURCE_KEY)
    if not isinstance(source, Mapping):
        source = sources.get(_LEGACY_PARENT_SOURCE_KEY)
    if not isinstance(source, Mapping):
        raise ValueError("Spatial-detail checkpoint has no parent model source")
    return source


def with_parent_checkpoint_source(
    sources: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(sources)
    updated.pop(_LEGACY_PARENT_SOURCE_KEY, None)
    updated[_PARENT_SOURCE_KEY] = dict(source)
    return updated


def migrate_structural_state_dict(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate legacy module prefixes without mutating the loaded payload."""

    migrated: dict[str, Any] = {}
    for name, value in state.items():
        target = (
            _DETAIL_STATE_PREFIX + name[len(_LEGACY_DETAIL_STATE_PREFIX) :]
            if name.startswith(_LEGACY_DETAIL_STATE_PREFIX)
            else name
        )
        migrated[target] = value
    return migrated


def baseline_controller_protocol(workspace: str | Path) -> Path:
    return Path(workspace) / _BASELINE_CONTROLLER_PROTOCOL


def shared_controller_feature_cache(workspace: str | Path) -> Path:
    return Path(workspace) / _SHARED_CONTROLLER_FEATURE_CACHE


def historical_model_training_manifests(
    workspace: str | Path,
) -> tuple[Path, ...]:
    root = Path(workspace)
    return tuple(root / path for path in _HISTORICAL_MODEL_TRAINING_MANIFESTS)


def controller_protocol_search_glob() -> str:
    return _CONTROLLER_PROTOCOL_SEARCH_GLOB


def historical_full_resolution_sources(
    workspace: str | Path,
) -> dict[str, Path]:
    root = Path(workspace)
    return {
        "semantic_base": root / _FULL_RESOLUTION_SEMANTIC_BASE,
        "geometry": root / _FULL_RESOLUTION_GEOMETRY,
    }


def detail_final_endpoint(
    endpoints: Mapping[str, Any],
) -> Mapping[str, Any]:
    endpoint = endpoints.get(_DETAIL_FINAL_ENDPOINT)
    if not isinstance(endpoint, Mapping):
        endpoint = endpoints.get(_LEGACY_DETAIL_FINAL_ENDPOINT)
    if not isinstance(endpoint, Mapping):
        raise ValueError("Interface report has no final spatial-detail endpoint")
    return endpoint


def high_resolution_protocol_name() -> str:
    return _HIGH_RESOLUTION_PROTOCOL_NAME


def is_high_resolution_protocol_name(value: object) -> bool:
    return value in {
        _HIGH_RESOLUTION_PROTOCOL_NAME,
        _LEGACY_HIGH_RESOLUTION_PROTOCOL_NAME,
    }


def normalize_fusion_mode(value: object) -> str:
    key = str(value).casefold()
    try:
        return _FUSION_MODE_ALIASES[key]
    except KeyError as exc:
        choices = sorted(set(_FUSION_MODE_ALIASES.values()))
        raise ValueError(
            f"Unknown fusion capability {value!r}; choose one of {choices}"
        ) from exc


def normalize_fusion_signature(value: object) -> str:
    key = str(value).casefold()
    try:
        return _FUSION_SIGNATURE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown fusion signature {value!r}") from exc


def is_lowrank_body_fusion_architecture(value: object) -> bool:
    return str(value).casefold() in _LOWRANK_BODY_FUSION_ARCHITECTURES


def acceptance_schema_for_protocol(value: object) -> int | None:
    return _ACCEPTANCE_SCHEMA_BY_PROTOCOL.get(str(value))


__all__ = [
    "acceptance_schema_for_protocol",
    "acceptance_path",
    "acceptance_protocol_name",
    "baseline_controller_protocol",
    "controller_protocol_search_glob",
    "detail_final_endpoint",
    "high_resolution_protocol_name",
    "is_high_resolution_protocol_name",
    "is_lowrank_body_fusion_architecture",
    "locked_protocol_paths",
    "detail_checkpoint_source",
    "historical_model_training_manifests",
    "historical_full_resolution_sources",
    "historical_artifact_path",
    "historical_run_path",
    "historical_purpose",
    "migrate_structural_state_dict",
    "normalize_fusion_mode",
    "normalize_fusion_signature",
    "parent_checkpoint_source",
    "shared_controller_feature_cache",
    "source_weight_lock",
    "with_parent_checkpoint_source",
]
