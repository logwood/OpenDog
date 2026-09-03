"""Resolve deployment roles to immutable artifacts at the release boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .workspace_paths import MODELS_ROOT, WORKSPACE_ROOT


DEPLOYMENT_MANIFEST = MODELS_ROOT / "deployment.json"
REGISTRY_PATH = MODELS_ROOT / "registry.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required model manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model manifest is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Model manifest must contain one JSON object: {path}")
    return value


def _optional_workspace_path(value: object, *, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Runtime profile {field} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Runtime profile {field} must be workspace-relative")
    resolved = (WORKSPACE_ROOT / relative).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Runtime profile {field} escapes the workspace") from exc
    return resolved


def _required_workspace_path(value: object, *, field: str) -> Path:
    resolved = _optional_workspace_path(value, field=field)
    if resolved is None:
        raise RuntimeError(f"Runtime profile {field} is required")
    return resolved


@dataclass(frozen=True)
class ModelProfile:
    """Version-independent runtime selection backed by release metadata."""

    name: str
    deployment_role: str
    release_role: str
    display_name: str
    summary: str
    capability: str
    backend: str
    runtime_backend: str
    model_package: str
    onnx: Path
    persistent_gallery: Path
    model_sha256: str
    single_graph: bool
    warmup_batches: tuple[int, ...]
    runtime_external_models: tuple[str, ...]
    requires_existing_gallery: bool
    config: Path | None = None
    identity_weights: Path | None = None
    seed_gallery: Path | None = None
    body_detector: Path | None = None
    expert_checkpoint: Path | None = None
    fusion_mode: str | None = None
    agent_mode: str | None = None

    @property
    def package_checkpoint_relative(self) -> Path:
        return Path("models/selected") / self.model_package / "model_final.pth"

    @property
    def package_checkpoint(self) -> Path:
        return WORKSPACE_ROOT / self.package_checkpoint_relative

    @classmethod
    def from_record(cls, name: str, record: Mapping[str, Any]) -> "ModelProfile":
        required_strings = (
            "deployment_role",
            "release_role",
            "display_name",
            "summary",
            "capability",
            "backend",
            "runtime_backend",
            "model_package",
            "onnx",
            "persistent_gallery",
            "model_sha256",
        )
        missing = [
            key for key in required_strings if not isinstance(record.get(key), str)
        ]
        if missing:
            raise RuntimeError(
                f"Runtime profile {name!r} is missing string fields: "
                + ", ".join(missing)
            )
        warmup = record.get("warmup_batches")
        if not isinstance(warmup, list) or not warmup or not all(
            isinstance(value, int) and value > 0 for value in warmup
        ):
            raise RuntimeError(f"Runtime profile {name!r} has invalid warmup_batches")
        fingerprint = str(record["model_sha256"]).casefold()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise RuntimeError(f"Runtime profile {name!r} has an invalid model_sha256")
        external = record.get("runtime_external_models", [])
        if not isinstance(external, list) or not all(
            isinstance(value, str) for value in external
        ):
            raise RuntimeError(
                f"Runtime profile {name!r} has invalid runtime_external_models"
            )
        return cls(
            name=name,
            deployment_role=str(record["deployment_role"]),
            release_role=str(record["release_role"]),
            display_name=str(record["display_name"]),
            summary=str(record["summary"]),
            capability=str(record["capability"]),
            backend=str(record["backend"]),
            runtime_backend=str(record["runtime_backend"]),
            model_package=str(record["model_package"]),
            onnx=_required_workspace_path(record["onnx"], field="onnx"),
            persistent_gallery=_required_workspace_path(
                record["persistent_gallery"], field="persistent_gallery"
            ),
            model_sha256=fingerprint,
            single_graph=bool(record.get("single_graph")),
            warmup_batches=tuple(warmup),
            runtime_external_models=tuple(external),
            requires_existing_gallery=bool(record.get("requires_existing_gallery")),
            config=_optional_workspace_path(record.get("config"), field="config"),
            identity_weights=_optional_workspace_path(
                record.get("identity_weights"), field="identity_weights"
            ),
            seed_gallery=_optional_workspace_path(
                record.get("seed_gallery"), field="seed_gallery"
            ),
            body_detector=_optional_workspace_path(
                record.get("body_detector"), field="body_detector"
            ),
            expert_checkpoint=_optional_workspace_path(
                record.get("expert_checkpoint"), field="expert_checkpoint"
            ),
            fusion_mode=(
                str(record["fusion_mode"]) if record.get("fusion_mode") else None
            ),
            agent_mode=(
                str(record["agent_mode"]) if record.get("agent_mode") else None
            ),
        )

    def public_metadata(self) -> dict[str, Any]:
        """Return role/capability labels without release-generation numbering."""

        return {
            "deployment_profile": self.name,
            "deployment_role": self.deployment_role,
            "release_role": self.release_role,
            "display_name": self.display_name,
            "summary": self.summary,
            "capability": self.capability,
            "model_package": self.model_package,
        }


def _profile_records() -> dict[str, Mapping[str, Any]]:
    registry = _load_json_object(REGISTRY_PATH)
    records = registry.get("runtime_profiles")
    if not isinstance(records, dict):
        records = _load_json_object(DEPLOYMENT_MANIFEST).get("runtime_profiles")
    if not isinstance(records, dict) or not records:
        raise RuntimeError("No runtime_profiles were found in release metadata")
    invalid = [
        name for name, record in records.items() if not isinstance(record, dict)
    ]
    if invalid:
        raise RuntimeError(f"Runtime profiles must be objects: {', '.join(invalid)}")
    return records


def runtime_profile_names() -> tuple[str, ...]:
    return tuple(_profile_records())


def get_runtime_profile(name: str) -> ModelProfile:
    records = _profile_records()
    try:
        record = records[name]
    except KeyError as exc:
        choices = ", ".join(records)
        raise ValueError(
            f"Unknown runtime profile {name!r}; choose one of: {choices}"
        ) from exc
    return ModelProfile.from_record(name, record)


def profile_for_backend(backend: str, *, agent: bool = False) -> ModelProfile:
    """Resolve compatibility backend flags to a role-based profile."""

    profiles = [get_runtime_profile(name) for name in runtime_profile_names()]
    if agent:
        profiles = [profile for profile in profiles if profile.agent_mode]
    else:
        profiles = [profile for profile in profiles if not profile.agent_mode]
    matches = [profile for profile in profiles if profile.backend == backend]
    if not matches:
        raise ValueError(f"No runtime profile provides backend {backend!r}")
    preferred = ("production", "candidate", "legacy-semantic", "research-bifor")
    return min(
        matches,
        key=lambda profile: (
            preferred.index(profile.name)
            if profile.name in preferred
            else len(preferred)
        ),
    )


def profile_for_model_path(model_path: str | Path) -> ModelProfile | None:
    resolved = Path(model_path).expanduser().resolve()
    for name in runtime_profile_names():
        profile = get_runtime_profile(name)
        if profile.onnx == resolved:
            return profile
    return None


__all__ = [
    "DEPLOYMENT_MANIFEST",
    "ModelProfile",
    "REGISTRY_PATH",
    "get_runtime_profile",
    "profile_for_backend",
    "profile_for_model_path",
    "runtime_profile_names",
]
