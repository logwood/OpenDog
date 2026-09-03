"""Canonical workspace paths and compatibility mapping for legacy configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]

# The vendored SAM 2 package keeps Hydra configuration files below the
# package's ``sam2/configs`` directory. Keep this root explicit so callers
# can safely turn a cleaned-workspace absolute path back into the config name
# expected by ``hydra.compose``.
SAM2_SOURCE_ROOT = SOURCE_ROOT / "third_party" / "sam2"
SAM2_CONFIG_ROOT = SAM2_SOURCE_ROOT / "sam2" / "configs"


def _discover_workspace_root() -> Path:
    override = os.environ.get("PET_REID_WORKSPACE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (SOURCE_ROOT, *SOURCE_ROOT.parents):
        source = candidate / "src" / "Pet-ReID-IMAG"
        if source.is_dir() and source.resolve() == SOURCE_ROOT:
            return candidate.resolve()
    # Preserve standalone-source compatibility for upstream development.
    return SOURCE_ROOT


WORKSPACE_ROOT = _discover_workspace_root()
DATA_ROOT = WORKSPACE_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed" / "pet-reid-imag"
LOCAL_GALLERY_ROOT = DATA_ROOT / "local_gallery"
QUERY_INBOX_ROOT = DATA_ROOT / "queries" / "inbox"
GALLERY_STORE_ROOT = DATA_ROOT / "gallery_store"
MODELS_ROOT = WORKSPACE_ROOT / "models"
PRETRAINED_MODELS_ROOT = MODELS_ROOT / "pretrained"
SELECTED_MODELS_ROOT = MODELS_ROOT / "selected"
ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
EVALUATIONS_ROOT = ARTIFACTS_ROOT / "evaluations"
REPORTS_ROOT = ARTIFACTS_ROOT / "reports"
WORKSPACE_LOGS_ROOT = ARTIFACTS_ROOT / "workspace_logs"
LEGACY_RUNS_ROOT = ARTIFACTS_ROOT / "runs" / "legacy"


def _slash(value: str | os.PathLike[str]) -> str:
    return os.fspath(value).replace("\\", "/")


def _strip_relative_prefix(value: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    return value


def _is_windows_absolute(normalized: str) -> bool:
    """Recognize a drive-qualified path even when running on POSIX."""

    return (
        len(normalized) >= 3
        and normalized[0].isalpha()
        and normalized[1:3] == ":/"
    )


def _relocated_windows_workspace_path(normalized: str) -> Path | None:
    """Map canonical paths embedded on Windows into this workspace.

    ``pathlib.Path`` on POSIX treats ``D:/...`` as a relative path. Released
    checkpoints may legitimately contain such a path in their audited source
    chain, so recognize stable workspace anchors instead of depending on the
    original checkout directory name.
    """

    if not _is_windows_absolute(normalized):
        return None
    lowered = normalized.casefold()
    for marker in ("/upstream/pet-reid-imag/", "/src/pet-reid-imag/"):
        if marker in lowered:
            index = lowered.index(marker) + len(marker)
            return _from_source_relative(normalized[index:])
    for marker in (
        "/artifacts/",
        "/models/selected/",
        "/models/pretrained/",
        "/data/raw/",
        "/data/processed/",
        "/data/local_gallery/",
        "/data/gallery_store/",
        "/data/queries/",
    ):
        if marker in lowered:
            index = lowered.index(marker) + 1
            return _from_workspace_relative(normalized[index:])
    return None


def _from_source_relative(normalized: str) -> Path:
    value = _strip_relative_prefix(normalized)
    stripped = value
    while stripped.startswith("../"):
        stripped = stripped[3:]

    lowered = stripped.casefold()
    source_prefix = "src/pet-reid-imag/"
    if lowered.startswith(source_prefix):
        return _from_source_relative(stripped[len(source_prefix) :])
    if lowered == "src/pet-reid-imag":
        return SOURCE_ROOT
    if lowered == "dog.pt":
        return PRETRAINED_MODELS_ROOT / "dog.pt"
    if lowered.startswith("dogfacenet_alignment/"):
        return DATA_ROOT / "raw" / "DogFaceNet_alignment" / stripped.split("/", 1)[1]
    if lowered == "dogfacenet_alignment":
        return DATA_ROOT / "raw" / "DogFaceNet_alignment"
    if lowered.startswith("new-images/") or lowered == "new-images":
        suffix = stripped[len("new-images") :].lstrip("/")
        return QUERY_INBOX_ROOT / suffix

    if lowered.startswith("data/processed/") or lowered.startswith("data/raw/"):
        return WORKSPACE_ROOT / stripped
    if lowered.startswith("data/local_gallery/") or lowered.startswith(
        "data/gallery_store/"
    ):
        return WORKSPACE_ROOT / stripped
    if lowered.startswith("data/queries/"):
        return WORKSPACE_ROOT / stripped
    # Before the cleanup, every source-local data/* path belonged to the
    # competition dataset tree. Canonical workspace data roots above are
    # intentionally handled first.
    if lowered.startswith("data/"):
        return PROCESSED_DATA_ROOT / stripped[len("data/") :]
    if lowered == "data":
        return PROCESSED_DATA_ROOT

    if lowered.startswith("artifacts/"):
        return WORKSPACE_ROOT / stripped
    if lowered.startswith("models/selected/") or lowered.startswith(
        "models/pretrained/"
    ):
        return WORKSPACE_ROOT / stripped
    if lowered.startswith("logs/"):
        return LEGACY_RUNS_ROOT / stripped[len("logs/") :]
    if lowered == "logs":
        return LEGACY_RUNS_ROOT
    if lowered.startswith("models/"):
        return SELECTED_MODELS_ROOT / stripped[len("models/") :]
    if lowered == "models":
        return SELECTED_MODELS_ROOT
    if lowered.startswith("pretrain/"):
        return PRETRAINED_MODELS_ROOT / stripped[len("pretrain/") :]
    if lowered.startswith("third_party/anyface/yolov5-face/weights/"):
        return PRETRAINED_MODELS_ROOT / "anyface" / Path(stripped).name
    if lowered.startswith("third_party/sam2/checkpoints/"):
        return PRETRAINED_MODELS_ROOT / "sam2" / Path(stripped).name
    if lowered.startswith("third_party/") or lowered.startswith("configs/"):
        return SOURCE_ROOT / stripped
    if lowered.startswith("tools/") or lowered.startswith("pet_id/"):
        return SOURCE_ROOT / stripped
    return (SOURCE_ROOT / value).resolve()


def _from_workspace_relative(normalized: str) -> Path:
    """Resolve a path that was rooted at either workspace layout."""

    stripped = _strip_relative_prefix(normalized).lstrip("/")
    lowered = stripped.casefold()
    legacy_source_prefix = "upstream/pet-reid-imag/"
    canonical_source_prefix = "src/pet-reid-imag/"
    if lowered.startswith(legacy_source_prefix):
        return _from_source_relative(stripped[len(legacy_source_prefix) :])
    if lowered.startswith(canonical_source_prefix):
        return _from_source_relative(stripped[len(canonical_source_prefix) :])
    if lowered in {"upstream/pet-reid-imag", "src/pet-reid-imag"}:
        return SOURCE_ROOT

    if lowered == "1" or lowered.startswith("1/"):
        return LOCAL_GALLERY_ROOT / "local-1" / stripped[1:].lstrip("/")
    if lowered == "2" or lowered.startswith("2/"):
        return LOCAL_GALLERY_ROOT / "local-2" / stripped[1:].lstrip("/")
    if lowered == "results" or lowered.startswith("results/"):
        return REPORTS_ROOT / stripped[len("results") :].lstrip("/")

    legacy_top_level = (
        "dogfacenet_alignment",
        "new-images",
        "logs",
        "pretrain",
    )
    if lowered == "dog.pt" or any(
        lowered == prefix or lowered.startswith(prefix + "/")
        for prefix in legacy_top_level
    ):
        return _from_source_relative(stripped)

    # Canonical workspace paths, including docs/, scripts/, models/ and data/,
    # must remain stable when an already-absolute path is normalized again.
    return WORKSPACE_ROOT / stripped


def resolve_legacy_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a config path while translating the pre-cleanup layout."""

    raw = _slash(value)
    path = Path(raw).expanduser()
    relocated = _relocated_windows_workspace_path(raw)
    if relocated is not None:
        return relocated.resolve()
    if not path.is_absolute():
        return _from_source_relative(raw).resolve()

    normalized = raw
    lowered = normalized.casefold()
    try:
        workspace_relative = path.resolve().relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        workspace_relative = None
    if workspace_relative is not None:
        return _from_workspace_relative(_slash(workspace_relative)).resolve()

    source_marker = "/upstream/pet-reid-imag/"
    if source_marker in lowered:
        index = lowered.index(source_marker) + len(source_marker)
        return _from_source_relative(normalized[index:]).resolve()

    return path.resolve()


def resolve_sam2_config_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a SAM 2 Hydra config to the vendored canonical path.

    SAM 2 accepts a *config name* (for example
    ``configs/sam2.1/sam2.1_hiera_t.yaml``), not an arbitrary filesystem path.
    Workspace cleanup nevertheless needs to normalize old absolute and
    relative references just like every other runtime asset. This helper
    performs that normalization while retaining a path that
    :class:`SAM2NoseSegmenter` can convert back to a Hydra config name.
    """

    raw = _slash(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        resolved = resolve_legacy_path(path)
        # A path already under the vendored config root is canonical. For
        # unrelated absolute paths, preserve the normal resolver's result so
        # the eventual constructor can report a useful missing-file error.
        return resolved

    normalized = _strip_relative_prefix(raw).lstrip("/")
    lowered = normalized.casefold()
    prefixes = (
        "third_party/sam2/sam2/configs/",
        "third_party/sam2/configs/",
        "sam2/configs/",
        "configs/",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            suffix = normalized[len(prefix) :]
            return (SAM2_CONFIG_ROOT / suffix).resolve()

    # A compact Hydra name such as ``sam2.1/foo.yaml`` is also common in
    # application configuration. Treat it as relative to ``configs``.
    if lowered.startswith(("sam2/", "sam2.1/")):
        return (SAM2_CONFIG_ROOT / normalized).resolve()

    # Last chance: preserve legacy source/workspace mappings. This keeps the
    # helper forward-compatible with a caller that supplies an already
    # canonical path in a different spelling.
    return resolve_legacy_path(raw)


def normalize_runtime_config(cfg: Any) -> Any:
    """Rewrite mutable FastReID path fields to canonical absolute paths."""

    was_frozen = bool(getattr(cfg, "is_frozen", lambda: False)())
    if was_frozen:
        cfg.defrost()
    try:
        if getattr(cfg, "OUTPUT_DIR", ""):
            cfg.OUTPUT_DIR = str(resolve_legacy_path(cfg.OUTPUT_DIR))
        model = getattr(cfg, "MODEL", None)
        if model is not None:
            if getattr(model, "WEIGHTS", ""):
                model.WEIGHTS = str(resolve_legacy_path(model.WEIGHTS))
            backbone = getattr(model, "BACKBONE", None)
            if backbone is not None and getattr(backbone, "PRETRAIN_PATH", ""):
                backbone.PRETRAIN_PATH = str(
                    resolve_legacy_path(backbone.PRETRAIN_PATH)
                )

        options = getattr(cfg, "MULTIMODAL", None)
        if options is not None:
            for field in (
                "NOSE_CONFIG",
                "NOSE_WEIGHTS",
                "IDENTITY_WEIGHTS",
                "ARCFACE_WEIGHTS",
                "ANYFACE_ROOT",
                "ANYFACE_WEIGHTS",
                "SAM2_CHECKPOINT",
                "SAM2_CONFIG",
                "CACHE_DIR",
            ):
                current = getattr(options, field, "")
                if current:
                    resolver = (
                        resolve_sam2_config_path
                        if field == "SAM2_CONFIG"
                        else resolve_legacy_path
                    )
                    setattr(options, field, str(resolver(current)))
    finally:
        if was_frozen:
            cfg.freeze()
    return cfg


__all__ = [
    "ARTIFACTS_ROOT",
    "DATA_ROOT",
    "EVALUATIONS_ROOT",
    "GALLERY_STORE_ROOT",
    "LEGACY_RUNS_ROOT",
    "LOCAL_GALLERY_ROOT",
    "MODELS_ROOT",
    "PRETRAINED_MODELS_ROOT",
    "PROCESSED_DATA_ROOT",
    "QUERY_INBOX_ROOT",
    "RAW_DATA_ROOT",
    "REPORTS_ROOT",
    "RUNS_ROOT",
    "SELECTED_MODELS_ROOT",
    "SOURCE_ROOT",
    "SAM2_CONFIG_ROOT",
    "SAM2_SOURCE_ROOT",
    "WORKSPACE_LOGS_ROOT",
    "WORKSPACE_ROOT",
    "normalize_runtime_config",
    "resolve_legacy_path",
    "resolve_sam2_config_path",
]
