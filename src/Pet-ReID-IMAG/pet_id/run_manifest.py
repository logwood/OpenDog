"""Reproducible run-directory and lifecycle metadata helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workspace_paths import (
    PROCESSED_DATA_ROOT,
    RUNS_ROOT,
    WORKSPACE_ROOT,
    resolve_legacy_path,
)


SCHEMA_VERSION = 1
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
_LAUNCHER_BOOTSTRAP_FILES = {"stdout.log"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str, *, field: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.strip()).strip("-.")
    if not slug or slug in {".", ".."}:
        raise ValueError(f"{field} must contain a usable name")
    return slug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(WORKSPACE_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def configure_standard_run(cfg: Any, args: Any) -> bool:
    """Optionally replace OUTPUT_DIR with a standard workstream/run-id path."""

    workstream_value = getattr(args, "run_workstream", "")
    if not workstream_value:
        return False
    workstream = safe_slug(workstream_value, field="--run-workstream")
    purpose = safe_slug(
        getattr(args, "run_purpose", "")
        or ("evaluation" if getattr(args, "eval_only", False) else "train"),
        field="--run-purpose",
    )
    model = safe_slug(
        str(getattr(cfg.MODEL, "META_ARCHITECTURE", "model")).casefold(),
        field="model name",
    )
    seed_value = int(getattr(cfg, "SEED", -1))
    seed = str(seed_value) if seed_value >= 0 else "random"
    run_id_value = getattr(args, "run_id", "")
    run_id = (
        safe_slug(run_id_value, field="--run-id")
        if run_id_value
        else datetime.now().strftime(f"%Y%m%d-%H%M_{model}_{purpose}_{seed}")
    )
    output_dir = (RUNS_ROOT / workstream / run_id).resolve()
    output_dir.relative_to(RUNS_ROOT.resolve())
    if output_dir.exists() and not getattr(args, "resume", False):
        # A PowerShell launcher may create stdout.log before Python parses the
        # config. No other pre-existing content is safe for a fresh run.
        existing = []
        for path in output_dir.iterdir():
            if path.name in _LAUNCHER_BOOTSTRAP_FILES:
                continue
            if path.name == "reports" and path.is_dir():
                report_files = {item.name for item in path.iterdir()}
                if report_files <= {"preflight.json"}:
                    continue
            existing.append(path)
        if existing:
            raise FileExistsError(
                f"standard run directory already contains files: {output_dir}; "
                "use --resume or choose another --run-id"
            )
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.COMPUTED_STANDARD_RUN = True
    return True


def initialize_run_manifest(cfg: Any, args: Any) -> Path:
    output_dir = Path(cfg.OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_run = bool(getattr(cfg, "COMPUTED_STANDARD_RUN", False))
    checkpoint_dir = output_dir / "checkpoints" if standard_run else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    if standard_run:
        (output_dir / "tensorboard").mkdir(exist_ok=True)

    resolved_config = output_dir / "resolved_config.yaml"
    resolved_config.write_text(cfg.dump(), encoding="utf-8")
    source_config = resolve_legacy_path(args.config_file)
    split_manifest = PROCESSED_DATA_ROOT / "splits" / "split_manifest.json"
    manifest_path = output_dir / "run_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    attempt = {
        "started_at_utc": utc_now(),
        "resume": bool(getattr(args, "resume", False)),
        "command": [sys.executable, *sys.argv],
    }
    attempts = list(previous.get("attempts", []))
    attempts.append(attempt)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": output_dir.name,
        "workstream": output_dir.parent.name,
        "purpose": (
            getattr(args, "run_purpose", "")
            or ("evaluation" if getattr(args, "eval_only", False) else "train")
        ),
        "standard_layout": standard_run,
        "status": "evaluating" if getattr(args, "eval_only", False) else "running",
        "started_at_utc": previous.get("started_at_utc", attempt["started_at_utc"]),
        "ended_at_utc": None,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
        },
        "command": attempt["command"],
        "attempts": attempts,
        "source_config": {
            "path": str(source_config),
            "sha256": sha256_file(source_config) if source_config.is_file() else None,
        },
        "resolved_config": str(resolved_config),
        "seed": int(getattr(cfg, "SEED", -1)),
        "datasets": {
            "train": list(getattr(cfg.DATASETS, "NAMES", ())),
            "test": list(getattr(cfg.DATASETS, "TESTS", ())),
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": (
                sha256_file(split_manifest) if split_manifest.is_file() else None
            ),
        },
        "paths": {
            "run": str(output_dir),
            "checkpoints": str(checkpoint_dir),
            "metrics": str(output_dir / "metrics.json"),
            "stdout": str(output_dir / "stdout.log"),
            "tensorboard": str(output_dir / "tensorboard"),
            "reports": str(output_dir / "reports"),
        },
        "checkpoint_policy": {
            "allow_intermediate_cleanup": bool(
                getattr(args, "allow_checkpoint_cleanup", False)
            ),
            "selected_checkpoint": None,
        },
        "result": None,
        "error": None,
    }
    _atomic_json(manifest_path, payload)
    return manifest_path


def _checkpoint_summary(output_dir: Path) -> dict[str, Any]:
    checkpoints = sorted(
        path
        for directory in (output_dir, output_dir / "checkpoints")
        if directory.is_dir()
        for path in directory.glob("*.pth")
    )
    selected = next(
        (
            path
            for name in ("model_best.pth", "model_final.pth")
            for path in checkpoints
            if path.name == name
        ),
        None,
    )
    return {
        "files": [str(path) for path in checkpoints],
        "selected_checkpoint": str(selected) if selected else None,
    }


def finalize_run_manifest(
    cfg: Any,
    *,
    status: str,
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    output_dir = Path(cfg.OUTPUT_DIR).resolve()
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoints = _checkpoint_summary(output_dir)
    payload["status"] = status
    payload["ended_at_utc"] = utc_now()
    payload["result"] = _json_safe(result)
    payload["error"] = (
        {"type": type(error).__name__, "message": str(error)} if error else None
    )
    payload["checkpoint_policy"]["selected_checkpoint"] = checkpoints[
        "selected_checkpoint"
    ]
    payload["checkpoints"] = checkpoints["files"]
    _atomic_json(manifest_path, payload)


__all__ = [
    "configure_standard_run",
    "finalize_run_manifest",
    "initialize_run_manifest",
    "safe_slug",
]
