#!/usr/bin/env python3
"""Refresh portable model metadata and produce cleanup/audit reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "src" / "Pet-ReID-IMAG"
LEGACY_RUNS_ROOT = WORKSPACE_ROOT / "artifacts" / "runs" / "legacy"
SELECTED_ROOT = WORKSPACE_ROOT / "models" / "selected"
PRETRAINED_ROOT = WORKSPACE_ROOT / "models" / "pretrained"
REPORTS_ROOT = WORKSPACE_ROOT / "artifacts" / "reports"
SNAPSHOT_HASHES = (
    WORKSPACE_ROOT
    / "archive"
    / "git"
    / "2026-08-28"
    / "model-archive-sha256.csv"
)
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".onnx", ".safetensors"}


PURPOSES = {
    "dogfacenet_semantic_v3_v1": (
        "production",
        "Default 800-class Semantic V3 joint embedding used by the API.",
    ),
    "dogfacenet_joint800_v1": (
        "rollback",
        "Locked 800-class joint-fusion rollback and comparison package.",
    ),
    "dogfacenet_shared_v2_fixed5000": (
        "research",
        "Fixed-budget shared-space V2 comparison package.",
    ),
    "dogfacenet_joint100_v1": (
        "research",
        "Locked 100-class joint-fusion protocol package.",
    ),
    "dogfacenet_shared_v2_smoke_10": (
        "smoke",
        "Ten-class ONNX systems-check export; not a production model.",
    ),
    "local_pet_gallery_semantic_v3_onnx_v1": (
        "production-gallery",
        "Seed prototype gallery encoded by the production Semantic V3 ONNX model.",
    ),
    "local_pet_gallery_joint800_onnx_v1": (
        "rollback-gallery",
        "Seed prototype gallery encoded by the Joint800 ONNX rollback model.",
    ),
    "local_pet_gallery_joint100_v1": (
        "research-gallery",
        "Prototype gallery encoded by the Joint100 model.",
    ),
    "local_pet_gallery_v1": (
        "legacy-gallery",
        "Original local prototype-gallery package retained for comparison.",
    ),
}


EXPLICIT_RETENTION = {
    "artifacts/runs/legacy/ablation_mesh_mix_fixed005_s101_224_d192/model_0001.pth",
    "artifacts/runs/legacy/ablation_mesh_mix_fixed005_s101_224_d192/model_0007.pth",
    "artifacts/runs/legacy/ablation_mesh_mix_fixed005_s101_224_d192/model_best.pth",
    "artifacts/runs/legacy/modern_latent_workspace_s101_224_d192/model_0001.pth",
    "artifacts/runs/legacy/modern_latent_workspace_s101_224_d192/model_best.pth",
    "artifacts/runs/legacy/modern_mesh_workspace_s101_224_d192_balanced/model_0001.pth",
    "artifacts/runs/legacy/modern_mesh_workspace_s101_224_d192_balanced/model_0007.pth",
    "artifacts/runs/legacy/modern_mesh_workspace_s101_224_d192_balanced/model_best.pth",
    "artifacts/runs/legacy/retrained_s101_224/model_recent_0.pth",
    "artifacts/runs/legacy/s101_224/model_final.pth",
    "artifacts/runs/legacy/s101_256/model_final.pth",
    "artifacts/runs/legacy/s101_288/model_final.pth",
    "artifacts/runs/legacy/s200_224/model_final.pth",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalize_metadata_path(value: str) -> str:
    slash = value.replace("\\", "/")
    lowered = slash.casefold()
    marker = "/upstream/pet-reid-imag/"
    if marker in lowered:
        slash = slash[lowered.index(marker) + len(marker) :]
        lowered = slash.casefold()
    if lowered.startswith("models/selected/pet_api_gallery_"):
        return "data/gallery_store/" + slash[len("models/selected/") :]
    if lowered.startswith(
        (
            "artifacts/",
            "data/processed/",
            "data/raw/",
            "data/local_gallery/",
            "data/gallery_store/",
            "models/selected/",
            "models/pretrained/",
            "src/pet-reid-imag/",
            "archive/",
        )
    ):
        return slash
    if lowered.startswith("models/pet_api_gallery_"):
        return "data/gallery_store/" + slash[len("models/") :]
    replacements = (
        ("logs/", "artifacts/runs/legacy/"),
        ("models/", "models/selected/"),
        ("configs/", "src/Pet-ReID-IMAG/configs/"),
        ("pretrain/", "models/pretrained/"),
        ("data/local_pet_gallery_v1/", "data/processed/pet-reid-imag/local_pet_gallery_v1/"),
        ("data/", "data/processed/pet-reid-imag/"),
        (
            "third_party/AnyFace/yolov5-face/weights/",
            "models/pretrained/anyface/",
        ),
        ("third_party/sam2/checkpoints/", "models/pretrained/sam2/"),
    )
    for old, new in replacements:
        if lowered.startswith(old.casefold()):
            return new + slash[len(old) :]
    return value


def normalize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_payload(item) for item in value]
    if isinstance(value, str):
        return normalize_metadata_path(value)
    return value


def resolve_metadata_path(value: str, metadata_file: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    workspace_candidate = (WORKSPACE_ROOT / path).resolve()
    if workspace_candidate.exists() or path.parts[:1] in {
        ("artifacts",),
        ("data",),
        ("models",),
        ("src",),
    }:
        return workspace_candidate
    return (metadata_file.parent / path).resolve()


def refresh_hash_pairs(payload: Any, metadata_file: Path) -> None:
    if isinstance(payload, list):
        for item in payload:
            refresh_hash_pairs(item, metadata_file)
        return
    if not isinstance(payload, dict):
        return
    pairs = (
        ("config_file", "config_sha256"),
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("source_checkpoint", "source_checkpoint_sha256"),
        ("dataset_manifest", "dataset_manifest_sha256"),
        ("model", "model_sha256"),
        ("metadata", "metadata_sha256"),
        ("report", "report_sha256"),
    )
    for path_key, hash_key in pairs:
        value = payload.get(path_key)
        if isinstance(value, str):
            candidate = resolve_metadata_path(value, metadata_file)
            if candidate.is_file():
                payload[hash_key] = sha256_file(candidate)
    path_value = payload.get("path")
    if isinstance(path_value, str) and "sha256" in payload:
        candidate = resolve_metadata_path(path_value, metadata_file)
        if candidate.is_file():
            payload["sha256"] = sha256_file(candidate)
    for item in payload.values():
        refresh_hash_pairs(item, metadata_file)


def refresh_selected_metadata() -> dict[str, int]:
    changed = 0
    json_files = sorted(SELECTED_ROOT.rglob("*.json"))
    for path in json_files:
        original = json.loads(path.read_text(encoding="utf-8"))
        payload = normalize_payload(original)
        refresh_hash_pairs(payload, path)
        if path.name == "metadata.json" and path.parent.name == "onnx":
            package = path.parents[1]
            onnx_path = path.parent / "pet_embedding.onnx"
            checkpoint = package / "model_final.pth"
            config = package / "config.yaml"
            if onnx_path.is_file():
                payload["model"] = relative(onnx_path)
                payload["onnx_sha256"] = sha256_file(onnx_path)
                payload["onnx_bytes"] = onnx_path.stat().st_size
            if checkpoint.is_file():
                payload["source_checkpoint"] = relative(checkpoint)
                payload["source_checkpoint_sha256"] = sha256_file(checkpoint)
            if config.is_file():
                payload["config_file"] = relative(config)
                payload["config_sha256"] = sha256_file(config)
        if payload != original:
            write_json(path, payload)
            changed += 1

    deployment = SELECTED_ROOT / "dogfacenet_semantic_v3_v1" / "deployment_record.json"
    if deployment.is_file():
        payload = json.loads(deployment.read_text(encoding="utf-8"))
        package = deployment.parent
        metadata = package / "onnx" / "metadata.json"
        validation = package / "onnx" / "validation.json"
        onnx_path = package / "onnx" / "pet_embedding.onnx"
        payload["rollback_package"] = "models/selected/dogfacenet_joint800_v1"
        payload["onnx"]["model"] = relative(onnx_path)
        payload["onnx"]["sha256"] = sha256_file(onnx_path)
        payload["onnx"]["metadata_sha256"] = sha256_file(metadata)
        payload["onnx"]["validation_sha256"] = sha256_file(validation)
        write_json(deployment, payload)

    # Deployment-record hashes changed after its final consistency refresh.
    return {"json_files": len(json_files), "changed": changed}


def refresh_local_gallery_manifest() -> dict[str, Any]:
    path = (
        WORKSPACE_ROOT
        / "data"
        / "processed"
        / "pet-reid-imag"
        / "local_pet_gallery_v1"
        / "dataset_manifest.json"
    )
    if not path.is_file():
        return {"path": relative(path), "changed": False, "records": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    workspace_marker = "/pet-reid-imag_repro_attempt_2026-08-09/"
    for record in payload.get("records", []):
        source = str(record.get("source_path", "")).replace("\\", "/")
        source_lower = source.casefold()
        if workspace_marker in source_lower:
            source = source[source_lower.index(workspace_marker) + len(workspace_marker) :]
        if source.startswith("1/"):
            updated_source = "data/local_gallery/local-1/" + source[2:]
        elif source.startswith("2/"):
            updated_source = "data/local_gallery/local-2/" + source[2:]
        else:
            updated_source = normalize_metadata_path(source)
        library = normalize_metadata_path(str(record.get("library_path", "")))
        if record.get("source_path") != updated_source:
            record["source_path"] = updated_source
            changed = True
        if record.get("library_path") != library:
            record["library_path"] = library
            changed = True
    if changed:
        write_json(path, payload)
    return {
        "path": relative(path),
        "changed": changed,
        "records": len(payload.get("records", [])),
        "sha256": sha256_file(path),
    }


def legacy_snapshot_mapping(value: str) -> str | None:
    slash = value.replace("\\", "/")
    mappings = (
        ("upstream/Pet-ReID-IMAG/logs/", "artifacts/runs/legacy/"),
        ("upstream/Pet-ReID-IMAG/models/", "models/selected/"),
        ("upstream/Pet-ReID-IMAG/pretrain/", "models/pretrained/"),
        (
            "upstream/Pet-ReID-IMAG/third_party/AnyFace/yolov5-face/weights/",
            "models/pretrained/anyface/",
        ),
        (
            "upstream/Pet-ReID-IMAG/third_party/sam2/checkpoints/",
            "models/pretrained/sam2/",
        ),
    )
    for old, new in mappings:
        if slash.startswith(old):
            return new + slash[len(old) :]
    if slash == "dog.pt":
        return "models/pretrained/dog.pt"
    if slash == "DogFaceNet_alignment.zip":
        return "archive/downloads/DogFaceNet_alignment.zip"
    return None


def load_hash_cache() -> dict[tuple[str, int], str]:
    cache: dict[tuple[str, int], str] = {}
    if not SNAPSHOT_HASHES.is_file():
        return cache
    with SNAPSHOT_HASHES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mapped = legacy_snapshot_mapping(row["relative_path"])
            if mapped:
                cache[(mapped, int(row["size_bytes"]))] = row["sha256"]
    return cache


def read_run_metrics(directory: Path) -> dict[str, float]:
    path = directory / "metrics.json"
    if not path.is_file():
        return {}
    values: dict[str, float] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in row.items():
            lowered = key.casefold()
            if not isinstance(value, (int, float)) or not any(
                token in lowered
                for token in ("roc_auc", "accuracy", "top1", "top5", "metric")
            ):
                continue
            values[key] = max(float(value), values.get(key, float("-inf")))
    return dict(sorted(values.items())[:12])


def classify_role(path: Path) -> tuple[str, int | None, int | None]:
    relative_path = relative(path)
    name = path.name.casefold()
    experiment = relative_path.casefold()
    epoch = None
    step = None
    if relative_path.startswith(("models/selected/", "models/pretrained/")):
        return "release", epoch, step
    if "failed" in experiment:
        return "failed", epoch, step
    if "smoke" in experiment:
        return "smoke", epoch, step
    if name == "model_best.pth":
        return "best", epoch, step
    if name in {"model_final.pth", "pet_embedding.onnx"}:
        return "final", epoch, step
    if "recent" in name:
        return "recent", epoch, step
    match = re.fullmatch(r"model_(\d+)\.pth", name)
    if match:
        epoch = int(match.group(1))
        return "milestone", epoch, step
    match = re.fullmatch(r"checkpoint_(\d+)\.pth", name)
    if match:
        step = int(match.group(1))
        return "milestone", epoch, step
    return "unknown", epoch, step


def inventory_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache = load_hash_cache()
    paths = sorted(
        path
        for root in (LEGACY_RUNS_ROOT, SELECTED_ROOT, PRETRAINED_ROOT)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in CHECKPOINT_SUFFIXES
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        rel = relative(path)
        size = path.stat().st_size
        cached_hash = cache.get((rel, size))
        digest = cached_hash or sha256_file(path)
        role, epoch, step = classify_role(path)
        if rel.startswith("artifacts/runs/legacy/"):
            parts = Path(rel).parts
            experiment = parts[3] if len(parts) > 4 else path.parent.name
            run_dir = LEGACY_RUNS_ROOT / experiment
        elif rel.startswith("models/selected/"):
            experiment = Path(rel).parts[2]
            run_dir = SELECTED_ROOT / experiment
        else:
            experiment = "pretrained"
            run_dir = PRETRAINED_ROOT
        config = run_dir / "config.yaml"
        records.append(
            {
                "path": rel,
                "size_bytes": size,
                "sha256": digest,
                "sha256_source": "snapshot" if cached_hash else "computed",
                "experiment": experiment,
                "config": relative(config) if config.is_file() else None,
                "epoch": epoch,
                "step": step,
                "metrics": read_run_metrics(run_dir),
                "role": role,
            }
        )

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    duplicates = [
        {
            "sha256": digest,
            "bytes_each": group[0]["size_bytes"],
            "paths": [item["path"] for item in group],
        }
        for digest, group in sorted(by_hash.items())
        if len(group) > 1
    ]
    release_hashes = {
        record["sha256"]
        for record in records
        if record["path"].startswith(("models/selected/", "models/pretrained/"))
    }
    for record in records:
        rel = record["path"]
        if rel in EXPLICIT_RETENTION:
            action, reason = "KEEP", "Explicitly retained by CHECKPOINT_RETENTION.md."
        elif rel.startswith(("models/selected/", "models/pretrained/")):
            action, reason = "KEEP", "Selected release or required pretrained model."
        elif record["sha256"] in release_hashes:
            action, reason = (
                "QUARANTINE",
                "Byte-identical selected/pretrained copy exists; preview only.",
            )
        elif record["role"] == "best":
            action, reason = "KEEP", "Best validation checkpoint for a completed run."
        elif record["role"] == "final" and not any(
            token in rel.casefold() for token in ("smoke", "failed")
        ):
            action, reason = "KEEP", "Final checkpoint without a safer replacement rule."
        elif record["role"] == "recent":
            action, reason = "KEEP", "Most recent recovery checkpoint."
        elif record["role"] in {"smoke", "failed"}:
            action, reason = (
                "QUARANTINE",
                "Smoke/failed-run checkpoint; retain logs and metrics instead.",
            )
        elif record["role"] == "milestone":
            action, reason = (
                "REVIEW",
                "Intermediate diagnostic checkpoint requires experiment-owner review.",
            )
        else:
            action, reason = "REVIEW", "Role is not documented strongly enough for automation."
        record["suggested_action"] = action
        record["reason"] = reason
    return records, duplicates


def write_checkpoint_reports(records: list[dict[str, Any]], duplicates: list[dict[str, Any]]) -> None:
    counts = Counter(record["suggested_action"] for record in records)
    bytes_by_action = Counter()
    for record in records:
        bytes_by_action[record["suggested_action"]] += record["size_bytes"]
    payload = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "destructive_actions_performed": False,
        "scope": [
            "artifacts/runs/legacy",
            "models/selected",
            "models/pretrained",
        ],
        "summary": {
            "files": len(records),
            "bytes": sum(record["size_bytes"] for record in records),
            "actions": dict(counts),
            "bytes_by_action": dict(bytes_by_action),
            "duplicate_groups": len(duplicates),
        },
        "records": records,
        "duplicate_groups": duplicates,
    }
    write_json(REPORTS_ROOT / "checkpoint_inventory.json", payload)

    lines = [
        "# Checkpoint cleanup preview",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "This is a preview only. No checkpoint was moved, quarantined, or deleted.",
        "",
        "## Summary",
        "",
        "| Action | Files | Bytes |",
        "|---|---:|---:|",
    ]
    for action in ("KEEP", "REVIEW", "QUARANTINE"):
        lines.append(
            f"| {action} | {counts[action]} | {bytes_by_action[action]:,} |"
        )
    lines.extend(
        [
            "",
            f"Duplicate hash groups: **{len(duplicates)}**.",
            "",
            "## Proposed quarantine candidates",
            "",
            "| Path | Size | Reason |",
            "|---|---:|---|",
        ]
    )
    for record in records:
        if record["suggested_action"] == "QUARANTINE":
            lines.append(
                f"| `{record['path']}` | {record['size_bytes']:,} | {record['reason']} |"
            )
    lines.extend(
        [
            "",
            "## Manual-review candidates",
            "",
            "| Path | Role | Size | Reason |",
            "|---|---|---:|---|",
        ]
    )
    for record in records:
        if record["suggested_action"] == "REVIEW":
            lines.append(
                f"| `{record['path']}` | {record['role']} | {record['size_bytes']:,} | {record['reason']} |"
            )
    (REPORTS_ROOT / "checkpoint_cleanup_preview.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evidence_for(package: Path) -> list[str]:
    names = (
        "deployment_record.json",
        "blind_completion.json",
        "fixed_budget_selection.json",
        "validation_selection.json",
        "lock_record.json",
        "model_lock.json",
    )
    return [relative(package / name) for name in names if (package / name).is_file()]


def metric_summary(package: Path) -> dict[str, Any]:
    deployment = package / "deployment_record.json"
    if deployment.is_file():
        payload = json.loads(deployment.read_text(encoding="utf-8"))
        return {
            "fresh_blind": payload.get("fresh_blind"),
            "development_conflict": payload.get("development_conflict"),
            "onnx_status": payload.get("onnx", {}).get("status"),
        }
    completion = package / "blind_completion.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        return {
            key: payload[key]
            for key in (
                "identities",
                "queries",
                "top1_accuracy",
                "top5_accuracy",
                "blind_top1_accuracy",
            )
            if key in payload
        }
    return {}


def generate_registry(records: list[dict[str, Any]]) -> None:
    checkpoint_hash = {record["path"]: record["sha256"] for record in records}
    packages = []
    for package in sorted(path for path in SELECTED_ROOT.iterdir() if path.is_dir()):
        role, purpose = PURPOSES.get(
            package.name,
            ("selected", "Selected package retained from the validated workspace."),
        )
        artifacts = []
        for path in sorted(package.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in {
                ".pth",
                ".onnx",
                ".npz",
                ".yaml",
                ".json",
            }:
                continue
            rel = relative(path)
            artifacts.append(
                {
                    "path": rel,
                    "bytes": path.stat().st_size,
                    "sha256": checkpoint_hash.get(rel) or sha256_file(path),
                }
            )
        config = package / "config.yaml"
        lock = next(
            (
                path
                for name in ("deployment_record.json", "lock_record.json", "model_lock.json")
                if (path := package / name).is_file()
            ),
            None,
        )
        source = None
        if lock:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            source = payload.get("source_checkpoint")
            if source is None:
                source = payload.get("source_artifacts", {}).get("checkpoint", {}).get("path")
        packages.append(
            {
                "name": package.name,
                "role": role,
                "purpose": purpose,
                "config": (
                    {"path": relative(config), "sha256": sha256_file(config)}
                    if config.is_file()
                    else None
                ),
                "source": source,
                "metrics": metric_summary(package),
                "evidence": evidence_for(package),
                "artifacts": artifacts,
            }
        )
    pretrained = []
    for record in records:
        if record["path"].startswith("models/pretrained/"):
            pretrained.append(
                {
                    "path": record["path"],
                    "bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                    "purpose": "Required pretrained geometry or identity encoder weight.",
                    "source": "SOURCES.md",
                }
            )
    registry_path = WORKSPACE_ROOT / "models" / "registry.json"
    registry = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "default_deployment": {
            "model_package": "dogfacenet_semantic_v3_v1",
            "config": "models/selected/dogfacenet_semantic_v3_v1/config.yaml",
            "checkpoint": "models/selected/dogfacenet_semantic_v3_v1/model_final.pth",
            "onnx": "models/selected/dogfacenet_semantic_v3_v1/onnx/pet_embedding.onnx",
            "seed_gallery": "models/selected/local_pet_gallery_semantic_v3_onnx_v1/gallery_model.json",
            "persistent_gallery": "data/gallery_store/pet_api_gallery_semantic_v3_v1",
        },
        "packages": packages,
        "pretrained": pretrained,
    }
    # Keep the registry stable when no model metadata changed. This file is
    # source-controlled, so replacing its timestamp on every audit would
    # create needless working-tree churn and make an audit non-idempotent.
    if registry_path.is_file():
        try:
            previous = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict):
            previous_body = dict(previous)
            previous_timestamp = previous_body.pop("generated_at_utc", None)
            current_body = dict(registry)
            current_body.pop("generated_at_utc", None)
            if previous_body == current_body and isinstance(previous_timestamp, str):
                registry["generated_at_utc"] = previous_timestamp
    write_json(registry_path, registry)


def directory_stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    if not path.is_dir():
        return 0, 0
    count = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            size += item.stat().st_size
    return count, size


def generate_move_map() -> None:
    destinations = (
        ("upstream/Pet-ReID-IMAG", "src/Pet-ReID-IMAG", "source"),
        ("1", "data/local_gallery/local-1", "local-gallery"),
        ("2", "data/local_gallery/local-2", "local-gallery"),
        ("new-images", "data/queries/inbox", "query-inbox"),
        ("DogFaceNet_alignment", "data/raw/DogFaceNet_alignment", "raw-data"),
        ("upstream/Pet-ReID-IMAG/data", "data/processed/pet-reid-imag", "processed-data"),
        ("upstream/Pet-ReID-IMAG/logs", "artifacts/runs/legacy", "legacy-runs"),
        ("logs", "artifacts/workspace_logs", "workspace-logs"),
        ("results", "artifacts/reports", "reports"),
        ("upstream/Pet-ReID-IMAG/models", "models/selected", "selected-models"),
        ("pretrain + vendor weights", "models/pretrained", "pretrained-models"),
        ("gallery databases", "data/gallery_store", "gallery-store"),
        ("DogFaceNet_alignment.zip", "archive/downloads/DogFaceNet_alignment.zip", "download-archive"),
        ("root design/repro documents", "docs", "documentation"),
    )
    output = WORKSPACE_ROOT / "archive" / "git" / "2026-08-28" / "data-moves.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source",
                "destination",
                "category",
                "destination_files",
                "destination_bytes",
                "source_absent",
                "verification",
            ),
        )
        writer.writeheader()
        for source, destination, category in destinations:
            target = WORKSPACE_ROOT / destination
            count, size = directory_stats(target)
            source_path = WORKSPACE_ROOT / source if "+" not in source else None
            writer.writerow(
                {
                    "source": source,
                    "destination": destination,
                    "category": category,
                    "destination_files": count,
                    "destination_bytes": size,
                    "source_absent": (
                        source_path is None or not source_path.exists()
                    ),
                    "verification": "destination-present" if target.exists() else "missing",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-metadata-refresh",
        action="store_true",
        help="do not normalize selected-model JSON paths and dependent hashes",
    )
    args = parser.parse_args()
    gallery_manifest = refresh_local_gallery_manifest()
    refresh = (
        {"json_files": 0, "changed": 0}
        if args.skip_metadata_refresh
        else refresh_selected_metadata()
    )
    records, duplicates = inventory_records()
    write_checkpoint_reports(records, duplicates)
    generate_registry(records)
    generate_move_map()
    print(
        json.dumps(
            {
                "metadata": refresh,
                "gallery_manifest": gallery_manifest,
                "checkpoint_files": len(records),
                "duplicate_groups": len(duplicates),
                "registry": relative(WORKSPACE_ROOT / "models" / "registry.json"),
                "inventory": relative(REPORTS_ROOT / "checkpoint_inventory.json"),
                "preview": relative(REPORTS_ROOT / "checkpoint_cleanup_preview.md"),
                "move_map": "archive/git/2026-08-28/data-moves.csv",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
