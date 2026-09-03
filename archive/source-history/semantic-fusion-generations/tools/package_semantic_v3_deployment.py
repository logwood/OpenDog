#!/usr/bin/env python3
"""Package the accepted semantic-residual v3 model without touching v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package_deployment(
    *,
    checkpoint: Path,
    config: Path,
    model_lock: Path,
    blind_completion: Path,
    output_dir: Path,
) -> dict:
    sources = {
        "checkpoint": checkpoint.resolve(),
        "config": config.resolve(),
        "model_lock": model_lock.resolve(),
        "blind_completion": blind_completion.resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing deployment inputs: {missing}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model package: {output_dir}")

    lock = _load_json(sources["model_lock"])
    completion = _load_json(sources["blind_completion"])
    checkpoint_hash = sha256_file(sources["checkpoint"])
    config_hash = sha256_file(sources["config"])
    lock_hash = sha256_file(sources["model_lock"])
    completion_hash = sha256_file(sources["blind_completion"])

    if lock.get("status") != "LOCKED_BEFORE_FRESH_BLIND_EVALUATION":
        raise RuntimeError("Model was not locked before fresh-blind evaluation")
    if lock["artifacts"]["model_final.pth"]["sha256"] != checkpoint_hash:
        raise RuntimeError("Checkpoint differs from the pre-blind model lock")
    if lock["artifacts"]["config"]["sha256"] != config_hash:
        raise RuntimeError("Config differs from the pre-blind model lock")
    selected = completion["selected_checkpoint"]
    if selected["sha256_before_and_after_evaluation"] != checkpoint_hash:
        raise RuntimeError("Completion record does not match the checkpoint")
    if not selected.get("unchanged"):
        raise RuntimeError("Checkpoint was not recorded as unchanged")
    if completion["model_lock"]["sha256"] != lock_hash:
        raise RuntimeError("Completion record does not match the model lock")
    acceptance = completion["acceptance"]
    if acceptance.get("selected_for_deployment") != "semantic_residual_v3":
        raise RuntimeError("Fresh-blind completion did not select semantic v3")
    if not completion.get("no_post_blind_training_or_model_selection"):
        raise RuntimeError("Post-blind immutability was not recorded")

    output_dir.mkdir(parents=True)
    destinations = {
        "model_final.pth": output_dir / "model_final.pth",
        "config.yaml": output_dir / "config.yaml",
        "model_lock.json": output_dir / "model_lock.json",
        "blind_completion.json": output_dir / "blind_completion.json",
    }
    for source_name, destination_name in (
        ("checkpoint", "model_final.pth"),
        ("config", "config.yaml"),
        ("model_lock", "model_lock.json"),
        ("blind_completion", "blind_completion.json"),
    ):
        shutil.copy2(sources[source_name], destinations[destination_name])

    packaged_hashes = {name: sha256_file(path) for name, path in destinations.items()}
    expected_hashes = {
        "model_final.pth": checkpoint_hash,
        "config.yaml": config_hash,
        "model_lock.json": lock_hash,
        "blind_completion.json": completion_hash,
    }
    if packaged_hashes != expected_hashes:
        raise RuntimeError("One or more packaged artifacts changed during copy")

    legacy = completion["evaluations"]["legacy_production"]["fused"]
    semantic = completion["evaluations"]["semantic_residual_v3"]["fused"]
    record = {
        "schema_version": 1,
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "name": "dogfacenet_semantic_v3_v1",
        "fusion_mode": "semantic_residual_v3",
        "embedding_dim": 512,
        "rollback_package": "models/selected/dogfacenet_joint800_v1",
        "source_artifacts": {
            name: {"path": str(path), "sha256": expected_hashes[target]}
            for name, path, target in (
                ("checkpoint", sources["checkpoint"], "model_final.pth"),
                ("config", sources["config"], "config.yaml"),
                ("model_lock", sources["model_lock"], "model_lock.json"),
                (
                    "blind_completion",
                    sources["blind_completion"],
                    "blind_completion.json",
                ),
            )
        },
        "packaged_artifacts": {
            name: {"path": str(path), "sha256": packaged_hashes[name]}
            for name, path in destinations.items()
        },
        "fresh_blind": {
            "identities": completion["identities"],
            "queries": completion["queries"],
            "legacy_top1_accuracy": legacy["top1_accuracy"],
            "semantic_v3_top1_accuracy": semantic["top1_accuracy"],
            "legacy_top5_accuracy": legacy["top5_accuracy"],
            "semantic_v3_top5_accuracy": semantic["top5_accuracy"],
        },
        "development_conflict": lock["decision"],
        "onnx": {"status": "pending_export"},
    }
    record_path = output_dir / "deployment_record.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# DogFaceNet semantic fusion v3 v1

This package is the accepted 512-dimensional semantic-residual fusion model.
It is installed beside, and does not overwrite, `dogfacenet_joint800_v1`.

## Locked evidence

- Fresh-blind protocol: {completion["identities"]} unseen identities,
  {completion["queries"]} held-out queries.
- Legacy fused Top-1/Top-5: {legacy["top1_accuracy"]:.4%} /
  {legacy["top5_accuracy"]:.4%}.
- Semantic-v3 fused Top-1/Top-5: {semantic["top1_accuracy"]:.4%} /
  {semantic["top5_accuracy"]:.4%}.
- Development cross-identity nose-conflict Top-1: 91.5% legacy versus
  96.5% semantic v3.
- Checkpoint SHA-256: `{checkpoint_hash}`.

`model_lock.json`, `blind_completion.json`, and `deployment_record.json`
contain the complete hashes and decision record. The `onnx` directory is
created only after export and numerical parity validation succeeds.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--blind-completion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = package_deployment(
        checkpoint=args.checkpoint,
        config=args.config_file,
        model_lock=args.model_lock,
        blind_completion=args.blind_completion,
        output_dir=args.output_dir,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
