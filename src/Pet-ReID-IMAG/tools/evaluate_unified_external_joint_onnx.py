#!/usr/bin/env python3
"""Evaluate external-joint ONNX retrieval and full-development PyTorch parity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_data import UnifiedRawManifestDataset  # noqa: E402
from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_runtime import UnifiedONNXRuntimePipeline  # noqa: E402
from pet_id.unified_training import retrieval_metrics  # noqa: E402
from pet_id.release_compatibility import acceptance_protocol_name  # noqa: E402


ACCEPTANCE_PROTOCOL = acceptance_protocol_name("external-development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    paths = {
        "model": args.model.expanduser().resolve(),
        "metadata": args.metadata.expanduser().resolve(),
        "acceptance": args.acceptance.expanduser().resolve(),
        "development": args.development_report.expanduser().resolve(),
        "reference": args.reference_cache.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    acceptance = load_json(paths["acceptance"])
    if (
        acceptance.get("schema_version") != 3
        or acceptance.get("protocol_name") != ACCEPTANCE_PROTOCOL
    ):
        raise RuntimeError("Unexpected external acceptance")
    manifest_path = Path(acceptance["development"]["path"]).expanduser().resolve()
    if sha256_file(manifest_path) != acceptance["development"]["sha256"]:
        raise RuntimeError("Development manifest differs from acceptance")
    metadata = load_json(paths["metadata"])
    if metadata.get("model_type") != "unified_external_joint_pet_reid":
        raise RuntimeError("Unexpected ONNX metadata model type")
    if metadata.get("onnx_sha256") != sha256_file(paths["model"]):
        raise RuntimeError("Metadata/ONNX hash mismatch")
    if metadata.get("external_models") != []:
        raise RuntimeError("Unified ONNX declares external runtime models")
    development = load_json(paths["development"])
    if development.get("passed") is not True:
        raise RuntimeError("PyTorch development report did not pass")
    if development.get("blind_data_used") is not False:
        raise RuntimeError("Development evidence must exclude blind data")
    if development.get("manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise RuntimeError("Development report/manifest hash mismatch")
    if metadata.get("source_checkpoint_sha256") != development.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Metadata/development checkpoint mismatch")
    cache_record = development.get("feature_cache") or {}
    if cache_record.get("sha256") != sha256_file(paths["reference"]):
        raise RuntimeError("Development report/reference cache hash mismatch")
    metadata_report = metadata.get("development_reports", {}).get("external", {})
    if metadata_report.get("sha256") != sha256_file(paths["development"]):
        raise RuntimeError("Metadata/development evidence hash mismatch")

    pipeline = UnifiedONNXRuntimePipeline(
        paths["model"],
        provider=args.provider,
        metadata_path=paths["metadata"],
        verify_hash=True,
    )
    dataset = UnifiedRawManifestDataset(
        manifest_path,
        input_size=pipeline.input_size,
        training=False,
        allow_letterbox_upscale=pipeline.letterbox_allow_upscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    outputs = []
    identities: list[str] = []
    source_paths: list[str] = []
    source_sha256: list[str] = []
    processed = 0
    for batch in loader:
        rgb = batch["rgb"].numpy().astype(np.float32, copy=False)
        outputs.append(pipeline._run(rgb))
        identities.extend(batch["identity"])
        source_paths.extend(batch["source_path"])
        source_sha256.extend(batch["source_sha256"])
        processed += int(rgb.shape[0])
        if processed == args.batch_size or processed % 64 == 0:
            print(
                f"external joint ONNX {args.provider}: {processed}/{len(dataset)}",
                flush=True,
            )
    embedding = torch.cat(outputs)
    metrics = retrieval_metrics(embedding, identities, source_paths)
    reference = np.load(paths["reference"], allow_pickle=False)
    if reference["source_sha256"].astype(str).tolist() != source_sha256:
        raise RuntimeError("Reference cache and manifest order differ")
    expected = torch.nn.functional.normalize(
        torch.from_numpy(reference["embedding"]).float(), dim=1
    )
    actual = torch.nn.functional.normalize(embedding.float(), dim=1)
    difference = (expected - actual).abs()
    cosine = (expected * actual).sum(dim=1)
    worst_indices = cosine.argsort()[:20].tolist()
    parity = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "below_minimum_cosine": int((cosine < args.minimum_cosine).sum()),
        "worst_samples": [
            {
                "index": int(index),
                "identity": identities[index],
                "source_path": source_paths[index],
                "cosine": float(cosine[index]),
                "max_abs_error": float(difference[index].max()),
            }
            for index in worst_indices
        ],
    }
    pytorch_metrics = development["candidate"]
    checks = {
        "top1": int(metrics["top1_correct"])
        >= int(pytorch_metrics["top1_correct"]),
        "top5": int(metrics["top5_correct"])
        >= int(pytorch_metrics["top5_correct"]),
        "parity": parity["minimum_cosine"] >= args.minimum_cosine
        and parity["below_minimum_cosine"] == 0,
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "external_joint_onnx_development",
        "model_type": "unified_external_joint_pet_reid_onnx",
        "model": str(paths["model"]),
        "model_sha256": sha256_file(paths["model"]),
        "metadata": str(paths["metadata"]),
        "metadata_sha256": sha256_file(paths["metadata"]),
        "acceptance": {
            "path": str(paths["acceptance"]),
            "sha256": sha256_file(paths["acceptance"]),
        },
        "development_report": {
            "path": str(paths["development"]),
            "sha256": sha256_file(paths["development"]),
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reference_cache": str(paths["reference"]),
        "reference_cache_sha256": sha256_file(paths["reference"]),
        "records": len(dataset),
        "provider": pipeline.backend_info(),
        "retrieval": metrics,
        "pytorch_retrieval": {
            "top1_correct": int(pytorch_metrics["top1_correct"]),
            "top5_correct": int(pytorch_metrics["top5_correct"]),
        },
        "parity_with_pytorch": parity,
        "thresholds": {"minimum_cosine": args.minimum_cosine},
        "checks": checks,
        "passed": all(checks.values()),
        "blind_data_used": False,
        "default_backend_changed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
