#!/usr/bin/env python3
"""Compare any one-graph unified ONNX against a locked external feature cache."""

from __future__ import annotations

import argparse
import json
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--reference-key", default="embedding")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "model": args.model.expanduser().resolve(),
        "metadata": args.metadata.expanduser().resolve(),
        "manifest": args.manifest.expanduser().resolve(),
        "reference": args.reference_cache.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    pipeline = UnifiedONNXRuntimePipeline(
        paths["model"],
        provider=args.provider,
        metadata_path=paths["metadata"],
        verify_hash=True,
    )
    dataset = UnifiedRawManifestDataset(
        paths["manifest"],
        input_size=pipeline.input_size,
        training=False,
        allow_letterbox_upscale=pipeline.letterbox_allow_upscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    outputs = []
    source_sha256: list[str] = []
    for batch in loader:
        outputs.append(
            pipeline._run(batch["rgb"].numpy().astype(np.float32, copy=False))
        )
        source_sha256.extend(batch["source_sha256"])
    actual = torch.nn.functional.normalize(torch.cat(outputs).float(), dim=1)
    cache = np.load(paths["reference"], allow_pickle=False)
    if args.reference_key not in cache.files:
        raise KeyError(
            f"{args.reference_key!r} is not in cache keys {cache.files}"
        )
    if cache["source_sha256"].astype(str).tolist() != source_sha256:
        raise RuntimeError("Reference cache and manifest order differ")
    expected = torch.nn.functional.normalize(
        torch.from_numpy(cache[args.reference_key]).float(), dim=1
    )
    cosine = (expected * actual).sum(dim=1)
    difference = (expected - actual).abs()
    worst = cosine.argsort()[:64]
    report = {
        "schema_version": 1,
        "model": str(paths["model"]),
        "model_sha256": sha256_file(paths["model"]),
        "metadata": str(paths["metadata"]),
        "metadata_sha256": sha256_file(paths["metadata"]),
        "manifest": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "reference_cache": str(paths["reference"]),
        "reference_cache_sha256": sha256_file(paths["reference"]),
        "reference_key": args.reference_key,
        "provider": pipeline.backend_info(),
        "records": len(dataset),
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "below_0_9999": int((cosine < 0.9999).sum()),
        "below_0_99": int((cosine < 0.99).sum()),
        "below_0_95": int((cosine < 0.95).sum()),
        "max_abs_error": float(difference.max()),
        "worst_indices": [int(index) for index in worst.tolist()],
        "worst_cosines": [float(cosine[index]) for index in worst.tolist()],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
