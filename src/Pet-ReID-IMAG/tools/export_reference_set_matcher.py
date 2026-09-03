#!/usr/bin/env python3
"""Export and verify a trained reference-set matcher as ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.reference_set_model import (  # noqa: E402
    ReferenceSetMatcherExport,
    build_reference_set_matcher_from_checkpoint,
)
from pet_id.reference_set_training import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "artifacts/runs/reference_set_matcher/onnx",
    )
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--validation-batch-size", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.opset < 17:
        raise ValueError("--opset must be at least 17")
    if args.validation_batch_size < 1:
        raise ValueError("--validation-batch-size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "reference_set_matcher.onnx"
    metadata_path = output_dir / "metadata.json"
    validation_path = output_dir / "validation.json"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(model_path)

    model, payload = build_reference_set_matcher_from_checkpoint(checkpoint)
    model.eval()
    wrapper = ReferenceSetMatcherExport(model).cpu().eval()
    batch = int(args.validation_batch_size)
    width = int(model.max_references)
    dimension = int(model.descriptor_dim)
    generator = torch.Generator().manual_seed(20260902)
    query = torch.randn(batch, dimension, generator=generator)
    references = torch.randn(batch, width, dimension, generator=generator)
    references = torch.nn.functional.normalize(references, dim=-1)
    mask = torch.ones(batch, width, dtype=torch.bool)
    mask[:, -1] = False
    with torch.inference_mode():
        expected = wrapper(query, references, mask).numpy()
    temporary = output_dir / "reference_set_matcher.exporting.onnx"
    if temporary.exists():
        temporary.unlink()
    torch.onnx.export(
        wrapper,
        (query, references, mask),
        temporary,
        input_names=["query", "references", "reference_mask"],
        output_names=["score"],
        opset_version=args.opset,
        dynamic_axes={
            "query": {0: "batch"},
            "references": {0: "batch"},
            "reference_mask": {0: "batch"},
            "score": {0: "batch"},
        },
        dynamo=False,
    )
    onnx.checker.check_model(str(temporary))
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["score"],
        {
            "query": query.numpy(),
            "references": references.numpy(),
            "reference_mask": mask.numpy(),
        },
    )[0]
    difference = np.abs(expected - actual)
    cosine = (expected * actual).sum() / max(
        float(np.linalg.norm(expected) * np.linalg.norm(actual)), 1e-12
    )
    validation = {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "cosine": float(cosine),
        "provider": "CPUExecutionProvider",
    }
    if validation["max_abs_error"] > 2e-4 or validation["cosine"] < 0.99999:
        raise RuntimeError(f"ONNX parity check failed: {validation}")
    temporary.replace(model_path)
    metadata = {
        "format": "reference-set-matcher-onnx",
        "onnx_sha256": sha256_file(model_path),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint": str(checkpoint),
        "encoder_fingerprint": payload.get("encoder_fingerprint"),
        "model_config": model.configuration(),
        "inputs": {
            "query": {"dtype": "float32", "shape": ["N", dimension]},
            "references": {
                "dtype": "float32",
                "shape": ["N", width, dimension],
            },
            "reference_mask": {"dtype": "bool", "shape": ["N", width]},
        },
        "outputs": {"score": {"dtype": "float32", "shape": ["N"]}},
        "validation": validation,
        "checkpoint_training": payload.get("training", {}),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
