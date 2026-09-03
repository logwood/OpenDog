#!/usr/bin/env python3
"""Export a fixed-width query/reference image-set ONNX graph.

This export is an experimental scoring graph.  The established single-image
RGB-to-descriptor graph is not modified; the new graph adds query RGB,
reference RGB, and a reference mask as inputs and emits one score per batch
row.
"""

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

from pet_id.reference_aware_model import (  # noqa: E402
    ReferenceAwarePetReID,
    ReferenceAwarePetReIDExport,
    build_reference_aware_encoder_from_checkpoint,
)
from pet_id.reference_set_model import (  # noqa: E402
    build_reference_set_matcher_from_checkpoint,
)
from pet_id.unified_training import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
        help=(
            "legacy ArcFace source for un-packaged base checkpoints; packaged "
            "checkpoints restore their verified source chain automatically"
        ),
    )
    parser.add_argument("--matcher-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "artifacts/runs/reference_aware_model/export",
    )
    parser.add_argument("--output-name", default="reference_aware_pet_reid.onnx")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.opset < 17:
        raise ValueError("--opset must be at least 17")
    device = torch.device(args.device)
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    arcface_checkpoint = args.arcface_checkpoint.expanduser().resolve()
    matcher_checkpoint = args.matcher_checkpoint.expanduser().resolve()
    for path in (base_checkpoint, matcher_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint is missing: {path}")
    encoder, base_payload = build_reference_aware_encoder_from_checkpoint(
        base_checkpoint,
        arcface_checkpoint,
        device=device,
    )
    if base_payload.get("model_type") == "unified_high_resolution_pet_reid":
        raise ValueError(
            "the fixed-width reference-aware export cannot consume a "
            "high-resolution raw checkpoint; export it through a dynamic "
            "raw-image path instead"
        )
    matcher, matcher_payload = build_reference_set_matcher_from_checkpoint(
        matcher_checkpoint,
        device=device,
    )
    model = ReferenceAwarePetReID(encoder, matcher).to(device).eval()
    wrapper = ReferenceAwarePetReIDExport(model).to(device).eval()
    input_size = model.input_size
    if input_size is None:
        raise ValueError("base model does not expose a fixed input_size")
    references = matcher.max_references
    query = torch.zeros((1, 3, input_size, input_size), device=device)
    reference_rgb = torch.zeros(
        (1, references, 3, input_size, input_size), device=device
    )
    reference_mask = torch.ones((1, references), dtype=torch.bool, device=device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.output_name
    temporary = destination.with_suffix(destination.suffix + ".exporting")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)
    if temporary.exists():
        temporary.unlink()
    with torch.inference_mode():
        expected = wrapper(query, reference_rgb, reference_mask).cpu().numpy()
    torch.onnx.export(
        wrapper,
        (query, reference_rgb, reference_mask),
        str(temporary),
        input_names=("query_rgb", "reference_rgb", "reference_mask"),
        output_names=("score",),
        dynamic_axes={
            "query_rgb": {0: "batch"},
            "reference_rgb": {0: "batch"},
            "reference_mask": {0: "batch"},
            "score": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(temporary))
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["score"],
        {
            "query_rgb": query.cpu().numpy(),
            "reference_rgb": reference_rgb.cpu().numpy(),
            "reference_mask": reference_mask.cpu().numpy(),
        },
    )[0]
    difference = np.abs(expected - actual)
    validation = {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "provider": "CPUExecutionProvider",
    }
    if validation["max_abs_error"] > 2e-4:
        raise RuntimeError(f"ONNX parity check failed: {validation}")
    temporary.replace(destination)
    metadata = {
        "format": "reference-aware-pet-reid-onnx",
        "model_config": model.configuration(),
        "input_contract": {
            "query_rgb": ["N", 3, input_size, input_size],
            "reference_rgb": ["N", references, 3, input_size, input_size],
            "reference_mask": ["N", references],
            "pixel_range": "0..255 float32",
        },
        "output_contract": {"score": ["N"], "dtype": "float32"},
        "onnx": str(destination.resolve()),
        "onnx_sha256": sha256_file(destination),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "matcher_checkpoint": str(matcher_checkpoint),
        "matcher_checkpoint_sha256": sha256_file(matcher_checkpoint),
        "encoder_fingerprint": (
            matcher_payload.get("encoder_fingerprint")
            or base_payload.get("model_sha256")
            or sha256_file(base_checkpoint)
        ),
        "single_image_graph_unchanged": True,
        "validation": validation,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not np.isfinite(expected).all():
        raise RuntimeError("source graph produced a non-finite score")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
