#!/usr/bin/env python3
"""Re-run the locked 20-identity protocol through the exported BIFOR ONNX."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.bifor_onnx import BIFOR_ONNX_INPUT_NAMES
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import DifferentiableROICropper
from pet_id.model_profiles import get_runtime_profile
from pet_id.release_compatibility import historical_run_path


EXPORT_SPEC = importlib.util.spec_from_file_location(
    "_bifor_onnx_export", ROOT / "tools/export_bifor_multimodal_onnx.py"
)
EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
assert EXPORT_SPEC.loader is not None
EXPORT_SPEC.loader.exec_module(EXPORT)

EVAL_SPEC = importlib.util.spec_from_file_location(
    "_body_primary_eval", ROOT / "tools/train_evaluate_body_primary_fusion.py"
)
EVAL = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(EVAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    profile = get_runtime_profile("research-bifor")
    parser.add_argument(
        "--model",
        type=Path,
        default=profile.onnx,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=historical_run_path(WORKSPACE, "joint-validation-manifest"),
    )
    parser.add_argument(
        "--body-metadata",
        type=Path,
        default=historical_run_path(WORKSPACE, "body-validation")
        / "body_semantic_features.npz",
    )
    parser.add_argument(
        "--expected-evaluation",
        type=Path,
        default=historical_run_path(WORKSPACE, "bifor-evaluation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=historical_run_path(WORKSPACE, "bifor-onnx-validation"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--strict-exact",
        action="store_true",
        help="fail unless FP32 ONNX metrics exactly match the historical BF16 cache",
    )
    return parser.parse_args()


def precropped_inputs(batch: dict, body_metadata: Path) -> tuple[np.ndarray, ...]:
    images = batch["images_0_255"].float()
    face_rois = batch["face_rois"].float()
    nose_rois = batch["nose_rois"].float()
    angles = batch["roll_angles_radians"].float()
    cropper = DifferentiableROICropper()
    face_crop = cropper(images, face_rois, angles, (224, 224))
    nose_crop = cropper(images, nose_rois, angles, (244, 244))
    mask_rois = nose_rois.clone()
    mask_rois[:, 0] = torch.arange(mask_rois.shape[0], dtype=mask_rois.dtype)
    nose_mask = cropper(
        batch["nose_masks"].float(), mask_rois, angles, (244, 244)
    ).clamp(0, 1)
    body_crop, _ = EXPORT.body_crops_from_metadata(
        images,
        list(batch["source_paths"]),
        body_metadata,
    )
    return tuple(
        value.numpy()
        for value in (
            nose_crop,
            face_crop,
            body_crop,
            nose_mask,
            batch["quality_signals"].float(),
            batch["viewpoint_signals"].float(),
            batch["branch_available"].bool(),
        )
    )


def main() -> None:
    args = parse_args()
    required = (
        args.model,
        args.manifest,
        args.body_metadata,
        args.expected_evaluation,
    )
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation inputs: {missing}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    provider = (
        "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    )
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(args.model.resolve()),
        sess_options=session_options,
        providers=[provider],
    )
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Requested {provider}, got {session.get_providers()}")
    if tuple(item.name for item in session.get_inputs()) != BIFOR_ONNX_INPUT_NAMES:
        raise RuntimeError("Unexpected exported input contract")

    dataset = PreparedDogFaceNetDataset(args.manifest.resolve(), training=False)
    features, identities, source_paths = [], [], []
    for start in range(0, len(dataset), args.batch_size):
        stop = min(start + args.batch_size, len(dataset))
        batch = collate_prepared_dogfacenet(
            [dataset[index] for index in range(start, stop)]
        )
        inputs = precropped_inputs(batch, args.body_metadata.resolve())
        feed = dict(zip(BIFOR_ONNX_INPUT_NAMES, inputs))
        features.append(session.run(["embedding"], feed)[0])
        identities.extend(value.casefold() for value in batch["identities"])
        source_paths.extend(
            str(Path(value).resolve()) for value in batch["source_paths"]
        )
        print(f"onnx: {stop}/{len(dataset)}", flush=True)
    feature_array = np.concatenate(features).astype(np.float32, copy=False)
    metrics = EVAL.evaluate_features(
        torch.from_numpy(feature_array), identities, source_paths
    )
    expected_report = json.loads(args.expected_evaluation.read_text(encoding="utf-8"))
    expected = expected_report["selected"]["metrics"]
    compared = {}
    for key in ("gallery_rank1", "leave_one_out_rank1", "auc"):
        compared[key] = {
            "onnx": metrics[key],
            "locked_pytorch": expected[key],
            "absolute_difference": abs(metrics[key] - expected[key]),
        }
    exact_metric_parity = not any(
        row["absolute_difference"] > 1e-12 for row in compared.values()
    )
    if args.strict_exact and not exact_metric_parity:
        raise RuntimeError(
            f"ONNX protocol metrics differ from locked PyTorch: {compared}"
        )
    norms = np.linalg.norm(feature_array, axis=1)
    report = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_sha256": EXPORT.sha256_file(args.model.resolve()),
        "provider": provider,
        "records": len(dataset),
        "identities": len(set(identities)),
        "embedding_shape": list(feature_array.shape),
        "embedding_norm_range": [float(norms.min()), float(norms.max())],
        "metrics": metrics,
        "locked_metric_parity": compared,
        "exact_metric_parity": exact_metric_parity,
        "precision_provenance": {
            "onnx": "float32",
            "locked_feature_cache": "bfloat16 autocast converted to float32",
            "interpretation": (
                "Gallery Rank-1 is reproduced exactly. Small LOO/AUC differences "
                "are expected because the selected projector was fit and evaluated "
                "on the historical BF16 feature cache, while production ONNX is FP32."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_name("features.npz"),
        features=feature_array,
        identities=np.asarray(identities),
        source_paths=np.asarray(source_paths),
    )
    print(json.dumps({"output": str(args.output.resolve()), **compared}, indent=2))


if __name__ == "__main__":
    main()
