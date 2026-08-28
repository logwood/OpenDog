#!/usr/bin/env python3
"""Evaluate the actual raw-image BIFOR ONNX runtime on a Re-ID manifest."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg
from pet_id.bifor_onnx_runtime import build_bifor_onnx_multimodal_pipeline
from pet_id.config import add_retri_config
from pet_id.gallery import descriptor_priority, load_exif_oriented_bgr
from pet_id.workspace_paths import normalize_runtime_config, resolve_legacy_path


EVAL_SPEC = importlib.util.spec_from_file_location(
    "_body_primary_eval", ROOT / "tools/train_evaluate_body_primary_fusion.py"
)
EVAL = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(EVAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_joint100_protocol_v1/validation_manifest.json",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=WORKSPACE
        / "models/selected/dogfacenet_semantic_v3_bifor_lowrank_v1/config.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=WORKSPACE
        / "models/selected/dogfacenet_semantic_v3_bifor_lowrank_v1/onnx/pet_embedding.onnx",
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=WORKSPACE
        / "models/pretrained/body_detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "artifacts/runs/bifor/onnx_raw_runtime_validation_v1/evaluation.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (args.manifest, args.config_file, args.model, args.body_detector)
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing runtime validation inputs: {missing}")
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(str(args.config_file.resolve()))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    pipeline = build_bifor_onnx_multimodal_pipeline(
        cfg,
        model_path=args.model,
        body_detector_checkpoint=args.body_detector,
        provider=args.provider,
        device=args.device,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    features, identities, source_paths = [], [], []
    branch_rows = []
    for index, record in enumerate(records, start=1):
        source = resolve_legacy_path(record["source_path"])
        descriptors = pipeline.encode_image(load_exif_oriented_bgr(source))
        if not descriptors:
            raise RuntimeError(f"No descriptor produced for {source}")
        selected = max(
            range(len(descriptors)),
            key=lambda row: descriptor_priority(descriptors[row]),
        )
        descriptor = descriptors[selected]
        features.append(descriptor.fused_feature.float())
        identities.append(record["identity"].casefold())
        source_paths.append(str(source.resolve()))
        branch_rows.append(list(descriptor.branch_available))
        print(f"raw runtime: {index}/{len(records)}", flush=True)
    feature_tensor = torch.stack(features)
    metrics = EVAL.evaluate_features(feature_tensor, identities, source_paths)
    norms = feature_tensor.norm(dim=1).numpy()
    report = {
        "schema_version": 1,
        "backend": pipeline.identity_model.backend_info(),
        "manifest": str(args.manifest.resolve()),
        "records": len(records),
        "identities": len(set(identities)),
        "embedding_shape": list(feature_tensor.shape),
        "embedding_norm_range": [float(norms.min()), float(norms.max())],
        "dual_branch_records": sum(all(row) for row in branch_rows),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_name("features.npz"),
        features=feature_tensor.numpy(),
        identities=np.asarray(identities),
        source_paths=np.asarray(source_paths),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "gallery_rank1": metrics["gallery_rank1"],
                "leave_one_out_rank1": metrics["leave_one_out_rank1"],
                "auc": metrics["auc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
