#!/usr/bin/env python3
"""Extract and evaluate BIFOR body features on an existing Re-ID protocol.

The script deliberately reuses body boxes and quality signals produced by the
existing frozen-body run.  Consequently the only experimental variable is the
body encoder: Swin V2-B (1024-D) is replaced by BIFOR f(2) (768-D).  The saved
archive follows the existing ``body_semantic_features.npz`` contract and can
be passed directly to ``train_evaluate_body_primary_fusion.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import functional as TVF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.bifor_backbone import FrozenBIFORBodyBackbone


EVALUATION_TOOL = ROOT / "tools" / "train_evaluate_body_primary_fusion.py"
SPEC = importlib.util.spec_from_file_location("_body_primary_eval", EVALUATION_TOOL)
EVALUATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATION)


def path_key(value: str) -> str:
    return str(Path(value).resolve()).replace("\\", "/").casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--body-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument(
        "--evaluation-purpose",
        choices=("development", "spent_test_diagnostic", "locked_final"),
        default="development",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    if not records:
        raise RuntimeError("Manifest contains no records")

    metadata = np.load(args.body_metadata, allow_pickle=False)
    metadata_rows = {
        path_key(path): index for index, path in enumerate(metadata["source_paths"].tolist())
    }
    missing = [
        record["source_path"]
        for record in records
        if path_key(record["source_path"]) not in metadata_rows
    ]
    if missing:
        raise ValueError(f"Body metadata is missing {len(missing)} manifest paths")
    order = np.asarray(
        [metadata_rows[path_key(record["source_path"])] for record in records],
        dtype=np.int64,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = FrozenBIFORBodyBackbone(args.checkpoint).to(device).eval()
    preprocess = model.preprocessing()
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )

    identities: list[str] = []
    source_paths: list[str] = []
    tensors: list[torch.Tensor] = []
    boxes = np.asarray(metadata["body_boxes_xyxy"][order], dtype=np.int32)
    for record, box in zip(records, boxes):
        image = read_image(record["source_path"], mode=ImageReadMode.RGB)
        target_width, target_height = (int(value) for value in record["resized_size"])
        if tuple(image.shape[-2:]) != (target_height, target_width):
            image = TVF.resize(image, [target_height, target_width], antialias=True)
        x1, y1, x2, y2 = (int(value) for value in box)
        crop = image[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            raise RuntimeError(f"Empty body crop for {record['source_path']}: {box.tolist()}")
        tensors.append(preprocess(crop))
        identities.append(record["identity"].casefold())
        source_paths.append(str(Path(record["source_path"]).resolve()))

    features: list[torch.Tensor] = []
    for start in range(0, len(tensors), args.batch_size):
        batch = torch.stack(tensors[start : start + args.batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            descriptor = model(batch)["global_features"]
            if not args.no_flip_tta:
                flipped = model(batch.flip(-1))["global_features"]
                descriptor = F.normalize(descriptor + flipped, dim=1)
        features.append(descriptor.float().cpu())
        processed = min(start + len(batch), len(tensors))
        print(f"bifor: {processed}/{len(tensors)}", flush=True)
    body_features = torch.cat(features)

    evaluation = EVALUATION.evaluate_features(
        body_features,
        identities,
        source_paths,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "body_semantic_features.npz"
    np.savez_compressed(
        feature_path,
        body_features=body_features.numpy(),
        identities=np.asarray(identities),
        source_paths=np.asarray(source_paths),
        body_boxes_xyxy=boxes,
        body_detected=np.asarray(metadata["body_detected"][order], dtype=np.bool_),
        body_detection_scores=np.asarray(
            metadata["body_detection_scores"][order], dtype=np.float32
        ),
        body_box_area_ratios=np.asarray(
            metadata["body_box_area_ratios"][order], dtype=np.float32
        ),
        face_to_body_area_ratios=np.asarray(
            metadata["face_to_body_area_ratios"][order], dtype=np.float32
        ),
        imagenet_dog_probabilities=np.asarray(
            metadata["imagenet_dog_probabilities"][order], dtype=np.float32
        ),
    )
    report = {
        "schema_version": 1,
        "purpose": args.evaluation_purpose,
        "manifest": str(args.manifest.resolve()),
        "records": len(records),
        "identities": len(set(identities)),
        "body_crop_source": str(args.body_metadata.resolve()),
        "encoder": {
            "name": "BIFOR f(2) / ConvNeXt-Small without classifier",
            "checkpoint": str(args.checkpoint.resolve()),
            "feature_dimensions": int(body_features.shape[1]),
            "input_size": [224, 224],
            "horizontal_flip_tta": not args.no_flip_tta,
            "frozen": True,
        },
        "single_branch": evaluation,
        "feature_archive": str(feature_path.resolve()),
        "output_contract": {
            "embedding_shape": list(body_features.shape),
            "unit_norm_max_error": float(
                (body_features.norm(dim=1) - 1.0).abs().max()
            ),
        },
    }
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(report_path.resolve()),
                "feature_archive": str(feature_path.resolve()),
                "records": len(records),
                "identities": len(set(identities)),
                "gallery_rank1": evaluation["gallery_rank1"],
                "leave_one_out_rank1": evaluation["leave_one_out_rank1"],
                "auc": evaluation["auc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
