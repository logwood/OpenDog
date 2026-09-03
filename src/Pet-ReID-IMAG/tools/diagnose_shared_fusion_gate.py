#!/usr/bin/env python3
"""Relate shared-space gate signals to branch and fused retrieval outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model
from pet_id.workspace_paths import normalize_runtime_config


QUALITY_NAMES = (
    "nose_quality",
    "face_quality",
    "detection_confidence",
    "segmentation_iou",
    "nose_resolution",
    "face_resolution",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def _query_map(evaluation: dict, branch: str) -> dict[int, dict]:
    return {
        int(row["query_index"]): row
        for row in evaluation["branches"][branch]["queries"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {args.output}")
    device = torch.device(args.device)
    dataset = PreparedDogFaceNetDataset(args.manifest, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prepared_dogfacenet,
    )

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=args.checkpoint,
    )
    model.eval()
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    gate_rows: list[dict] = []
    for batch in loader:
        inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"targets", "identities", "source_paths"}
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = model(**inputs)
        weights = output["fusion_weights"].float().cpu().numpy()
        frontality = output["viewpoint_frontality"].float().cpu().numpy()
        available = output["effective_branch_available"].cpu().numpy()
        quality = batch["quality_signals"].numpy()
        for row_index in range(weights.shape[0]):
            row = {
                "nose_weight": float(weights[row_index, 0]),
                "face_weight": float(weights[row_index, 1]),
                "viewpoint_frontality": float(frontality[row_index]),
                "available": [bool(value) for value in available[row_index]],
            }
            row.update(
                {
                    name: float(quality[row_index, column])
                    for column, name in enumerate(QUALITY_NAMES)
                }
            )
            gate_rows.append(row)

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    fused = _query_map(evaluation, "fused")
    nose = _query_map(evaluation, "nose")
    face = _query_map(evaluation, "face")
    categories: dict[str, list[dict]] = defaultdict(list)
    query_rows = []
    for query_index, fused_result in sorted(fused.items()):
        nose_correct = bool(nose[query_index]["correct"])
        face_correct = bool(face[query_index]["correct"])
        fused_correct = bool(fused_result["correct"])
        if nose_correct and face_correct:
            branch_state = "both_correct"
        elif nose_correct:
            branch_state = "nose_only_correct"
        elif face_correct:
            branch_state = "face_only_correct"
        else:
            branch_state = "both_wrong"
        category = f"{branch_state}__fused_{'correct' if fused_correct else 'wrong'}"
        diagnostic = {
            "query_index": query_index,
            "identity": fused_result["query_identity"],
            "true_identity_rank": int(fused_result["true_identity_rank"]),
            **gate_rows[query_index],
        }
        categories[category].append(diagnostic)
        query_rows.append({"category": category, **diagnostic})

    metric_names = (
        "nose_weight",
        "face_weight",
        "nose_quality",
        "face_quality",
        "detection_confidence",
        "segmentation_iou",
        "nose_resolution",
        "face_resolution",
        "viewpoint_frontality",
    )
    category_summary = {}
    for name, rows in sorted(categories.items()):
        category_summary[name] = {
            "count": len(rows),
            "signals": {
                metric: _stats([float(row[metric]) for row in rows])
                for metric in metric_names
            },
            "availability": dict(
                Counter("nose+face" if row["available"] == [True, True] else "nose_only"
                        if row["available"] == [True, False] else "face_only"
                        if row["available"] == [False, True] else "none"
                        for row in rows)
            ),
        }

    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "evaluation": str(args.evaluation.resolve()),
        "records": len(gate_rows),
        "queries": len(query_rows),
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "categories": category_summary,
        "query_rows": query_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "categories": {
        name: {"count": row["count"], "nose_weight": row["signals"]["nose_weight"]}
        for name, row in category_summary.items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
