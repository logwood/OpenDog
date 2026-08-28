#!/usr/bin/env python3
"""Evaluate robustness to a deliberately mismatched cross-identity nose crop.

The clean gallery and the query face remain unchanged. Each held-out query
receives the nose crop, mask, and nose-specific quality fields from a
deterministically selected different identity. This is a development-only
stress test and intentionally refuses the protocol's locked fresh-blind split.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from fastreid.config import get_cfg

from pet_id import add_retri_config
from pet_id.dogfacenet_alignment import (
    PreparedDogFaceNetDataset,
    collate_prepared_dogfacenet,
)
from pet_id.multimodal import build_local_identity_model
from pet_id.onnx_export import (
    PreCroppedPetEmbeddingModel,
    extract_precropped_onnx_inputs,
)
from pet_id.workspace_paths import normalize_runtime_config
from evaluate_multimodal_dogfacenet_large import (
    evaluate_branch,
    summarize,
)


NOSE_QUALITY_COLUMNS = (0, 3, 4)


@dataclass(frozen=True)
class ConflictPair:
    query_index: int
    donor_index: int
    query_identity: str
    donor_identity: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_development_manifest(manifest: dict) -> None:
    split = str(manifest.get("protocol_split", "")).casefold()
    policy = str(manifest.get("usage_policy", "")).casefold()
    if split == "fresh_blind" or "single_final_evaluation" in policy:
        raise RuntimeError(
            "Conflict robustness is a development-only experiment; refusing "
            "to open the locked fresh-blind manifest."
        )


def build_conflict_pairs(
    identities: list[str],
    gallery_per_identity: int,
) -> list[ConflictPair]:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(identity.casefold(), []).append(index)
    ordered_identities = sorted(grouped)
    if len(ordered_identities) < 2:
        raise ValueError("At least two identities are required for conflicts")

    pairs: list[ConflictPair] = []
    for identity_position, identity in enumerate(ordered_identities):
        query_indices = grouped[identity][gallery_per_identity:]
        if not query_indices:
            raise ValueError(f"Identity {identity!r} has no held-out queries")
        donor_identity = ordered_identities[
            (identity_position + 1) % len(ordered_identities)
        ]
        donor_indices = grouped[donor_identity][gallery_per_identity:]
        if not donor_indices:
            raise ValueError(
                f"Donor identity {donor_identity!r} has no held-out queries"
            )
        for offset, query_index in enumerate(query_indices):
            pairs.append(
                ConflictPair(
                    query_index=query_index,
                    donor_index=donor_indices[offset % len(donor_indices)],
                    query_identity=identity,
                    donor_identity=donor_identity,
                )
            )
    return pairs


def replace_nose_quality(
    query_quality: torch.Tensor,
    donor_quality: torch.Tensor,
) -> torch.Tensor:
    if query_quality.shape != donor_quality.shape:
        raise ValueError("Query and donor quality tensors must have equal shape")
    mixed = query_quality.clone()
    mixed[:, NOSE_QUALITY_COLUMNS] = donor_quality[:, NOSE_QUALITY_COLUMNS]
    return mixed


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        **summarize(array.tolist()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def transition_summary(
    clean_evaluation: dict,
    corrupted_evaluation: dict,
    pairs: list[ConflictPair],
) -> dict:
    clean_by_index = {
        int(row["query_index"]): row for row in clean_evaluation["queries"]
    }
    corrupted_by_index = {
        int(row["query_index"]): row for row in corrupted_evaluation["queries"]
    }

    retained = regressed = recovered = both_wrong = 0
    donor_hijacked = changed = 0
    rank_deltas: list[float] = []
    rows = []
    for pair in pairs:
        clean = clean_by_index[pair.query_index]
        corrupted = corrupted_by_index[pair.query_index]
        clean_correct = bool(clean["correct"])
        corrupted_correct = bool(corrupted["correct"])
        if clean_correct and corrupted_correct:
            retained += 1
        elif clean_correct:
            regressed += 1
        elif corrupted_correct:
            recovered += 1
        else:
            both_wrong += 1

        clean_top1 = clean["top5"][0]["identity"]
        corrupted_top1 = corrupted["top5"][0]["identity"]
        is_donor_hijacked = corrupted_top1 == pair.donor_identity
        donor_hijacked += int(is_donor_hijacked)
        changed += int(clean_top1 != corrupted_top1)
        rank_delta = int(corrupted["true_identity_rank"]) - int(
            clean["true_identity_rank"]
        )
        rank_deltas.append(float(rank_delta))
        rows.append(
            {
                **asdict(pair),
                "query_source_path": clean["query_source_path"],
                "clean_correct": clean_correct,
                "corrupted_correct": corrupted_correct,
                "clean_true_rank": int(clean["true_identity_rank"]),
                "corrupted_true_rank": int(corrupted["true_identity_rank"]),
                "true_rank_delta": rank_delta,
                "clean_top1_identity": clean_top1,
                "corrupted_top1_identity": corrupted_top1,
                "donor_hijacked": is_donor_hijacked,
                "corrupted_top5": corrupted["top5"],
            }
        )

    query_count = len(pairs)
    corrupted_wrong = query_count - int(corrupted_evaluation["top1_correct"])
    return {
        "query_count": query_count,
        "clean_top1_accuracy": float(clean_evaluation["top1_accuracy"]),
        "corrupted_top1_accuracy": float(corrupted_evaluation["top1_accuracy"]),
        "absolute_top1_drop": float(
            clean_evaluation["top1_accuracy"] - corrupted_evaluation["top1_accuracy"]
        ),
        "retained_correct": retained,
        "regressed": regressed,
        "recovered": recovered,
        "both_wrong": both_wrong,
        "top1_identity_changed": changed,
        "top1_identity_changed_rate": changed / query_count,
        "donor_hijacked": donor_hijacked,
        "donor_hijack_rate": donor_hijacked / query_count,
        "donor_hijack_rate_among_corrupted_errors": (
            donor_hijacked / corrupted_wrong if corrupted_wrong else 0.0
        ),
        "true_rank_delta": distribution(rank_deltas),
        "queries": rows,
    }


def gate_summary(
    clean_nose_weights: torch.Tensor,
    corrupted_nose_weights: torch.Tensor,
    corrupted_evaluation: dict,
) -> dict:
    clean_values = clean_nose_weights.float().cpu().tolist()
    corrupted_values = corrupted_nose_weights.float().cpu().tolist()
    deltas = [
        corrupted - clean for clean, corrupted in zip(clean_values, corrupted_values)
    ]
    correct_mask = [bool(row["correct"]) for row in corrupted_evaluation["queries"]]
    return {
        "clean_nose_weight": distribution(clean_values),
        "corrupted_nose_weight": distribution(corrupted_values),
        "corrupted_minus_clean": distribution(deltas),
        "decreased_count": sum(delta < 0 for delta in deltas),
        "decreased_rate": sum(delta < 0 for delta in deltas) / len(deltas),
        "corrupted_correct_nose_weight": distribution(
            [value for value, correct in zip(corrupted_values, correct_mask) if correct]
        ),
        "corrupted_wrong_nose_weight": distribution(
            [
                value
                for value, correct in zip(corrupted_values, correct_mask)
                if not correct
            ]
        ),
    }


def to_model_inputs(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in {"targets", "identities", "source_paths"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_development_manifest(manifest)
    output_path = args.output_dir / "conflict_evaluation.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace"
        )

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
    cfg.merge_from_file(str(args.config_file))
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()
    model = build_local_identity_model(
        cfg,
        device=device,
        for_training=False,
        identity_weights=str(args.checkpoint),
    ).eval()
    wrapper = PreCroppedPetEmbeddingModel(model).to(device).eval()

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    clean_features: list[torch.Tensor] = []
    clean_weights: list[torch.Tensor] = []
    identities: list[str] = []
    source_paths: list[str] = []
    for batch in loader:
        inputs = to_model_inputs(batch, device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ),
        ):
            output = model(**inputs)
        clean_features.append(output["features"].float().cpu())
        clean_weights.append(output["fusion_weights"].float().cpu())
        identities.extend(identity.casefold() for identity in batch["identities"])
        source_paths.extend(batch["source_paths"])

    clean_feature_matrix = torch.cat(clean_features)
    clean_weight_matrix = torch.cat(clean_weights)
    pairs = build_conflict_pairs(
        identities,
        args.gallery_images_per_identity,
    )
    query_indices = torch.tensor([pair.query_index for pair in pairs], dtype=torch.long)

    corrupted_feature_rows: list[torch.Tensor] = []
    corrupted_weight_rows: list[torch.Tensor] = []
    for start in range(0, len(pairs), args.batch_size):
        pair_batch = pairs[start : start + args.batch_size]
        query_batch = collate_prepared_dogfacenet(
            [dataset[pair.query_index] for pair in pair_batch]
        )
        donor_batch = collate_prepared_dogfacenet(
            [dataset[pair.donor_index] for pair in pair_batch]
        )
        query_inputs = to_model_inputs(query_batch, device)
        donor_inputs = to_model_inputs(donor_batch, device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ),
        ):
            query_crops = extract_precropped_onnx_inputs(model, **query_inputs)
            donor_crops = extract_precropped_onnx_inputs(model, **donor_inputs)
            mixed_quality = replace_nose_quality(query_crops[3], donor_crops[3])
            mixed_available = torch.stack(
                (donor_crops[5][:, 0], query_crops[5][:, 1]),
                dim=1,
            )
            corrupted_output = wrapper(
                donor_crops[0],
                query_crops[1],
                donor_crops[2],
                mixed_quality,
                query_crops[4],
                mixed_available,
            )
        corrupted_feature_rows.append(corrupted_output[0].float().cpu())
        corrupted_weight_rows.append(corrupted_output[3][:, 0].float().cpu())

    corrupted_query_features = torch.cat(corrupted_feature_rows)
    corrupted_nose_weights = torch.cat(corrupted_weight_rows)
    corrupted_all_features = clean_feature_matrix.clone()
    corrupted_all_features.index_copy_(0, query_indices, corrupted_query_features)

    clean_evaluation = evaluate_branch(
        clean_feature_matrix,
        identities,
        source_paths,
        args.gallery_images_per_identity,
    )
    corrupted_evaluation = evaluate_branch(
        corrupted_all_features,
        identities,
        source_paths,
        args.gallery_images_per_identity,
    )
    transition = transition_summary(
        clean_evaluation,
        corrupted_evaluation,
        pairs,
    )
    gates = gate_summary(
        clean_weight_matrix.index_select(0, query_indices)[:, 0],
        corrupted_nose_weights,
        corrupted_evaluation,
    )

    summary = {
        "schema_version": 1,
        "experiment": "cross_identity_nose_injection",
        "protocol_guard": "development_only_fresh_blind_refused",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_split": manifest.get("protocol_split"),
        "usage_policy": manifest.get("usage_policy"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.config_file.resolve()),
        "config_sha256": sha256_file(args.config_file),
        "fusion_mode": model.fusion_mode,
        "records": len(dataset),
        "identities": len(set(identities)),
        "amp_dtype": (str(amp_dtype).removeprefix("torch.") if use_amp else "float32"),
        "corruption": {
            "gallery": "clean_identity_prototypes",
            "query_face": "unchanged",
            "query_viewpoint": "unchanged",
            "nose_crop": "next_sorted_different_identity_query",
            "nose_mask": "same_donor_as_nose_crop",
            "nose_quality_columns": list(NOSE_QUALITY_COLUMNS),
            "nose_quality_names": [
                "nose_quality",
                "segmentation_predicted_iou",
                "nose_resolution",
            ],
            "donor_assignment": ("cyclic_next_sorted_identity_same_query_offset"),
        },
        "clean": clean_evaluation,
        "corrupted": corrupted_evaluation,
        "transition": transition,
        "gate": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    concise = {
        "output": str(output_path.resolve()),
        "fusion_mode": model.fusion_mode,
        "query_count": transition["query_count"],
        "clean_top1": clean_evaluation["top1_accuracy"],
        "corrupted_top1": corrupted_evaluation["top1_accuracy"],
        "absolute_top1_drop": transition["absolute_top1_drop"],
        "regressed": transition["regressed"],
        "donor_hijack_rate": transition["donor_hijack_rate"],
        "clean_mean_nose_weight": gates["clean_nose_weight"]["mean"],
        "corrupted_mean_nose_weight": gates["corrupted_nose_weight"]["mean"],
        "gate_decreased_rate": gates["decreased_rate"],
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
