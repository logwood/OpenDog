"""Evaluate BIFOR dog matching when some identities are absent from gallery.

The closed-set ReID evaluator always has a positive gallery identity for each
query.  This protocol holds out identities from the gallery, calibrates an
identity score and margin on a development partition, and reports open-set
false accepts on a disjoint evaluation partition.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastreid.config import get_cfg
from fastreid.data.build import build_reid_test_loader
from fastreid.data.common import CommDataset
from fastreid.data.transforms import build_transforms
from fastreid.evaluation import DatasetEvaluator, inference_on_dataset
from fastreid.utils.checkpoint import Checkpointer

from pet_id import add_retri_config
from pet_id.dataset import BIFOR_ROOT
from pet_id.train_net import Trainer
from pet_id.workspace_paths import normalize_runtime_config


LOGGER = logging.getLogger("pet_id.open_set")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _identity_records() -> dict[int, dict[str, list[Path]]]:
    root = BIFOR_ROOT / "test"
    identities: dict[int, dict[str, list[Path]]] = {}
    for directory in sorted(root.iterdir(), key=lambda path: int(path.name)):
        if not directory.is_dir():
            continue
        files = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name,
        )
        if len(files) < 2:
            continue
        midpoint = max(1, len(files) // 2)
        if midpoint >= len(files):
            continue
        identities[int(directory.name)] = {
            "query": files[:midpoint],
            "gallery": files[midpoint:],
        }
    if not identities:
        raise RuntimeError(f"No valid BIFOR test identities found below {root}")
    return identities


def _items_for_ids(
    identities: dict[int, dict[str, list[Path]]],
    ids: list[int],
    split: str,
    camera: int,
) -> list[tuple[str, int, int]]:
    items: list[tuple[str, int, int]] = []
    for identity in ids:
        for path in identities[identity][split]:
            items.append((str(path), int(identity), int(camera)))
    return items


class _FeatureCollector(DatasetEvaluator):
    def __init__(self) -> None:
        self.features: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.features = {}

    def process(self, inputs: dict[str, Any], outputs: torch.Tensor) -> None:
        batch_features = outputs.detach().float().cpu().numpy()
        for path, feature in zip(inputs["img_paths"], batch_features):
            self.features[str(path)] = feature

    def evaluate(self) -> None:
        return None


def _extract_features(cfg, weight_path: Path, items: list[tuple[str, int, int]]) -> dict[str, np.ndarray]:
    dataset = CommDataset(items, build_transforms(cfg, is_train=False), relabel=False)
    loader, _ = build_reid_test_loader(
        cfg,
        test_set=dataset,
        num_query=len(items),
    )

    model = Trainer.build_model(cfg)
    Checkpointer(model, save_dir=str(weight_path.parent)).load(str(weight_path))
    collector = _FeatureCollector()
    inference_on_dataset(model, loader, collector)
    if len(collector.features) != len(items):
        raise RuntimeError(
            f"Feature extraction returned {len(collector.features)} paths for {len(items)} items"
        )
    return collector.features


def _score_queries(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_identity_order: list[int],
    top_k: int,
) -> dict[str, np.ndarray]:
    query_features = query_features / np.maximum(
        np.linalg.norm(query_features, axis=1, keepdims=True), 1e-12
    )
    gallery_features = gallery_features / np.maximum(
        np.linalg.norm(gallery_features, axis=1, keepdims=True), 1e-12
    )
    similarities = query_features @ gallery_features.T
    identity_columns: list[np.ndarray] = []
    for identity in gallery_identity_order:
        indices = np.flatnonzero(gallery_ids == identity)
        values = np.sort(similarities[:, indices], axis=1)
        keep = min(int(top_k), values.shape[1])
        identity_columns.append(values[:, -keep:].mean(axis=1))
    identity_scores = np.stack(identity_columns, axis=1)
    order = np.argsort(-identity_scores, axis=1)
    row = np.arange(identity_scores.shape[0])
    best_index = order[:, 0]
    second_index = order[:, 1]
    best_scores = identity_scores[row, best_index]
    second_scores = identity_scores[row, second_index]
    return {
        "predicted_ids": np.asarray(gallery_identity_order, dtype=np.int64)[best_index],
        "best_scores": best_scores,
        "margins": best_scores - second_scores,
    }


def _calibrate_gate(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
    target_false_accept_rate: float,
) -> dict[str, float]:
    score_values = np.concatenate((known["best_scores"], unknown["best_scores"]))
    margin_values = np.concatenate((known["margins"], unknown["margins"]))
    score_candidates = np.unique(np.quantile(score_values, np.linspace(0.0, 1.0, 161)))
    margin_candidates = np.unique(np.quantile(margin_values, np.linspace(0.0, 1.0, 81)))

    best: tuple[float, float, float, float, float] | None = None
    fallback: tuple[float, float, float, float, float] | None = None
    for score_threshold in score_candidates:
        for margin_threshold in margin_candidates:
            known_accept = (
                (known["best_scores"] >= score_threshold)
                & (known["margins"] >= margin_threshold)
            )
            unknown_accept = (
                (unknown["best_scores"] >= score_threshold)
                & (unknown["margins"] >= margin_threshold)
            )
            true_accept_rate = float(known_accept.mean())
            false_accept_rate = float(unknown_accept.mean())
            balanced = (true_accept_rate + (1.0 - false_accept_rate)) / 2.0
            candidate = (
                balanced,
                true_accept_rate,
                -false_accept_rate,
                float(score_threshold),
                float(margin_threshold),
            )
            if fallback is None or candidate > fallback:
                fallback = candidate
            if false_accept_rate <= target_false_accept_rate and (
                best is None or candidate > best
            ):
                best = candidate

    selected = best or fallback
    if selected is None:
        raise RuntimeError("Could not calibrate an open-set gate")
    return {
        "score_threshold": selected[3],
        "margin_threshold": selected[4],
        "calibration_known_accept_rate": selected[1],
        "calibration_unknown_false_accept_rate": -selected[2],
        "calibration_balanced_accuracy": selected[0],
        "target_false_accept_rate": float(target_false_accept_rate),
    }


def _summarize(
    result: dict[str, np.ndarray],
    true_ids: np.ndarray | None,
    gate: dict[str, float],
) -> dict[str, float]:
    accepted = (
        (result["best_scores"] >= gate["score_threshold"])
        & (result["margins"] >= gate["margin_threshold"])
    )
    output = {
        "queries": float(len(accepted)),
        "accept_rate": float(accepted.mean()),
    }
    if true_ids is None:
        output["unknown_false_accept_rate"] = float(accepted.mean())
        output["unknown_reject_rate"] = float((~accepted).mean())
        return output

    correct = result["predicted_ids"] == true_ids
    output.update(
        {
            "top1_accuracy": float(correct.mean()),
            "known_accept_rate": float(accepted.mean()),
            "known_correct_and_accepted_rate": float((correct & accepted).mean()),
            "accepted_identity_accuracy": (
                float(correct[accepted].mean()) if accepted.any() else 0.0
            ),
        }
    )
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--calibration-known", type=int, default=180)
    parser.add_argument("--calibration-unknown", type=int, default=180)
    parser.add_argument("--evaluation-known", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--target-far", type=float, default=0.01)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identities = _identity_records()
    identity_ids = sorted(identities)
    random.Random(args.seed).shuffle(identity_ids)
    required = args.calibration_known + args.calibration_unknown + args.evaluation_known
    if required >= len(identity_ids):
        raise ValueError(f"Need at least one evaluation-unknown identity; got {required} of {len(identity_ids)}")

    cursor = 0
    calibration_known_ids = identity_ids[cursor : cursor + args.calibration_known]
    cursor += args.calibration_known
    calibration_unknown_ids = identity_ids[cursor : cursor + args.calibration_unknown]
    cursor += args.calibration_unknown
    evaluation_known_ids = identity_ids[cursor : cursor + args.evaluation_known]
    cursor += args.evaluation_known
    evaluation_unknown_ids = identity_ids[cursor:]
    gallery_ids = sorted(calibration_known_ids + evaluation_known_ids)

    query_items = _items_for_ids(
        identities,
        calibration_known_ids
        + calibration_unknown_ids
        + evaluation_known_ids
        + evaluation_unknown_ids,
        "query",
        0,
    )
    gallery_items = _items_for_ids(identities, gallery_ids, "gallery", 1)
    all_items = query_items + gallery_items

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.HEADS.NUM_CLASSES = 2000
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = str(Path(args.weights).expanduser().resolve())
    cfg.DATALOADER.NUM_WORKERS = min(int(cfg.DATALOADER.NUM_WORKERS), 8)
    normalize_runtime_config(cfg)
    cfg.freeze()

    LOGGER.info(
        "Extracting %d query and %d gallery images for %d gallery identities",
        len(query_items),
        len(gallery_items),
        len(gallery_ids),
    )
    feature_map = _extract_features(cfg, Path(args.weights), all_items)
    query_features = np.stack([feature_map[str(item[0])] for item in query_items])
    gallery_features = np.stack([feature_map[str(item[0])] for item in gallery_items])
    query_ids = np.asarray([item[1] for item in query_items], dtype=np.int64)
    gallery_pid_array = np.asarray([item[1] for item in gallery_items], dtype=np.int64)

    query_offsets: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(query_items):
        query_offsets[int(item[1])].append(index)

    def query_result(ids: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        indices = np.asarray(
            [index for identity in ids for index in query_offsets[identity]], dtype=np.int64
        )
        return (
            _score_queries(
                query_features[indices],
                gallery_features,
                gallery_pid_array,
                gallery_ids,
                args.top_k,
            ),
            query_ids[indices],
        )

    calibration_known, calibration_known_true = query_result(calibration_known_ids)
    calibration_unknown, _ = query_result(calibration_unknown_ids)
    evaluation_known, evaluation_known_true = query_result(evaluation_known_ids)
    evaluation_unknown, _ = query_result(evaluation_unknown_ids)
    gate = _calibrate_gate(
        calibration_known,
        calibration_unknown,
        args.target_far,
    )

    summary = {
        "protocol": {
            "dataset": "YT-BB-Dog",
            "total_identities": len(identity_ids),
            "gallery_identities": len(gallery_ids),
            "calibration_known_identities": len(calibration_known_ids),
            "calibration_unknown_identities": len(calibration_unknown_ids),
            "evaluation_known_identities": len(evaluation_known_ids),
            "evaluation_unknown_identities": len(evaluation_unknown_ids),
            "top_k_gallery_references": int(args.top_k),
            "seed": int(args.seed),
        },
        "gate": gate,
        "calibration": {
            "known": _summarize(calibration_known, calibration_known_true, gate),
            "unknown": _summarize(calibration_unknown, None, gate),
        },
        "evaluation": {
            "known": _summarize(evaluation_known, evaluation_known_true, gate),
            "unknown": _summarize(evaluation_unknown, None, gate),
        },
    }

    unknown_accept = (
        (evaluation_unknown["best_scores"] >= gate["score_threshold"])
        & (evaluation_unknown["margins"] >= gate["margin_threshold"])
    )
    accepted_indices = np.flatnonzero(unknown_accept)
    hard_confusers = []
    unknown_query_indices = [
        index for identity in evaluation_unknown_ids for index in query_offsets[identity]
    ]
    unknown_query_paths = [query_items[index][0] for index in unknown_query_indices]
    unknown_query_true_ids = query_ids[np.asarray(unknown_query_indices, dtype=np.int64)]
    for local_index in accepted_indices[np.argsort(-evaluation_unknown["best_scores"][accepted_indices])[:20]]:
        hard_confusers.append(
            {
                "query": unknown_query_paths[int(local_index)],
                "true_unregistered_id": int(unknown_query_true_ids[local_index]),
                "predicted_gallery_id": int(evaluation_unknown["predicted_ids"][local_index]),
                "score": float(evaluation_unknown["best_scores"][local_index]),
                "margin": float(evaluation_unknown["margins"][local_index]),
            }
        )
    summary["hard_unknown_false_accepts"] = hard_confusers

    (output_dir / "open_set_result.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "open_set_features.npz",
        query_features=query_features,
        gallery_features=gallery_features,
        query_ids=query_ids,
        gallery_ids=gallery_pid_array,
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
