"""Evaluate spatial-query main/query branches and descriptor fusion without repeated inference."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from fastreid.config import get_cfg
from fastreid.evaluation import DatasetEvaluator, inference_on_dataset
from fastreid.utils.checkpoint import Checkpointer

from pet_id import add_retri_config
from pet_id.train_net import Trainer
from pet_id.workspace_paths import normalize_runtime_config


class _SpatialQueryBranchExtractor(nn.Module):
    """Return both descriptors from one spatial-query backbone/workspace forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batched_inputs):
        images = self.model.preprocess_image(batched_inputs)
        features, latents = self.model._forward_workspace_backbone(
            images, collect_diagnostics=False, return_latents=True
        )
        main_descriptor = self.model.heads(features)
        query_descriptor = self.model.identity_query_head(
            latents, collect_diagnostics=False
        )
        return {"main": main_descriptor, "query": query_descriptor}


class _BranchCollector(DatasetEvaluator):
    def reset(self):
        self.main = []
        self.query = []
        self.targets = []
        self.pair_ids = []

    def process(self, inputs, outputs):
        self.main.append(outputs["main"].detach().to("cpu", torch.float32))
        self.query.append(outputs["query"].detach().to("cpu", torch.float32))
        self.targets.append(inputs["targets"].detach().to("cpu"))
        self.pair_ids.append(inputs["camids"].detach().to("cpu"))

    def evaluate(self):
        return {}

    def tensors(self):
        return (
            torch.cat(self.main),
            torch.cat(self.query),
            torch.cat(self.targets),
            torch.cat(self.pair_ids),
        )


def _paired_scores(features, targets, pair_ids, num_query):
    if features.shape[0] != num_query * 2:
        raise ValueError(
            f"Expected {num_query * 2} descriptors, found {features.shape[0]}"
        )
    if not torch.equal(targets[:num_query], targets[num_query:]):
        raise ValueError("Validation query/gallery labels are misaligned")
    if not torch.equal(pair_ids[:num_query], pair_ids[num_query:]):
        raise ValueError("Validation query/gallery pair IDs are misaligned")

    query = F.normalize(features[:num_query].float(), dim=1)
    gallery = F.normalize(features[num_query:].float(), dim=1)
    labels = targets[:num_query].numpy().astype(np.int64, copy=False)
    scores = (query * gallery).sum(dim=1).numpy()
    return labels, scores


def _verification_metrics(features, targets, pair_ids, num_query):
    labels, scores = _paired_scores(features, targets, pair_ids, num_query)
    auc = float(roc_auc_score(labels, scores))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    best_index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold = float(thresholds[best_index])
    accuracy = float(np.mean((scores >= threshold).astype(np.int64) == labels))
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    return {
        "ROC_AUC": auc,
        "accuracy_at_best_threshold": accuracy,
        "best_threshold": threshold,
        "positive_score_mean": float(positive_scores.mean()),
        "positive_score_std": float(positive_scores.std()),
        "negative_score_mean": float(negative_scores.mean()),
        "negative_score_std": float(negative_scores.std()),
        "mean_score_margin": float(positive_scores.mean() - negative_scores.mean()),
        "pairs": int(labels.size),
    }


def _fuse(main, query, weight):
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Fusion weight must be in [0, 1], found {weight}")
    return torch.cat(
        (
            math.sqrt(1.0 - weight) * F.normalize(main.float(), dim=1),
            math.sqrt(weight) * F.normalize(query.float(), dim=1),
        ),
        dim=1,
    )


def _checkpoint_epoch(path):
    match = re.search(r"model_(\d+)\.pth$", path.name)
    return int(match.group(1)) + 1 if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    normalize_runtime_config(cfg)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = 0
    cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASETS.TESTS = ("PetIDValidation",)
    cfg.OUTPUT_DIR = str(output_path.parent)
    cfg.freeze()

    model = Trainer.build_model(cfg)
    if model.__class__.__name__ != "SpatialQueryLatentWorkspace":
        raise TypeError(f"Expected spatial-query model, found {model.__class__.__name__}")
    if model.identity_query_head is None:
        raise ValueError("Spatial-query identity head is disabled")
    extractor = _SpatialQueryBranchExtractor(model)
    loader, num_query = Trainer.build_test_loader(cfg, "PetIDValidation")

    report = {
        "config": args.config_file,
        "batch_size": args.batch_size,
        "fixed_fusion_weights": args.weights,
        "checkpoints": [],
    }
    for checkpoint_value in args.checkpoints:
        checkpoint = Path(checkpoint_value)
        checkpoint_data = Checkpointer(model).load(str(checkpoint))
        collector = _BranchCollector()
        inference_on_dataset(extractor, loader, collector)
        main, query, targets, pair_ids = collector.tensors()

        learned_weight = float(model._fusion_weight().detach().cpu())
        variants = {
            "main_only": _verification_metrics(
                main, targets, pair_ids, num_query
            ),
            "query_only": _verification_metrics(
                query, targets, pair_ids, num_query
            ),
            "checkpoint_fused": _verification_metrics(
                _fuse(main, query, learned_weight), targets, pair_ids, num_query
            ),
        }
        for weight in args.weights:
            name = f"fixed_weight_{weight:g}"
            variants[name] = _verification_metrics(
                _fuse(main, query, weight), targets, pair_ids, num_query
            )

        _, main_scores = _paired_scores(main, targets, pair_ids, num_query)
        _, query_scores = _paired_scores(query, targets, pair_ids, num_query)
        result = {
            "checkpoint": str(checkpoint),
            "epoch": _checkpoint_epoch(checkpoint),
            "checkpoint_iteration": checkpoint_data.get("iteration"),
            "learned_fusion_weight": learned_weight,
            "descriptor_shapes": {
                "main": list(main.shape),
                "query": list(query.shape),
            },
            "main_query_score_correlation": float(
                np.corrcoef(main_scores, query_scores)[0, 1]
            ),
            "variants": variants,
        }
        report["checkpoints"].append(result)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    print(f"Saved exact spatial-query fusion ablation report to {output_path}")


if __name__ == "__main__":
    main()
