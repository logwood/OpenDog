"""Evaluate inference-path ablations from trained Latent Workspace V2 checkpoints."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from fastreid.config import get_cfg
from fastreid.evaluation import DatasetEvaluator, inference_on_dataset
from fastreid.utils.checkpoint import Checkpointer

from pet_id import add_retri_config
from pet_id.train_net import Trainer


class _DescriptorCollector(DatasetEvaluator):
    def reset(self):
        self.features = []
        self.targets = []
        self.pair_ids = []

    def process(self, inputs, outputs):
        self.features.append(outputs.detach().to("cpu", torch.float32))
        self.targets.append(inputs["targets"].detach().to("cpu"))
        self.pair_ids.append(inputs["camids"].detach().to("cpu"))

    def evaluate(self):
        return {}

    def tensors(self):
        return (
            torch.cat(self.features),
            torch.cat(self.targets),
            torch.cat(self.pair_ids),
        )


def _verification_metrics(features, targets, pair_ids, num_query):
    if features.shape[0] != num_query * 2:
        raise ValueError(
            f"Expected {num_query * 2} descriptors, found {features.shape[0]}"
        )
    query = F.normalize(features[:num_query].float(), dim=1)
    gallery = F.normalize(features[num_query:].float(), dim=1)
    if not torch.equal(targets[:num_query], targets[num_query:]):
        raise ValueError("Validation query/gallery labels are misaligned")
    if not torch.equal(pair_ids[:num_query], pair_ids[num_query:]):
        raise ValueError("Validation query/gallery pair IDs are misaligned")

    labels = targets[:num_query].numpy().astype(np.int64, copy=False)
    scores = (query * gallery).sum(dim=1).numpy()
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


def _collect(model, cfg, feature_fusion_scale, inference_fusion_weight):
    model.slot_feature_fusion_scale = feature_fusion_scale
    model.slot_inference_fusion_weight = inference_fusion_weight
    loader, num_query = Trainer.build_test_loader(cfg, "PetIDValidation")
    collector = _DescriptorCollector()
    inference_on_dataset(model, loader, collector)
    return (*collector.tensors(), num_query)


def _checkpoint_epoch(path):
    match = re.search(r"model_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = 0
    cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASETS.TESTS = ("PetIDValidation",)
    cfg.OUTPUT_DIR = str(output_path.parent)
    cfg.freeze()

    model = Trainer.build_model(cfg)
    original_feature_weight = float(
        cfg.MODEL.LATENT_WORKSPACE.SLOT_FEATURE_FUSION_SCALE
    )
    original_descriptor_weight = float(
        cfg.MODEL.LATENT_WORKSPACE.SLOT_INFERENCE_FUSION_WEIGHT
    )
    if not 0.0 < original_descriptor_weight < 1.0:
        raise ValueError("Descriptor fusion weight must be strictly between zero and one")

    report = {
        "config": args.config_file,
        "batch_size": args.batch_size,
        "feature_fusion_scale": original_feature_weight,
        "descriptor_fusion_weight": original_descriptor_weight,
        "checkpoints": [],
    }
    for checkpoint_value in args.checkpoints:
        checkpoint = Path(checkpoint_value)
        Checkpointer(model).load(str(checkpoint))

        full, targets, pair_ids, num_query = _collect(
            model,
            cfg,
            original_feature_weight,
            original_descriptor_weight,
        )
        no_c5, targets_no_c5, pair_ids_no_c5, num_query_no_c5 = _collect(
            model,
            cfg,
            0.0,
            original_descriptor_weight,
        )
        if num_query != num_query_no_c5:
            raise ValueError("Ablation passes produced different query counts")
        if not torch.equal(targets, targets_no_c5) or not torch.equal(
            pair_ids, pair_ids_no_c5
        ):
            raise ValueError("Ablation passes produced different sample order")

        main_dim = int(cfg.MODEL.BACKBONE.FEAT_DIM)
        slot_dim = int(
            cfg.MODEL.LATENT_WORKSPACE.NUM_SLOTS
            * cfg.MODEL.LATENT_WORKSPACE.SLOT_SET_DIM_PER_SLOT
        )
        if full.shape[1] != main_dim + slot_dim or no_c5.shape[1] != full.shape[1]:
            raise ValueError(
                f"Unexpected descriptor shapes: {tuple(full.shape)}, {tuple(no_c5.shape)}"
            )
        main_scale = math.sqrt(1.0 - original_descriptor_weight)
        slot_scale = math.sqrt(original_descriptor_weight)
        variants = {
            "full_v2": full,
            "no_slot_descriptor": full[:, :main_dim] / main_scale,
            "no_c5_slot_fusion": no_c5,
            "writeback_only": no_c5[:, :main_dim] / main_scale,
            "slot_only": full[:, main_dim:] / slot_scale,
        }
        result = {
            "checkpoint": str(checkpoint),
            "epoch": _checkpoint_epoch(checkpoint),
            "variants": {
                name: _verification_metrics(
                    descriptors, targets, pair_ids, num_query
                )
                for name, descriptors in variants.items()
            },
        }
        report["checkpoints"].append(result)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    print(f"Saved exact ablation report to {output_path}")


if __name__ == "__main__":
    main()
