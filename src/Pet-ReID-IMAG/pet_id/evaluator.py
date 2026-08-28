# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os
import itertools
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from fastreid.evaluation import DatasetEvaluator
from fastreid.evaluation.query_expansion import aqe
from fastreid.utils import comm

from .workspace_paths import LEGACY_RUNS_ROOT, PROCESSED_DATA_ROOT
from fastreid.utils.compute_dist import build_dist

logger = logging.getLogger("fastreid.pet_id_submission")


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def get_test_pairs(df: pd.DataFrame):
    pairs = []
    for idx, col in df.iterrows():
        imageA, imageB = col
        pairs.append((imageA, imageB))
    return pairs


def write_txt(l: list, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in l:
            f.write(str(item) + '\n')


class _PetIDBaseEvaluator(DatasetEvaluator):
    """Evaluator base that preserves string filenames used as test IDs."""

    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir or cfg.OUTPUT_DIR
        self._cpu_device = torch.device('cpu')
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device)
            if isinstance(inputs['targets'], torch.Tensor)
            else inputs['targets'],
            'camids': inputs['camids'].to(self._cpu_device)
            if isinstance(inputs['camids'], torch.Tensor)
            else inputs['camids'],
        }
        self._predictions.append(prediction)

    def _gather_predictions(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            if not comm.is_main_process():
                return None
            return list(itertools.chain(*predictions))
        return self._predictions


class PetIDVerificationEvaluator(_PetIDBaseEvaluator):
    """Compute true ROC-AUC on the deterministic held-out verification pairs."""

    def evaluate(self):
        predictions = self._gather_predictions()
        if predictions is None:
            return {}

        features = torch.cat([prediction['feats'] for prediction in predictions], dim=0)
        targets = torch.cat([prediction['pids'] for prediction in predictions], dim=0)
        pair_ids = torch.cat([prediction['camids'] for prediction in predictions], dim=0)
        expected = self._num_query * 2
        if features.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} validation descriptors, found {features.shape[0]}"
            )

        query_features = F.normalize(features[:self._num_query], dim=1)
        gallery_features = F.normalize(features[self._num_query:], dim=1)
        query_targets = targets[:self._num_query]
        gallery_targets = targets[self._num_query:]
        query_pair_ids = pair_ids[:self._num_query]
        gallery_pair_ids = pair_ids[self._num_query:]
        if not torch.equal(query_targets, gallery_targets):
            raise ValueError("Validation query/gallery labels are misaligned")
        if not torch.equal(query_pair_ids, gallery_pair_ids):
            raise ValueError("Validation query/gallery pair IDs are misaligned")

        labels = query_targets.numpy().astype(np.int64, copy=False)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Validation requires both positive and negative pairs")
        scores = (query_features * gallery_features).sum(dim=1).numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError("Validation similarity scores contain NaN or infinity")

        auc = float(roc_auc_score(labels, scores))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
        best_index = int(np.argmax(true_positive_rate - false_positive_rate))
        threshold = float(thresholds[best_index])
        predictions_at_threshold = (scores >= threshold).astype(np.int64)
        accuracy = float(np.mean(predictions_at_threshold == labels))

        return OrderedDict(
            metric=auc,
            ROC_AUC=auc,
            best_threshold=threshold,
            accuracy_at_best_threshold=accuracy,
            pairs=float(labels.size),
            positive_pairs=float(labels.sum()),
            negative_pairs=float(labels.size - labels.sum()),
        )


class PetIDFeatureEvaluator(_PetIDBaseEvaluator):
    """Export one model's query/gallery descriptors and filename mapping."""

    def evaluate(self):
        predictions = self._gather_predictions()
        if predictions is None:
            return {}

        features = torch.cat([p['feats'] for p in predictions], dim=0).numpy()
        filenames = []
        for prediction in predictions:
            filenames.extend(prediction['pids'])

        expected = features.shape[0]
        if len(filenames) != expected or self._num_query * 2 != expected:
            raise ValueError(
                'Unexpected PetIDTest layout: '
                f'{len(filenames)=}, {features.shape=}, {self._num_query=}'
            )

        query_features = features[:self._num_query]
        gallery_features = features[self._num_query:]
        query_filename = filenames[:self._num_query]
        gallery_filename = filenames[self._num_query:]

        os.makedirs(self._output_dir, exist_ok=True)
        np.save(os.path.join(self._output_dir, 'query_f.npy'), query_features)
        np.save(os.path.join(self._output_dir, 'gallery_f.npy'), gallery_features)
        write_txt(query_filename, os.path.join(self._output_dir, 'query_filename.txt'))
        write_txt(gallery_filename, os.path.join(self._output_dir, 'gallery_filename.txt'))

        return OrderedDict(
            feature_export={
                'query_shape': list(query_features.shape),
                'gallery_shape': list(gallery_features.shape),
                'output_dir': self._output_dir,
            }
        )


class PetIDEvaluator(_PetIDBaseEvaluator):

    # def evaluate(self):
    #     if comm.get_world_size() > 1:
    #         comm.synchronize()
    #         predictions = comm.gather(self._predictions, dst=0)
    #         predictions = list(itertools.chain(*predictions))
    #         if not comm.is_main_process():
    #             return {}
    #     else:
    #         predictions = self._predictions

    #     features = []
    #     pids = []
    #     camids = []
    #     for prediction in predictions:
    #         features.append(prediction['feats'])
    #         pids.extend(prediction['pids'])
    #         camids.extend(prediction['camids'])

    #     features = torch.cat(features, dim=0)
    #     query_features = features[:self._num_query]
    #     gallery_features = features[self._num_query:]

    #     query_filename = pids[:self._num_query]
    #     gallery_filename = pids[self._num_query:]
    #     np.save(os.path.join(self.cfg.OUTPUT_DIR, 'query_f.npy'), query_features)
    #     np.save(os.path.join(self.cfg.OUTPUT_DIR, 'gallery_f.npy'), gallery_features)
    #     if self.cfg.TEST.AQE.ENABLED:
    #         logger.info("Test with AQE setting")
    #         qe_time = self.cfg.TEST.AQE.QE_TIME
    #         qe_k = self.cfg.TEST.AQE.QE_K
    #         alpha = self.cfg.TEST.AQE.ALPHA
    #         query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

    #     dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

    #     if self.cfg.TEST.RERANK.ENABLED:
    #         logger.info("Test with rerank setting")
    #         k1 = self.cfg.TEST.RERANK.K1
    #         k2 = self.cfg.TEST.RERANK.K2
    #         lambda_value = self.cfg.TEST.RERANK.LAMBDA

    #         if self.cfg.TEST.METRIC == "cosine":
    #             query_features = F.normalize(query_features, dim=1)
    #             gallery_features = F.normalize(gallery_features, dim=1)

    #         rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
    #         dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

    #     if self.cfg.TEST.SAVE_DIST.ENABLED:
    #         #save_dist = np.copy(dist).astype(np.float16)
    #         np.save(os.path.join(self.cfg.OUTPUT_DIR, 'dist.npy'), dist)
    #         write_txt(query_filename, os.path.join(self.cfg.OUTPUT_DIR, 'query_filename.txt'))
    #         write_txt(gallery_filename, os.path.join(self.cfg.OUTPUT_DIR, 'gallery_filename.txt'))

    #     submit = pd.read_csv('/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/test/test_data.csv')
        
    #     test_pair = get_test_pairs(submit)

    #     prediction = []
    #     for imageA, imageB in test_pair:
    #         #print(imageA, imageB)
    #         row = query_filename.index(imageA)
    #         column = gallery_filename.index(imageB)
    #         score = (1 - dist[row][column])
    #         prediction.append(score)

    #     submit['prediction'] = prediction
    #     submit.to_csv(os.path.join(os.path.join(self.cfg.OUTPUT_DIR, 'submit.csv')), index=False)

    #     return OrderedDict(submit='finished')


    def evaluate(self):
        predictions = self._gather_predictions()
        if predictions is None:
            return {}

        pids = []
        for prediction in predictions:
            pids.extend(prediction['pids'])

        query_filename = pids[:self._num_query]
        gallery_filename = pids[self._num_query:]

        if len(query_filename) != self._num_query or len(gallery_filename) != self._num_query:
            raise ValueError('Query/gallery filename mapping is incomplete')


        branch_features = []
        for branch in ("s101_224", "s101_256", "s101_288", "s200_224"):
            branch_dir = LEGACY_RUNS_ROOT / branch
            query_path = branch_dir / "query_f.npy"
            gallery_path = branch_dir / "gallery_f.npy"
            if not query_path.is_file() or not gallery_path.is_file():
                raise FileNotFoundError(
                    f"Missing exported features for {branch}: "
                    f"{query_path}, {gallery_path}"
                )
            branch_features.append(
                (
                    torch.from_numpy(np.load(query_path)),
                    torch.from_numpy(np.load(gallery_path)),
                )
            )

       
        query_features = torch.cat([pair[0] for pair in branch_features], dim=1)
        gallery_features = torch.cat([pair[1] for pair in branch_features], dim=1)

        if query_features.shape[0] != len(query_filename) or gallery_features.shape[0] != len(gallery_filename):
            raise ValueError(
                'Feature/filename count mismatch: '
                f'{query_features.shape[0]=}, {len(query_filename)=}, '
                f'{gallery_features.shape[0]=}, {len(gallery_filename)=}'
            )
 
        print(query_features.shape, gallery_features.shape)
        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        submit = pd.read_csv(PROCESSED_DATA_ROOT / "test" / "test_data.csv")
        test_pair = get_test_pairs(submit)

        query_index = {name: idx for idx, name in enumerate(query_filename)}
        gallery_index = {name: idx for idx, name in enumerate(gallery_filename)}
        prediction = []
        for imageA, imageB in test_pair:
            row = query_index[imageA]
            column = gallery_index[imageB]
            score = (1 - dist[row][column]) * 100
            prediction.append(score)

        submit['prediction'] = prediction
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        submit.to_csv(os.path.join(os.path.join(self.cfg.OUTPUT_DIR, 'submit.csv')), index=False)

        return OrderedDict(submit='finished')
