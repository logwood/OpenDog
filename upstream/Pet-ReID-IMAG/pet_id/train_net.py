# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append(".")

from fastreid.config import get_cfg

from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer
from fastreid.evaluation import ReidEvaluator

from pet_id import (
    PetIDEvaluator,
    PetIDFeatureEvaluator,
    PetIDVerificationEvaluator,
    add_retri_config,
)
from pet_id.latent_hooks import LatentHealthHook


class Trainer(DefaultTrainer):
    def build_hooks(self):
        trainer_hooks = super().build_hooks()
        if self.cfg.MODEL.META_ARCHITECTURE in {
            "LatentWorkspaceBaseline",
            "LatentWorkspaceV2Baseline",
            "LatentWorkspaceV3Baseline",
        }:
            workspace = self.cfg.MODEL.LATENT_WORKSPACE
            health_hook = LatentHealthHook(
                self.model,
                workspace.HEALTH_PERIOD,
                early_abort_enabled=workspace.EARLY_ABORT_ENABLED,
                early_abort_warmup_iters=workspace.EARLY_ABORT_WARMUP_ITERS,
                early_abort_patience=workspace.EARLY_ABORT_PATIENCE,
                slot_cosine_max=workspace.EARLY_ABORT_SLOT_COSINE_MAX,
                min_effective_rank=workspace.EARLY_ABORT_MIN_EFFECTIVE_RANK,
                query_cosine_max=workspace.EARLY_ABORT_QUERY_COSINE_MAX,
                min_query_rank=workspace.EARLY_ABORT_MIN_QUERY_RANK,
            )
            # Metrics must be present before PeriodicWriter flushes this step.
            writer_index = next(
                (
                    index
                    for index, trainer_hook in enumerate(trainer_hooks)
                    if trainer_hook.__class__.__name__ == "PeriodicWriter"
                ),
                len(trainer_hooks),
            )
            trainer_hooks.insert(writer_index, health_hook)
        return trainer_hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        if dataset_name in {"PetIDValidation", "PetIDValidationSmoke"}:
            return data_loader, PetIDVerificationEvaluator(cfg, num_query, output_dir)
        return data_loader, ReidEvaluator(cfg, num_query, output_dir)


class Committer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        return data_loader, PetIDEvaluator(cfg, num_query, output_dir)


class FeatureExtractor(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        return data_loader, PetIDFeatureEvaluator(cfg, num_query, output_dir)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.skip_final_eval:
        # The released PetID dataset class reuses the training set for query
        # and gallery with identical camera IDs. FastReID removes every such
        # match, so the forced end-of-training evaluation is not a valid
        # validation protocol and may fail after the last epoch.
        cfg.DATASETS.TESTS = ()
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Trainer.build_model(cfg)

        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(
            cfg.MODEL.WEIGHTS
        )  # load trained model

        if args.save_features:
            res = FeatureExtractor.test(cfg, model)
        elif args.commit:
            res = Committer.test(cfg, model)
        else:
            res = Trainer.test(cfg, model)

        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--commit", action="store_true", help="submission testing results"
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="export query/gallery .npy features and filename mappings for one model",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="skip the released train-as-query/gallery evaluation after the final epoch",
    )
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
