# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import sys

sys.path.append(".")

from fastreid.config import get_cfg

from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer
from fastreid.evaluation import ReidEvaluator
from fastreid.utils import comm
from fastreid.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

from pet_id import (
    PetIDEvaluator,
    PetIDFeatureEvaluator,
    PetIDVerificationEvaluator,
    add_retri_config,
)
from pet_id.latent_hooks import LatentHealthHook
from pet_id.run_manifest import (
    configure_standard_run,
    finalize_run_manifest,
    initialize_run_manifest,
)
from pet_id.workspace_paths import normalize_runtime_config


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if bool(getattr(cfg, "COMPUTED_STANDARD_RUN", False)):
            checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpointer.save_dir = checkpoint_dir

    def build_writers(self):
        if not bool(getattr(self.cfg, "COMPUTED_STANDARD_RUN", False)):
            return super().build_writers()
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(os.path.join(self.cfg.OUTPUT_DIR, "tensorboard")),
        ]

    def build_hooks(self):
        trainer_hooks = super().build_hooks()
        if self.cfg.MODEL.META_ARCHITECTURE in {
            "LatentWorkspaceBaseline",
            "RoleAnchoredLatentWorkspace",
            "SpatialQueryLatentWorkspace",
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
    normalize_runtime_config(cfg)
    configure_standard_run(cfg, args)
    if args.skip_final_eval:
        # The released PetID dataset class reuses the training set for query
        # and gallery with identical camera IDs. FastReID removes every such
        # match, so the forced end-of-training evaluation is not a valid
        # validation protocol and may fail after the last epoch.
        cfg.DATASETS.TESTS = ()
    cfg.freeze()
    default_setup(cfg, args)
    if comm.is_main_process():
        initialize_run_manifest(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    try:
        if args.eval_only:
            cfg.defrost()
            cfg.MODEL.BACKBONE.PRETRAIN = False
            model = Trainer.build_model(cfg)

            Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(
                cfg.MODEL.WEIGHTS
            )  # load trained model

            if args.save_features:
                result = FeatureExtractor.test(cfg, model)
            elif args.commit:
                result = Committer.test(cfg, model)
            else:
                result = Trainer.test(cfg, model)
        else:
            trainer = Trainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            result = trainer.train()
        if comm.is_main_process():
            finalize_run_manifest(cfg, status="completed", result=result)
        return result
    except BaseException as error:
        if comm.is_main_process():
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            finalize_run_manifest(cfg, status=status, error=error)
        raise


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
    parser.add_argument(
        "--run-workstream",
        default="",
        help="opt in to artifacts/runs/<workstream>/<run-id> standard layout",
    )
    parser.add_argument("--run-id", default="", help="explicit standard run id")
    parser.add_argument("--run-purpose", default="", help="purpose token for generated run id")
    parser.add_argument(
        "--allow-checkpoint-cleanup",
        action="store_true",
        help="record that intermediate checkpoints may be reviewed for cleanup",
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
