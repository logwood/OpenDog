#!/usr/bin/env python3
"""Run either the residual control or integrated one-pass identity backbone.

``--integrated-token-fusion`` replaces the outer bounded residual with one
self-attention identity backbone over the protected detail hierarchy, ArcFace,
native nose, detail, and continuous geometry tokens. The training-only
classifier is normalized and scaled so the fused space receives useful identity
gradients immediately.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

import train_unified_highres_structural as trainer
from pet_id.unified_highres_structural_onepass import (
    build_integrated_from_detail_checkpoint,
    build_onepass_from_detail_checkpoint,
)


_onepass_builder = build_onepass_from_detail_checkpoint

_original_identity_batches = trainer.identity_batches
_original_evaluate = trainer.evaluate
_original_run_training = trainer.run_training
_original_render_markdown = trainer.render_markdown
_original_build_optimizer = trainer.build_optimizer

_active_train_targets: tuple[int, ...] | None = None
_active_eval_targets: tuple[int, ...] | None = None


def _render_markdown_with_selection_compatibility(report):
    if "selection" in report:
        return _original_render_markdown(report)
    compatible = dict(report)
    compatible["selection"] = {
        "epoch": int(report["config"]["selected_epoch"]),
    }
    return _original_render_markdown(compatible)


trainer.render_markdown = _render_markdown_with_selection_compatibility


class _ScaledCosineClassifier(torch.nn.Module):
    """Training-only normalized identity head with a learnable score scale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        feature_gradient_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("The scaled cosine identity head does not use bias")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.feature_gradient_scale = float(feature_gradient_scale)
        if not 0.0 < self.feature_gradient_scale <= 1.0:
            raise ValueError("feature_gradient_scale must be in (0, 1]")
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.logit_scale_log = torch.nn.Parameter(
            torch.tensor(math.log(32.0), dtype=torch.float32)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale_log.clamp(math.log(8.0), math.log(64.0)).exp()
        normalized = F.normalize(embedding.float(), dim=1)
        classifier_input = normalized.detach() + self.feature_gradient_scale * (
            normalized - normalized.detach()
        )
        return scale * F.linear(
            classifier_input,
            F.normalize(self.weight.float(), dim=1),
        )


def _enable_scaled_cosine_classifier() -> None:
    # The trainer only uses its local ``nn`` binding to construct and initialize
    # the disposable identity classifier.  A proxy avoids mutating torch.nn.
    trainer.nn = types.SimpleNamespace(
        Linear=_ScaledCosineClassifier,
        init=torch.nn.init,
        Module=torch.nn.Module,
        Parameter=torch.nn.Parameter,
    )


def _build_integrated_optimizer(model, classifier, args):
    core = []
    bridges = []
    nose = []
    other = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "nose_encoder.model" in name:
            nose.append(parameter)
        elif name.startswith("structural_residual."):
            core.append(parameter)
        elif name.startswith(("global_bridge.", "detail_bridge.")):
            bridges.append(parameter)
        else:
            other.append(parameter)
    if other:
        core.extend(other)
    if not core or not bridges or not nose:
        raise RuntimeError(
            "Integrated optimizer expected non-empty core, bridge, and nose groups"
        )
    base_lr = float(args.learning_rate)
    return torch.optim.AdamW(
        [
            {"params": core, "lr": base_lr, "name": "identity_token_core"},
            {
                "params": bridges,
                "lr": 0.5 * base_lr,
                "name": "pretrained_dual_space_bridges",
            },
            {
                "params": nose,
                "lr": float(args.nose_learning_rate),
                "name": "nose_tail",
            },
            {
                "params": list(classifier.parameters()),
                "lr": 5.0 * base_lr,
                "name": "training_identity_classifier",
            },
        ],
        weight_decay=float(args.weight_decay),
    )


class _IdentityBatchView:
    def __init__(self, dataset, targets):
        self.indices_by_target = {
            int(target): dataset.indices_by_target[int(target)]
            for target in targets
            if int(target) in dataset.indices_by_target
        }


class _EvaluationSubset:
    def __init__(self, dataset, targets):
        wanted = {int(target) for target in targets}
        self._dataset = dataset
        self._indices = [
            index for index, target in enumerate(dataset.targets)
            if int(target) in wanted
        ]
        ordered = sorted(wanted)
        target_map = {target: index for index, target in enumerate(ordered)}
        self.targets = [
            target_map[int(dataset.targets[index])] for index in self._indices
        ]

    def load(self, index):
        return self._dataset.load(self._indices[index])


def _identity_batches_with_split(
    dataset,
    *,
    identities_per_batch,
    images_per_identity,
    seed,
    epoch,
):
    if _active_train_targets is None:
        return _original_identity_batches(
            dataset,
            identities_per_batch=identities_per_batch,
            images_per_identity=images_per_identity,
            seed=seed,
            epoch=epoch,
        )
    view = _IdentityBatchView(dataset, _active_train_targets)
    if len(view.indices_by_target) != len(_active_train_targets):
        raise RuntimeError("Training target subset is not present in the dataset")
    return _original_identity_batches(
        view,
        identities_per_batch=identities_per_batch,
        images_per_identity=images_per_identity,
        seed=seed,
        epoch=epoch,
    )


def _evaluate_with_split(
    model,
    dataset,
    *,
    device,
    amp_dtype,
    use_amp,
    gallery_images,
):
    if _active_eval_targets is None:
        return _original_evaluate(
            model,
            dataset,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            gallery_images=gallery_images,
        )
    view = _EvaluationSubset(dataset, _active_eval_targets)
    if len(set(view.targets)) != len(_active_eval_targets):
        raise RuntimeError("Evaluation target subset is not present in the dataset")
    return _original_evaluate(
        model,
        view,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        gallery_images=gallery_images,
    )


def _run_training_with_split(model, dataset, train_targets, *args, **kwargs):
    global _active_train_targets, _active_eval_targets
    previous_train = _active_train_targets
    previous_eval = _active_eval_targets
    _active_train_targets = tuple(int(target) for target in train_targets)
    validation_targets = kwargs.get("validation_targets")
    _active_eval_targets = (
        None
        if validation_targets is None
        else tuple(int(target) for target in validation_targets)
    )
    try:
        return _original_run_training(model, dataset, train_targets, *args, **kwargs)
    finally:
        _active_train_targets = previous_train
        _active_eval_targets = previous_eval


trainer.identity_batches = _identity_batches_with_split
trainer.evaluate = _evaluate_with_split
trainer.run_training = _run_training_with_split


def _active_builder(*args, **kwargs):
    model, payload = _onepass_builder(*args, **kwargs)
    gain_logit = getattr(model.structural_residual, "gain_logit", None)
    if gain_logit is not None:
        # Residual-control compatibility.  The integrated token model has no
        # residual gain or runtime output cap.
        with torch.no_grad():
            gain_logit.fill_(0.2)
    return model, payload


trainer.build_structural_from_detail_checkpoint = _active_builder


def _enable_legacy_800_200_protocol():
    """Use all 800 train identities and the former 200 identities for selection."""
    original_dataset = trainer.LockedDetailDataset
    original_checkpoint_builder = trainer.create_structural_checkpoint
    state = {"development_dataset_replaced": False}

    def legacy_split(targets, *, dev_count, seed):
        if len(targets) != 800:
            raise RuntimeError(
                f"Legacy 800/200 mode expected 800 training identities, got {len(targets)}"
            )
        return list(targets), list(range(200))

    def legacy_dataset(manifest_path, *args, **kwargs):
        path = Path(manifest_path)
        if (
            not kwargs.get("training", False)
            and not state["development_dataset_replaced"]
            and path.name == "train.manifest.json"
        ):
            candidate = path.with_name("validation.manifest.json")
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            manifest_path = candidate
            state["development_dataset_replaced"] = True
        return original_dataset(manifest_path, *args, **kwargs)

    def legacy_checkpoint_builder(*args, **kwargs):
        selection = kwargs.get("selection")
        if selection is not None:
            selection = dict(selection)
            selection["source"] = "locked_200_development_selection"
            selection["locked_validation_used_for_selection"] = True
            kwargs["selection"] = selection
        return original_checkpoint_builder(*args, **kwargs)

    trainer.split_identity_targets = legacy_split
    trainer.LockedDetailDataset = legacy_dataset
    trainer.create_structural_checkpoint = legacy_checkpoint_builder
    print(
        json.dumps(
            {
                "protocol_mode": "legacy_800_train_200_validation",
                "training_identities": 800,
                "selection_identities": 200,
                "locked_validation_is_blind_test": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _mark_legacy_report():
    output_value = None
    for index, value in enumerate(sys.argv):
        if value == "--output-dir" and index + 1 < len(sys.argv):
            output_value = sys.argv[index + 1]
            break
    if output_value is None:
        return
    output_dir = trainer.workspace_path(Path(output_value))
    report_path = output_dir / "report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = report.setdefault("protocol", {})
    protocol["locked_validation_used_for_selection"] = True
    protocol["locked_validation_is_blind_test"] = False
    protocol["selection_protocol"] = "legacy_800_train_200_validation"
    selection = report.setdefault("selection", {})
    selection["source"] = "locked_200_development_selection"
    selection["locked_validation_used_for_selection"] = True
    report["legacy_800_200_mode"] = True
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "report.md"
    if markdown_path.is_file():
        markdown = markdown_path.read_text(encoding="utf-8")
        markdown = markdown.replace(
            "chosen without reading the locked validation partition",
            "chosen on the locked 200-identity development partition",
        )
        markdown_path.write_text(markdown, encoding="utf-8")


def _append_default_argument(option: str, value: str) -> None:
    if option not in sys.argv:
        sys.argv.extend((option, value))


def _enable_integrated_training_defaults() -> None:
    # These are optimization settings, not inference-time branch thresholds.
    # Explicit command-line values always win.
    _append_default_argument("--learning-rate", "5.0e-5")
    _append_default_argument("--nose-learning-rate", "5.0e-6")
    _append_default_argument("--anchor-weight", "0.0")
    _append_default_argument("--distill-weight", "0.02")


def _mark_integrated_report() -> None:
    output_value = None
    for index, value in enumerate(sys.argv):
        if value == "--output-dir" and index + 1 < len(sys.argv):
            output_value = sys.argv[index + 1]
            break
    if output_value is None:
        return
    report_path = trainer.workspace_path(Path(output_value)) / "report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    design = report.setdefault("design", {})
    design.update(
        {
            "fusion_backbone": "end_to_end_structural_identity_transformer",
            "input_to_embedding_single_graph": True,
            "manual_branch_thresholds": False,
            "fixed_expert_weights": False,
            "bounded_output_residual": False,
            "protected_anchor_zero_initialized_residual": False,
            "protected_detail_role": (
                "identity_token_initialization_and_weak_distillation"
            ),
            "training_identity_head": "learnable_scaled_cosine",
            "classifier_feature_gradient_scale": 0.25,
            "differential_learning_rates": True,
        }
    )
    report["integrated_token_fusion"] = True
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    integrated_mode = "--integrated-token-fusion" in sys.argv
    if integrated_mode:
        _onepass_builder = build_integrated_from_detail_checkpoint
        _enable_scaled_cosine_classifier()
        trainer.build_optimizer = _build_integrated_optimizer
        _enable_integrated_training_defaults()
        sys.argv = [
            value for value in sys.argv if value != "--integrated-token-fusion"
        ]
        print(
            json.dumps(
                {
                    "fusion_mode": "end_to_end_structural_identity_transformer",
                    "manual_branch_thresholds": False,
                    "bounded_output_residual": False,
                    "training_identity_head": "learnable_scaled_cosine",
                    "classifier_feature_gradient_scale": 0.25,
                    "optimizer_groups": {
                        "identity_token_core": "1.0x",
                        "pretrained_dual_space_bridges": "0.5x",
                        "nose_tail": "--nose-learning-rate",
                        "training_identity_classifier": "5.0x",
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    legacy_mode = "--legacy-800-200" in sys.argv
    if legacy_mode:
        _enable_legacy_800_200_protocol()
        sys.argv = [value for value in sys.argv if value != "--legacy-800-200"]
    trainer.main()
    if legacy_mode:
        _mark_legacy_report()
    if integrated_mode:
        _mark_integrated_report()
