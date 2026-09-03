#!/usr/bin/env python3
"""Train a monotonic residual controller and evaluate on fresh 3-image dogs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import build_alignment_index  # noqa: E402
from pet_id.evidence_controller import build_evidence_arrays  # noqa: E402
from pet_id.monotonic_evidence_controller import (  # noqa: E402
    DEFAULT_ACTION_COSTS,
    OUTPUT_NAMES,
    SCALAR_FEATURE_NAMES,
    MonotonicResidualController,
    ScalarNormalizer,
    choose_action,
    probabilities_from_logits,
    scalarize_evidence,
)
from pet_id.gallery import build_pipeline  # noqa: E402
from pet_id.gallery_service import MultimodalPipelineEncoder  # noqa: E402
from pet_id.model_profiles import get_runtime_profile  # noqa: E402
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.recognition_agent import AgentFeatureEncoder, MegaDescriptorEncoder  # noqa: E402
from pet_id.release_compatibility import (  # noqa: E402
    baseline_controller_protocol,
    controller_protocol_search_glob,
    historical_model_training_manifests,
    shared_controller_feature_cache,
)


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FORMAL = load_file_module(
    "pet_reid_formal_for_monotonic_controller",
    ROOT / "tools/evaluate_agent_formal_protocol.py",
)
BASELINE_CONTROLLER = load_file_module(
    "pet_reid_baseline_controller_tool",
    ROOT / "tools/train_evaluate_learned_controller.py",
)
BIFOR_PROFILE = get_runtime_profile("research-bifor")
AGENT_PROFILE = get_runtime_profile("research-agent")

CachedFeature = FORMAL.CachedFeature
FeatureCache = FORMAL.FeatureCache
exact_auc = FORMAL.exact_auc
extract_features = FORMAL.extract_features
sha256_file = FORMAL.sha256_file
sha256_json = FORMAL.sha256_json
stable_order = FORMAL.stable_order
write_json = FORMAL.write_json

DEFAULT_OUTPUT = WORKSPACE / "artifacts/runs/controllers/monotonic"
DEFAULT_BASELINE_PROTOCOL = baseline_controller_protocol(WORKSPACE)
DEFAULT_SHARED_CACHE = shared_controller_feature_cache(WORKSPACE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_manifest_identities(
    paths: Iterable[Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    identities: set[str] = set()
    evidence = []
    for path in sorted(set(path.resolve() for path in paths if path.is_file())):
        document = read_json(path)
        records = document.get("records")
        if not isinstance(records, list):
            continue
        current = {str(row["identity"]).casefold() for row in records}
        identities.update(current)
        evidence.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "identities": len(current),
                "records": len(records),
            }
        )
    return identities, evidence


def collect_protocol_identities(
    paths: Iterable[Path], skip: Path
) -> tuple[set[str], list[dict[str, Any]]]:
    identities: set[str] = set()
    evidence = []
    skip = skip.resolve()
    for path in sorted(set(path.resolve() for path in paths if path.is_file())):
        if path == skip:
            continue
        document = read_json(path)
        splits = document.get("splits")
        if not isinstance(splits, dict):
            continue
        current = {
            str(row["identity"]).casefold()
            for rows in splits.values()
            if isinstance(rows, list)
            for row in rows
            if isinstance(row, dict) and "identity" in row
        }
        identities.update(current)
        evidence.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "identities": len(current),
            }
        )
    return identities, evidence


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    baseline_protocol = read_json(args.baseline_protocol)
    baseline_payload = dict(baseline_protocol)
    baseline_hash = baseline_payload.pop("protocol_sha256")
    if sha256_json(baseline_payload) != baseline_hash:
        raise RuntimeError("Baseline controller protocol hash mismatch")
    development = [
        dict(row)
        for name in ("controller_train", "controller_validation")
        for row in baseline_protocol["splits"][name]
    ]
    dev_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in development:
        dev_grouped[row["identity"]].append(row)
    dev_ids = sorted(
        dev_grouped,
        key=lambda identity: stable_order(
            args.seed, "monotonic-development", identity
        ),
    )
    if len(dev_ids) != args.fit_identities + args.calibration_identities:
        raise ValueError(
            f"Expected {args.fit_identities + args.calibration_identities} "
            f"baseline development identities, found {len(dev_ids)}"
        )
    fit_ids = set(dev_ids[: args.fit_identities])
    calibration_ids = set(dev_ids[args.fit_identities :])

    def dev_rows(identities: set[str], origin: str) -> list[dict[str, Any]]:
        rows = []
        for identity in sorted(identities):
            current = sorted(
                dev_grouped[identity],
                key=lambda row: row["identity_record_index"],
            )
            if len(current) != 4:
                raise ValueError(f"Development identity {identity} lacks four images")
            for row in current:
                copied = dict(row)
                copied["origin"] = origin
                rows.append(copied)
        return rows

    alignment_records, alignment_audit = build_alignment_index(args.dataset_root)
    raw_grouped: dict[str, list[Any]] = defaultdict(list)
    for record in alignment_records:
        raw_grouped[record.identity.casefold()].append(record)
    manifest_excluded, manifest_evidence = collect_manifest_identities(
        (WORKSPACE / "artifacts/runs/legacy").rglob("*manifest.json")
    )
    protocol_paths = (WORKSPACE / "artifacts/runs").glob(
        controller_protocol_search_glob()
    )
    protocol_excluded, protocol_evidence = collect_protocol_identities(
        protocol_paths, args.output_dir / "protocol.json"
    )
    excluded = manifest_excluded | protocol_excluded
    eligible = [
        identity
        for identity, rows in raw_grouped.items()
        if identity not in excluded and len(rows) == 3
    ]
    eligible.sort(
        key=lambda identity: stable_order(
            args.seed, "monotonic-fresh-test", identity
        )
    )
    required = args.test_known_identities + args.test_unknown_identities
    if len(eligible) < required:
        raise ValueError(f"Need {required} fresh three-image identities, found {len(eligible)}")
    selected = eligible[:required]
    test_known_ids = selected[: args.test_known_identities]
    test_unknown_ids = selected[args.test_known_identities :]

    def ordered_raw(identity: str) -> list[Any]:
        return sorted(
            raw_grouped[identity],
            key=lambda row: stable_order(
                args.seed,
                "monotonic-fresh-image",
                identity,
                row.canonical_filename,
                row.source_path.name,
            ),
        )

    def raw_record(
        row: Any,
        identity: str,
        index: int,
        origin: str,
        role: str,
    ) -> dict[str, Any]:
        return {
            "identity": identity,
            "display_identity": row.identity,
            "source_path": str(row.source_path.resolve()),
            "canonical_filename": row.canonical_filename,
            "source_sha256": sha256_file(row.source_path),
            "origin": origin,
            "role": role,
            "identity_record_index": index,
        }

    test_known = []
    for identity in test_known_ids:
        for index, row in enumerate(ordered_raw(identity)):
            test_known.append(
                raw_record(
                    row,
                    identity,
                    index,
                    "locked_test_known",
                    "gallery" if index < 2 else "query",
                )
            )
    test_unknown = [
        raw_record(
            ordered_raw(identity)[0],
            identity,
            0,
            "locked_test_unknown",
            "unknown_query",
        )
        for identity in test_unknown_ids
    ]
    splits = {
        "controller_fit": dev_rows(fit_ids, "controller_fit"),
        "controller_calibration": dev_rows(
            calibration_ids, "controller_calibration"
        ),
        "locked_test_known": test_known,
        "locked_test_unknown": test_unknown,
    }
    split_ids = {
        name: {row["identity"] for row in rows}
        for name, rows in splits.items()
    }
    model_train_ids, model_evidence = collect_manifest_identities(
        historical_model_training_manifests(WORKSPACE)
    )
    protocol = {
        "schema_version": 1,
        "name": "monotonic_controller_fresh_three_image_test",
        "created_at": utc_now(),
        "seed": args.seed,
        "policy": {
            "controller_fit": (
                "80 baseline development identities; no baseline test identity"
            ),
            "controller_calibration": (
                "16 identity-disjoint baseline development identities"
            ),
            "locked_test_known": "fresh identities, two gallery plus one query",
            "locked_test_unknown": "fresh identities, one query per identity",
            "recapture_locked_test": False,
            "test_tuning": False,
            "base_models_frozen": True,
        },
        "source": {
            "baseline_controller_protocol": {
                "path": str(args.baseline_protocol.resolve()),
                "sha256": sha256_file(args.baseline_protocol),
                "protocol_sha256": baseline_hash,
            },
            "manifest_exclusions": manifest_evidence,
            "protocol_exclusions": protocol_evidence,
            "model_training_protocols": model_evidence,
        },
        "splits": splits,
        "audit": {
            "alignment_index": alignment_audit,
            "excluded_identities": len(excluded),
            "eligible_exactly_three_images": len(eligible),
            "selected_fresh_test_identities": len(selected),
            "split_identity_counts": {
                name: len(values) for name, values in split_ids.items()
            },
            "split_record_counts": {
                name: len(values) for name, values in splits.items()
            },
            "pairwise_identity_overlap": {
                f"{left}__{right}": len(split_ids[left] & split_ids[right])
                for index, left in enumerate(split_ids)
                for right in list(split_ids)[index + 1 :]
            },
            "model_training_overlap": {
                name: sorted(values & model_train_ids)
                for name, values in split_ids.items()
            },
        },
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    return protocol


def load_or_build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_dir / "protocol.json"
    if path.is_file() and not args.rebuild_protocol:
        protocol = read_json(path)
        payload = dict(protocol)
        expected = payload.pop("protocol_sha256")
        if sha256_json(payload) != expected:
            raise RuntimeError("Monotonic controller protocol hash mismatch")
        return protocol
    protocol = build_protocol(args)
    write_json(path, protocol)
    (args.output_dir / "protocol.sha256").write_text(
        protocol["protocol_sha256"] + "  protocol.json\n", encoding="ascii"
    )
    return protocol


@dataclass
class ScalarExample:
    example_id: str
    episode_id: str
    identity: str
    known: bool
    expert_available: bool
    evidence: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    facts: dict[str, Any]


BASELINE_OUTPUT_INDEX = {
    name: index for index, name in enumerate(BASELINE_CONTROLLER.OUTPUT_NAMES)
}


def convert_baseline_examples(
    rows: Sequence[Any],
) -> list[ScalarExample]:
    output = []
    for row in rows:
        targets = np.asarray(
            [row.targets[BASELINE_OUTPUT_INDEX[name]] for name in OUTPUT_NAMES],
            dtype=np.float32,
        )
        output.append(
            ScalarExample(
                example_id=row.example_id,
                episode_id=row.episode_id,
                identity=row.true_identity,
                known=row.known,
                expert_available=row.expert_available,
                evidence=scalarize_evidence(row.candidates, row.context),
                targets=targets,
                target_mask=np.ones(len(OUTPUT_NAMES), dtype=np.float32),
                facts=dict(row.facts),
            )
        )
    return output


def development_examples(
    records: Sequence[dict[str, Any]],
    features: Mapping[str, Any],
    *,
    folds: int,
    seed: int,
    top_candidates: int,
    split: str,
) -> list[ScalarExample]:
    return convert_baseline_examples(
        BASELINE_CONTROLLER.make_development_examples(
            split=split,
            records=records,
            features=features,
            folds=folds,
            seed=seed,
            top_candidates=top_candidates,
        )
    )


def locked_test_examples(
    protocol: dict[str, Any],
    features: Mapping[str, Any],
    *,
    top_candidates: int,
) -> list[ScalarExample]:
    known_grouped = BASELINE_CONTROLLER.records_by_identity(
        protocol["splits"]["locked_test_known"]
    )
    known_ids = sorted(known_grouped)
    gallery = BASELINE_CONTROLLER.build_gallery(
        known_ids, known_grouped, features
    )
    queries = [
        (known_grouped[identity][2], identity, True)
        for identity in known_ids
    ] + [
        (row, row["identity"], False)
        for row in protocol["splits"]["locked_test_unknown"]
    ]
    output = []
    for query, identity, known in queries:
        feature = features[query["source_sha256"]]
        bifor_scores = gallery.bifor_prototypes @ feature.bifor
        mega_scores = gallery.mega_prototypes @ feature.mega
        bifor_reference_scores = np.einsum(
            "nrd,d->nr", gallery.bifor_references, feature.bifor
        )
        mega_reference_scores = np.einsum(
            "nrd,d->nr", gallery.mega_references, feature.mega
        )
        bifor_index = int(np.argmax(bifor_scores))
        mega_index = int(np.argmax(mega_scores))
        bifor_order = np.argsort(-bifor_scores)
        mega_order = np.argsort(-mega_scores)
        bifor_prediction = gallery.identities[bifor_index]
        mega_prediction = gallery.identities[mega_index]
        bifor_correct = bool(known and bifor_prediction == identity)
        mega_correct = bool(known and mega_prediction == identity)
        stable = bool(
            bifor_index == int(np.argmax(bifor_reference_scores[:, 0]))
            and bifor_index == int(np.argmax(bifor_reference_scores[:, 1]))
        )
        target_map = {
            "bifor_correct": bifor_correct,
            "mega_correct": mega_correct,
            "unknown": not known,
            "expert_gain": mega_correct and not bifor_correct,
            "recapture_gain": False,
            "gallery_stable": stable,
            "temporal_consistency": False,
        }
        target_mask = np.ones(len(OUTPUT_NAMES), dtype=np.float32)
        target_mask[OUTPUT_NAMES.index("recapture_gain")] = 0.0
        target_mask[OUTPUT_NAMES.index("temporal_consistency")] = 0.0
        targets = np.asarray(
            [target_map[name] for name in OUTPUT_NAMES], dtype=np.float32
        )
        facts = {
            "known": known,
            "true_identity": identity,
            "query_sha256": query["source_sha256"],
            "bifor_prediction": bifor_prediction,
            "mega_prediction": mega_prediction,
            "bifor_top1_score": float(bifor_scores[bifor_index]),
            "bifor_margin": float(
                bifor_scores[bifor_index]
                - bifor_scores[int(bifor_order[1])]
            ),
            "mega_top1_score": float(mega_scores[mega_index]),
            "mega_margin": float(
                mega_scores[mega_index] - mega_scores[int(mega_order[1])]
            ),
            "targets": {name: bool(value) for name, value in target_map.items()},
        }
        episode_id = f"locked_test:{'known' if known else 'unknown'}:{identity}"
        for expert_available in (False, True):
            candidates, _, context = build_evidence_arrays(
                bifor_scores=bifor_scores,
                mega_scores=mega_scores,
                bifor_reference_scores=bifor_reference_scores,
                mega_reference_scores=mega_reference_scores,
                bifor_gallery_consistency=gallery.bifor_consistency,
                mega_gallery_consistency=gallery.mega_consistency,
                metadata=feature.metadata,
                expert_available=expert_available,
                top_candidates=top_candidates,
            )
            stage = "post_expert" if expert_available else "pre_expert"
            output.append(
                ScalarExample(
                    example_id=f"{episode_id}:{stage}",
                    episode_id=episode_id,
                    identity=identity,
                    known=known,
                    expert_available=expert_available,
                    evidence=scalarize_evidence(candidates, context),
                    targets=targets,
                    target_mask=target_mask,
                    facts=facts,
                )
            )
    return output


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def fit_model(
    train_rows: Sequence[ScalarExample],
    validation_rows: Sequence[ScalarExample],
    *,
    weight_decay: float,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[
    MonotonicResidualController,
    ScalarNormalizer,
    dict[str, Any],
]:
    torch.manual_seed(args.seed + seed_offset)
    train_x = np.stack([row.evidence for row in train_rows])
    train_y = np.stack([row.targets for row in train_rows])
    train_mask = np.stack([row.target_mask for row in train_rows])
    validation_x = np.stack([row.evidence for row in validation_rows])
    validation_y = np.stack([row.targets for row in validation_rows])
    validation_mask = np.stack([row.target_mask for row in validation_rows])
    normalizer = ScalarNormalizer.fit(train_x)
    train_x = normalizer.normalize(train_x)
    validation_x = normalizer.normalize(validation_x)
    device = torch.device(args.controller_device)
    model = MonotonicResidualController().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=weight_decay,
    )
    positive = (train_y * train_mask).sum(axis=0)
    negative = ((1.0 - train_y) * train_mask).sum(axis=0)
    pos_weight = np.clip(
        negative / np.maximum(positive, 1.0), 1.0, args.max_pos_weight
    ).astype(np.float32)
    pos_weight_tensor = torch.from_numpy(pos_weight).to(device)
    tensors = (
        torch.from_numpy(train_x).to(device),
        torch.from_numpy(train_y).to(device),
        torch.from_numpy(train_mask).to(device),
        torch.from_numpy(validation_x).to(device),
        torch.from_numpy(validation_y).to(device),
        torch.from_numpy(validation_mask).to(device),
    )
    train_tx, train_ty, train_tm, val_tx, val_ty, val_tm = tensors
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_tx.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(
                model(train_tx[batch]),
                train_ty[batch],
                train_tm[batch],
                pos_weight_tensor,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        train_loss = total_loss / len(order)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                masked_bce(model(val_tx), val_ty, val_tm)
            )
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "validation_bce": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No monotonic controller checkpoint was selected")
    model.load_state_dict(best_state)
    return model, normalizer, {
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "best_validation_bce": best_loss,
        "epochs_run": len(history),
        "positive_counts": {
            name: int(positive[index])
            for index, name in enumerate(OUTPUT_NAMES)
        },
        "positive_weights": {
            name: float(pos_weight[index])
            for index, name in enumerate(OUTPUT_NAMES)
        },
        "history": history,
    }


def infer_logits(
    model: MonotonicResidualController,
    normalizer: ScalarNormalizer,
    rows: Sequence[ScalarExample],
    device: torch.device,
) -> np.ndarray:
    values = normalizer.normalize(np.stack([row.evidence for row in rows]))
    model.eval()
    with torch.inference_mode():
        return (
            model(torch.from_numpy(values).to(device))
            .detach()
            .cpu()
            .numpy()
        )


def masked_log_loss(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> float:
    loss = (
        np.maximum(logits, 0.0)
        - logits * targets
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return float((loss * mask).sum() / mask.sum())


def calibrate_temperatures(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    output = np.ones(len(OUTPUT_NAMES), dtype=np.float32)
    candidates = np.geomspace(0.25, 4.0, 121)
    for index in range(len(OUTPUT_NAMES)):
        valid = mask[:, index].astype(bool)
        if not valid.any():
            continue
        current = logits[valid, index]
        labels = targets[valid, index]
        losses = []
        for temperature in candidates:
            scaled = current / temperature
            losses.append(
                float(
                    np.mean(
                        np.maximum(scaled, 0.0)
                        - scaled * labels
                        + np.log1p(np.exp(-np.abs(scaled)))
                    )
                )
            )
        output[index] = float(candidates[int(np.argmin(losses))])
    return output


def probability_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    report = {}
    for index, name in enumerate(OUTPUT_NAMES):
        valid = mask[:, index].astype(bool)
        labels = targets[valid, index].astype(bool)
        scores = probabilities[valid, index]
        positive = scores[labels].tolist()
        negative = scores[~labels].tolist()
        report[name] = {
            "valid": int(valid.sum()),
            "positive": int(labels.sum()),
            "negative": int((~labels).sum()),
            "auroc": (
                exact_auc(positive, negative)
                if positive and negative
                else None
            ),
            "brier": (
                float(np.mean((scores - labels.astype(np.float32)) ** 2))
                if scores.size
                else None
            ),
        }
    return report


def select_weight_decay(
    records: Sequence[dict[str, Any]],
    features: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    grouped = BASELINE_CONTROLLER.records_by_identity(records)
    identities = sorted(
        grouped,
        key=lambda identity: stable_order(
            args.seed, "monotonic-cross-validation", identity
        ),
    )
    identity_folds = [
        set(
            identity
            for index, identity in enumerate(identities)
            if index % args.cv_folds == fold
        )
        for fold in range(args.cv_folds)
    ]
    prepared = []
    for fold, validation_ids in enumerate(identity_folds):
        train_records = [
            row for row in records if row["identity"] not in validation_ids
        ]
        validation_records = [
            row for row in records if row["identity"] in validation_ids
        ]
        prepared.append(
            (
                development_examples(
                    train_records,
                    features,
                    folds=args.episode_folds,
                    seed=args.seed + fold,
                    top_candidates=args.top_candidates,
                    split=f"cv{fold}_train",
                ),
                development_examples(
                    validation_records,
                    features,
                    folds=args.episode_folds,
                    seed=args.seed + fold,
                    top_candidates=args.top_candidates,
                    split=f"cv{fold}_validation",
                ),
            )
        )
    rows = []
    for decay_index, weight_decay in enumerate(args.weight_decays):
        fold_losses = []
        for fold, (train_rows, validation_rows) in enumerate(prepared):
            _, _, fit = fit_model(
                train_rows,
                validation_rows,
                weight_decay=weight_decay,
                args=args,
                seed_offset=100 * decay_index + fold,
            )
            fold_losses.append(fit["best_validation_bce"])
            print(
                f"cv weight_decay={weight_decay:g} fold={fold} "
                f"bce={fold_losses[-1]:.6f}",
                flush=True,
            )
        rows.append(
            {
                "weight_decay": weight_decay,
                "fold_validation_bce": fold_losses,
                "mean_validation_bce": float(np.mean(fold_losses)),
                "std_validation_bce": float(np.std(fold_losses)),
            }
        )
    selected = min(
        rows, key=lambda row: (row["mean_validation_bce"], -row["weight_decay"])
    )
    return float(selected["weight_decay"]), {
        "selection": "minimum mean identity-fold validation BCE",
        "candidates": rows,
        "selected_weight_decay": selected["weight_decay"],
    }


def group_stages(
    rows: Sequence[ScalarExample],
) -> dict[str, dict[bool, ScalarExample]]:
    grouped: dict[str, dict[bool, ScalarExample]] = defaultdict(dict)
    for row in rows:
        grouped[row.episode_id][row.expert_available] = row
    return dict(grouped)


def evaluate_policy(
    rows: Sequence[ScalarExample],
    probabilities: np.ndarray,
    costs: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probability_map = {
        row.example_id: {
            name: float(probabilities[index, head])
            for head, name in enumerate(OUTPUT_NAMES)
        }
        for index, row in enumerate(rows)
    }
    decisions = []
    for episode_id, stages in sorted(group_stages(rows).items()):
        pre = stages[False]
        post = stages[True]
        pre_probabilities = probability_map[pre.example_id]
        pre_action = choose_action(
            pre_probabilities,
            expert_available=False,
            costs=costs,
        )
        consulted = pre_action["action"] == "consult_expert"
        final_probabilities = (
            probability_map[post.example_id]
            if consulted
            else pre_probabilities
        )
        final_action = (
            choose_action(
                final_probabilities,
                expert_available=True,
                costs=costs,
            )
            if consulted
            else pre_action
        )
        action = str(final_action["action"])
        if action == "defer_review":
            correct = True
        elif action == "accept_bifor":
            correct = bool(pre.facts["targets"]["bifor_correct"])
        elif action == "accept_mega":
            correct = bool(pre.facts["targets"]["mega_correct"])
        elif action == "reject_unknown":
            correct = bool(pre.facts["targets"]["unknown"])
        else:
            raise RuntimeError(f"Unexpected final action: {action}")
        decisions.append(
            {
                "episode_id": episode_id,
                "identity": pre.identity,
                "known": pre.known,
                "facts": pre.facts,
                "pre_probabilities": pre_probabilities,
                "post_probabilities": (
                    probability_map[post.example_id] if consulted else None
                ),
                "pre_action": pre_action["action"],
                "consulted": consulted,
                "final_action": action,
                "autonomous": action != "defer_review",
                "correct": correct,
                "utilities": final_action["utilities"],
            }
        )
    known = [row for row in decisions if row["known"]]
    unknown = [row for row in decisions if not row["known"]]
    autonomous = [row for row in decisions if row["autonomous"]]
    raw_ranked = sorted(
        decisions,
        key=lambda row: row["facts"]["bifor_top1_score"],
        reverse=True,
    )
    raw_same_coverage = raw_ranked[: len(autonomous)]
    actions = Counter(row["final_action"] for row in decisions)
    report = {
        "episodes": len(decisions),
        "known_episodes": len(known),
        "unknown_episodes": len(unknown),
        "bifor_known_top1": sum(
            row["facts"]["targets"]["bifor_correct"] for row in known
        )
        / len(known),
        "mega_known_top1": sum(
            row["facts"]["targets"]["mega_correct"] for row in known
        )
        / len(known),
        "autonomous_coverage": len(autonomous) / len(decisions),
        "autonomous_accuracy": (
            sum(row["correct"] for row in autonomous) / len(autonomous)
            if autonomous
            else None
        ),
        "autonomous_errors": sum(not row["correct"] for row in autonomous),
        "review_rate": actions["defer_review"] / len(decisions),
        "consultation_rate": sum(row["consulted"] for row in decisions)
        / len(decisions),
        "unknown_rejection_rate": sum(
            row["final_action"] == "reject_unknown" for row in unknown
        )
        / len(unknown),
        "known_false_rejection_rate": sum(
            row["final_action"] == "reject_unknown" for row in known
        )
        / len(known),
        "overall_accuracy_with_review_assumed_correct": sum(
            row["correct"] for row in decisions
        )
        / len(decisions),
        "final_actions": dict(actions),
        "costs": dict(costs),
        "strong_baselines": {
            "bifor_known_vs_unknown_top1_auroc": exact_auc(
                [row["facts"]["bifor_top1_score"] for row in known],
                [row["facts"]["bifor_top1_score"] for row in unknown],
            ),
            "mega_known_vs_unknown_top1_auroc": exact_auc(
                [row["facts"]["mega_top1_score"] for row in known],
                [row["facts"]["mega_top1_score"] for row in unknown],
            ),
            "learned_unknown_preexpert_auroc": exact_auc(
                [row["pre_probabilities"]["unknown"] for row in unknown],
                [row["pre_probabilities"]["unknown"] for row in known],
            ),
            "raw_bifor_selective_at_learned_coverage": {
                "coverage_count": len(raw_same_coverage),
                "coverage": len(raw_same_coverage) / len(decisions),
                "correct": sum(
                    row["facts"]["targets"]["bifor_correct"]
                    for row in raw_same_coverage
                ),
                "accuracy": (
                    sum(
                        row["facts"]["targets"]["bifor_correct"]
                        for row in raw_same_coverage
                    )
                    / len(raw_same_coverage)
                    if raw_same_coverage
                    else None
                ),
            },
        },
    }
    return report, decisions


class DeployableMonotonicController(nn.Module):
    def __init__(
        self,
        model: MonotonicResidualController,
        normalizer: ScalarNormalizer,
        temperatures: np.ndarray,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(normalizer.mean))
        self.register_buffer("std", torch.from_numpy(normalizer.std))
        self.register_buffer(
            "temperatures", torch.from_numpy(temperatures.astype(np.float32))
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return probabilities_from_logits(
            self.model((evidence - self.mean) / self.std),
            self.temperatures,
        )


def export_onnx(
    model: MonotonicResidualController,
    normalizer: ScalarNormalizer,
    temperatures: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    wrapper = DeployableMonotonicController(
        model.cpu().eval(), normalizer, temperatures
    ).eval()
    sample = torch.zeros(2, len(SCALAR_FEATURE_NAMES), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (sample,),
        path,
        input_names=("evidence",),
        output_names=("probabilities",),
        dynamic_axes={"evidence": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "output_names": list(OUTPUT_NAMES),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "data/raw/DogFaceNet_alignment")
    parser.add_argument(
        "--baseline-protocol",
        type=Path,
        default=DEFAULT_BASELINE_PROTOCOL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_SHARED_CACHE)
    parser.add_argument("--config-file", type=Path, default=BIFOR_PROFILE.config)
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=BIFOR_PROFILE.identity_weights,
    )
    parser.add_argument("--onnx-model", type=Path, default=BIFOR_PROFILE.onnx)
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=BIFOR_PROFILE.body_detector,
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        default=AGENT_PROFILE.expert_checkpoint,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megadescriptor-device")
    parser.add_argument("--controller-device", default="cuda")
    parser.add_argument("--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument("--fit-identities", type=int, default=80)
    parser.add_argument("--calibration-identities", type=int, default=16)
    parser.add_argument("--test-known-identities", type=int, default=80)
    parser.add_argument("--test-unknown-identities", type=int, default=30)
    parser.add_argument("--episode-folds", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2, 1e-1))
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-pos-weight", type=float, default=6.0)
    parser.add_argument("--consult-cost", type=float, default=DEFAULT_ACTION_COSTS["consult_expert"])
    parser.add_argument("--reject-cost", type=float, default=DEFAULT_ACTION_COSTS["reject_unknown"])
    parser.add_argument("--review-cost", type=float, default=DEFAULT_ACTION_COSTS["defer_review"])
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--rebuild-protocol", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--skip-onnx-export", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.baseline_protocol = args.baseline_protocol.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    protocol = load_or_build_protocol(args)
    print(
        json.dumps(
            {
                "protocol": str(args.output_dir / "protocol.json"),
                "protocol_sha256": protocol["protocol_sha256"],
                "audit": protocol["audit"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.protocol_only:
        return

    pipeline = build_pipeline(
        args.config_file.expanduser().resolve(),
        args.identity_weights.expanduser().resolve(),
        args.device,
        backend="onnx-bifor",
        onnx_model=args.onnx_model.expanduser().resolve(),
        onnx_provider=args.onnx_provider,
        onnx_warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
        verify_onnx_source_checkpoint=True,
        body_detector=args.body_detector.expanduser().resolve(),
    )
    encoder = AgentFeatureEncoder(
        MultimodalPipelineEncoder(pipeline),
        [
            MegaDescriptorEncoder(
                args.megadescriptor_checkpoint.expanduser().resolve(),
                device=args.megadescriptor_device or args.device,
            )
        ],
    )
    backend = encoder.backend_info()
    cache = FeatureCache(args.feature_cache.expanduser().resolve())
    try:
        features, extraction = extract_features(
            protocol, encoder, cache, backend
        )
    finally:
        cache.close()

    selected_decay, cross_validation = select_weight_decay(
        protocol["splits"]["controller_fit"], features, args
    )
    fit_examples = development_examples(
        protocol["splits"]["controller_fit"],
        features,
        folds=args.episode_folds,
        seed=args.seed,
        top_candidates=args.top_candidates,
        split="controller_fit",
    )
    calibration_examples = development_examples(
        protocol["splits"]["controller_calibration"],
        features,
        folds=args.episode_folds,
        seed=args.seed,
        top_candidates=args.top_candidates,
        split="controller_calibration",
    )
    started = time.perf_counter()
    model, normalizer, training = fit_model(
        fit_examples,
        calibration_examples,
        weight_decay=selected_decay,
        args=args,
        seed_offset=999,
    )
    training["wall_seconds"] = time.perf_counter() - started
    device = torch.device(args.controller_device)
    calibration_logits = infer_logits(
        model, normalizer, calibration_examples, device
    )
    calibration_targets = np.stack(
        [row.targets for row in calibration_examples]
    )
    calibration_mask = np.stack(
        [row.target_mask for row in calibration_examples]
    )
    temperatures = calibrate_temperatures(
        calibration_logits, calibration_targets, calibration_mask
    )
    calibration_probabilities = 1.0 / (
        1.0 + np.exp(-calibration_logits / temperatures)
    )

    test_examples = locked_test_examples(
        protocol, features, top_candidates=args.top_candidates
    )
    test_logits = infer_logits(model, normalizer, test_examples, device)
    test_targets = np.stack([row.targets for row in test_examples])
    test_mask = np.stack([row.target_mask for row in test_examples])
    test_probabilities = 1.0 / (
        1.0 + np.exp(-test_logits / temperatures)
    )
    costs = {
        **DEFAULT_ACTION_COSTS,
        "consult_expert": args.consult_cost,
        "reject_unknown": args.reject_cost,
        "defer_review": args.review_cost,
    }
    calibration_policy, _ = evaluate_policy(
        calibration_examples, calibration_probabilities, costs
    )
    test_policy, test_decisions = evaluate_policy(
        test_examples, test_probabilities, costs
    )

    checkpoint = args.output_dir / "monotonic_controller.pt"
    torch.save(
        {
            "schema_version": 2,
            "model": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "temperatures": temperatures.tolist(),
            "feature_names": list(SCALAR_FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "costs": costs,
            "selected_weight_decay": selected_decay,
            "protocol_sha256": protocol["protocol_sha256"],
        },
        checkpoint,
    )
    onnx = None
    if not args.skip_onnx_export:
        onnx = export_onnx(
            model,
            normalizer,
            temperatures,
            args.output_dir / "monotonic_controller.onnx",
        )
    write_json(args.output_dir / "test_decisions.json", test_decisions)
    pre_calibration = np.asarray(
        [not row.expert_available for row in calibration_examples],
        dtype=np.bool_,
    )
    pre_test = np.asarray(
        [not row.expert_available for row in test_examples], dtype=np.bool_
    )
    report = {
        "schema_version": 2,
        "created_at": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "method": {
            "name": "monotonic BIFOR calibration plus linear residual",
            "base_models_frozen": True,
            "test_reused_from_baseline": False,
            "recapture_tested": False,
            "handwritten_match_threshold": False,
            "action_rule": "argmax learned success probability minus explicit cost",
            "feature_names": list(SCALAR_FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
        },
        "backend": backend,
        "extraction": extraction,
        "cross_validation": cross_validation,
        "training": training,
        "calibration": {
            "identity_count": args.calibration_identities,
            "temperatures": {
                name: float(temperatures[index])
                for index, name in enumerate(OUTPUT_NAMES)
            },
            "heads_pre_expert": probability_metrics(
                calibration_probabilities[pre_calibration],
                calibration_targets[pre_calibration],
                calibration_mask[pre_calibration],
            ),
            "policy": calibration_policy,
        },
        "locked_test": {
            "heads_pre_expert": probability_metrics(
                test_probabilities[pre_test],
                test_targets[pre_test],
                test_mask[pre_test],
            ),
            "heads_post_expert": probability_metrics(
                test_probabilities[~pre_test],
                test_targets[~pre_test],
                test_mask[~pre_test],
            ),
            "policy": test_policy,
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
            },
            "onnx": onnx,
            "test_decisions": str(
                (args.output_dir / "test_decisions.json").resolve()
            ),
        },
        "limitations": [
            "Fresh locked identities have three images, so recapture and temporal heads are not evaluated.",
            "Action costs are explicit deployment preferences, not visual thresholds.",
            "MegaDescriptor-B-224 weights are non-commercial CC BY-NC 4.0.",
        ],
    }
    write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(report["locked_test"], ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
