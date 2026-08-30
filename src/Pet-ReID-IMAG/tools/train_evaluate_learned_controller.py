#!/usr/bin/env python3
"""Train and evaluate an identity-disjoint learned evidence controller.

This experiment freezes BIFOR and MegaDescriptor.  It creates automatic
counterfactual labels from fixed gallery/query episodes, trains a small
DeepSets controller, calibrates its independent heads on validation identities,
and evaluates once on locked identities that neither base model nor controller
training has seen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import build_alignment_index  # noqa: E402
from pet_id.evidence_controller import (  # noqa: E402
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    DEFAULT_ACTION_COSTS,
    OUTPUT_NAMES,
    EvidenceNet,
    EvidenceNormalizer,
    build_evidence_arrays,
    calibrated_probabilities,
    choose_action,
    learned_judgments,
)
from pet_id.gallery import build_pipeline  # noqa: E402
from pet_id.gallery_service import MultimodalPipelineEncoder, normalize_feature  # noqa: E402
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.recognition_agent import AgentFeatureEncoder, MegaDescriptorEncoder  # noqa: E402


# Import by file rather than through tools.__init__.  The upstream tools package
# imports optional clustering dependencies that are unrelated to this experiment.
_FORMAL_PATH = ROOT / "tools/evaluate_agent_formal_protocol.py"
_FORMAL_SPEC = importlib.util.spec_from_file_location(
    "pet_reid_agent_formal_protocol", _FORMAL_PATH
)
if _FORMAL_SPEC is None or _FORMAL_SPEC.loader is None:
    raise RuntimeError(f"Cannot load formal evaluator: {_FORMAL_PATH}")
_FORMAL = importlib.util.module_from_spec(_FORMAL_SPEC)
sys.modules[_FORMAL_SPEC.name] = _FORMAL
_FORMAL_SPEC.loader.exec_module(_FORMAL)
DEFAULT_BIFOR_PACKAGE = _FORMAL.DEFAULT_BIFOR_PACKAGE
DEFAULT_SEMANTIC_PACKAGE = _FORMAL.DEFAULT_SEMANTIC_PACKAGE
CachedFeature = _FORMAL.CachedFeature
FeatureCache = _FORMAL.FeatureCache
exact_auc = _FORMAL.exact_auc
extract_features = _FORMAL.extract_features
json_default = _FORMAL.json_default
sha256_file = _FORMAL.sha256_file
sha256_json = _FORMAL.sha256_json
stable_order = _FORMAL.stable_order
write_json = _FORMAL.write_json


DEFAULT_OUTPUT = WORKSPACE / "artifacts/runs/agent_v2/learned_controller_v1"
DEFAULT_SHARED_CACHE = (
    WORKSPACE / "artifacts/runs/agent_v1/formal_joint100_20_v1/feature_cache.sqlite3"
)
MEGA_ID = "megadescriptor_b224"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_mean(rows: Sequence[np.ndarray]) -> np.ndarray:
    return normalize_feature(np.mean(np.stack(rows), axis=0), "episode mean")


def collect_manifest_identities(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
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


def collect_protocol_identities(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    identities: set[str] = set()
    evidence = []
    for path in sorted(set(path.resolve() for path in paths if path.is_file())):
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


def build_controller_protocol(args: argparse.Namespace) -> dict[str, Any]:
    alignment_records, alignment_audit = build_alignment_index(args.dataset_root)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in alignment_records:
        grouped[record.identity.casefold()].append(record)

    manifest_paths = list((WORKSPACE / "artifacts/runs/legacy").rglob("*manifest.json"))
    manifest_excluded, manifest_evidence = collect_manifest_identities(manifest_paths)
    protocol_paths = list((WORKSPACE / "artifacts/runs/agent_v1").glob("formal_*/protocol.json"))
    protocol_excluded, protocol_evidence = collect_protocol_identities(protocol_paths)
    excluded = manifest_excluded | protocol_excluded
    eligible = [
        identity
        for identity, rows in grouped.items()
        if identity not in excluded and len(rows) >= args.images_per_identity
    ]
    eligible.sort(key=lambda identity: stable_order(args.seed, "controller-identity", identity))
    required = args.train_identities + args.validation_identities + args.test_identities
    if len(eligible) < required:
        raise ValueError(
            f"Need {required} untouched identities with {args.images_per_identity} images; "
            f"found {len(eligible)}"
        )
    selected = eligible[:required]
    train_ids = selected[: args.train_identities]
    validation_end = args.train_identities + args.validation_identities
    validation_ids = selected[args.train_identities : validation_end]
    test_ids = selected[validation_end:]
    if not 0 < args.test_unknown_identities < len(test_ids):
        raise ValueError("test-unknown-identities must leave at least one known test identity")
    test_known_ids = test_ids[: len(test_ids) - args.test_unknown_identities]
    test_unknown_ids = test_ids[len(test_ids) - args.test_unknown_identities :]

    def records_for(identities: Sequence[str], split: str) -> list[dict[str, Any]]:
        output = []
        for identity in identities:
            ordered = sorted(
                grouped[identity],
                key=lambda row: stable_order(
                    args.seed,
                    "controller-image",
                    identity,
                    row.canonical_filename,
                    row.source_path.name,
                ),
            )[: args.images_per_identity]
            for index, row in enumerate(ordered):
                output.append(
                    {
                        "identity": identity,
                        "display_identity": row.identity,
                        "source_path": str(row.source_path.resolve()),
                        "canonical_filename": row.canonical_filename,
                        "source_sha256": sha256_file(row.source_path),
                        "origin": split,
                        "identity_record_index": index,
                    }
                )
        return output

    splits = {
        "controller_train": records_for(train_ids, "controller_train"),
        "controller_validation": records_for(validation_ids, "controller_validation"),
        "locked_test_known": records_for(test_known_ids, "locked_test_known"),
        "locked_test_unknown": records_for(test_unknown_ids, "locked_test_unknown"),
    }
    model_train_paths = [
        WORKSPACE / "artifacts/runs/legacy/dogfacenet_joint100_protocol_v1/train_manifest.json",
        WORKSPACE / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/train_manifest.json",
    ]
    model_train_ids, model_train_evidence = collect_manifest_identities(model_train_paths)
    identity_sets = {
        name: {row["identity"] for row in rows} for name, rows in splits.items()
    }
    protocol = {
        "schema_version": 1,
        "name": "learned_evidence_controller_identity_disjoint_v1",
        "created_at": utc_now(),
        "seed": args.seed,
        "policy": {
            "identity_unit": "dog identity",
            "images_per_identity": args.images_per_identity,
            "gallery_images_per_known_identity": 2,
            "queries_per_identity": 2,
            "controller_train_panels": args.episode_folds,
            "controller_validation_panels": args.episode_folds,
            "locked_test_tuning": False,
            "base_models_frozen": True,
            "historical_protocols_excluded": True,
        },
        "source": {
            "dataset_root": str(args.dataset_root.resolve()),
            "manifest_exclusions": manifest_evidence,
            "agent_protocol_exclusions": protocol_evidence,
            "model_training_protocols": model_train_evidence,
        },
        "splits": splits,
        "audit": {
            "alignment_index": alignment_audit,
            "excluded_identities": len(excluded),
            "eligible_identities": len(eligible),
            "selected_identities": len(selected),
            "split_identity_counts": {name: len(values) for name, values in identity_sets.items()},
            "split_record_counts": {name: len(values) for name, values in splits.items()},
            "pairwise_identity_overlap": {
                f"{left}__{right}": len(identity_sets[left] & identity_sets[right])
                for index, left in enumerate(identity_sets)
                for right in list(identity_sets)[index + 1 :]
            },
            "model_training_overlap": {
                name: sorted(values & model_train_ids) for name, values in identity_sets.items()
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
        actual = sha256_json(payload)
        if expected != actual:
            raise RuntimeError(f"Protocol hash mismatch: expected {expected}, got {actual}")
        return protocol
    protocol = build_controller_protocol(args)
    write_json(path, protocol)
    (args.output_dir / "protocol.sha256").write_text(
        protocol["protocol_sha256"] + "  protocol.json\n", encoding="ascii"
    )
    return protocol


@dataclass(frozen=True)
class GalleryState:
    identities: tuple[str, ...]
    bifor_references: np.ndarray
    mega_references: np.ndarray
    bifor_prototypes: np.ndarray
    mega_prototypes: np.ndarray
    bifor_consistency: np.ndarray
    mega_consistency: np.ndarray


@dataclass
class EvidenceExample:
    example_id: str
    episode_id: str
    split: str
    expert_available: bool
    known: bool
    true_identity: str
    current_sha256: str
    recapture_sha256: str
    candidates: np.ndarray
    mask: np.ndarray
    context: np.ndarray
    targets: np.ndarray
    facts: dict[str, Any]


def records_by_identity(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["identity"]].append(row)
    for identity in grouped:
        grouped[identity].sort(key=lambda row: row["identity_record_index"])
    return dict(grouped)


def build_gallery(
    identities: Sequence[str],
    grouped: Mapping[str, Sequence[dict[str, Any]]],
    features: Mapping[str, CachedFeature],
) -> GalleryState:
    ordered = tuple(sorted(identities))
    bifor_references = np.stack(
        [
            np.stack([features[row["source_sha256"]].bifor for row in grouped[identity][:2]])
            for identity in ordered
        ]
    )
    mega_references = np.stack(
        [
            np.stack([features[row["source_sha256"]].mega for row in grouped[identity][:2]])
            for identity in ordered
        ]
    )
    bifor_prototypes = np.stack(
        [normalized_mean(list(values)) for values in bifor_references]
    )
    mega_prototypes = np.stack(
        [normalized_mean(list(values)) for values in mega_references]
    )
    return GalleryState(
        identities=ordered,
        bifor_references=bifor_references,
        mega_references=mega_references,
        bifor_prototypes=bifor_prototypes,
        mega_prototypes=mega_prototypes,
        bifor_consistency=np.sum(bifor_references[:, 0] * bifor_references[:, 1], axis=1),
        mega_consistency=np.sum(mega_references[:, 0] * mega_references[:, 1], axis=1),
    )


def make_episode_examples(
    *,
    episode_id: str,
    split: str,
    current: dict[str, Any],
    recapture: dict[str, Any],
    true_identity: str,
    known: bool,
    gallery: GalleryState,
    features: Mapping[str, CachedFeature],
    top_candidates: int,
) -> list[EvidenceExample]:
    current_feature = features[current["source_sha256"]]
    recapture_feature = features[recapture["source_sha256"]]
    bifor_scores = gallery.bifor_prototypes @ current_feature.bifor
    mega_scores = gallery.mega_prototypes @ current_feature.mega
    bifor_reference_scores = np.einsum(
        "nrd,d->nr", gallery.bifor_references, current_feature.bifor
    )
    mega_reference_scores = np.einsum(
        "nrd,d->nr", gallery.mega_references, current_feature.mega
    )
    bifor_index = int(np.argmax(bifor_scores))
    mega_index = int(np.argmax(mega_scores))
    bifor_prediction = gallery.identities[bifor_index]
    mega_prediction = gallery.identities[mega_index]
    bifor_order = np.argsort(-bifor_scores)
    mega_order = np.argsort(-mega_scores)
    bifor_runner_up = int(bifor_order[1]) if len(bifor_order) > 1 else bifor_index
    mega_runner_up = int(mega_order[1]) if len(mega_order) > 1 else mega_index
    bifor_correct = bool(known and bifor_prediction == true_identity)
    mega_correct = bool(known and mega_prediction == true_identity)

    recapture_bifor = normalized_mean([current_feature.bifor, recapture_feature.bifor])
    recapture_scores = gallery.bifor_prototypes @ recapture_bifor
    recapture_prediction = gallery.identities[int(np.argmax(recapture_scores))]
    recapture_correct = bool(known and recapture_prediction == true_identity)
    recapture_single_scores = gallery.bifor_prototypes @ recapture_feature.bifor
    recapture_single_prediction = gallery.identities[int(np.argmax(recapture_single_scores))]
    gallery_stable = bool(
        bifor_index == int(np.argmax(bifor_reference_scores[:, 0]))
        and bifor_index == int(np.argmax(bifor_reference_scores[:, 1]))
    )
    targets_by_name = {
        "bifor_correct": bifor_correct,
        "mega_correct": mega_correct,
        "consult_success": bifor_correct or mega_correct,
        "recapture_correct": recapture_correct,
        "unknown": not known,
        "gallery_stable": gallery_stable,
        "expert_gain": mega_correct and not bifor_correct,
        "recapture_gain": recapture_correct and not bifor_correct,
        "temporal_consistency": bifor_prediction == recapture_single_prediction,
    }
    targets = np.asarray([targets_by_name[name] for name in OUTPUT_NAMES], dtype=np.float32)
    facts = {
        "known": known,
        "true_identity": true_identity,
        "bifor_prediction": bifor_prediction,
        "bifor_top1_score": float(bifor_scores[bifor_index]),
        "bifor_margin": float(
            bifor_scores[bifor_index] - bifor_scores[bifor_runner_up]
        ),
        "mega_prediction": mega_prediction,
        "mega_top1_score": float(mega_scores[mega_index]),
        "mega_margin": float(mega_scores[mega_index] - mega_scores[mega_runner_up]),
        "recapture_prediction": recapture_prediction,
        "recapture_top1_score": float(np.max(recapture_scores)),
        "recapture_single_prediction": recapture_single_prediction,
        "targets": {key: bool(value) for key, value in targets_by_name.items()},
    }
    output = []
    for expert_available in (False, True):
        candidates, mask, context = build_evidence_arrays(
            bifor_scores=bifor_scores,
            mega_scores=mega_scores,
            bifor_reference_scores=bifor_reference_scores,
            mega_reference_scores=mega_reference_scores,
            bifor_gallery_consistency=gallery.bifor_consistency,
            mega_gallery_consistency=gallery.mega_consistency,
            metadata=current_feature.metadata,
            expert_available=expert_available,
            top_candidates=top_candidates,
        )
        stage = "post_expert" if expert_available else "pre_expert"
        output.append(
            EvidenceExample(
                example_id=f"{episode_id}:{stage}",
                episode_id=episode_id,
                split=split,
                expert_available=expert_available,
                known=known,
                true_identity=true_identity,
                current_sha256=current["source_sha256"],
                recapture_sha256=recapture["source_sha256"],
                candidates=candidates,
                mask=mask,
                context=context,
                targets=targets,
                facts=facts,
            )
        )
    return output


def make_development_examples(
    *,
    split: str,
    records: Sequence[dict[str, Any]],
    features: Mapping[str, CachedFeature],
    folds: int,
    seed: int,
    top_candidates: int,
) -> list[EvidenceExample]:
    grouped = records_by_identity(records)
    identities = sorted(
        grouped, key=lambda identity: stable_order(seed, split, "panel", identity)
    )
    examples = []
    for fold in range(folds):
        unknown = {identity for index, identity in enumerate(identities) if index % folds == fold}
        known = [identity for identity in identities if identity not in unknown]
        if not known or not unknown:
            raise ValueError(f"Fold {fold} in {split} has no known or unknown identity")
        gallery = build_gallery(known, grouped, features)
        for identity in known:
            rows = grouped[identity]
            for direction, (current_index, recapture_index) in enumerate(((2, 3), (3, 2))):
                episode_id = f"{split}:fold{fold}:known:{identity}:{direction}"
                examples.extend(
                    make_episode_examples(
                        episode_id=episode_id,
                        split=split,
                        current=rows[current_index],
                        recapture=rows[recapture_index],
                        true_identity=identity,
                        known=True,
                        gallery=gallery,
                        features=features,
                        top_candidates=top_candidates,
                    )
                )
        for identity in sorted(unknown):
            rows = grouped[identity]
            for direction, (current_index, recapture_index) in enumerate(((0, 1), (2, 3))):
                episode_id = f"{split}:fold{fold}:unknown:{identity}:{direction}"
                examples.extend(
                    make_episode_examples(
                        episode_id=episode_id,
                        split=split,
                        current=rows[current_index],
                        recapture=rows[recapture_index],
                        true_identity=identity,
                        known=False,
                        gallery=gallery,
                        features=features,
                        top_candidates=top_candidates,
                    )
                )
    return examples


def make_locked_test_examples(
    *,
    known_records: Sequence[dict[str, Any]],
    unknown_records: Sequence[dict[str, Any]],
    features: Mapping[str, CachedFeature],
    top_candidates: int,
) -> list[EvidenceExample]:
    known_grouped = records_by_identity(known_records)
    unknown_grouped = records_by_identity(unknown_records)
    gallery = build_gallery(sorted(known_grouped), known_grouped, features)
    examples = []
    for identity, rows in sorted(known_grouped.items()):
        for direction, (current_index, recapture_index) in enumerate(((2, 3), (3, 2))):
            examples.extend(
                make_episode_examples(
                    episode_id=f"locked_test:known:{identity}:{direction}",
                    split="locked_test",
                    current=rows[current_index],
                    recapture=rows[recapture_index],
                    true_identity=identity,
                    known=True,
                    gallery=gallery,
                    features=features,
                    top_candidates=top_candidates,
                )
            )
    for identity, rows in sorted(unknown_grouped.items()):
        for direction, (current_index, recapture_index) in enumerate(((0, 1), (2, 3))):
            examples.extend(
                make_episode_examples(
                    episode_id=f"locked_test:unknown:{identity}:{direction}",
                    split="locked_test",
                    current=rows[current_index],
                    recapture=rows[recapture_index],
                    true_identity=identity,
                    known=False,
                    gallery=gallery,
                    features=features,
                    top_candidates=top_candidates,
                )
            )
    return examples


class ExampleDataset(Dataset):
    def __init__(self, examples: Sequence[EvidenceExample], normalizer: EvidenceNormalizer):
        self.rows = []
        for example in examples:
            candidates, context = normalizer.normalize(example.candidates, example.context)
            self.rows.append((candidates.astype(np.float32), example.mask, context.astype(np.float32), example.targets))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate_examples(rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    max_candidates = max(row[0].shape[0] for row in rows)
    candidates = np.zeros((len(rows), max_candidates, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros((len(rows), max_candidates), dtype=np.bool_)
    contexts = np.stack([row[2] for row in rows])
    targets = np.stack([row[3] for row in rows])
    for index, (values, mask, _, _) in enumerate(rows):
        candidates[index, : values.shape[0]] = values
        masks[index, : values.shape[0]] = mask
    return (
        torch.from_numpy(candidates),
        torch.from_numpy(masks),
        torch.from_numpy(contexts),
        torch.from_numpy(targets),
    )


def loss_for(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def infer_logits(
    model: EvidenceNet,
    examples: Sequence[EvidenceExample],
    normalizer: EvidenceNormalizer,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        ExampleDataset(examples, normalizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    output = []
    model.eval()
    with torch.inference_mode():
        for candidates, mask, context, _ in loader:
            logits = model(candidates.to(device), mask.to(device), context.to(device))
            output.append(logits.cpu().numpy())
    return np.concatenate(output, axis=0)


def train_controller(
    train_examples: Sequence[EvidenceExample],
    validation_examples: Sequence[EvidenceExample],
    args: argparse.Namespace,
) -> tuple[EvidenceNet, EvidenceNormalizer, dict[str, Any]]:
    normalizer = EvidenceNormalizer.fit(
        [row.candidates for row in train_examples], [row.context for row in train_examples]
    )
    device = torch.device(args.controller_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Controller requested CUDA but CUDA is unavailable")
    model = EvidenceNet(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_targets = np.stack([row.targets for row in train_examples])
    positive = train_targets.sum(axis=0)
    negative = train_targets.shape[0] - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1.0), 0.25, 8.0)
    pos_weight_tensor = torch.from_numpy(pos_weight.astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ExampleDataset(train_examples, normalizer),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_examples,
    )
    validation_loader = DataLoader(
        ExampleDataset(validation_examples, normalizer),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for candidates, mask, context, targets in train_loader:
            candidates = candidates.to(device)
            mask = mask.to(device)
            context = context.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(candidates, mask, context)
            loss = loss_for(logits, targets, pos_weight_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * targets.shape[0]
            train_count += targets.shape[0]
        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.inference_mode():
            for candidates, mask, context, targets in validation_loader:
                targets = targets.to(device)
                logits = model(candidates.to(device), mask.to(device), context.to(device))
                loss = loss_for(logits, targets)
                validation_loss += float(loss) * targets.shape[0]
                validation_count += targets.shape[0]
        train_loss /= train_count
        validation_loss /= validation_count
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_bce={train_loss:.6f} val_bce={validation_loss:.6f}",
                flush=True,
            )
        history.append({"epoch": epoch, "train_bce": train_loss, "validation_bce": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Controller training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, normalizer, {
        "device": str(device),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "positive_counts": {name: int(positive[index]) for index, name in enumerate(OUTPUT_NAMES)},
        "positive_weights": {name: float(pos_weight[index]) for index, name in enumerate(OUTPUT_NAMES)},
        "best_epoch": best_epoch,
        "best_validation_bce": best_loss,
        "epochs_run": len(history),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
    }


def calibrate_temperatures(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    temperatures = np.ones(len(OUTPUT_NAMES), dtype=np.float32)
    candidates = np.geomspace(0.25, 4.0, 121)
    for index in range(len(OUTPUT_NAMES)):
        labels = targets[:, index]
        best = (float("inf"), 1.0)
        for temperature in candidates:
            scaled = logits[:, index] / temperature
            loss = np.mean(np.maximum(scaled, 0.0) - scaled * labels + np.log1p(np.exp(-np.abs(scaled))))
            if loss < best[0]:
                best = (float(loss), float(temperature))
        temperatures[index] = best[1]
    return temperatures


def probability_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    report = {}
    epsilon = 1e-7
    for index, name in enumerate(OUTPUT_NAMES):
        scores = probabilities[:, index]
        labels = targets[:, index].astype(bool)
        positive = scores[labels].tolist()
        negative = scores[~labels].tolist()
        report[name] = {
            "positive": int(labels.sum()),
            "negative": int((~labels).sum()),
            "auroc": exact_auc(positive, negative) if positive and negative else None,
            "brier": float(np.mean((scores - labels.astype(np.float32)) ** 2)),
            "log_loss": float(
                -np.mean(
                    labels * np.log(np.clip(scores, epsilon, 1.0))
                    + (~labels) * np.log(np.clip(1.0 - scores, epsilon, 1.0))
                )
            ),
        }
    return report


def examples_by_episode(examples: Sequence[EvidenceExample]) -> dict[str, dict[bool, EvidenceExample]]:
    grouped: dict[str, dict[bool, EvidenceExample]] = defaultdict(dict)
    for example in examples:
        grouped[example.episode_id][example.expert_available] = example
    if any(set(stages) != {False, True} for stages in grouped.values()):
        raise ValueError("Every episode must contain pre- and post-expert examples")
    return dict(grouped)


def evaluate_policy(
    examples: Sequence[EvidenceExample],
    probabilities: np.ndarray,
    costs: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probability_rows = {
        example.example_id: {name: float(probabilities[index, head]) for head, name in enumerate(OUTPUT_NAMES)}
        for index, example in enumerate(examples)
    }
    grouped = examples_by_episode(examples)
    decisions = []
    for episode_id, stages in sorted(grouped.items()):
        pre = stages[False]
        post = stages[True]
        pre_probabilities = probability_rows[pre.example_id]
        pre_decision = choose_action(pre_probabilities, expert_available=False, costs=costs)
        final_example = pre
        final_probabilities = pre_probabilities
        final_decision = pre_decision
        consulted = pre_decision["action"] == "consult_expert"
        if consulted:
            final_example = post
            final_probabilities = probability_rows[post.example_id]
            final_decision = choose_action(final_probabilities, expert_available=True, costs=costs)
        action = str(final_decision["action"])
        target_name = {
            "accept_bifor": "bifor_correct",
            "accept_mega": "mega_correct",
            "recapture": "recapture_correct",
            "reject_unknown": "unknown",
        }.get(action)
        correct = True if action == "defer_review" else bool(final_example.facts["targets"][target_name])
        autonomous = action != "defer_review"
        decisions.append(
            {
                "episode_id": episode_id,
                "known": pre.known,
                "true_identity": pre.true_identity,
                "current_sha256": pre.current_sha256,
                "recapture_sha256": pre.recapture_sha256,
                "pre_action": pre_decision["action"],
                "consulted": consulted,
                "final_action": action,
                "autonomous": autonomous,
                "correct": correct,
                "facts": pre.facts,
                "pre_probabilities": pre_probabilities,
                "post_probabilities": probability_rows[post.example_id] if consulted else None,
                "learned_judgments": learned_judgments(
                    final_probabilities, expert_available=consulted
                ),
                "utilities": final_decision["utilities"],
            }
        )
    total = len(decisions)
    autonomous_rows = [row for row in decisions if row["autonomous"]]
    known_rows = [row for row in decisions if row["known"]]
    unknown_rows = [row for row in decisions if not row["known"]]
    known_bifor_correct = sum(row["facts"]["targets"]["bifor_correct"] for row in known_rows)
    actions = Counter(row["final_action"] for row in decisions)
    pre_actions = Counter(row["pre_action"] for row in decisions)
    raw_bifor_ranked = sorted(
        decisions,
        key=lambda row: row["facts"]["bifor_top1_score"],
        reverse=True,
    )
    raw_at_same_coverage = raw_bifor_ranked[: len(autonomous_rows)]
    raw_correct_positive = [
        row["facts"]["bifor_top1_score"]
        for row in decisions
        if row["facts"]["targets"]["bifor_correct"]
    ]
    raw_correct_negative = [
        row["facts"]["bifor_top1_score"]
        for row in decisions
        if not row["facts"]["targets"]["bifor_correct"]
    ]
    raw_margin_positive = [
        row["facts"]["bifor_margin"]
        for row in decisions
        if row["facts"]["targets"]["bifor_correct"]
    ]
    raw_margin_negative = [
        row["facts"]["bifor_margin"]
        for row in decisions
        if not row["facts"]["targets"]["bifor_correct"]
    ]
    learned_correct_positive = [
        row["pre_probabilities"]["bifor_correct"]
        for row in decisions
        if row["facts"]["targets"]["bifor_correct"]
    ]
    learned_correct_negative = [
        row["pre_probabilities"]["bifor_correct"]
        for row in decisions
        if not row["facts"]["targets"]["bifor_correct"]
    ]
    learned_unknown_positive = [
        row["pre_probabilities"]["unknown"] for row in unknown_rows
    ]
    learned_unknown_negative = [
        row["pre_probabilities"]["unknown"] for row in known_rows
    ]
    report = {
        "episodes": total,
        "known_episodes": len(known_rows),
        "unknown_episodes": len(unknown_rows),
        "bifor_known_top1": known_bifor_correct / len(known_rows),
        "bifor_known_correct": known_bifor_correct,
        "autonomous_coverage": len(autonomous_rows) / total,
        "autonomous_accuracy": (
            sum(row["correct"] for row in autonomous_rows) / len(autonomous_rows)
            if autonomous_rows
            else None
        ),
        "autonomous_errors": sum(not row["correct"] for row in autonomous_rows),
        "overall_accuracy_with_review_assumed_correct": sum(row["correct"] for row in decisions) / total,
        "known_episode_success": sum(row["correct"] for row in known_rows) / len(known_rows),
        "unknown_episode_success": sum(row["correct"] for row in unknown_rows) / len(unknown_rows),
        "consultation_rate": sum(row["consulted"] for row in decisions) / total,
        "recapture_rate": actions["recapture"] / total,
        "review_rate": actions["defer_review"] / total,
        "unknown_rejection_rate": (
            sum(row["final_action"] == "reject_unknown" for row in unknown_rows) / len(unknown_rows)
        ),
        "known_false_rejection_rate": (
            sum(row["final_action"] == "reject_unknown" for row in known_rows) / len(known_rows)
        ),
        "strong_baselines": {
            "bifor_known_vs_unknown_top1_score_auroc": exact_auc(
                [row["facts"]["bifor_top1_score"] for row in known_rows],
                [row["facts"]["bifor_top1_score"] for row in unknown_rows],
            ),
            "mega_known_vs_unknown_top1_score_auroc": exact_auc(
                [row["facts"]["mega_top1_score"] for row in known_rows],
                [row["facts"]["mega_top1_score"] for row in unknown_rows],
            ),
            "learned_unknown_preexpert_auroc": exact_auc(
                learned_unknown_positive, learned_unknown_negative
            ),
            "bifor_top1_score_auc_for_bifor_correct": exact_auc(
                raw_correct_positive, raw_correct_negative
            ),
            "bifor_margin_auc_for_bifor_correct": exact_auc(
                raw_margin_positive, raw_margin_negative
            ),
            "learned_preexpert_auc_for_bifor_correct": exact_auc(
                learned_correct_positive, learned_correct_negative
            ),
            "raw_bifor_selective_at_learned_coverage": {
                "coverage_count": len(raw_at_same_coverage),
                "coverage": len(raw_at_same_coverage) / total,
                "correct": sum(
                    row["facts"]["targets"]["bifor_correct"]
                    for row in raw_at_same_coverage
                ),
                "accuracy": (
                    sum(
                        row["facts"]["targets"]["bifor_correct"]
                        for row in raw_at_same_coverage
                    )
                    / len(raw_at_same_coverage)
                    if raw_at_same_coverage
                    else None
                ),
            },
        },
        "pre_actions": dict(pre_actions),
        "final_actions": dict(actions),
        "costs": dict(costs),
    }
    return report, decisions


class DeployableController(nn.Module):
    def __init__(
        self,
        model: EvidenceNet,
        normalizer: EvidenceNormalizer,
        temperatures: np.ndarray,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("candidate_mean", torch.from_numpy(normalizer.candidate_mean))
        self.register_buffer("candidate_std", torch.from_numpy(normalizer.candidate_std))
        self.register_buffer("context_mean", torch.from_numpy(normalizer.context_mean))
        self.register_buffer("context_std", torch.from_numpy(normalizer.context_std))
        self.register_buffer("temperatures", torch.from_numpy(temperatures.astype(np.float32)))

    def forward(self, candidates: torch.Tensor, candidate_mask: torch.Tensor, context: torch.Tensor):
        candidates = (candidates - self.candidate_mean) / self.candidate_std
        context = (context - self.context_mean) / self.context_std
        return calibrated_probabilities(
            self.model(candidates, candidate_mask, context), self.temperatures
        )


def export_onnx(
    model: EvidenceNet,
    normalizer: EvidenceNormalizer,
    temperatures: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    wrapper = DeployableController(model.cpu().eval(), normalizer, temperatures).eval()
    candidates = torch.zeros(1, 5, len(CANDIDATE_FEATURE_NAMES), dtype=torch.float32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    context = torch.zeros(1, len(CONTEXT_FEATURE_NAMES), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (candidates, mask, context),
        path,
        input_names=("candidates", "candidate_mask", "context"),
        output_names=("probabilities",),
        dynamic_axes={
            "candidates": {0: "batch", 1: "candidate_count"},
            "candidate_mask": {0: "batch", 1: "candidate_count"},
            "context": {0: "batch"},
            "probabilities": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "outputs": list(OUTPUT_NAMES)}


def markdown_summary(report: dict[str, Any]) -> str:
    locked = report["locked_test"]["policy"]
    heads = report["locked_test"]["heads"]
    lines = [
        "# Learned Evidence Controller V1",
        "",
        f"- Protocol SHA-256: `{report['protocol_sha256']}`",
        f"- Locked test episodes: {locked['episodes']} "
        f"({locked['known_episodes']} known / {locked['unknown_episodes']} unknown)",
        f"- BIFOR known Top-1: {locked['bifor_known_top1']:.2%}",
        f"- Autonomous coverage: {locked['autonomous_coverage']:.2%}",
        f"- Autonomous accuracy: {locked['autonomous_accuracy']:.2%}" if locked["autonomous_accuracy"] is not None else "- Autonomous accuracy: n/a",
        f"- Review rate: {locked['review_rate']:.2%}",
        f"- Consultation rate: {locked['consultation_rate']:.2%}",
        f"- Unknown rejection rate: {locked['unknown_rejection_rate']:.2%}",
        f"- Known false rejection rate: {locked['known_false_rejection_rate']:.2%}",
        "",
        "| Learned head | AUROC | Brier |",
        "|---|---:|---:|",
    ]
    for name in OUTPUT_NAMES:
        row = heads[name]
        auc = "n/a" if row["auroc"] is None else f"{row['auroc']:.4f}"
        lines.append(f"| {name} | {auc} | {row['brier']:.4f} |")
    lines.extend(
        [
            "",
            "The action is the argmax of learned expected success minus explicit business cost; no hand-written match threshold is used.",
            "The historical unseen57 protocol was not used for controller training, calibration, or locked testing.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "data/raw/DogFaceNet_alignment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_SHARED_CACHE)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_BIFOR_PACKAGE / "config.yaml")
    parser.add_argument("--identity-weights", type=Path, default=DEFAULT_SEMANTIC_PACKAGE / "model_final.pth")
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_BIFOR_PACKAGE / "onnx/pet_embedding.onnx")
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=WORKSPACE / "models/pretrained/body_detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/megadescriptor/MegaDescriptor-B-224/pytorch_model.bin",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megadescriptor-device")
    parser.add_argument("--controller-device", default="cuda")
    parser.add_argument("--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument("--train-identities", type=int, default=72)
    parser.add_argument("--validation-identities", type=int, default=24)
    parser.add_argument("--test-identities", type=int, default=38)
    parser.add_argument("--test-unknown-identities", type=int, default=10)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--episode-folds", type=int, default=4)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--consult-cost", type=float, default=DEFAULT_ACTION_COSTS["consult_expert"])
    parser.add_argument("--recapture-cost", type=float, default=DEFAULT_ACTION_COSTS["recapture"])
    parser.add_argument("--reject-cost", type=float, default=DEFAULT_ACTION_COSTS["reject_unknown"])
    parser.add_argument("--review-cost", type=float, default=DEFAULT_ACTION_COSTS["defer_review"])
    parser.add_argument("--rebuild-protocol", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--skip-onnx-export", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args()
    if args.images_per_identity != 4:
        raise ValueError("V1 episode construction requires exactly four images per identity")
    if args.episode_folds < 2:
        raise ValueError("episode-folds must be at least two")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
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
    backend_info = encoder.backend_info()
    cache = FeatureCache(args.feature_cache.expanduser().resolve())
    try:
        features, extraction = extract_features(protocol, encoder, cache, backend_info)
    finally:
        cache.close()

    train_examples = make_development_examples(
        split="controller_train",
        records=protocol["splits"]["controller_train"],
        features=features,
        folds=args.episode_folds,
        seed=args.seed,
        top_candidates=args.top_candidates,
    )
    validation_examples = make_development_examples(
        split="controller_validation",
        records=protocol["splits"]["controller_validation"],
        features=features,
        folds=args.episode_folds,
        seed=args.seed,
        top_candidates=args.top_candidates,
    )
    test_examples = make_locked_test_examples(
        known_records=protocol["splits"]["locked_test_known"],
        unknown_records=protocol["splits"]["locked_test_unknown"],
        features=features,
        top_candidates=args.top_candidates,
    )
    model, normalizer, training = train_controller(train_examples, validation_examples, args)
    controller_device = torch.device(args.controller_device)
    validation_logits = infer_logits(
        model, validation_examples, normalizer, batch_size=args.batch_size, device=controller_device
    )
    validation_targets = np.stack([row.targets for row in validation_examples])
    temperatures = calibrate_temperatures(validation_logits, validation_targets)
    validation_probabilities = 1.0 / (1.0 + np.exp(-validation_logits / temperatures))
    test_logits = infer_logits(
        model, test_examples, normalizer, batch_size=args.batch_size, device=controller_device
    )
    test_targets = np.stack([row.targets for row in test_examples])
    test_probabilities = 1.0 / (1.0 + np.exp(-test_logits / temperatures))
    validation_pre = np.asarray(
        [not row.expert_available for row in validation_examples], dtype=np.bool_
    )
    test_pre = np.asarray(
        [not row.expert_available for row in test_examples], dtype=np.bool_
    )
    costs = {
        **DEFAULT_ACTION_COSTS,
        "consult_expert": args.consult_cost,
        "recapture": args.recapture_cost,
        "reject_unknown": args.reject_cost,
        "defer_review": args.review_cost,
    }
    validation_policy, _ = evaluate_policy(validation_examples, validation_probabilities, costs)
    test_policy, test_decisions = evaluate_policy(test_examples, test_probabilities, costs)

    checkpoint_path = args.output_dir / "controller.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "model_config": {"hidden_dim": args.hidden_dim, "dropout": args.dropout},
            "normalizer": normalizer.state_dict(),
            "temperatures": temperatures.tolist(),
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "action_costs": costs,
            "protocol_sha256": protocol["protocol_sha256"],
        },
        checkpoint_path,
    )
    onnx = None
    if not args.skip_onnx_export:
        onnx = export_onnx(model, normalizer, temperatures, args.output_dir / "controller.onnx")
    write_json(args.output_dir / "test_decisions.json", test_decisions)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "method": {
            "name": "DeepSets OvA learned evidence controller",
            "base_models_frozen": True,
            "two_stage_expert_masking": True,
            "handwritten_match_threshold": False,
            "action_rule": "argmax(predicted_success_probability - explicit_action_cost)",
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
        },
        "backend": backend_info,
        "extraction": extraction,
        "episodes": {
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "locked_test_examples": len(test_examples),
            "examples_per_episode": 2,
        },
        "training": training,
        "calibration": {
            "source": "controller_validation identities only",
            "temperatures": {name: float(temperatures[index]) for index, name in enumerate(OUTPUT_NAMES)},
        },
        "validation": {
            "heads": probability_metrics(validation_probabilities, validation_targets),
            "heads_pre_expert": probability_metrics(
                validation_probabilities[validation_pre],
                validation_targets[validation_pre],
            ),
            "heads_post_expert": probability_metrics(
                validation_probabilities[~validation_pre],
                validation_targets[~validation_pre],
            ),
            "policy": validation_policy,
        },
        "locked_test": {
            "heads": probability_metrics(test_probabilities, test_targets),
            "heads_pre_expert": probability_metrics(
                test_probabilities[test_pre], test_targets[test_pre]
            ),
            "heads_post_expert": probability_metrics(
                test_probabilities[~test_pre], test_targets[~test_pre]
            ),
            "policy": test_policy,
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
            },
            "onnx": onnx,
            "test_decisions": str((args.output_dir / "test_decisions.json").resolve()),
        },
        "limitations": [
            "Action costs encode deployment preferences and are intentionally explicit rather than learned from this dataset.",
            "A recapture action is evaluated as BIFOR on the normalized mean of two query embeddings.",
            "A deferred review is counted as correct for system-level accuracy; autonomous accuracy is reported separately.",
            "MegaDescriptor-B-224 weights are non-commercial CC BY-NC 4.0.",
        ],
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "SUMMARY.md").write_text(markdown_summary(report), encoding="utf-8")
    print(json.dumps(report["locked_test"], ensure_ascii=False, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
