"""Evaluate the token-level query/reference model on a held-out manifest.

The evaluator keeps the descriptor and token grids together.  It never
constructs a reference set from the query row and it never reads a blind split.
The descriptor centroid score is reported beside the learned token score
so a short structural experiment cannot be mistaken for a parameter sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.reference_aware_model import (  # noqa: E402
    build_reference_aware_encoder_from_checkpoint,
)
from pet_id.reference_aware_training import (  # noqa: E402
    paired_retrieval_error_summary,
    validate_reference_image_manifest,
)
from pet_id.reference_token_model import (  # noqa: E402
    MODEL_FORMAT,
    build_token_reference_aware_model_from_checkpoint,
    catalog_confidence_gate_from_scores,
)
from pet_id.unified_data import UnifiedManifestDataset  # noqa: E402
from pet_id.unified_highres_data import (  # noqa: E402
    UnifiedHighResolutionReferenceDataset,
    load_highres_manifest,
)
from pet_id.unified_training import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--matcher-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=WORKSPACE / "models/pretrained/dog.pt",
    )
    parser.add_argument("--reference-counts", default="1,2,3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--identity-chunk", type=int, default=32)
    parser.add_argument(
        "--highres-image-size",
        type=int,
        default=2048,
        help="square canvas used for a high-resolution base checkpoint",
    )
    parser.add_argument(
        "--highres-maximum-side",
        type=int,
        default=4096,
        help="maximum raw source side accepted by the high-resolution evaluator",
    )
    parser.add_argument("--open-set-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def resolve_device(value: str) -> torch.device:
    requested = str(value).casefold()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(sorted({int(item.strip()) for item in str(value).split(",")}))
    except ValueError as error:
        raise ValueError("--reference-counts must contain positive integers") from error
    if not counts or any(count < 1 for count in counts):
        raise ValueError("--reference-counts must contain positive integers")
    return counts


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _uses_highres_protocol(path: Path) -> bool:
    """Return whether a manifest follows the dynamic spatial-detail schema."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read manifest: {path}") from error
    protocol = str(payload.get("protocol_name", "")).casefold()
    return "high_resolution" in protocol or "highres" in protocol


def _group_manifest_indices(
    dataset: UnifiedManifestDataset,
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[str(record["identity"]).casefold()].append(index)
    return {identity: tuple(rows) for identity, rows in sorted(grouped.items())}


def encode_manifest_features(
    model: torch.nn.Module,
    dataset: UnifiedManifestDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder = getattr(model, "encode_image_features", None)
    if not callable(encoder):
        raise TypeError("token model does not expose encode_image_features")
    descriptors: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            images = torch.stack(
                [dataset[index]["rgb"] for index in range(start, stop)]
            )
            descriptor, token = encoder(images.to(device))
            descriptors.append(F.normalize(descriptor.float(), dim=1).cpu())
            tokens.append(F.normalize(token.float(), dim=-1).cpu())
    if not descriptors:
        raise RuntimeError("manifest has no records")
    return torch.cat(descriptors), torch.cat(tokens)


def _rank_metrics(scores: torch.Tensor, targets: Sequence[int]) -> dict[str, Any]:
    if scores.ndim != 2 or scores.shape[0] != len(targets):
        raise ValueError("scores and targets have incompatible shapes")
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    ranking = scores.argsort(dim=1, descending=True, stable=True)
    ranks: list[int] = []
    for row, target in zip(ranking, target_tensor):
        matches = (row == target).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise RuntimeError("target identity is missing from score columns")
        ranks.append(int(matches[0].item()) + 1)
    return {
        "query_records": len(ranks),
        "identity_count": int(scores.shape[1]),
        "top1_accuracy": float(np.mean(np.asarray(ranks) == 1)),
        "top5_accuracy": float(np.mean(np.asarray(ranks) <= 5)),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
    }


def _hard_negative_metrics(
    scores: torch.Tensor,
    targets: Sequence[int],
) -> dict[str, Any]:
    """Report the margin against the strongest non-target identity.

    This is intentionally derived from the same score matrix as ranking. It
    does not introduce a second threshold or a tuned negative sampler, so a
    structural gain cannot be hidden behind a parameter choice.
    """

    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=scores.device)
    if scores.ndim != 2 or scores.shape[0] != target_tensor.numel():
        raise ValueError("scores and targets have incompatible shapes")
    rows = torch.arange(scores.shape[0], device=scores.device)
    positive = scores[rows, target_tensor]
    impostor = scores.clone()
    impostor[rows, target_tensor] = -float("inf")
    hard_negative = impostor.max(dim=1).values
    margin = positive - hard_negative
    return {
        "positive_score_mean": float(positive.float().mean().cpu()),
        "hard_negative_score_mean": float(hard_negative.float().mean().cpu()),
        "positive_minus_hard_negative_mean": float(margin.float().mean().cpu()),
        "positive_minus_hard_negative_p05": float(
            np.percentile(margin.detach().float().cpu().numpy(), 5.0)
        ),
        "positive_beats_hard_negative_rate": float((margin > 0).float().mean().cpu()),
    }


def _catalog_confidence_gate_metrics(
    gate: torch.Tensor,
) -> dict[str, float]:
    if gate.ndim != 1 or gate.numel() < 1:
        raise ValueError("catalog confidence gate must be a non-empty vector")
    gate = gate.detach().float()
    if not bool(torch.isfinite(gate).all()):
        raise ValueError("catalog confidence gate must contain finite values")
    return {
        "mean": float(gate.mean()),
        "closed_fraction": float((gate <= 0.0).float().mean()),
        "active_fraction": float((gate > 0.0).float().mean()),
    }


def _score_sets(
    model: torch.nn.Module,
    query_descriptor: torch.Tensor,
    query_tokens: torch.Tensor,
    reference_descriptors: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    device: torch.device,
    identity_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if identity_chunk < 1:
        raise ValueError("identity_chunk must be positive")
    identity_count, reference_count, descriptor_dim = reference_descriptors.shape
    if query_descriptor.shape[1] != descriptor_dim:
        raise ValueError("query/reference descriptor widths differ")
    learned_chunks: list[torch.Tensor] = []
    baseline_chunks: list[torch.Tensor] = []
    query_descriptor = F.normalize(query_descriptor.float(), dim=1).to(device)
    query_tokens = F.normalize(query_tokens.float(), dim=-1).to(device)
    reference_descriptors = F.normalize(reference_descriptors.float(), dim=2).to(device)
    reference_tokens = F.normalize(reference_tokens.float(), dim=-1).to(device)
    with torch.inference_mode():
        centroids = F.normalize(reference_descriptors.mean(dim=1), dim=1)
        centroid_scores = torch.einsum("qd,id->qi", query_descriptor, centroids)
        query_catalog_gate = catalog_confidence_gate_from_scores(centroid_scores)
        for start in range(0, identity_count, identity_chunk):
            stop = min(start + identity_chunk, identity_count)
            ref_descriptor_chunk = reference_descriptors[start:stop]
            ref_token_chunk = reference_tokens[start:stop]
            batch = query_descriptor.shape[0]
            expanded_query = (
                query_descriptor[:, None, :]
                .expand(batch, stop - start, -1)
                .reshape(-1, descriptor_dim)
            )
            expanded_query_tokens = (
                query_tokens[:, None, :, :]
                .expand(batch, stop - start, -1, -1)
                .reshape(-1, query_tokens.shape[1], query_tokens.shape[2])
            )
            expanded_references = (
                ref_descriptor_chunk[None, ...]
                .expand(batch, -1, -1, -1)
                .reshape(-1, reference_count, descriptor_dim)
            )
            expanded_reference_tokens = (
                ref_token_chunk[None, ...]
                .expand(batch, -1, -1, -1, -1)
                .reshape(
                    -1,
                    reference_count,
                    reference_tokens.shape[2],
                    reference_tokens.shape[3],
                )
            )
            mask = torch.ones(
                (expanded_query.shape[0], reference_count),
                dtype=torch.bool,
                device=device,
            )
            expanded_catalog_gate = (
                query_catalog_gate[:, None].expand(batch, stop - start).reshape(-1)
            )
            output = model.forward_encoded(
                expanded_query,
                expanded_references,
                mask,
                query_tokens=expanded_query_tokens,
                reference_tokens=expanded_reference_tokens,
                catalog_confidence_gate=expanded_catalog_gate,
                return_aux=True,
            )
            if not isinstance(output, dict):
                raise RuntimeError("token model returned no auxiliary output")
            learned = output.get("score")
            matcher_baseline = output.get("baseline_score")
            if not isinstance(learned, torch.Tensor) or not isinstance(
                matcher_baseline, torch.Tensor
            ):
                raise RuntimeError(
                    "token model auxiliary output is missing score tensors"
                )
            learned_chunks.append(learned.reshape(batch, stop - start).cpu())
            baseline_chunks.append(
                matcher_baseline.reshape(batch, stop - start).cpu()
            )
    learned_matrix = torch.cat(learned_chunks, dim=1)
    baseline_matrix = torch.cat(baseline_chunks, dim=1)
    if reference_count == 1 and not torch.equal(learned_matrix, baseline_matrix):
        raise RuntimeError(
            "singleton token matching must equal the centroid baseline exactly"
        )
    return learned_matrix, baseline_matrix, query_catalog_gate.cpu()


def _evaluate_selected_reference_sets(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    tokens: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    selections: Sequence[tuple[str, int, Sequence[int], Sequence[int]]],
    device: torch.device,
    identity_chunk: int,
) -> dict[str, Any]:
    """Score one explicitly selected reference set per identity.

    Each selection is ``(identity, query_row, reference_rows, metadata)``. The
    helper keeps candidate-specific reference rows intact and only requires
    that all candidates in one evaluation have the same reference count.
    """

    if not selections:
        return {"status": "skipped", "reason": "no eligible identities"}
    identities = [str(item[0]) for item in selections]
    query_rows = [int(item[1]) for item in selections]
    reference_rows = [tuple(int(row) for row in item[2]) for item in selections]
    counts = {len(rows) for rows in reference_rows}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise ValueError("selected reference sets must have one positive width")
    reference_count = next(iter(counts))
    if any(query in refs for query, refs in zip(query_rows, reference_rows)):
        raise ValueError("selected query/reference rows overlap")
    reference_descriptors = torch.stack(
        [descriptors[list(rows)] for rows in reference_rows]
    )
    reference_tokens = torch.stack([tokens[list(rows)] for rows in reference_rows])
    learned, baseline, catalog_gate = _score_sets(
        model,
        descriptors[query_rows],
        tokens[query_rows],
        reference_descriptors,
        reference_tokens,
        device=device,
        identity_chunk=identity_chunk,
    )
    targets = list(range(len(identities)))
    return {
        "status": "ok",
        "identity_count": len(identities),
        "reference_count": reference_count,
        "token_matcher": _rank_metrics(learned, targets),
        "centroid_baseline": _rank_metrics(baseline, targets),
        "paired_top1": paired_retrieval_error_summary(
            learned,
            baseline,
            torch.as_tensor(targets, dtype=torch.long),
            catalog_gate,
        ),
        "token_matcher_hard_negative": _hard_negative_metrics(learned, targets),
        "centroid_baseline_hard_negative": _hard_negative_metrics(baseline, targets),
        "catalog_confidence_gate": _catalog_confidence_gate_metrics(catalog_gate),
        "reference_query_overlap": False,
    }


def evaluate_leave_one_view_out(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    tokens: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    reference_count: int,
    device: torch.device,
    identity_chunk: int,
) -> dict[str, Any]:
    """Evaluate every held-out view with the other views as references."""

    reference_count = int(reference_count)
    eligible = [
        identity
        for identity, rows in grouped.items()
        if len(rows) >= reference_count + 1
    ]
    if not eligible:
        return {
            "status": "skipped",
            "reason": "each identity needs reference_count + 1 images",
            "reference_count": reference_count,
        }
    eligible = sorted(eligible)
    rotations: list[dict[str, Any]] = []
    learned_matrices: list[torch.Tensor] = []
    baseline_matrices: list[torch.Tensor] = []
    catalog_gates: list[torch.Tensor] = []
    for heldout_slot in range(reference_count + 1):
        selections = []
        for identity in eligible:
            rows = tuple(grouped[identity][: reference_count + 1])
            query = rows[heldout_slot]
            refs = rows[:heldout_slot] + rows[heldout_slot + 1 :]
            selections.append((identity, query, refs, (heldout_slot,)))
        query_rows = [int(item[1]) for item in selections]
        ref_rows = [tuple(int(row) for row in item[2]) for item in selections]
        reference_descriptors = torch.stack(
            [descriptors[list(rows)] for rows in ref_rows]
        )
        reference_tokens = torch.stack([tokens[list(rows)] for rows in ref_rows])
        learned, baseline, catalog_gate = _score_sets(
            model,
            descriptors[query_rows],
            tokens[query_rows],
            reference_descriptors,
            reference_tokens,
            device=device,
            identity_chunk=identity_chunk,
        )
        targets = list(range(len(eligible)))
        learned_matrices.append(learned)
        baseline_matrices.append(baseline)
        catalog_gates.append(catalog_gate)
        rotations.append(
            {
                "heldout_slot": heldout_slot,
                "token_matcher": _rank_metrics(learned, targets),
                "centroid_baseline": _rank_metrics(baseline, targets),
                "paired_top1": paired_retrieval_error_summary(
                    learned,
                    baseline,
                    torch.as_tensor(targets, dtype=torch.long),
                    catalog_gate,
                ),
                "catalog_confidence_gate": _catalog_confidence_gate_metrics(
                    catalog_gate
                ),
            }
        )
    learned_all = torch.cat(learned_matrices, dim=0)
    baseline_all = torch.cat(baseline_matrices, dim=0)
    catalog_gate_all = torch.cat(catalog_gates, dim=0)
    targets_all = list(range(len(eligible))) * (reference_count + 1)
    return {
        "status": "ok",
        "identity_count": len(eligible),
        "reference_count": reference_count,
        "heldout_views_per_identity": reference_count + 1,
        "token_matcher": _rank_metrics(learned_all, targets_all),
        "centroid_baseline": _rank_metrics(baseline_all, targets_all),
        "paired_top1": paired_retrieval_error_summary(
            learned_all,
            baseline_all,
            torch.as_tensor(targets_all, dtype=torch.long),
            catalog_gate_all,
        ),
        "token_matcher_hard_negative": _hard_negative_metrics(learned_all, targets_all),
        "centroid_baseline_hard_negative": _hard_negative_metrics(
            baseline_all, targets_all
        ),
        "catalog_confidence_gate": _catalog_confidence_gate_metrics(catalog_gate_all),
        "rotations": rotations,
        "reference_query_overlap": False,
    }


def evaluate_view_diversity_subsets(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    tokens: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    device: torch.device,
    identity_chunk: int,
) -> dict[str, Any]:
    """Compare deliberately repeated and complementary two-view references.

    Pair selection uses only frozen base-descriptor cosine, and is reported as
    an exploratory diagnostic rather than a tuned production policy. The
    query is a deterministic remaining view, so it never appears in its own
    reference set.
    """

    eligible = sorted(identity for identity, rows in grouped.items() if len(rows) >= 4)
    if not eligible:
        return {"status": "skipped", "reason": "each identity needs four images"}
    conditions: dict[str, Any] = {}
    for mode in ("repeated", "complementary"):
        selections: list[tuple[str, int, Sequence[int], Sequence[int]]] = []
        pair_cosines: list[float] = []
        for identity in eligible:
            rows = tuple(grouped[identity][:4])
            local = F.normalize(descriptors[list(rows)].float(), dim=1)
            pair_candidates: list[tuple[float, int, int]] = []
            for left in range(4):
                for right in range(left + 1, 4):
                    pair_candidates.append(
                        (float((local[left] * local[right]).sum()), left, right)
                    )
            pair = max(pair_candidates) if mode == "repeated" else min(pair_candidates)
            _cosine, left, right = pair
            refs = (rows[left], rows[right])
            remaining = [row for row in rows if row not in refs]
            query = remaining[0]
            selections.append((identity, query, refs, (left, right)))
            pair_cosines.append(float(_cosine))
        result = _evaluate_selected_reference_sets(
            model,
            descriptors,
            tokens,
            grouped,
            selections=selections,
            device=device,
            identity_chunk=identity_chunk,
        )
        result["selection_basis"] = "frozen_base_descriptor_cosine"
        result["reference_pair_cosine_mean"] = float(np.mean(pair_cosines))
        result["reference_pair_cosine_min"] = float(np.min(pair_cosines))
        result["reference_pair_cosine_max"] = float(np.max(pair_cosines))
        conditions[mode] = result
    return {
        "status": "ok",
        "identity_count": len(eligible),
        "conditions": conditions,
        "reference_query_overlap": False,
    }


def evaluate_reference_count(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    tokens: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    reference_count: int,
    device: torch.device,
    identity_chunk: int,
) -> dict[str, Any]:
    reference_count = int(reference_count)
    eligible = sorted(
        identity for identity, rows in grouped.items() if len(rows) > reference_count
    )
    if not eligible:
        return {
            "status": "skipped",
            "reason": "each identity needs at least reference_count + 1 images",
            "reference_count": reference_count,
        }
    positions = {identity: index for index, identity in enumerate(eligible)}
    rotation_count = max(len(grouped[identity]) for identity in eligible)
    learned_rows: list[torch.Tensor] = []
    baseline_rows: list[torch.Tensor] = []
    gate_rows: list[torch.Tensor] = []
    targets_all: list[int] = []
    rotations: list[dict[str, Any]] = []
    for rotation_index in range(rotation_count):
        query_identities = [
            identity
            for identity in eligible
            if rotation_index < len(grouped[identity])
        ]
        query_rows = [
            grouped[identity][rotation_index] for identity in query_identities
        ]
        targets = [positions[identity] for identity in query_identities]
        selected_references: list[tuple[int, ...]] = []
        for identity in eligible:
            rows = grouped[identity]
            start = (rotation_index + 1) % len(rows)
            rotated = rows[start:] + rows[:start]
            references = rotated[:reference_count]
            if (
                rotation_index < len(rows)
                and rows[rotation_index] in references
            ):
                raise RuntimeError("rotating query appeared in its reference set")
            selected_references.append(references)
        reference_descriptors = torch.stack(
            [descriptors[list(rows)] for rows in selected_references]
        )
        reference_tokens = torch.stack(
            [tokens[list(rows)] for rows in selected_references]
        )
        learned, baseline, catalog_gate = _score_sets(
            model,
            descriptors[query_rows],
            tokens[query_rows],
            reference_descriptors,
            reference_tokens,
            device=device,
            identity_chunk=identity_chunk,
        )
        learned_rows.append(learned)
        baseline_rows.append(baseline)
        gate_rows.append(catalog_gate)
        targets_all.extend(targets)
        rotations.append(
            {
                "rotation_index": rotation_index,
                "query_records": len(query_rows),
                "token_matcher": _rank_metrics(learned, targets),
                "centroid_baseline": _rank_metrics(baseline, targets),
                "paired_top1": paired_retrieval_error_summary(
                    learned,
                    baseline,
                    torch.as_tensor(targets, dtype=torch.long),
                    catalog_gate,
                ),
                "catalog_confidence_gate": _catalog_confidence_gate_metrics(
                    catalog_gate
                ),
            }
        )
    learned = torch.cat(learned_rows, dim=0)
    baseline = torch.cat(baseline_rows, dim=0)
    catalog_gate = torch.cat(gate_rows, dim=0)
    expected_queries = sum(len(grouped[identity]) for identity in eligible)
    if len(targets_all) != expected_queries:
        raise RuntimeError("rotating evaluation did not cover every eligible row")
    return {
        "status": "ok",
        "reference_count": reference_count,
        "identity_count": len(eligible),
        "query_assignment": "deterministic_rotating_folds",
        "coverage": {
            "eligible_manifest_records": expected_queries,
            "query_records": len(targets_all),
            "rotation_count": rotation_count,
            "complete": len(targets_all) == expected_queries,
        },
        "token_matcher": _rank_metrics(learned, targets_all),
        "centroid_baseline": _rank_metrics(baseline, targets_all),
        "paired_top1": paired_retrieval_error_summary(
            learned,
            baseline,
            torch.as_tensor(targets_all, dtype=torch.long),
            catalog_gate,
        ),
        "token_matcher_hard_negative": _hard_negative_metrics(
            learned, targets_all
        ),
        "centroid_baseline_hard_negative": _hard_negative_metrics(
            baseline, targets_all
        ),
        "catalog_confidence_gate": _catalog_confidence_gate_metrics(catalog_gate),
        "rotations": rotations,
        "reference_query_overlap": False,
    }


def evaluate_open_set(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    tokens: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    reference_count: int,
    fraction: float,
    device: torch.device,
    identity_chunk: int,
) -> dict[str, Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("open-set fraction must be between 0 and 1")
    eligible = [
        identity for identity, rows in grouped.items() if len(rows) > reference_count
    ]
    split = (
        max(1, min(len(eligible) - 1, int(round(len(eligible) * fraction))))
        if len(eligible) >= 2
        else 0
    )
    if split <= 0 or split >= len(eligible):
        return {"status": "skipped", "reason": "not enough identities"}
    known, unknown = eligible[:split], eligible[split:]
    reference_descriptors = torch.stack(
        [descriptors[list(grouped[identity][:reference_count])] for identity in known]
    )
    reference_tokens = torch.stack(
        [tokens[list(grouped[identity][:reference_count])] for identity in known]
    )
    known_rows: list[int] = []
    known_targets: list[int] = []
    for target, identity in enumerate(known):
        rows = grouped[identity][reference_count:]
        known_rows.extend(rows)
        known_targets.extend([target] * len(rows))
    unknown_rows = [grouped[identity][reference_count] for identity in unknown]
    known_scores, known_baseline, known_catalog_gate = _score_sets(
        model,
        descriptors[known_rows],
        tokens[known_rows],
        reference_descriptors,
        reference_tokens,
        device=device,
        identity_chunk=identity_chunk,
    )
    unknown_scores, unknown_baseline, unknown_catalog_gate = _score_sets(
        model,
        descriptors[unknown_rows],
        tokens[unknown_rows],
        reference_descriptors,
        reference_tokens,
        device=device,
        identity_chunk=identity_chunk,
    )
    result: dict[str, Any] = {
        "status": "ok",
        "reference_count": reference_count,
        "known_identities": len(known),
        "unknown_identities": len(unknown),
        "known_split_fraction": fraction,
        "catalog_confidence_gate": {
            "known": _catalog_confidence_gate_metrics(known_catalog_gate),
            "unknown": _catalog_confidence_gate_metrics(unknown_catalog_gate),
        },
        "known_paired_top1": (
            paired_retrieval_error_summary(
                known_scores,
                known_baseline,
                torch.as_tensor(known_targets, dtype=torch.long),
                known_catalog_gate,
            )
            if known_scores.shape[1] >= 2
            else {
                "status": "skipped",
                "reason": "paired retrieval needs at least two known identities",
            }
        ),
    }
    for name, known_matrix, unknown_matrix in (
        ("token_matcher", known_scores, unknown_scores),
        ("centroid_baseline", known_baseline, unknown_baseline),
    ):
        target_tensor = torch.as_tensor(known_targets, dtype=torch.long)
        positive = known_matrix[torch.arange(known_matrix.shape[0]), target_tensor]
        impostor = known_matrix.clone()
        impostor[torch.arange(impostor.shape[0]), target_tensor] = -float("inf")
        threshold = float(np.percentile(positive.numpy(), 5.0))
        unknown_max = unknown_matrix.max(dim=1).values
        result[name] = {
            "known_top1_accuracy": _rank_metrics(known_matrix, known_targets)[
                "top1_accuracy"
            ],
            "known_positive_score_p05": threshold,
            "known_impostor_max_mean": float(impostor.max(dim=1).values.mean()),
            "unknown_max_score_p95": float(np.percentile(unknown_max.numpy(), 95.0)),
            "unknown_false_accept_rate_at_p05": float(
                (unknown_max >= threshold).float().mean()
            ),
        }
    return result


def main() -> None:
    args = build_parser().parse_args()
    counts = parse_counts(args.reference_counts)
    device = resolve_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    matcher_checkpoint = args.matcher_checkpoint.expanduser().resolve()
    arcface_checkpoint = args.arcface_checkpoint.expanduser().resolve()
    for path in (manifest, base_checkpoint, matcher_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = torch.load(matcher_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
        raise ValueError("matcher-checkpoint is not a token reference-aware checkpoint")
    encoder, base_payload = build_reference_aware_encoder_from_checkpoint(
        base_checkpoint, arcface_checkpoint, device=device
    )
    model, matcher_payload = build_token_reference_aware_model_from_checkpoint(
        matcher_checkpoint, encoder, device=device
    )
    model.eval()
    base_is_highres = (
        base_payload.get("model_type") == "unified_high_resolution_pet_reid"
    )
    manifest_is_highres = _uses_highres_protocol(manifest)
    if manifest_is_highres and not base_is_highres:
        raise ValueError(
            "high-resolution reference manifests require a high-resolution base checkpoint"
        )
    highres_mode = base_is_highres and manifest_is_highres
    if highres_mode:
        load_highres_manifest(manifest, expected_split="development")
        dataset = UnifiedHighResolutionReferenceDataset(
            manifest,
            image_size=args.highres_image_size,
            expected_split="development",
            training=False,
            maximum_side=args.highres_maximum_side,
        )
    else:
        validate_reference_image_manifest(manifest)
        input_size = getattr(model, "input_size", None)
        if input_size is None:
            raise TypeError("selected encoder does not expose input_size")
        dataset = UnifiedManifestDataset(
            manifest,
            input_size=int(input_size),
            training=False,
            min_images_per_identity=1,
        )
    grouped = _group_manifest_indices(dataset)
    descriptors, tokens = encode_manifest_features(
        model, dataset, batch_size=int(args.batch_size), device=device
    )
    evaluations = {
        str(count): evaluate_reference_count(
            model,
            descriptors,
            tokens,
            grouped,
            reference_count=count,
            device=device,
            identity_chunk=int(args.identity_chunk),
        )
        for count in counts
    }
    open_set_count = next(
        (count for count in counts if evaluations[str(count)].get("status") == "ok"),
        None,
    )
    open_set = (
        evaluate_open_set(
            model,
            descriptors,
            tokens,
            grouped,
            reference_count=int(open_set_count),
            fraction=float(args.open_set_fraction),
            device=device,
            identity_chunk=int(args.identity_chunk),
        )
        if open_set_count is not None
        else {"status": "skipped", "reason": "no feasible reference count"}
    )
    feasible_view_counts = [
        len(rows) - 1 for rows in grouped.values() if len(rows) >= 2
    ]
    leave_one_view_out = (
        evaluate_leave_one_view_out(
            model,
            descriptors,
            tokens,
            grouped,
            reference_count=min(
                int(getattr(model, "max_references", 3)),
                min(feasible_view_counts),
            ),
            device=device,
            identity_chunk=int(args.identity_chunk),
        )
        if feasible_view_counts
        else {"status": "skipped", "reason": "not enough views"}
    )
    view_diversity = evaluate_view_diversity_subsets(
        model,
        descriptors,
        tokens,
        grouped,
        device=device,
        identity_chunk=int(args.identity_chunk),
    )
    result = {
        "format": "reference-token-structural-evaluation",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "matcher_checkpoint": str(matcher_checkpoint),
        "matcher_checkpoint_sha256": sha256_file(matcher_checkpoint),
        "matcher_training": matcher_payload.get("training", {}),
        "device": str(device),
        "data_mode": "highres" if highres_mode else "fixed",
        "image_size": (
            int(args.highres_image_size)
            if highres_mode
            else int(getattr(dataset, "input_size", 0))
        ),
        "records": len(dataset),
        "identities": len(grouped),
        "reference_counts": evaluations,
        "open_set": open_set,
        "leave_one_view_out": leave_one_view_out,
        "view_diversity_subsets": view_diversity,
        "blind_split_used": False,
        "reference_query_overlap": False,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else matcher_checkpoint.parent / "evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
