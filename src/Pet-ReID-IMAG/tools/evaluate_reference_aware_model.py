#!/usr/bin/env python3
"""Evaluate a trained query/reference image-set model on a held-out manifest.

The evaluator encodes each image once, then compares the learned
query-conditioned matcher with the transparent centroid-plus-top-k baseline.
Reference and query rows are always disjoint.  The optional open-set report
uses one half of the held-out identities as an enrolled gallery and the other
half as unknown queries; no blind split is read by this tool.
"""

from __future__ import annotations

import argparse
import json
import math
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
    build_reference_aware_model_from_checkpoint,
)
from pet_id.reference_aware_training import validate_reference_image_manifest  # noqa: E402
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
        help=(
            "legacy ArcFace source for an un-packaged base checkpoint; "
            "packaged checkpoints restore their own source chain"
        ),
    )
    parser.add_argument(
        "--reference-counts",
        default="1,2,3,4",
        help="comma-separated reference counts to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=8)
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


def parse_reference_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(sorted({int(item.strip()) for item in str(value).split(",")}))
    except ValueError as exc:
        raise ValueError("--reference-counts must contain positive integers") from exc
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


def _group_manifest_indices(dataset: UnifiedManifestDataset) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[str(record["identity"]).casefold()].append(index)
    return {identity: tuple(indices) for identity, indices in sorted(grouped.items())}


def encode_manifest(
    model: torch.nn.Module,
    dataset: UnifiedManifestDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode all manifest rows once and return CPU-normalized descriptors."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder = getattr(model, "encode_images", None)
    if not callable(encoder):
        raise TypeError("reference-aware model does not expose encode_images")
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            images = torch.stack(
                [dataset[index]["rgb"] for index in range(start, min(start + batch_size, len(dataset)))]
            ).to(device)
            descriptors = encoder(images)
            chunks.append(F.normalize(descriptors.float(), dim=1).cpu())
    if not chunks:
        raise RuntimeError("manifest has no records")
    return torch.cat(chunks, dim=0)


def _exact_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    negative_sorted = sorted(float(value) for value in negative)
    wins = 0.0
    for value in positive:
        left = np.searchsorted(negative_sorted, float(value), side="left")
        right = np.searchsorted(negative_sorted, float(value), side="right")
        wins += float(left) + 0.5 * float(right - left)
    return float(wins / (len(positive) * len(negative_sorted)))


def _rank_metrics(scores: torch.Tensor, targets: Sequence[int]) -> dict[str, Any]:
    if scores.ndim != 2 or scores.shape[0] != len(targets):
        raise ValueError("scores and targets have incompatible shapes")
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    ranking = scores.argsort(dim=1, descending=True)
    ranks: list[int] = []
    for row, target in zip(ranking, target_tensor):
        matches = (row == target).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise RuntimeError("target identity is missing from score columns")
        ranks.append(int(matches[0].item()) + 1)
    positive = scores[torch.arange(scores.shape[0]), target_tensor]
    negative_values: list[float] = []
    for row, target in zip(scores, target_tensor):
        negative_values.extend(row[torch.arange(row.shape[0]) != target].tolist())
    return {
        "query_records": len(ranks),
        "identity_count": int(scores.shape[1]),
        "top1_correct": int(sum(rank == 1 for rank in ranks)),
        "top1_accuracy": float(sum(rank == 1 for rank in ranks) / len(ranks)),
        "top5_correct": int(sum(rank <= 5 for rank in ranks)),
        "top5_accuracy": float(sum(rank <= 5 for rank in ranks) / len(ranks)),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "positive_score_mean": float(positive.mean()),
        "positive_score_min": float(positive.min()),
        "negative_score_mean": float(np.mean(negative_values)) if negative_values else None,
        "auc": _exact_auc(positive.tolist(), negative_values),
    }


def _score_sets(
    matcher: torch.nn.Module,
    query: torch.Tensor,
    reference_sets: Sequence[torch.Tensor],
    *,
    device: torch.device,
    identity_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return learned and transparent baseline score matrices on CPU."""

    if not reference_sets:
        raise ValueError("at least one reference set is required")
    query = F.normalize(query.float(), dim=1).to(device)
    references = F.normalize(torch.stack(list(reference_sets)).float(), dim=2).to(device)
    identity_count, reference_count, descriptor_dim = references.shape
    if query.shape[1] != descriptor_dim:
        raise ValueError("query and reference descriptor widths differ")
    mask = torch.ones(
        (1, reference_count), dtype=torch.bool, device=device
    )
    learned_chunks: list[torch.Tensor] = []
    baseline_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        similarities = torch.einsum("qd,ikd->qik", query, references)
        centroids = F.normalize(references.mean(dim=1), dim=1)
        centroid_scores = torch.einsum("qd,id->qi", query, centroids)
        top_k = min(int(getattr(matcher, "reference_top_k", reference_count)), reference_count)
        top_k_scores = similarities.topk(top_k, dim=2).values.mean(dim=2)
        weight = float(getattr(matcher, "reference_score_weight", 0.4))
        baseline = (1.0 - weight) * centroid_scores + weight * top_k_scores
        for start in range(0, identity_count, identity_chunk):
            stop = min(start + identity_chunk, identity_count)
            ref_chunk = references[start:stop]
            batch = query.shape[0]
            expanded_query = query[:, None, :].expand(batch, stop - start, -1).reshape(-1, descriptor_dim)
            expanded_refs = ref_chunk[None, ...].expand(batch, -1, -1, -1).reshape(
                -1, reference_count, descriptor_dim
            )
            expanded_mask = mask.expand(expanded_query.shape[0], -1)
            output = matcher(expanded_query, expanded_refs, expanded_mask)
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("matcher returned auxiliary output unexpectedly")
            learned_chunks.append(output.reshape(batch, stop - start).cpu())
            baseline_chunks.append(baseline[:, start:stop].cpu())
    return torch.cat(learned_chunks, dim=1), torch.cat(baseline_chunks, dim=1)


def evaluate_reference_count(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    reference_count: int,
    device: torch.device,
) -> dict[str, Any]:
    eligible = [identity for identity, rows in grouped.items() if len(rows) > reference_count]
    if not eligible:
        return {
            "status": "skipped",
            "reason": "each identity needs at least reference_count + 1 images",
            "reference_count": reference_count,
        }
    references = [descriptors[list(grouped[identity][:reference_count])] for identity in eligible]
    query_rows: list[int] = []
    targets: list[int] = []
    for target, identity in enumerate(eligible):
        query_rows.extend(grouped[identity][reference_count:])
        targets.extend([target] * (len(grouped[identity]) - reference_count))
    query = descriptors[query_rows]
    learned, baseline = _score_sets(model.matcher, query, references, device=device)
    return {
        "status": "ok",
        "reference_count": reference_count,
        "identity_count": len(eligible),
        "learned_matcher": _rank_metrics(learned, targets),
        "centroid_top_k_baseline": _rank_metrics(baseline, targets),
        "reference_query_overlap": False,
    }


def evaluate_open_set(
    model: torch.nn.Module,
    descriptors: torch.Tensor,
    grouped: dict[str, tuple[int, ...]],
    *,
    reference_count: int,
    fraction: float,
    device: torch.device,
) -> dict[str, Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("open-set fraction must be between 0 and 1")
    eligible = [identity for identity, rows in grouped.items() if len(rows) > reference_count]
    split = max(1, min(len(eligible) - 1, int(round(len(eligible) * fraction)))) if len(eligible) >= 2 else 0
    if split <= 0 or split >= len(eligible):
        return {"status": "skipped", "reason": "not enough identities for open-set split"}
    known = eligible[:split]
    unknown = eligible[split:]
    references = [descriptors[list(grouped[identity][:reference_count])] for identity in known]

    known_query_rows: list[int] = []
    known_targets: list[int] = []
    for target, identity in enumerate(known):
        rows = grouped[identity][reference_count:]
        known_query_rows.extend(rows)
        known_targets.extend([target] * len(rows))
    unknown_query_rows = [grouped[identity][reference_count] for identity in unknown]
    known_query = descriptors[known_query_rows]
    unknown_query = descriptors[unknown_query_rows]
    known_learned, known_baseline = _score_sets(model.matcher, known_query, references, device=device)
    unknown_learned, unknown_baseline = _score_sets(model.matcher, unknown_query, references, device=device)

    result: dict[str, Any] = {
        "status": "ok",
        "reference_count": reference_count,
        "known_identities": len(known),
        "unknown_identities": len(unknown),
        "known_split_fraction": fraction,
    }
    for name, known_scores, unknown_scores in (
        ("learned_matcher", known_learned, unknown_learned),
        ("centroid_top_k_baseline", known_baseline, unknown_baseline),
    ):
        positive = known_scores[torch.arange(known_scores.shape[0]), torch.as_tensor(known_targets)]
        impostor = known_scores.clone()
        impostor[torch.arange(impostor.shape[0]), torch.as_tensor(known_targets)] = -float("inf")
        known_impostor = impostor.max(dim=1).values
        unknown_max = unknown_scores.max(dim=1).values
        threshold = float(np.percentile(positive.numpy(), 5.0))
        result[name] = {
            "known_top1_accuracy": _rank_metrics(known_scores, known_targets)["top1_accuracy"],
            "known_positive_score_p05": threshold,
            "known_positive_score_mean": float(positive.mean()),
            "known_impostor_max_mean": float(known_impostor.mean()),
            "known_auc": _exact_auc(positive.tolist(), known_impostor.tolist()),
            "threshold_false_reject_rate": float((positive < threshold).float().mean()),
            "unknown_max_score_mean": float(unknown_max.mean()),
            "unknown_max_score_p95": float(np.percentile(unknown_max.numpy(), 95.0)),
            "unknown_false_accept_rate_at_p05": float((unknown_max >= threshold).float().mean()),
        }
    return result


def main() -> None:
    args = build_parser().parse_args()
    counts = parse_reference_counts(args.reference_counts)
    device = resolve_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    matcher_checkpoint = args.matcher_checkpoint.expanduser().resolve()
    arcface_checkpoint = args.arcface_checkpoint.expanduser().resolve()
    for path in (manifest, base_checkpoint, matcher_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    encoder, base_payload = build_reference_aware_encoder_from_checkpoint(
        base_checkpoint,
        arcface_checkpoint,
        device=device,
    )
    model, matcher_payload = build_reference_aware_model_from_checkpoint(
        matcher_checkpoint,
        encoder,
        device=device,
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
    descriptors = encode_manifest(
        model,
        dataset,
        batch_size=int(args.batch_size),
        device=device,
    )
    evaluations = {
        str(count): evaluate_reference_count(
            model,
            descriptors,
            grouped,
            reference_count=count,
            device=device,
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
            grouped,
            reference_count=int(open_set_count),
            fraction=float(args.open_set_fraction),
            device=device,
        )
        if open_set_count is not None
        else {"status": "skipped", "reason": "no feasible reference count"}
    )
    result = {
        "format": "reference-aware-evaluation",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "base_model_type": base_payload.get("model_type"),
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
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
