#!/usr/bin/env python3
"""Formal closed/open-set evaluation for the deployed BIFOR + Mega Agent.

The validation identities are used only for threshold calibration. The blind
identities and a deterministic, identity-disjoint unknown set are used only
for the final report. Image features are cached in SQLite after every sample
so an interrupted raw-image run can resume without re-encoding completed rows.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.dogfacenet_alignment import build_alignment_index  # noqa: E402
from pet_id.gallery import build_pipeline  # noqa: E402
from pet_id.gallery_service import (  # noqa: E402
    EncodedPetImage,
    MultimodalPipelineEncoder,
    normalize_feature,
)
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.recognition_agent import (  # noqa: E402
    AgentFeatureEncoder,
    MEGADESCRIPTOR_EXPERT_ID,
    MegaDescriptorEncoder,
    expert_reliabilities,
)


DEFAULT_PROTOCOL_ROOT = (
    WORKSPACE / "artifacts/runs/legacy/dogfacenet_joint100_protocol_v1"
)
DEFAULT_BIFOR_PACKAGE = (
    WORKSPACE / "models/selected/dogfacenet_semantic_v3_bifor_lowrank_v1"
)
DEFAULT_SEMANTIC_PACKAGE = (
    WORKSPACE / "models/selected/dogfacenet_semantic_v3_v1"
)
DEFAULT_OUTPUT = WORKSPACE / "artifacts/runs/agent_v1/formal_joint100_20_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_order(seed: int, *parts: str) -> str:
    text = ":".join((str(seed), *(str(part).casefold() for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_mean(rows: Sequence[np.ndarray]) -> np.ndarray:
    return normalize_feature(np.mean(np.stack(rows), axis=0), "prototype")


def exact_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        raise ValueError("AUC needs both positive and negative scores")
    negative_sorted = sorted(float(value) for value in negative)
    wins = 0.0
    for value in positive:
        left = bisect_left(negative_sorted, float(value))
        right = bisect_right(negative_sorted, float(value))
        wins += left + 0.5 * (right - left)
    return float(wins / (len(positive) * len(negative_sorted)))


def summarize(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if total < 1:
        raise ValueError("Wilson interval needs at least one observation")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def exact_mcnemar_pvalue(base_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar test over discordant query outcomes."""
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(base_only, candidate_only) + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * tail))


def paired_correctness(
    metrics: dict[str, dict[str, Any]],
    *,
    base: str,
    candidate: str,
) -> dict[str, Any]:
    base_rows = {
        row["source_sha256"]: row for row in metrics[base]["query_results"]
    }
    candidate_rows = {
        row["source_sha256"]: row
        for row in metrics[candidate]["query_results"]
    }
    if base_rows.keys() != candidate_rows.keys():
        raise ValueError("Paired comparison query sets differ")
    both_correct = base_only = candidate_only = both_wrong = 0
    for key, base_row in base_rows.items():
        candidate_row = candidate_rows[key]
        pair = (bool(base_row["correct"]), bool(candidate_row["correct"]))
        if pair == (True, True):
            both_correct += 1
        elif pair == (True, False):
            base_only += 1
        elif pair == (False, True):
            candidate_only += 1
        else:
            both_wrong += 1
    total = len(base_rows)
    base_correct = both_correct + base_only
    candidate_correct = both_correct + candidate_only
    return {
        "base": base,
        "candidate": candidate,
        "queries": total,
        "both_correct": both_correct,
        "base_only_correct": base_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "base_accuracy": base_correct / total,
        "candidate_accuracy": candidate_correct / total,
        "accuracy_delta": (candidate_correct - base_correct) / total,
        "base_accuracy_wilson_95": wilson_interval(base_correct, total),
        "candidate_accuracy_wilson_95": wilson_interval(candidate_correct, total),
        "exact_mcnemar_pvalue_two_sided": exact_mcnemar_pvalue(
            base_only, candidate_only
        ),
        "interpretation": (
            "p<0.05 supports a paired accuracy difference; intervals and the test "
            "do not account for within-identity query correlation"
        ),
    }


def threshold_for_known_recall(
    known_scores: Sequence[float], target_recall: float
) -> dict[str, float]:
    descending = sorted((float(value) for value in known_scores), reverse=True)
    accepted = min(len(descending), max(1, math.ceil(target_recall * len(descending))))
    threshold = descending[accepted - 1]
    achieved = sum(value >= threshold for value in descending) / len(descending)
    return {
        "threshold": float(threshold),
        "target_known_recall": float(target_recall),
        "calibration_known_recall": float(achieved),
    }


def best_balanced_threshold(
    positive: Sequence[float], negative: Sequence[float]
) -> dict[str, float]:
    positives = [float(value) for value in positive]
    negatives = [float(value) for value in negative]
    rows = sorted(
        [(value, 1) for value in positives] + [(value, 0) for value in negatives],
        reverse=True,
    )
    best = (-1.0, 0.0, 0.0, rows[0][0] + 1e-6)
    true_positives = 0
    false_positives = 0
    cursor = 0
    while cursor < len(rows):
        score = rows[cursor][0]
        next_cursor = cursor
        while next_cursor < len(rows) and rows[next_cursor][0] == score:
            if rows[next_cursor][1]:
                true_positives += 1
            else:
                false_positives += 1
            next_cursor += 1
        next_score = rows[next_cursor][0] if next_cursor < len(rows) else score - 2e-6
        threshold = 0.5 * (score + next_score)
        known_recall = true_positives / len(positives)
        unknown_recall = (len(negatives) - false_positives) / len(negatives)
        candidate = (
            0.5 * (known_recall + unknown_recall),
            known_recall,
            unknown_recall,
            threshold,
        )
        if candidate > best:
            best = candidate
        cursor = next_cursor
    return {
        "threshold": float(best[3]),
        "balanced_accuracy": float(best[0]),
        "known_recall": float(best[1]),
        "unknown_recall": float(best[2]),
    }


def load_manifest_records(path: Path, split: str) -> list[dict[str, Any]]:
    document = read_json(path)
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records: {path}")
    rows = []
    for record in records:
        source = Path(record["source_path"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {source}")
        rows.append(
            {
                "identity": str(record["identity"]).casefold(),
                "display_identity": str(record["identity"]),
                "source_path": str(source),
                "canonical_filename": str(record.get("canonical_filename") or source.name),
                "source_sha256": str(record.get("source_sha256") or sha256_file(source)),
                "origin": split,
            }
        )
    return rows


def assign_known_roles(
    records: Sequence[dict[str, Any]], gallery_per_identity: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["identity"]].append(dict(record))
    output = []
    for identity in sorted(grouped):
        rows = grouped[identity]
        if len(rows) <= gallery_per_identity:
            raise ValueError(f"Identity {identity} has no held-out query")
        for index, record in enumerate(rows):
            record["role"] = "gallery" if index < gallery_per_identity else "query"
            record["identity_record_index"] = index
            output.append(record)
    return output


def manifest_identities(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    identities: set[str] = set()
    evidence = []
    for path in paths:
        if not path.is_file():
            continue
        document = read_json(path)
        records = document.get("records")
        if not isinstance(records, list):
            continue
        current = {str(row["identity"]).casefold() for row in records}
        identities.update(current)
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "identities": len(current),
                "records": len(records),
            }
        )
    return identities, evidence


def select_unknown_records(
    dataset_root: Path,
    *,
    excluded_identities: set[str],
    calibration_identities: int,
    test_identities: int,
    images_per_identity: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    alignment_records, index_report = build_alignment_index(dataset_root)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in alignment_records:
        identity = record.identity.casefold()
        if identity not in excluded_identities:
            grouped[identity].append(record)
    eligible = [
        identity
        for identity, rows in grouped.items()
        if len(rows) >= images_per_identity
    ]
    eligible.sort(key=lambda identity: stable_order(seed, "identity", identity))
    needed = calibration_identities + test_identities
    if len(eligible) < needed:
        raise ValueError(
            f"Need {needed} unknown identities with {images_per_identity} images, "
            f"found {len(eligible)}"
        )

    def choose(identity: str, origin: str) -> list[dict[str, Any]]:
        rows = sorted(
            grouped[identity],
            key=lambda row: stable_order(
                seed, "image", identity, row.canonical_filename, row.source_path.name
            ),
        )[:images_per_identity]
        return [
            {
                "identity": identity,
                "display_identity": row.identity,
                "source_path": str(row.source_path.resolve()),
                "canonical_filename": row.canonical_filename,
                "source_sha256": sha256_file(row.source_path),
                "origin": origin,
                "role": "unknown_query",
                "identity_record_index": index,
            }
            for index, row in enumerate(rows)
        ]

    calibration = [
        row
        for identity in eligible[:calibration_identities]
        for row in choose(identity, "calibration_unknown")
    ]
    test = [
        row
        for identity in eligible[calibration_identities:needed]
        for row in choose(identity, "test_unknown")
    ]
    audit = {
        "alignment_index": index_report,
        "excluded_identities": len(excluded_identities),
        "eligible_identities": len(eligible),
        "calibration_identities": calibration_identities,
        "test_identities": test_identities,
        "images_per_identity": images_per_identity,
        "identity_overlap": len(
            {row["identity"] for row in calibration}
            & {row["identity"] for row in test}
        ),
    }
    return calibration, test, audit


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    validation = assign_known_roles(
        load_manifest_records(args.validation_manifest, "calibration_known"),
        args.gallery_images_per_identity,
    )
    test = assign_known_roles(
        load_manifest_records(args.test_manifest, "test_known"),
        args.gallery_images_per_identity,
    )
    test_manifest_identities = {row["identity"] for row in test}
    training_paths = [
        DEFAULT_PROTOCOL_ROOT / "train_manifest.json",
        WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/train_manifest.json",
    ]
    training_identities, training_evidence = manifest_identities(training_paths)
    excluded_test_identities = sorted(test_manifest_identities & training_identities)
    if args.require_unseen_test:
        test = [
            row for row in test if row["identity"] not in training_identities
        ]
        if not test:
            raise ValueError("No identity-disjoint test records remain")
    calibration_identities = {row["identity"] for row in validation}
    test_identities = {row["identity"] for row in test}
    calibration_training_overlap = sorted(
        calibration_identities & training_identities
    )
    test_training_overlap = sorted(test_identities & training_identities)
    exclusion_paths = [
        *training_paths,
        args.validation_manifest,
        args.test_manifest,
        WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/validation_manifest.json",
        WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_joint800_protocol_v1/blind_test_manifest.json",
        WORKSPACE
        / "artifacts/runs/legacy/dogfacenet_shared_v3_protocol_v1/fresh_blind_manifest.json",
    ]
    excluded, exclusion_evidence = manifest_identities(exclusion_paths)
    excluded.update(row["identity"] for row in validation)
    excluded.update(row["identity"] for row in test)
    calibration_unknown, test_unknown, unknown_audit = select_unknown_records(
        args.dataset_root,
        excluded_identities=excluded,
        calibration_identities=args.unknown_calibration_identities,
        test_identities=args.unknown_test_identities,
        images_per_identity=args.unknown_images_per_identity,
        seed=args.seed,
    )
    protocol = {
        "schema_version": 1,
        "name": "agent_v1_fixed_gallery_open_set_v1",
        "created_at": utc_now(),
        "seed": args.seed,
        "policy": {
            "calibration_known": "joint100 validation; thresholds only",
            "test_known": "joint100 blind; final metrics only",
            "unknown": "identity-disjoint raw DogFaceNet samples excluded from locked model protocols",
            "gallery_images_per_identity": args.gallery_images_per_identity,
            "threshold_target_known_recall": args.target_known_recall,
            "test_threshold_tuning": False,
        },
        "source_manifests": {
            "validation": {
                "path": str(args.validation_manifest.resolve()),
                "sha256": sha256_file(args.validation_manifest),
            },
            "test": {
                "path": str(args.test_manifest.resolve()),
                "sha256": sha256_file(args.test_manifest),
            },
            "identity_exclusions": exclusion_evidence,
        },
        "splits": {
            "calibration_known": validation,
            "calibration_unknown": calibration_unknown,
            "test_known": test,
            "test_unknown": test_unknown,
        },
        "audit": {
            "model_training_protocols": training_evidence,
            "model_training_identities": len(training_identities),
            "calibration_known_identities": len(calibration_identities),
            "test_known_identities": len(test_identities),
            "test_manifest_identities_before_unseen_filter": len(
                test_manifest_identities
            ),
            "require_unseen_test": bool(args.require_unseen_test),
            "test_identities_removed_by_unseen_filter": (
                excluded_test_identities if args.require_unseen_test else []
            ),
            "calibration_known_training_overlap_count": len(
                calibration_training_overlap
            ),
            "calibration_known_training_overlap": calibration_training_overlap,
            "test_known_training_overlap_count": len(test_training_overlap),
            "test_known_training_overlap": test_training_overlap,
            "test_identity_generalization_status": (
                "identity_disjoint_from_model_training"
                if not test_training_overlap
                else "training_identity_overlap_not_unseen_generalization"
            ),
            "calibration_test_known_overlap": len(
                {row["identity"] for row in validation}
                & {row["identity"] for row in test}
            ),
            "known_unknown_overlap": len(
                ({row["identity"] for row in validation + test})
                & {
                    row["identity"]
                    for row in calibration_unknown + test_unknown
                }
            ),
            "unknown_selection": unknown_audit,
        },
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    return protocol


class FeatureCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                cache_key TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                bifor_model_sha256 TEXT NOT NULL,
                mega_model_sha256 TEXT NOT NULL,
                bifor_dim INTEGER NOT NULL,
                bifor BLOB NOT NULL,
                mega_dim INTEGER NOT NULL,
                mega BLOB NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def key(
        source_sha256: str, bifor_model_sha256: str, mega_model_sha256: str
    ) -> str:
        return hashlib.sha256(
            f"{source_sha256}:{bifor_model_sha256}:{mega_model_sha256}".encode(
                "ascii"
            )
        ).hexdigest()

    def get(self, cache_key: str) -> "CachedFeature | None":
        row = self.connection.execute(
            "SELECT * FROM features WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        bifor = np.frombuffer(row["bifor"], dtype=np.float32).copy()
        mega = np.frombuffer(row["mega"], dtype=np.float32).copy()
        if bifor.size != row["bifor_dim"] or mega.size != row["mega_dim"]:
            raise RuntimeError(f"Corrupt cache row: {cache_key}")
        return CachedFeature(
            bifor=normalize_feature(bifor, "cached bifor"),
            mega=normalize_feature(mega, "cached mega"),
            metadata=json.loads(row["metadata_json"]),
        )

    def put(
        self,
        cache_key: str,
        record: dict[str, Any],
        feature: "CachedFeature",
        bifor_model_sha256: str,
        mega_model_sha256: str,
    ) -> None:
        bifor = np.ascontiguousarray(feature.bifor, dtype=np.float32)
        mega = np.ascontiguousarray(feature.mega, dtype=np.float32)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO features (
                cache_key, source_path, source_sha256,
                bifor_model_sha256, mega_model_sha256,
                bifor_dim, bifor, mega_dim, mega, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                record["source_path"],
                record["source_sha256"],
                bifor_model_sha256,
                mega_model_sha256,
                bifor.size,
                bifor.tobytes(),
                mega.size,
                mega.tobytes(),
                json.dumps(feature.metadata, ensure_ascii=False, default=json_default),
                utc_now(),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


@dataclass(frozen=True)
class CachedFeature:
    bifor: np.ndarray
    mega: np.ndarray
    metadata: dict[str, Any]


def feature_from_encoded(encoded: EncodedPetImage) -> CachedFeature:
    mega = encoded.expert_features.get(MEGADESCRIPTOR_EXPERT_ID)
    if mega is None:
        raise RuntimeError("MegaDescriptor feature is missing")
    return CachedFeature(
        bifor=normalize_feature(encoded.fused, "bifor"),
        mega=normalize_feature(mega, "megadescriptor"),
        metadata={
            "primary": encoded.metadata,
            "experts": encoded.expert_metadata,
        },
    )


def reliability_weights(feature: CachedFeature) -> dict[str, float]:
    encoded = EncodedPetImage(
        fused=feature.bifor,
        nose=np.empty(0, dtype=np.float32),
        face=np.empty(0, dtype=np.float32),
        metadata=feature.metadata["primary"],
        expert_features={MEGADESCRIPTOR_EXPERT_ID: feature.mega},
        expert_metadata=feature.metadata["experts"],
    )
    return expert_reliabilities(encoded, [MEGADESCRIPTOR_EXPERT_ID])


def all_protocol_records(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for records in protocol["splits"].values():
        for record in records:
            key = record["source_sha256"]
            previous = unique.get(key)
            if previous and previous["source_path"] != record["source_path"]:
                raise ValueError(f"Duplicate content uses different paths: {key}")
            unique[key] = record
    return list(unique.values())


def extract_features(
    protocol: dict[str, Any],
    encoder: AgentFeatureEncoder,
    cache: FeatureCache,
    backend_info: dict[str, Any],
) -> tuple[dict[str, CachedFeature], dict[str, Any]]:
    bifor_sha = str(backend_info["model_sha256"])
    mega_sha = str(
        backend_info["experts"][MEGADESCRIPTOR_EXPERT_ID]["model_sha256"]
    )
    records = all_protocol_records(protocol)
    features: dict[str, CachedFeature] = {}
    cache_hits = 0
    encoded_count = 0
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        cache_key = cache.key(record["source_sha256"], bifor_sha, mega_sha)
        cached = cache.get(cache_key)
        if cached is None:
            item_started = time.perf_counter()
            encoded = encoder.encode_file(Path(record["source_path"]))
            cached = feature_from_encoded(encoded)
            cache.put(cache_key, record, cached, bifor_sha, mega_sha)
            encoded_count += 1
            status = f"encoded {time.perf_counter() - item_started:.2f}s"
        else:
            cache_hits += 1
            status = "cache"
        features[record["source_sha256"]] = cached
        print(
            f"[{index:03d}/{len(records):03d}] {status:>14} "
            f"{record['origin']:<20} {record['canonical_filename']}",
            flush=True,
        )
    return features, {
        "cache_path": str(cache.path.resolve()),
        "unique_images": len(records),
        "cache_hits": cache_hits,
        "encoded": encoded_count,
        "wall_seconds": time.perf_counter() - started,
    }


def split_records(
    protocol: dict[str, Any], known_name: str, unknown_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    known = protocol["splits"][known_name]
    gallery = [row for row in known if row["role"] == "gallery"]
    queries = [row for row in known if row["role"] == "query"]
    unknown = protocol["splits"][unknown_name]
    return gallery, queries, unknown


def build_prototypes(
    gallery: Sequence[dict[str, Any]], features: dict[str, CachedFeature]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, list[CachedFeature]] = defaultdict(list)
    for record in gallery:
        grouped[record["identity"]].append(features[record["source_sha256"]])
    identities = sorted(grouped)
    bifor = np.stack(
        [normalized_mean([row.bifor for row in grouped[name]]) for name in identities]
    )
    mega = np.stack(
        [normalized_mean([row.mega for row in grouped[name]]) for name in identities]
    )
    return identities, bifor, mega


def score_queries(
    records: Sequence[dict[str, Any]],
    features: dict[str, CachedFeature],
    prototype_identities: Sequence[str],
    bifor_prototypes: np.ndarray,
    mega_prototypes: np.ndarray,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        feature = features[record["source_sha256"]]
        bifor_scores = feature.bifor @ bifor_prototypes.T
        mega_scores = feature.mega @ mega_prototypes.T
        weights = reliability_weights(feature)
        agent_scores = (
            weights["bifor"] * bifor_scores
            + weights[MEGADESCRIPTOR_EXPERT_ID] * mega_scores
        )
        output[record["source_sha256"]] = {
            "record": record,
            "bifor": np.asarray(bifor_scores, dtype=np.float32),
            "mega": np.asarray(mega_scores, dtype=np.float32),
            "agent": np.asarray(agent_scores, dtype=np.float32),
            "weights": weights,
            "prototype_identities": list(prototype_identities),
        }
    return output


def closed_set_metrics(
    scores: dict[str, dict[str, Any]], method: str
) -> dict[str, Any]:
    queries = []
    ranks = []
    positive_scores = []
    negative_scores = []
    for row in scores.values():
        record = row["record"]
        values = row[method]
        identities = row["prototype_identities"]
        order = np.argsort(-values).tolist()
        true_column = identities.index(record["identity"])
        rank = order.index(true_column) + 1
        ranks.append(rank)
        positive_scores.append(float(values[true_column]))
        negative_scores.extend(
            float(value)
            for column, value in enumerate(values)
            if column != true_column
        )
        queries.append(
            {
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "query_identity": record["identity"],
                "predicted_identity": identities[order[0]],
                "true_rank": rank,
                "correct": rank == 1,
                "top1_score": float(values[order[0]]),
                "true_score": float(values[true_column]),
                "top5": [
                    {"identity": identities[column], "score": float(values[column])}
                    for column in order[:5]
                ],
                "weights": row["weights"] if method == "agent" else None,
            }
        )
    count = len(queries)
    reciprocal = [1.0 / rank for rank in ranks]
    return {
        "queries": count,
        "top1_correct": sum(rank == 1 for rank in ranks),
        "top1_accuracy": sum(rank == 1 for rank in ranks) / count,
        "top5_correct": sum(rank <= 5 for rank in ranks),
        "top5_accuracy": sum(rank <= 5 for rank in ranks) / count,
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "mean_average_precision": float(np.mean(reciprocal)),
        "map_note": "one relevant identity prototype per query, so AP=reciprocal rank",
        "true_rank": summarize(ranks),
        "same_score": summarize(positive_scores),
        "different_score": summarize(negative_scores),
        "verification_auc": exact_auc(positive_scores, negative_scores),
        "query_results": queries,
    }


def open_score_rows(
    known_scores: dict[str, dict[str, Any]],
    unknown_scores: dict[str, dict[str, Any]],
    method: str,
) -> tuple[list[float], list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    known = []
    known_rows = []
    for row in known_scores.values():
        values = row[method]
        best = int(np.argmax(values))
        score = float(values[best])
        known.append(score)
        known_rows.append(
            {
                "source_path": row["record"]["source_path"],
                "source_sha256": row["record"]["source_sha256"],
                "true_identity": row["record"]["identity"],
                "predicted_identity": row["prototype_identities"][best],
                "top1_score": score,
                "top1_correct": (
                    row["prototype_identities"][best]
                    == row["record"]["identity"]
                ),
            }
        )
    unknown = []
    unknown_rows = []
    for row in unknown_scores.values():
        values = row[method]
        best = int(np.argmax(values))
        score = float(values[best])
        unknown.append(score)
        unknown_rows.append(
            {
                "source_path": row["record"]["source_path"],
                "source_sha256": row["record"]["source_sha256"],
                "unknown_identity": row["record"]["identity"],
                "predicted_identity": row["prototype_identities"][best],
                "top1_score": score,
            }
        )
    return known, unknown, known_rows, unknown_rows


def apply_open_threshold(
    known_rows: Sequence[dict[str, Any]],
    unknown_rows: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    accepted_known = [row for row in known_rows if row["top1_score"] >= threshold]
    rejected_known = [row for row in known_rows if row["top1_score"] < threshold]
    false_accepts = [row for row in unknown_rows if row["top1_score"] >= threshold]
    correct_accepted = [row for row in accepted_known if row["top1_correct"]]
    return {
        "threshold": float(threshold),
        "known_count": len(known_rows),
        "unknown_count": len(unknown_rows),
        "false_accept_rate": len(false_accepts) / len(unknown_rows),
        "false_reject_rate": len(rejected_known) / len(known_rows),
        "unknown_rejection_rate": 1.0 - len(false_accepts) / len(unknown_rows),
        "known_acceptance_rate": len(accepted_known) / len(known_rows),
        "known_acceptance_accuracy": len(correct_accepted) / len(known_rows),
        "accepted_known_classification_accuracy": (
            len(correct_accepted) / len(accepted_known) if accepted_known else 0.0
        ),
        "false_accepts": false_accepts,
        "false_rejects": rejected_known,
    }


def open_set_metrics(
    calibration_known: dict[str, dict[str, Any]],
    calibration_unknown: dict[str, dict[str, Any]],
    test_known: dict[str, dict[str, Any]],
    test_unknown: dict[str, dict[str, Any]],
    method: str,
    target_known_recall: float,
) -> dict[str, Any]:
    cal_known, cal_unknown, _, _ = open_score_rows(
        calibration_known, calibration_unknown, method
    )
    test_known_values, test_unknown_values, known_rows, unknown_rows = open_score_rows(
        test_known, test_unknown, method
    )
    recall_threshold = threshold_for_known_recall(cal_known, target_known_recall)
    balanced = best_balanced_threshold(cal_known, cal_unknown)
    return {
        "score": "maximum identity-prototype cosine",
        "test_auroc": exact_auc(test_known_values, test_unknown_values),
        "calibration": {
            "known_score": summarize(cal_known),
            "unknown_score": summarize(cal_unknown),
            "auroc": exact_auc(cal_known, cal_unknown),
            "known_recall_threshold": recall_threshold,
            "best_balanced_threshold": balanced,
        },
        "test": {
            "known_score": summarize(test_known_values),
            "unknown_score": summarize(test_unknown_values),
            "at_calibrated_95_known_recall": apply_open_threshold(
                known_rows, unknown_rows, recall_threshold["threshold"]
            ),
            "at_calibrated_balanced_threshold": apply_open_threshold(
                known_rows, unknown_rows, balanced["threshold"]
            ),
        },
    }


def quality_bucket(feature: CachedFeature) -> set[str]:
    mega = feature.metadata.get("experts", {}).get(MEGADESCRIPTOR_EXPERT_ID, {})
    quality = mega.get("quality", {}) if isinstance(mega, dict) else {}
    luminance = float(quality.get("luminance_mean", 0.5))
    sharpness = float(quality.get("sharpness", 0.5))
    buckets = {"all"}
    if luminance < 0.25:
        buckets.add("low_light")
    elif luminance > 0.75:
        buckets.add("high_light")
    else:
        buckets.add("mid_light")
    buckets.add("blur_or_low_detail" if sharpness < 0.35 else "sharp")
    buckets.add("body_detected" if mega.get("body_detected") else "body_fallback")
    return buckets


def quality_slices(
    test_scores: dict[str, dict[str, Any]],
    features: dict[str, CachedFeature],
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, row in test_scores.items():
        for bucket in quality_bucket(features[key]):
            buckets[bucket].append(row)
    output = {}
    for bucket, rows in sorted(buckets.items()):
        methods = {}
        for method in ("bifor", "mega", "agent"):
            correct = 0
            for row in rows:
                best = int(np.argmax(row[method]))
                correct += (
                    row["prototype_identities"][best] == row["record"]["identity"]
                )
            methods[method] = {
                "top1_correct": int(correct),
                "top1_accuracy": correct / len(rows),
            }
        output[bucket] = {"queries": len(rows), "methods": methods}
    return output


def comparison_cases(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_method = {
        method: {row["source_sha256"]: row for row in result["query_results"]}
        for method, result in metrics.items()
    }
    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, bifor in by_method["bifor"].items():
        mega = by_method["mega"][key]
        agent = by_method["agent"][key]
        summary = {
            "source_path": bifor["source_path"],
            "source_sha256": key,
            "true_identity": bifor["query_identity"],
            "bifor_prediction": bifor["predicted_identity"],
            "bifor_score": bifor["top1_score"],
            "mega_prediction": mega["predicted_identity"],
            "mega_score": mega["top1_score"],
            "agent_prediction": agent["predicted_identity"],
            "agent_score": agent["top1_score"],
            "agent_weights": agent["weights"],
        }
        if bifor["correct"] and not mega["correct"]:
            cases["bifor_correct_mega_wrong"].append(summary)
        if mega["correct"] and not bifor["correct"]:
            cases["mega_correct_bifor_wrong"].append(summary)
        if agent["correct"] and not bifor["correct"]:
            cases["agent_fixes_bifor"].append(summary)
        if not agent["correct"] and bifor["correct"]:
            cases["agent_regresses_bifor"].append(summary)
        if bifor["predicted_identity"] != mega["predicted_identity"]:
            cases["expert_disagreement"].append(summary)
        if not agent["correct"]:
            cases["agent_errors"].append(summary)
    for name in (
        "bifor_correct_mega_wrong",
        "mega_correct_bifor_wrong",
        "agent_fixes_bifor",
        "agent_regresses_bifor",
        "expert_disagreement",
        "agent_errors",
    ):
        cases.setdefault(name, [])
    return dict(cases)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_contact_sheet(
    rows: Sequence[dict[str, Any]], output: Path, *, limit: int = 24
) -> Path | None:
    if not rows:
        return None
    rows = list(rows[:limit])
    columns = 4
    cell_width, image_height, text_height = 300, 230, 92
    cell_height = image_height + text_height
    sheet = Image.new(
        "RGB",
        (columns * cell_width, math.ceil(len(rows) / columns) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(15)
    small = load_font(13)
    for index, row in enumerate(rows):
        left = (index % columns) * cell_width
        top = (index // columns) * cell_height
        with Image.open(row["source_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
        x = left + (cell_width - image.width) // 2
        y = top + (image_height - image.height) // 2
        sheet.paste(image, (x, y))
        label = (
            f"true={row.get('true_identity', row.get('unknown_identity', 'unknown'))}\n"
            f"B={row.get('bifor_prediction', '-')}  "
            f"M={row.get('mega_prediction', '-')}\n"
            f"A={row.get('agent_prediction', row.get('predicted_identity', '-'))}  "
            f"score={row.get('agent_score', row.get('top1_score', 0.0)):.3f}"
        )
        draw.rectangle(
            (left, top + image_height, left + cell_width, top + cell_height),
            fill=(248, 248, 248),
        )
        draw.text((left + 7, top + image_height + 5), label, fill="black", font=small)
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline=(190, 190, 190),
            width=1,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return output


def concise_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "queries",
        "top1_correct",
        "top1_accuracy",
        "top5_correct",
        "top5_accuracy",
        "mean_reciprocal_rank",
        "mean_average_precision",
        "verification_auc",
    )
    return {key: metrics[key] for key in keys}


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    test = report["closed_set"]["test"]
    calibration = report["closed_set"]["calibration"]
    paired = report["paired_comparisons"]["test_agent_vs_bifor"]
    audit = report["protocol"]["audit"]
    calibration_identities = audit["calibration_known_identities"]
    test_identities = audit["test_known_identities"]
    test_overlap = audit["test_known_training_overlap_count"]
    test_status = audit["test_identity_generalization_status"]
    lines = [
        "# Agent V1 正式准确率评测",
        "",
        (
            f"测试协议：{calibration_identities} 个校准身份仅用于阈值；"
            f"{test_identities} 个测试身份用于最终闭集结果。"
        ),
        (
            f"测试身份与当前模型训练身份重叠 {test_overlap}/{test_identities}；"
            f"协议状态：{test_status}。"
        ),
        "",
        "| 方法 | 校准 Top-1 | 盲测 Top-1 | 盲测 Top-5 | MRR | mAP | 同异样本 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"bifor": "BIFOR-only", "mega": "MegaDescriptor-only", "agent": "Agent 融合"}
    for method in ("bifor", "mega", "agent"):
        row = test[method]
        lines.append(
            f"| {labels[method]} | {calibration[method]['top1_accuracy']:.4%} | "
            f"{row['top1_accuracy']:.4%} | "
            f"{row['top5_accuracy']:.4%} | {row['mean_reciprocal_rank']:.6f} | "
            f"{row['mean_average_precision']:.6f} | {row['verification_auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Agent 相对 BIFOR 的盲测配对差异为 "
                f"{paired['candidate_only_correct']} 张修正 / "
                f"{paired['base_only_correct']} 张退化，"
                f"exact McNemar p={paired['exact_mcnemar_pvalue_two_sided']:.6f}；"
                "当前样本量不足以证明稳定提升。"
            ),
            "",
            "开放集阈值只由验证集和独立校准 unknown 身份确定，未在盲测集调参。",
            "",
        ]
    )
    lines.extend(
        [
            "| 方法 | AUROC | FAR@95%校准已知召回 | FRR | 库外拒识率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("bifor", "mega", "agent"):
        row = report["open_set"][method]
        threshold = row["test"]["at_calibrated_95_known_recall"]
        lines.append(
            f"| {labels[method]} | {row['test_auroc']:.6f} | "
            f"{threshold['false_accept_rate']:.4%} | "
            f"{threshold['false_reject_rate']:.4%} | "
            f"{threshold['unknown_rejection_rate']:.4%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_ROOT / "validation_manifest.json",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_ROOT / "blind_test_manifest.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE / "data/raw/DogFaceNet_alignment",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="optional shared SQLite cache; defaults to OUTPUT_DIR/feature_cache.sqlite3",
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_BIFOR_PACKAGE / "config.yaml")
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=DEFAULT_SEMANTIC_PACKAGE / "model_final.pth",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=DEFAULT_BIFOR_PACKAGE / "onnx/pet_embedding.onnx",
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=WORKSPACE
        / "models/pretrained/body_detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        default=WORKSPACE
        / "models/pretrained/megadescriptor/MegaDescriptor-B-224/pytorch_model.bin",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megadescriptor-device")
    parser.add_argument("--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument("--gallery-images-per-identity", type=int, default=2)
    parser.add_argument("--unknown-calibration-identities", type=int, default=20)
    parser.add_argument("--unknown-test-identities", type=int, default=20)
    parser.add_argument("--unknown-images-per-identity", type=int, default=3)
    parser.add_argument("--target-known-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--require-unseen-test",
        action="store_true",
        help="remove every test identity present in either locked training protocol",
    )
    parser.add_argument("--rebuild-protocol", action="store_true")
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="write and audit the fixed protocol without loading any model",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gallery_images_per_identity < 1:
        raise ValueError("gallery-images-per-identity must be positive")
    if not 0.0 < args.target_known_recall <= 1.0:
        raise ValueError("target-known-recall must be in (0, 1]")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.is_file() and not args.rebuild_protocol:
        protocol = read_json(protocol_path)
        expected = protocol.pop("protocol_sha256")
        actual = sha256_json(protocol)
        protocol["protocol_sha256"] = expected
        if actual != expected:
            raise RuntimeError(
                f"Protocol hash mismatch: expected {expected}, got {actual}"
            )
    else:
        protocol = build_protocol(args)
        write_json(protocol_path, protocol)
    (args.output_dir / "protocol.sha256").write_text(
        protocol["protocol_sha256"] + "  protocol.json\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "protocol": str(protocol_path),
                "protocol_sha256": protocol["protocol_sha256"],
                "audit": protocol["audit"],
                "records": {
                    name: len(rows) for name, rows in protocol["splits"].items()
                },
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
                args.megadescriptor_checkpoint,
                device=args.megadescriptor_device or args.device,
            )
        ],
    )
    backend_info = encoder.backend_info()
    cache = FeatureCache(
        args.feature_cache.expanduser().resolve()
        if args.feature_cache
        else args.output_dir / "feature_cache.sqlite3"
    )
    try:
        features, extraction = extract_features(
            protocol, encoder, cache, backend_info
        )
    finally:
        cache.close()

    calibration_gallery, calibration_queries, calibration_unknown_records = split_records(
        protocol, "calibration_known", "calibration_unknown"
    )
    test_gallery, test_queries, test_unknown_records = split_records(
        protocol, "test_known", "test_unknown"
    )
    cal_ids, cal_bifor, cal_mega = build_prototypes(calibration_gallery, features)
    test_ids, test_bifor, test_mega = build_prototypes(test_gallery, features)
    calibration_known_scores = score_queries(
        calibration_queries, features, cal_ids, cal_bifor, cal_mega
    )
    calibration_unknown_scores = score_queries(
        calibration_unknown_records, features, cal_ids, cal_bifor, cal_mega
    )
    test_known_scores = score_queries(
        test_queries, features, test_ids, test_bifor, test_mega
    )
    test_unknown_scores = score_queries(
        test_unknown_records, features, test_ids, test_bifor, test_mega
    )
    calibration_closed = {
        method: closed_set_metrics(calibration_known_scores, method)
        for method in ("bifor", "mega", "agent")
    }
    test_closed = {
        method: closed_set_metrics(test_known_scores, method)
        for method in ("bifor", "mega", "agent")
    }
    open_set = {
        method: open_set_metrics(
            calibration_known_scores,
            calibration_unknown_scores,
            test_known_scores,
            test_unknown_scores,
            method,
            args.target_known_recall,
        )
        for method in ("bifor", "mega", "agent")
    }
    cases = comparison_cases(test_closed)
    sheets = {}
    for name, rows in cases.items():
        rendered = render_contact_sheet(
            rows, args.output_dir / "failure_cases" / f"{name}.jpg"
        )
        sheets[name] = str(rendered.resolve()) if rendered else None
    false_accepts = open_set["agent"]["test"][
        "at_calibrated_95_known_recall"
    ]["false_accepts"]
    rendered = render_contact_sheet(
        false_accepts,
        args.output_dir / "failure_cases/agent_unknown_false_accepts.jpg",
    )
    sheets["agent_unknown_false_accepts"] = (
        str(rendered.resolve()) if rendered else None
    )

    report = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol["protocol_sha256"],
            "audit": protocol["audit"],
            "policy": protocol["policy"],
        },
        "models": backend_info,
        "extraction": extraction,
        "closed_set": {
            "calibration": calibration_closed,
            "test": test_closed,
        },
        "paired_comparisons": {
            "calibration_agent_vs_bifor": paired_correctness(
                calibration_closed, base="bifor", candidate="agent"
            ),
            "test_agent_vs_bifor": paired_correctness(
                test_closed, base="bifor", candidate="agent"
            ),
            "test_mega_vs_bifor": paired_correctness(
                test_closed, base="bifor", candidate="mega"
            ),
        },
        "open_set": open_set,
        "quality_slices": quality_slices(test_known_scores, features),
        "comparisons": {
            name: {"count": len(rows), "rows": rows}
            for name, rows in cases.items()
        },
        "failure_case_sheets": sheets,
        "limitations": [
            "DogFaceNet is face-centric; many images do not contain a clean full-body view.",
            "The current Agent fusion is zero-shot quality weighting, not probability calibration.",
            "MegaDescriptor-B-224 weights are CC BY-NC 4.0 and non-commercial.",
            "mAP equals MRR here because the gallery contains one prototype per identity.",
        ],
    }
    report_path = args.output_dir / "report.json"
    write_json(report_path, report)
    summary_path = args.output_dir / "SUMMARY.md"
    write_markdown_summary(summary_path, report)
    concise = {
        "report": str(report_path),
        "summary": str(summary_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "extraction": extraction,
        "closed_set_test": {
            method: concise_metrics(metrics) for method, metrics in test_closed.items()
        },
        "open_set_test": {
            method: {
                "auroc": metrics["test_auroc"],
                **{
                    key: value
                    for key, value in metrics["test"][
                        "at_calibrated_95_known_recall"
                    ].items()
                    if key
                    in {
                        "threshold",
                        "false_accept_rate",
                        "false_reject_rate",
                        "unknown_rejection_rate",
                        "known_acceptance_accuracy",
                    }
                },
            }
            for method, metrics in open_set.items()
        },
        "comparisons": {name: len(rows) for name, rows in cases.items()},
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
