"""Strict, provenance-aware supervision for model-local face/nose usability.

No detector/landmark/segmentation heuristic is silently promoted to ground truth.
Labels refer to the *model's local crops*, not merely presence in the source RGB.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .unified_training import sha256_file
from .workspace_paths import WORKSPACE_ROOT, resolve_legacy_path

SCHEMA_NAME = "model_local_part_visibility"
PARTS = ("face", "nose")
PROVENANCES = {"human", "teacher_weak", "synthetic", "unknown"}
TRAIN_SPLITS = {"train", "dev_train", "training_extension"}
VALIDATION_SPLITS = {"validation", "dev_validation", "development"}
# Historical lineage, not a file-name convention. Renaming a consumed manifest
# cannot make it eligible for training, annotation, or another selection round.
CONSUMED_MANIFEST_SHA256 = {
    "33dbb399506b2122f1fca77b579336962b9071a1fe9be8d3e53c8a71ff5da236",
    "6f7ec3164e7f1535cc9eaf26c0cfa57abb0f62e40a6307eecb3ab034a9137d5a",
}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Unknown or missing {name} fields")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA-256")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text")
    return value


def _path(value):
    path = Path(value)
    return resolve_legacy_path(path) if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _box(value, size):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Expected complete part extent / retained crop xyxy")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Nonfinite or nonnumeric crop coordinates")
    left, top, right, bottom = value
    if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]):
        raise ValueError("Crop or complete part extent lies outside parent image")
    return value


def validate_label(label, *, identity):
    _keys(label, {"usable", "provenance", "evidence"}, "visibility label")
    provenance, usable, evidence = label["provenance"], label["usable"], label["evidence"]
    if provenance not in PROVENANCES:
        raise ValueError("Unknown visibility provenance")
    if provenance == "unknown":
        if usable is not None or evidence is not None:
            raise ValueError("Unknown observations must be null, never negatives")
        return
    if type(usable) is not bool:
        raise ValueError("Known usability must be an explicit boolean")
    if provenance == "human":
        _keys(evidence, {"annotation_id", "annotator", "scope"}, "human evidence")
        _text(evidence["annotation_id"])
        _text(evidence["annotator"])
        if evidence["scope"] != "model_local_crop":
            raise ValueError("Human usability must assess the model-local crop")
    elif provenance == "teacher_weak":
        _keys(evidence, {"teacher_id", "checkpoint_sha256", "observation", "target_matched"}, "teacher evidence")
        _text(evidence["teacher_id"])
        _digest(evidence["checkpoint_sha256"])
        if (usable is not True or evidence["observation"] != "positive_detection"
                or evidence["target_matched"] is not True):
            raise ValueError("Teacher missing/failed detections are unknown, not negatives")
    else:
        _keys(evidence, {
            "transform", "parent_source_sha256", "parent_identity", "parent_size",
            "complete_part_box_xyxy", "retained_crop_xyxy", "annotation_id",
            "part_extent_provenance", "complete_part_extent",
        }, "synthetic evidence")
        if usable is not False or evidence["transform"] != "crop_excludes_complete_part":
            raise ValueError("Only proven complete-part exclusion creates a synthetic negative")
        _digest(evidence["parent_source_sha256"])
        if _text(evidence["parent_identity"]).strip().casefold() != identity.strip().casefold():
            raise ValueError("Synthetic identity differs from its parent")
        _text(evidence["annotation_id"])
        if evidence["part_extent_provenance"] != "human" or evidence["complete_part_extent"] is not True:
            raise ValueError("A landmark/teacher box does not certify the complete part extent")
        size = evidence["parent_size"]
        if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x < 2 for x in size):
            raise ValueError("Invalid parent size")
        part = _box(evidence["complete_part_box_xyxy"], size)
        crop = _box(evidence["retained_crop_xyxy"], size)
        if any(type(x) is not int for x in crop):
            raise ValueError("Synthetic retained crop must use integer pixel edges")
        if min(part[2], crop[2]) > max(part[0], crop[0]) and min(part[3], crop[3]) > max(part[1], crop[1]):
            raise ValueError("Retained crop still intersects the annotated part")


def load_visibility_manifest(path, *, expected_split, expected_target_graph_sha256):
    """Validate split authority, membership, provenance and target-graph binding."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    _keys(payload, {
        "schema_version", "schema_name", "split", "target_graph_sha256",
        "source_manifests", "records",
    }, "visibility manifest")
    if payload["schema_version"] != 1 or payload["schema_name"] != SCHEMA_NAME:
        raise ValueError("Unsupported visibility manifest schema")
    if expected_split not in {"train", "validation"} or payload["split"] != expected_split:
        raise ValueError("Visibility split is not authorized for this operation")
    if _digest(payload["target_graph_sha256"]) != _digest(expected_target_graph_sha256):
        raise ValueError("Visibility labels refer to a different local-crop graph")
    sources = {}
    if not isinstance(payload["source_manifests"], list) or not payload["source_manifests"]:
        raise ValueError("Visibility manifest requires explicit source manifests")
    for source in payload["source_manifests"]:
        _keys(source, {"path", "sha256"}, "source manifest")
        digest = _digest(source["sha256"])
        if digest in CONSUMED_MANIFEST_SHA256:
            raise ValueError("Consumed blind lineage is forbidden")
        if digest in sources:
            raise ValueError("Duplicate source manifest")
        source_path = _path(source["path"])
        if sha256_file(source_path) != digest:
            raise ValueError("Source manifest hash mismatch")
        manifest = json.loads(source_path.read_text(encoding="utf-8-sig"))
        # A visibility validation split is identity-held-out *from an authorized
        # training-pool source*, not the historical model-selection manifests.
        # Development/validation source files stay protected and are never
        # accepted as a shortcut to create new labels.
        allowed = TRAIN_SPLITS
        if manifest.get("protocol_split") not in allowed:
            raise ValueError("Source manifest split cannot be used for this operation")
        entries = {}
        for record in manifest["records"]:
            key = (_text(record["identity"]).strip().casefold(), _digest(record["source_sha256"]))
            entries[key] = _path(record["source_path"])
        sources[digest] = entries
    records, seen = payload["records"], set()
    if not isinstance(records, list) or not records:
        raise ValueError("Visibility manifest has no records")
    for record in records:
        _keys(record, {
            "identity", "source_path", "source_sha256", "source_manifest_sha256", "local_visibility",
        }, "visibility record")
        identity = _text(record["identity"]).strip().casefold()
        source_hash = _digest(record["source_sha256"])
        if source_hash in seen:
            raise ValueError("Duplicate image in visibility manifest")
        seen.add(source_hash)
        source_entries = sources.get(_digest(record["source_manifest_sha256"]))
        if source_entries is None:
            raise ValueError("Record refers to an undeclared source manifest")
        _keys(record["local_visibility"], PARTS, "part labels")
        synthetic = []
        for part in PARTS:
            label = record["local_visibility"][part]
            validate_label(label, identity=identity)
            if label["provenance"] == "synthetic":
                synthetic.append(label["evidence"])
        parent_hashes = {e["parent_source_sha256"] for e in synthetic}
        if len(parent_hashes) > 1:
            raise ValueError("Synthetic parts have inconsistent parents")
        membership_hash = next(iter(parent_hashes), source_hash)
        if (identity, membership_hash) not in source_entries:
            raise ValueError("Image/identity is not a member of the authorized source manifest")
        if synthetic:
            transforms = {tuple(e["retained_crop_xyxy"]) for e in synthetic}
            sizes = {tuple(e["parent_size"]) for e in synthetic}
            if len(transforms) != 1 or len(sizes) != 1:
                raise ValueError("Synthetic labels disagree on the actual transform")
            record["_parent_path"] = str(source_entries[(identity, membership_hash)])
        record["_source_path"] = str(_path(record["source_path"]))
    return {**payload, "manifest_path": str(path), "manifest_sha256": sha256_file(path)}


def lineage_sets(manifest):
    identities, hashes = set(), set()
    for record in manifest["records"]:
        identities.add(record["identity"].strip().casefold())
        hashes.add(record["source_sha256"])
        for label in record["local_visibility"].values():
            if label["provenance"] == "synthetic":
                hashes.add(label["evidence"]["parent_source_sha256"])
    return identities, hashes


def require_independent_visibility_splits(train, validation, *, protected_records=()):
    """Fail closed before training: identity AND file/parent disjointness."""
    if train["split"] != "train" or validation["split"] != "validation":
        raise ValueError("Expected a training/validation pair")
    if train["target_graph_sha256"] != validation["target_graph_sha256"]:
        raise ValueError("Splits label different target graphs")
    train_ids, train_hashes = lineage_sets(train)
    validation_ids, validation_hashes = lineage_sets(validation)
    protected_ids = {r["identity"].strip().casefold() for r in protected_records}
    protected_hashes = {r["source_sha256"].lower() for r in protected_records}
    if train_ids & (validation_ids | protected_ids) or train_hashes & (validation_hashes | protected_hashes):
        raise ValueError("Training overlaps protected development/validation identities or source lineage")
    # Necessary, not sufficient sample-size criteria. Synthetic holdout negatives
    # do not establish correctness on real occlusion/domain-shift cases.
    for part in PARTS:
        training_classes = {
            r["local_visibility"][part]["usable"] for r in train["records"]
            if r["local_visibility"][part]["provenance"] in {"human", "synthetic"}
        }
        holdout_classes = {
            r["local_visibility"][part]["usable"] for r in validation["records"]
            if r["local_visibility"][part]["provenance"] == "human"
        }
        if training_classes != {False, True} or holdout_classes != {False, True}:
            raise ValueError(f"Missing real positive/negative independent visibility supervision: {part}")


def masked_visibility_loss(logits, targets, weights):
    if logits.ndim != 2 or logits.shape[1] != 2 or targets.shape != logits.shape or weights.shape != logits.shape:
        raise ValueError("Visibility loss expects matching [N,2] tensors")
    if not torch.isfinite(targets).all() or not torch.isfinite(weights).all():
        raise ValueError("Visibility targets/weights must be finite")
    if ((targets != 0) & (targets != 1)).any() or (weights < 0).any():
        raise ValueError("Invalid visibility targets/weights")
    loss = F.binary_cross_entropy_with_logits(logits, targets.to(logits), reduction="none")
    weights = weights.to(logits)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8).copy()


class LocalVisibilityDataset(Dataset):
    """Raw variable-size images; teacher-weak supervision is disabled by default."""

    def __init__(self, manifest, *, teacher_weak_weight=0.0):
        if not 0 <= teacher_weak_weight <= 1:
            raise ValueError("teacher_weak_weight must be in [0,1]")
        self.records = manifest["records"]
        self.weights = {"human": 1.0, "synthetic": 1.0, "teacher_weak": float(teacher_weak_weight), "unknown": 0.0}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["_source_path"])
        if sha256_file(path) != record["source_sha256"]:
            raise ValueError("Visibility image hash mismatch")
        rgb = _read_rgb(path)
        for label in record["local_visibility"].values():
            if label["provenance"] != "synthetic":
                continue
            evidence = label["evidence"]
            parent = Path(record["_parent_path"])
            if sha256_file(parent) != evidence["parent_source_sha256"]:
                raise ValueError("Synthetic parent image hash mismatch")
            original = _read_rgb(parent)
            if list(original.shape[1::-1]) != evidence["parent_size"]:
                raise ValueError("Synthetic parent size differs from evidence")
            left, top, right, bottom = evidence["retained_crop_xyxy"]
            if not np.array_equal(rgb, original[top:bottom, left:right]):
                raise ValueError("Synthetic pixels do not implement the declared complete-part exclusion")
        if min(rgb.shape[:2]) < 2:
            raise ValueError("Visibility images require minimum side two")
        labels = [record["local_visibility"][part] for part in PARTS]
        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
            "targets": torch.tensor([float(label["usable"] is True) for label in labels]),
            "weights": torch.tensor([self.weights[label["provenance"]] for label in labels]),
            "provenance": [label["provenance"] for label in labels],
            "identity": record["identity"],
            "source_sha256": record["source_sha256"],
        }
