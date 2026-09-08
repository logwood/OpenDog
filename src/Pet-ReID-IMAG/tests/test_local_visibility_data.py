"""Provenance/split guards. All images and annotations here are test fixtures."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from pet_id.local_visibility_data import (
    CONSUMED_MANIFEST_SHA256,
    LocalVisibilityDataset,
    SCHEMA_NAME,
    load_visibility_manifest,
    masked_visibility_loss,
    require_independent_visibility_splits,
    validate_label,
)
from pet_id.unified_training import sha256_file

GRAPH = "e" * 64


def unknown():
    return {"usable": None, "provenance": "unknown", "evidence": None}


def human(value=True):
    return {"usable": value, "provenance": "human", "evidence": {
        "annotation_id": "unit-test-fixture", "annotator": "test-not-a-real-annotation",
        "scope": "model_local_crop",
    }}


def teacher():
    return {"usable": True, "provenance": "teacher_weak", "evidence": {
        "teacher_id": "unit-test", "checkpoint_sha256": "f" * 64,
        "observation": "positive_detection", "target_matched": True,
    }}


def write_fixture(tmp_path, *, source_split="dev_train", labels=None, synthetic=False):
    source = tmp_path / "source.png"
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    Image.fromarray(pixels).save(source)
    source_hash = sha256_file(source)
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps({"protocol_split": source_split, "records": [{
        "identity": "Dog-A", "source_path": str(source), "source_sha256": source_hash,
    }]}), encoding="utf-8")
    labels = labels or {"face": human(), "nose": unknown()}
    if synthetic:
        source = tmp_path / "synthetic.png"
        Image.fromarray(pixels[:, :4]).save(source)
        labels = {"face": unknown(), "nose": {"usable": False, "provenance": "synthetic", "evidence": {
            "transform": "crop_excludes_complete_part", "parent_source_sha256": source_hash,
            "parent_identity": "dog-a", "parent_size": [16, 12],
            "complete_part_box_xyxy": [8., 3., 12., 8.], "retained_crop_xyxy": [0, 0, 4, 12],
            "annotation_id": "unit-test-full-part", "part_extent_provenance": "human",
            "complete_part_extent": True,
        }}}
    payload = {
        "schema_version": 1, "schema_name": SCHEMA_NAME, "split": "train", "target_graph_sha256": GRAPH,
        "source_manifests": [{"path": str(manifest_path), "sha256": sha256_file(manifest_path)}],
        "records": [{"identity": "dog-a", "source_path": str(source), "source_sha256": sha256_file(source),
                     "source_manifest_sha256": sha256_file(manifest_path), "local_visibility": labels}],
    }
    path = tmp_path / "visibility.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def load(path):
    return load_visibility_manifest(path, expected_split="train", expected_target_graph_sha256=GRAPH)


def test_unknown_mask_and_weak_teacher_are_not_ground_truth(tmp_path):
    path, _ = write_fixture(tmp_path, labels={"face": teacher(), "nose": unknown()})
    manifest = load(path)
    sample = LocalVisibilityDataset(manifest)[0]
    assert sample["rgb"].shape == (3, 12, 16)
    assert sample["rgb"].dtype == torch.float32
    assert torch.equal(sample["weights"], torch.zeros(2))
    assert torch.equal(sample["targets"], torch.tensor([1., 0.]))
    enabled = LocalVisibilityDataset(manifest, teacher_weak_weight=0.25)[0]
    assert torch.equal(enabled["weights"], torch.tensor([0.25, 0.]))
    logits = torch.randn(1, 2, requires_grad=True)
    loss = masked_visibility_loss(logits, sample["targets"][None], sample["weights"][None])
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_human_positive_and_unknown_have_different_loss_gradients():
    logits = torch.zeros(1, 2, requires_grad=True)
    loss = masked_visibility_loss(logits, torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] == 0


@pytest.mark.parametrize("label", [
    {"usable": False, "provenance": "unknown", "evidence": None},
    {**teacher(), "usable": False},
    {**teacher(), "evidence": {**teacher()["evidence"], "observation": "no_detection"}},
    {**human(), "evidence": {**human()["evidence"], "scope": "source_image_presence"}},
    {**human(), "provenance": "development_score_derived"},
    {**human(), "unexpected_confidence": 1.0},
])
def test_unsafe_or_ambiguous_labels_are_rejected(label):
    with pytest.raises(ValueError):
        validate_label(label, identity="dog-a")


def test_source_split_cannot_be_laundered_by_renaming_visibility_file(tmp_path):
    path, _ = write_fixture(tmp_path, source_split="dev_validation")
    with pytest.raises(ValueError, match="Source manifest split"):
        load(path)


def test_visibility_validation_may_be_held_out_from_training_pool_only(tmp_path):
    path, payload = write_fixture(tmp_path, source_split="dev_train")
    payload["split"] = "validation"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_visibility_manifest(path, expected_split="validation", expected_target_graph_sha256=GRAPH)
    assert loaded["split"] == "validation"
    payload["source_manifests"][0]["path"] = str(tmp_path / "protected.json")
    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps({"protocol_split": "dev_validation", "records": payload["records"]}), encoding="utf-8")
    payload["source_manifests"][0]["sha256"] = sha256_file(protected)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Source manifest split"):
        load_visibility_manifest(path, expected_split="validation", expected_target_graph_sha256=GRAPH)


def test_consumed_source_fingerprint_is_forbidden_before_opening_records(tmp_path):
    path, payload = write_fixture(tmp_path)
    payload["source_manifests"][0]["sha256"] = next(iter(CONSUMED_MANIFEST_SHA256))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Consumed blind"):
        load(path)


def test_graph_binding_unknown_fields_and_source_membership_are_strict(tmp_path):
    path, original = write_fixture(tmp_path)
    for mutation, message in [
        (lambda p: p.update(target_graph_sha256="b" * 64), "different local-crop graph"),
        (lambda p: p.update(unknown_option=True), "fields"),
        (lambda p: p["records"][0].update(identity="another-dog"), "not a member"),
    ]:
        payload = copy.deepcopy(original)
        mutation(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load(path)


def test_source_image_hash_is_checked_on_loading_pixels(tmp_path):
    path, _ = write_fixture(tmp_path)
    manifest = load(path)
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(tmp_path / "source.png")
    with pytest.raises(ValueError, match="image hash mismatch"):
        LocalVisibilityDataset(manifest)[0]


def test_proven_synthetic_exclusion_is_verified_against_actual_pixels(tmp_path):
    path, payload = write_fixture(tmp_path, synthetic=True)
    sample = LocalVisibilityDataset(load(path))[0]
    assert sample["rgb"].shape == (3, 12, 4)
    assert torch.equal(sample["weights"], torch.tensor([0., 1.]))
    # A matching derivative hash is not sufficient: the derivative must actually
    # implement the declared parent crop, not an unrelated image.
    changed = tmp_path / "synthetic.png"
    Image.fromarray(np.zeros((12, 4, 3), dtype=np.uint8)).save(changed)
    payload["records"][0]["source_sha256"] = sha256_file(changed)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pixels do not implement"):
        LocalVisibilityDataset(load(path))[0]


@pytest.mark.parametrize("change", [
    {"retained_crop_xyxy": [0, 0, 16, 12]},
    {"complete_part_extent": False},
    {"part_extent_provenance": "teacher_weak"},
    {"parent_identity": "different-parent"},
])
def test_unproven_synthetic_negative_is_rejected(tmp_path, change):
    path, payload = write_fixture(tmp_path, synthetic=True)
    payload["records"][0]["local_visibility"]["nose"]["evidence"].update(change)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def split_fixture(split, prefix):
    return {"split": split, "target_graph_sha256": GRAPH, "records": [
        {"identity": f"{prefix}-{index}", "source_sha256": f"{prefix}-{index}",
         "local_visibility": {"face": human(value), "nose": human(value)}}
        for index, value in enumerate((False, True))
    ]}


def test_readiness_requires_identity_and_hash_disjoint_real_holdout():
    train, validation = split_fixture("train", "a"), split_fixture("validation", "b")
    require_independent_visibility_splits(train, validation)
    validation["records"][0]["identity"] = "A-0"
    with pytest.raises(ValueError, match="overlaps"):
        require_independent_visibility_splits(train, validation)
    validation = split_fixture("validation", "b")
    validation["records"][0]["source_sha256"] = "a-0"
    with pytest.raises(ValueError, match="overlaps"):
        require_independent_visibility_splits(train, validation)
    validation = split_fixture("validation", "b")
    validation["records"][0]["local_visibility"]["nose"] = unknown()
    with pytest.raises(ValueError, match="Missing real"):
        require_independent_visibility_splits(train, validation)


def test_synthetic_parent_and_other_protected_development_cannot_leak():
    train, validation = split_fixture("train", "a"), split_fixture("validation", "b")
    train["records"][0]["local_visibility"]["nose"] = {
        "usable": False, "provenance": "synthetic", "evidence": {"parent_source_sha256": "b-1"}
    }
    with pytest.raises(ValueError, match="overlaps"):
        require_independent_visibility_splits(train, validation)
    train = split_fixture("train", "a")
    with pytest.raises(ValueError, match="overlaps"):
        require_independent_visibility_splits(train, validation, protected_records=[{"identity": "A-1", "source_sha256": "other"}])


def test_audit_does_not_convert_geometry_proxies_or_missing_detection_to_labels():
    path = Path(__file__).resolve().parents[1] / "tools/audit_local_visibility_data.py"
    spec = importlib.util.spec_from_file_location("visibility_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.inventory_manifest({"records": [
        {"identity": "dog-a", "source_sha256": "a", "detection": {"confidence": 0.99}},
        {"identity": "dog-b", "source_sha256": "b", "detection": None},
    ]})
    assert result["explicit_visibility_labels"]["face"] == {"unknown": 2}
    assert result["explicit_visibility_labels"]["nose"] == {"unknown": 2}
    assert result["nonempty_fields"]["detection"] == 1
