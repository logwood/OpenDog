"""Integrity checks for the validated spatial-detail candidate package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pet_id.model_profiles import get_runtime_profile
from pet_id.workspace_paths import WORKSPACE_ROOT


CANDIDATE_PROFILE = get_runtime_profile("candidate")
PRODUCTION_PROFILE = get_runtime_profile("production")
CANDIDATE_PACKAGE = (
    WORKSPACE_ROOT
    / "models"
    / "selected"
    / CANDIDATE_PROFILE.model_package
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_candidate_package_is_single_dynamic_graph_and_blind_noninferior() -> None:
    deployment = read_json(CANDIDATE_PACKAGE / "deployment_record.json")
    metadata = read_json(CANDIDATE_PACKAGE / "onnx/metadata.json")
    blind = deployment["blind"]
    assert deployment["name"] == CANDIDATE_PROFILE.model_package
    assert deployment["status"] == "validated_optional_candidate"
    assert deployment["architecture"]["single_onnx_graph"] is True
    assert deployment["architecture"]["runtime_external_models"] == []
    assert deployment["deployment"]["default_changed"] is False
    assert deployment["deployment"]["current_default_package"] == (
        PRODUCTION_PROFILE.model_package
    )
    assert deployment["deployment"]["optional_backend"] == "onnx-highres"
    assert blind["passed"] is True
    assert blind["post_blind_tuning_permitted"] is False
    assert blind["candidate_top1_correct"] >= blind["parent_top1_correct"]
    assert blind["candidate_top5_correct"] >= blind["parent_top5_correct"]
    assert metadata["runtime_contract"]["inputs"]["rgb"]["shape"] == [
        "N",
        3,
        "H",
        "W",
    ]
    assert metadata["runtime_contract"]["outputs"]["embedding"]["shape"] == [
        "N",
        512,
    ]
    assert metadata["external_models"] == []
    assert (
        sha256_file(CANDIDATE_PROFILE.onnx)
        == metadata["onnx_sha256"]
    )


def test_candidate_package_selected_artifacts_are_immutable_and_registered() -> None:
    deployment = read_json(CANDIDATE_PACKAGE / "deployment_record.json")
    for record in deployment["selected_artifacts"].values():
        path = WORKSPACE_ROOT / record["path"]
        assert path.is_file(), path
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]

    blind_report = deployment["selected_artifacts"]["blind_report"]
    assert deployment["blind"]["report"] == blind_report["path"]
    assert deployment["blind"]["report_sha256"] == blind_report["sha256"]

    registry = read_json(WORKSPACE_ROOT / "models/registry.json")
    roles = registry["model_roles"]
    assert roles["current_development"]["model_package"] == (
        CANDIDATE_PROFILE.model_package
    )
    assert roles["production_baseline"]["model_package"] == (
        PRODUCTION_PROFILE.model_package
    )
    default = registry["default_deployment"]
    assert default["model_package"] == PRODUCTION_PROFILE.model_package
    assert default["backend"] == "unified-onnx"
    package = next(
        item
        for item in registry["packages"]
        if item["name"] == CANDIDATE_PROFILE.model_package
    )
    assert package["role"] == "validated-candidate"
    assert package["deployment_role"] == "current_development"
    assert package["metrics"]["blind"]["passed"] is True
