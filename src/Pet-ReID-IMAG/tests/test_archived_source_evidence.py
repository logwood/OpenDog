from __future__ import annotations

import importlib.util
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
METADATA_SCRIPT = WORKSPACE_ROOT / "scripts" / "generate_workspace_metadata.py"


def load_metadata_module():
    spec = importlib.util.spec_from_file_location(
        "workspace_metadata_generator", METADATA_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relocated_locked_source_evidence_is_byte_identical() -> None:
    metadata = load_metadata_module()
    result = metadata.verify_archived_source_evidence()

    assert result["evidence_files"] > 0
    assert result["code_records"] >= result["relocated_sources"]
    assert result["relocated_sources"] == result["verified"]
