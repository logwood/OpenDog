#!/usr/bin/env python3
"""Atomically package one blind-approved UnifiedPetReID V4 candidate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_external_model import sha256_file  # noqa: E402
from pet_id.unified_highres import MODEL_TYPE  # noqa: E402
from pet_id.unified_highres_protocol import PROTOCOL_NAME  # noqa: E402


PACKAGE_NAME = "unified_pet_reid_v4_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--blind-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def verified_file(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    source = file(record.get(path_key, ""))
    expected = str(record.get(hash_key, "")).casefold()
    actual = sha256_file(source)
    if not expected or actual != expected:
        raise RuntimeError(f"Locked artifact hash mismatch: {source}")
    return source


def relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def copy_verified(
    source: Path, destination: Path, expected_hash: str
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(destination)
    if digest != expected_hash:
        raise RuntimeError(f"Packaged copy hash mismatch: {destination}")
    return {
        "path": relative(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def main() -> None:
    args = parse_args()
    lock_path = file(args.candidate_lock)
    blind_path = file(args.blind_report)
    marker_path = file(args.blind_marker)
    output = args.output_dir.expanduser().resolve()
    if output.name != PACKAGE_NAME:
        raise RuntimeError(f"V4 package directory must be named {PACKAGE_NAME!r}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")

    lock = read_json(lock_path)
    blind = read_json(blind_path)
    marker = read_json(marker_path)
    if lock.get("schema_version") != 1 or lock.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 candidate lock")
    if lock.get("status") != "LOCKED_UNSCORED":
        raise RuntimeError("V4 candidate lock status changed")
    if blind.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected V4 blind report")
    if blind.get("purpose") != (
        "single_aggregate_unified_v4_vs_production_v3_blind_comparison"
    ):
        raise RuntimeError("Unexpected V4 blind report purpose")
    if blind.get("passed") is not True or blind.get("promotion_eligible") is not True:
        raise RuntimeError("V4 blind comparison did not pass")
    if blind.get("post_blind_tuning_permitted") is not False:
        raise RuntimeError("V4 blind report permits forbidden post-blind tuning")
    if blind.get("feature_cache_persisted") is not False:
        raise RuntimeError("V4 blind report persisted protected features")
    if blind.get("per_query_results_stored") is not False:
        raise RuntimeError("V4 blind report persisted per-query results")
    if blind.get("candidate_lock", {}).get("sha256") != sha256_file(lock_path):
        raise RuntimeError("V4 blind report candidate lock mismatch")

    if marker.get("status") != "COMPLETED" or marker.get("single_attempt") is not True:
        raise RuntimeError("V4 blind marker is not permanently completed")
    if marker.get("report_sha256") != sha256_file(blind_path):
        raise RuntimeError("V4 blind marker report hash mismatch")
    if marker.get("passed") is not True:
        raise RuntimeError("V4 blind marker did not record a pass")
    if marker.get("post_blind_tuning_permitted") is not False:
        raise RuntimeError("V4 blind marker permits forbidden post-blind tuning")
    if Path(str(blind.get("attempt_marker", ""))).expanduser().resolve() != marker_path:
        raise RuntimeError("V4 blind report marker path mismatch")

    candidate_metrics = blind["candidate"]["retrieval"]
    parent_metrics = blind["parent_production_v3"]["retrieval"]
    if int(candidate_metrics["top1_correct"]) < int(parent_metrics["top1_correct"]):
        raise RuntimeError("V4 blind Top-1 is below production V3")
    if int(candidate_metrics["top5_correct"]) < int(parent_metrics["top5_correct"]):
        raise RuntimeError("V4 blind Top-5 is below production V3")

    candidate = lock["candidate"]
    checkpoint = verified_file(candidate, "checkpoint", "checkpoint_sha256")
    model = verified_file(candidate, "onnx", "onnx_sha256")
    metadata = verified_file(candidate, "metadata", "metadata_sha256")
    metadata_payload = read_json(metadata)
    if metadata_payload.get("model_type") != MODEL_TYPE:
        raise RuntimeError("Unexpected V4 metadata model type")
    if metadata_payload.get("onnx_sha256") != candidate["onnx_sha256"]:
        raise RuntimeError("V4 metadata ONNX hash mismatch")
    if (
        metadata_payload.get("source_checkpoint_sha256")
        != candidate["checkpoint_sha256"]
    ):
        raise RuntimeError("V4 metadata checkpoint hash mismatch")
    if metadata_payload.get("external_models") != []:
        raise RuntimeError("V4 metadata declares external runtime models")
    if candidate.get("single_onnx_graph") is not True:
        raise RuntimeError("V4 package is not a single ONNX graph")
    if candidate.get("dynamic_raw_spatial_input") is not True:
        raise RuntimeError("V4 package does not expose dynamic raw input")
    if int(candidate.get("output_dimension", 0)) != 512:
        raise RuntimeError("V4 package output dimension changed")
    if candidate.get("external_models") != []:
        raise RuntimeError("V4 package declares external runtime models")

    protocol_record = lock["protocol_lock"]
    protocol_path = file(protocol_record["path"])
    if sha256_file(protocol_path) != protocol_record["sha256"]:
        raise RuntimeError("V4 protocol lock changed")
    protocol = read_json(protocol_path)
    if (
        Path(str(protocol.get("blind_attempt_marker", ""))).expanduser().resolve()
        != marker_path
    ):
        raise RuntimeError("V4 protocol marker path mismatch")

    evidence: dict[str, tuple[Path, str]] = {}
    for name, record in lock["development_evidence"].items():
        source = file(record["path"])
        digest = sha256_file(source)
        if digest != record["sha256"]:
            raise RuntimeError(f"V4 development evidence changed: {name}")
        evidence[name] = (source, digest)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.package-{uuid.uuid4().hex}.tmp")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        staging.mkdir(parents=False)
        selected: dict[str, dict[str, Any]] = {}
        selected["checkpoint"] = copy_verified(
            checkpoint,
            staging / "model_final.pth",
            candidate["checkpoint_sha256"],
        )
        selected["onnx"] = copy_verified(
            model,
            staging / "onnx" / "unified_pet_reid_v4.onnx",
            candidate["onnx_sha256"],
        )
        selected["metadata"] = copy_verified(
            metadata,
            staging / "onnx" / "metadata.json",
            candidate["metadata_sha256"],
        )
        selected["candidate_lock"] = copy_verified(
            lock_path,
            staging / "candidate_lock_v4.json",
            sha256_file(lock_path),
        )
        selected["blind_report"] = copy_verified(
            blind_path,
            staging / "blind_v4.json",
            sha256_file(blind_path),
        )
        selected["blind_marker"] = copy_verified(
            marker_path,
            staging / "blind_v4.attempt.json",
            sha256_file(marker_path),
        )
        selected["protocol_lock"] = copy_verified(
            protocol_path,
            staging / "protocol_lock.json",
            sha256_file(protocol_path),
        )

        evidence_destinations = {
            "highres_development_pytorch": staging / "development.json",
            "v3_legacy_guard": staging / "development_v3_legacy_guard.json",
            "highres_development_onnx_cpu": staging / "onnx" / "development_cpu.json",
            "highres_development_onnx_cuda": staging / "onnx" / "development_cuda.json",
            "export_validation": staging / "onnx" / "validation.json",
            "benchmark": staging / "onnx" / "benchmark.json",
        }
        for name, destination in evidence_destinations.items():
            source, digest = evidence[name]
            selected[name] = copy_verified(source, destination, digest)

        # Records describe the final paths after atomic publication, not staging.
        staging_prefix = relative(staging) + "/"
        output_prefix = relative(output) + "/"
        for record in selected.values():
            if not record["path"].startswith(staging_prefix):
                raise RuntimeError("Packaged artifact escaped the staging directory")
            record["path"] = output_prefix + record["path"][len(staging_prefix) :]

        development = read_json(evidence["highres_development_pytorch"][0])
        guard = read_json(evidence["v3_legacy_guard"][0])
        cpu_report = read_json(evidence["highres_development_onnx_cpu"][0])
        cuda_report = read_json(evidence["highres_development_onnx_cuda"][0])
        benchmark = read_json(evidence["benchmark"][0])
        deployment = {
            "schema_version": 1,
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "name": PACKAGE_NAME,
            "display_name": "UnifiedPetReID V4 High Resolution",
            "status": "validated_optional_candidate",
            "architecture": {
                "single_onnx_graph": True,
                "runtime_external_models": [],
                "input": "float32 RGB [N,3,H,W] in 0..255; dynamic H/W",
                "output": "L2-normalized float32 [N,512]",
                "maximum_input_side": int(
                    metadata_payload["model_config"]["maximum_input_side"]
                ),
                "oversize_policy": "resize long side before the single ONNX graph",
                "internal_global_view": "centered black square to 1280",
                "high_resolution_detail_sampling": "inside the same ONNX graph",
            },
            "selected_artifacts": selected,
            "development": {
                "queries": int(development["candidate_metrics"]["query_records"]),
                "candidate_top1_correct": int(
                    development["candidate_metrics"]["top1_correct"]
                ),
                "candidate_top5_correct": int(
                    development["candidate_metrics"]["top5_correct"]
                ),
                "parent_top1_correct": int(
                    development["parent_production_metrics"]["top1_correct"]
                ),
                "parent_top5_correct": int(
                    development["parent_production_metrics"]["top5_correct"]
                ),
                "minimum_cross_resolution_cosine": development["cross_resolution"][
                    "minimum_cosine"
                ],
                "noninferiority_passed": bool(development["noninferiority"]["passed"]),
            },
            "blind": {
                "single_attempt": True,
                "queries": int(candidate_metrics["query_records"]),
                "candidate_top1_correct": int(candidate_metrics["top1_correct"]),
                "candidate_top5_correct": int(candidate_metrics["top5_correct"]),
                "parent_top1_correct": int(parent_metrics["top1_correct"]),
                "parent_top5_correct": int(parent_metrics["top5_correct"]),
                "candidate_auc": float(candidate_metrics["auc"]),
                "parent_auc": float(parent_metrics["auc"]),
                "passed": True,
                "post_blind_tuning_permitted": False,
                "report": selected["blind_report"]["path"],
                "report_sha256": selected["blind_report"]["sha256"],
            },
            "legacy_regression": {
                "v3_development_top1_correct": int(
                    guard["v3_development"]["candidate_clean"]["top1_correct"]
                ),
                "v3_development_top5_correct": int(
                    guard["v3_development"]["candidate_clean"]["top5_correct"]
                ),
                "legacy_clean_top1_correct": int(
                    guard["legacy_clean_conflict"]["candidate_clean"]["top1_correct"]
                ),
                "legacy_clean_top5_correct": int(
                    guard["legacy_clean_conflict"]["candidate_clean"]["top5_correct"]
                ),
                "legacy_conflict_top1_correct": int(
                    guard["legacy_clean_conflict"]["candidate_conflict"]["top1_correct"]
                ),
                "legacy_conflict_top5_correct": int(
                    guard["legacy_clean_conflict"]["candidate_conflict"]["top5_correct"]
                ),
            },
            "onnx": {
                "status": "validated",
                "dynamic_input": ["N", 3, "H", "W"],
                "output": ["N", 512],
                "external_tensor_files": [],
                "development_minimum_cosine_cpu": cpu_report["parity_with_pytorch"][
                    "minimum_cosine"
                ],
                "development_minimum_cosine_cuda": cuda_report["parity_with_pytorch"][
                    "minimum_cosine"
                ],
            },
            "benchmark": {
                "report": selected["benchmark"]["path"],
                "report_sha256": selected["benchmark"]["sha256"],
                "results": benchmark.get("results", benchmark.get("benchmarks")),
            },
            "deployment": {
                "default_changed": False,
                "current_default_package": "unified_pet_reid_v3_v1",
                "optional_backend": "onnx-highres",
                "independent_gallery": "data/gallery_store/pet_api_gallery_unified_v4_v1",
                "gallery_reencoding_required": True,
                "rollback_package": "models/selected/unified_pet_reid_v3_v1",
            },
        }
        deployment_path = staging / "deployment_record.json"
        deployment_path.write_text(
            json.dumps(deployment, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        readme = """# UnifiedPetReID V4 High Resolution — 当前研发主线（包修订 v1）

这是当前研发主线的已验证高分辨率候选包。这里的“包修订 v1”是文件包编号，不是
模型代际。它通过一次性 blind 非劣验证，但尚未执行生产激活；运行时仍然只加载一张
ONNX 图：

```text
RGB [N, 3, H, W] -> UnifiedPetReID V4 -> L2-normalized 512D
```

- 动态原图输入，长边上限 4096；更大的图像仅在进入 ONNX 前按长边等比缩小。
- 1280 全局视图、脸部细节采样、鼻部细节采样和融合计算全部位于同一张 ONNX 图内。
- 不加载 AnyFace、SAM2、身体检测器或第二个身份模型。
- development：V4 与生产 V3 都是 Top-1/Top-5 `16/16`。
- 唯一一次 blind：V4 与生产 V3 都是 Top-1 `15/16`、Top-5 `16/16`，非劣验证通过。
- blind 之后禁止继续用该 split 调参。

## 使用

CUDA 快速启动：仓库根目录 `start-pet-reid-highres.cmd`。

CPU 快速启动：仓库根目录 `start-pet-reid-highres-cpu.cmd`。

也可以直接启动 Python API：

```powershell
python tools/serve_pet_api.py --backend onnx-highres --onnx-provider cuda
```

V4 使用独立图库 `data/gallery_store/pet_api_gallery_unified_v4_v1`。V3 图库不能直接混用；
需要重新录入，或使用 `tools/migrate_unified_highres_gallery.py` 从原始录入图片安全重算。

V3 仍是生产基线、默认部署和回滚模型，V4 不会覆盖它。运行时唯一必需的模型文件是
`onnx/unified_pet_reid_v4.onnx`；`model_final.pth` 仅保留用于来源追踪和复现导出。
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")

        # Verify every deployment-record path before the atomic publication.
        for record in selected.values():
            packaged = WORKSPACE / record["path"]
            # Paths point at the final directory; map them back to staging now.
            relative_in_package = packaged.relative_to(output)
            staged = staging / relative_in_package
            if not staged.is_file() or sha256_file(staged) != record["sha256"]:
                raise RuntimeError(f"Final package verification failed: {staged}")
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "package": str(output),
                "name": PACKAGE_NAME,
                "status": "validated_optional_candidate",
                "onnx_sha256": candidate["onnx_sha256"],
                "blind": deployment["blind"],
                "default_changed": False,
                "optional_backend": "onnx-highres",
                "gallery": "data/gallery_store/pet_api_gallery_unified_v4_v1",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
