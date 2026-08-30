#!/usr/bin/env python3
"""Validate Agent V1 with real independent queries through the FastAPI route."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.api import create_app  # noqa: E402
from pet_id.gallery import build_pipeline  # noqa: E402
from pet_id.gallery_service import (  # noqa: E402
    MultimodalPipelineEncoder,
    PetGalleryStore,
    PetIdentificationService,
)
from pet_id.onnx_runtime import parse_warmup_batches  # noqa: E402
from pet_id.recognition_agent import (  # noqa: E402
    AgentFeatureEncoder,
    MEGADESCRIPTOR_EXPERT_ID,
    MegaDescriptorEncoder,
)
from pet_id.workspace_paths import GALLERY_STORE_ROOT, SELECTED_MODELS_ROOT  # noqa: E402


BIFOR_PACKAGE = SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_bifor_lowrank_v1"
BIFOR_SHA256 = "63f53f32e4c9ad0c26d8ab3d91cdbfb462e920b66424656a22ff62d02b209c22"
MEGA_SHA256 = "655791158167f07773a890368f7db2fced85d569b9bccbbe7e5194e5051e2459"


def labelled_image(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("query must be PET_ID=IMAGE_PATH")
    pet_id, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not pet_id or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid labelled query {value!r}")
    return pet_id, path


def checked_json(response, operation: str) -> dict:
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"{operation} failed with HTTP {response.status_code}: {response.text}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{operation} did not return a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gallery",
        type=Path,
        default=GALLERY_STORE_ROOT / "pet_api_gallery_agent_v1",
    )
    parser.add_argument("--query", action="append", type=labelled_image, required=True)
    parser.add_argument("--config-file", type=Path, default=BIFOR_PACKAGE / "config.yaml")
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=SELECTED_MODELS_ROOT / "dogfacenet_semantic_v3_v1" / "model_final.pth",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=BIFOR_PACKAGE / "onnx" / "pet_embedding.onnx",
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=SELECTED_MODELS_ROOT.parent
        / "pretrained"
        / "body_detection"
        / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    )
    parser.add_argument(
        "--megadescriptor-checkpoint",
        type=Path,
        default=SELECTED_MODELS_ROOT.parent
        / "pretrained"
        / "megadescriptor"
        / "MegaDescriptor-B-224"
        / "pytorch_model.bin",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megadescriptor-device")
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent.parent
        / "artifacts"
        / "runs"
        / "agent_v1"
        / "gallery_api"
        / "report.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    store = PetGalleryStore(args.gallery)
    service = PetIdentificationService(store, encoder)
    results: list[dict] = []
    with TestClient(create_app(service)) as client:
        health = checked_json(client.get("/health"), "health")
        for expected_pet_id, path in args.query:
            result = checked_json(
                client.post(
                    "/v1/identify",
                    params={"top_k": min(5, health["gallery"]["pets"])},
                    files={
                        "file": (
                            path.name,
                            path.read_bytes(),
                            mimetypes.guess_type(path.name)[0]
                            or "application/octet-stream",
                        )
                    },
                ),
                f"identify {path.name}",
            )
            score = result.get("top1_score")
            descriptor = result["query"]["inference"]["descriptor"]
            agent = result.get("agent") or {}
            top_candidate = (result.get("candidates") or [{}])[0]
            results.append(
                {
                    "path": str(path),
                    "expected_pet_id": expected_pet_id,
                    "top1_pet_id": top_candidate.get("pet_id"),
                    "predicted_pet_id": result.get("predicted_pet_id"),
                    "top1_correct": top_candidate.get("pet_id") == expected_pet_id,
                    "accepted": result.get("accepted"),
                    "agent_decision": agent.get("decision"),
                    "expert_agreement": agent.get("expert_agreement"),
                    "top1_score": score,
                    "margin": result.get("margin"),
                    "finite_score": isinstance(score, (int, float))
                    and math.isfinite(float(score)),
                    "expert_weights": agent.get("expert_weights"),
                    "expert_results": agent.get("expert_results"),
                    "reasons": agent.get("reasons"),
                    "capture_recommendations": agent.get(
                        "capture_recommendations"
                    ),
                    "candidate_expert_scores": top_candidate.get("expert_scores"),
                    "branch_available": descriptor.get("branch_available"),
                    "runtime_diagnostics": descriptor.get("runtime_diagnostics"),
                }
            )
    expert_models = store.expert_models()
    fingerprints_ok = bool(
        service.model_fingerprint == BIFOR_SHA256
        and expert_models.get(MEGADESCRIPTOR_EXPERT_ID, {}).get("model_sha256")
        == MEGA_SHA256
    )
    structure_ok = all(
        item["finite_score"]
        and item["agent_decision"]
        in {"matched", "needs_more_evidence", "possible_unknown"}
        and set(item["expert_weights"] or {})
        == {"bifor", MEGADESCRIPTOR_EXPERT_ID}
        and set(item["candidate_expert_scores"] or {})
        == {"bifor", MEGADESCRIPTOR_EXPERT_ID}
        and (item["runtime_diagnostics"] or {}).get("body", {}).get("detected")
        is True
        for item in results
    )
    passed = bool(
        fingerprints_ok
        and structure_ok
        and results
        and all(item["top1_correct"] for item in results)
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "gallery": str(Path(args.gallery).resolve()),
        "fingerprints_verified": fingerprints_ok,
        "queries": len(results),
        "top1_correct": sum(int(item["top1_correct"]) for item in results),
        "top1_accuracy": (
            sum(int(item["top1_correct"]) for item in results) / len(results)
            if results
            else None
        ),
        "accepted": sum(int(bool(item["accepted"])) for item in results),
        "expert_agreement": sum(
            int(bool(item["expert_agreement"])) for item in results
        ),
        "health": health,
        "results": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
