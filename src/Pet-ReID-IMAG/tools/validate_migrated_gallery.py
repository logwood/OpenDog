#!/usr/bin/env python3
"""Validate a migrated Gallery through the real FastAPI identification route."""

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
from pet_id.model_profiles import get_runtime_profile  # noqa: E402


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
    bifor = get_runtime_profile("research-bifor")
    legacy = get_runtime_profile("legacy-semantic")
    parser.add_argument(
        "--gallery",
        type=Path,
        default=bifor.persistent_gallery,
    )
    parser.add_argument(
        "--query", action="append", type=labelled_image, required=True
    )
    parser.add_argument("--config-file", type=Path, default=bifor.config)
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=legacy.identity_weights,
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=bifor.onnx,
    )
    parser.add_argument(
        "--body-detector",
        type=Path,
        default=bifor.body_detector,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1")
    parser.add_argument(
        "--expected-model-sha256",
        default=bifor.model_sha256,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/bifor/migrated_gallery_api/report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gallery = args.gallery.expanduser().resolve()
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
    service = PetIdentificationService(
        PetGalleryStore(gallery), MultimodalPipelineEncoder(pipeline)
    )
    results: list[dict] = []
    with TestClient(create_app(service)) as client:
        health = checked_json(client.get("/health"), "health")
        for expected_pet_id, path in args.query:
            response = client.post(
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
            )
            result = checked_json(response, f"identify {path.name}")
            score = result.get("top1_score")
            descriptor = result["query"]["inference"]["descriptor"]
            results.append(
                {
                    "path": str(path),
                    "expected_pet_id": expected_pet_id,
                    "predicted_pet_id": result.get("predicted_pet_id"),
                    "correct": result.get("predicted_pet_id") == expected_pet_id,
                    "accepted": result.get("accepted"),
                    "top1_score": score,
                    "margin": result.get("margin"),
                    "finite_score": isinstance(score, (int, float))
                    and math.isfinite(float(score)),
                    "branch_available": descriptor.get("branch_available"),
                    "runtime_diagnostics": descriptor.get("runtime_diagnostics"),
                    "model_fingerprint": result.get("model_fingerprint"),
                }
            )
    fingerprint_ok = (
        health.get("model_fingerprint") == args.expected_model_sha256
        and service.store.metadata().get("model_fingerprint")
        == args.expected_model_sha256
    )
    passed = bool(
        fingerprint_ok
        and health["gallery"]["pets"] > 0
        and health["gallery"]["reference_images"] > 0
        and results
        and all(
            item["correct"]
            and item["accepted"]
            and item["finite_score"]
            and (item["runtime_diagnostics"] or {}).get("body", {}).get("detected")
            is True
            and item["model_fingerprint"] == args.expected_model_sha256
            for item in results
        )
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "gallery": str(gallery),
        "model_fingerprint_verified": fingerprint_ok,
        "queries": len(results),
        "correct": sum(int(item["correct"]) for item in results),
        "top1_accuracy": (
            sum(int(item["correct"]) for item in results) / len(results)
            if results
            else None
        ),
        "health": health,
        "results": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
