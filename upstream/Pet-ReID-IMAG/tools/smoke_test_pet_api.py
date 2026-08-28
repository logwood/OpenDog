#!/usr/bin/env python3
"""Exercise enrollment and identification through the API with a real backend."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
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


def enrollment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("enrollment must be PET_ID=IMAGE_PATH")
    pet_id, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not pet_id or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid enrollment {value!r}")
    return pet_id, path


def checked_json(response, operation: str) -> dict:
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"{operation} failed with HTTP {response.status_code}: {response.text}"
        )
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enroll",
        action="append",
        type=enrollment,
        required=True,
        metavar="PET_ID=IMAGE_PATH",
        help="repeat for every reference image",
    )
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--expected-pet-id")
    parser.add_argument(
        "--require-query-dual-branch",
        action="store_true",
        help=(
            "fail unless the query used a real detector result with both nose and "
            "face branches available and finite, positive fusion weights"
        ),
    )
    parser.add_argument("--storage-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("models/dogfacenet_semantic_v3_v1/config.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--identity-weights",
        type=Path,
        default=Path("models/dogfacenet_semantic_v3_v1/model_final.pth"),
        help="source checkpoint used to verify the ONNX deployment metadata",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("models/dogfacenet_semantic_v3_v1/onnx/pet_embedding.onnx"),
    )
    parser.add_argument(
        "--onnx-provider", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--onnx-warmup-batches", default="1,4,8")
    args = parser.parse_args()

    query = args.query.expanduser().resolve()
    if not query.is_file():
        raise FileNotFoundError(query)
    temporary = tempfile.TemporaryDirectory(prefix="pet-api-smoke-")
    storage_context = temporary if args.storage_dir is None else nullcontext(None)
    try:
        with storage_context:
            storage = (
                Path(temporary.name)
                if args.storage_dir is None
                else args.storage_dir.expanduser().resolve()
            )
            pipeline = build_pipeline(
                args.config_file.expanduser().resolve(),
                args.identity_weights.expanduser().resolve(),
                args.device,
                backend="onnx",
                onnx_model=args.onnx_model.expanduser().resolve(),
                onnx_provider=args.onnx_provider,
                onnx_warmup_batches=parse_warmup_batches(args.onnx_warmup_batches),
                verify_onnx_source_checkpoint=True,
            )
            service = PetIdentificationService(
                PetGalleryStore(storage), MultimodalPipelineEncoder(pipeline)
            )
            grouped: dict[str, list[Path]] = defaultdict(list)
            for pet_id, path in args.enroll:
                grouped[pet_id].append(path)
            enrollment_results = []
            with TestClient(create_app(service)) as client:
                for pet_id, paths in grouped.items():
                    files = [
                        (
                            "files",
                            (
                                path.name,
                                path.read_bytes(),
                                mimetypes.guess_type(path.name)[0]
                                or "application/octet-stream",
                            ),
                        )
                        for path in paths
                    ]
                    response = client.post(f"/v1/pets/{pet_id}/images", files=files)
                    enrollment_results.append(
                        checked_json(response, f"enroll {pet_id}")
                    )
                response = client.post(
                    f"/v1/identify?top_k={min(5, len(grouped))}",
                    files={
                        "file": (
                            query.name,
                            query.read_bytes(),
                            mimetypes.guess_type(query.name)[0]
                            or "application/octet-stream",
                        )
                    },
                )
                identification = checked_json(response, "identify")
                health = checked_json(client.get("/health"), "health")
            identity_passed = (
                args.expected_pet_id is None
                or identification["predicted_pet_id"] == args.expected_pet_id
            )
            descriptor = identification["query"]["inference"]["descriptor"]
            branch_available = descriptor.get("branch_available")
            fusion_weights = descriptor.get("fusion_weights")
            dual_branch_observed = (
                branch_available == [True, True]
                and descriptor.get("detection") is not None
                and isinstance(fusion_weights, list)
                and len(fusion_weights) == 2
                and all(
                    isinstance(weight, (int, float))
                    and math.isfinite(float(weight))
                    and float(weight) > 0.0
                    for weight in fusion_weights
                )
            )
            dual_branch_passed = (
                not args.require_query_dual_branch or dual_branch_observed
            )
            passed = identity_passed and dual_branch_passed
            report = {
                "passed": passed,
                "identity_passed": identity_passed,
                "expected_pet_id": args.expected_pet_id,
                "predicted_pet_id": identification["predicted_pet_id"],
                "query_dual_branch_required": args.require_query_dual_branch,
                "query_dual_branch_observed": dual_branch_observed,
                "enrollment": enrollment_results,
                "identification": identification,
                "health": health,
                "temporary_gallery": args.storage_dir is None,
            }
            if args.output is not None:
                output = args.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not passed:
                raise SystemExit(1)
    finally:
        if args.storage_dir is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
