#!/usr/bin/env python3
"""Run a real HTTP smoke test through the Java gateway and Python ONNX API.

The test deliberately uses an isolated gallery.  It exercises the contracts
that are easy to miss when only testing the Python service in-process:

* Java -> Python health and model metadata;
* gallery enrollment and idempotent duplicate handling;
* identification, persisted history, image download and review;
* administrator authentication, asynchronous batches and CSV export;
* hard-case collection; and
* model-bound gallery backup/merge-restore.

Start the stack separately with ``scripts/pet-reid-stack.ps1`` and pass the
same run directory via ``--run-dir``.  A JSON report and the downloaded backup
are left in that directory even when an assertion fails.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import mimetypes
import sys
import time
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import requests
except ImportError as error:  # pragma: no cover - exercised by installation errors
    raise SystemExit(
        "The live-stack smoke test requires requests; install "
        "src/Pet-ReID-IMAG/requirements-api.txt first."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
LIVE_RUNS_ROOT = (ROOT / "artifacts" / "runs" / "live-stack-e2e").resolve()
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "DogFaceNet_alignment" / "images"
DEFAULT_ADMIN_KEY_FILE = (
    ROOT / "artifacts" / "workspace_logs" / "quick_start" / "admin-key.txt"
)
DEFAULT_RUNTIME_FILE = (
    ROOT / "artifacts" / "workspace_logs" / "quick_start" / "runtime.json"
)


class SmokeFailure(RuntimeError):
    """A failed live-stack assertion with a user-facing error message."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return LIVE_RUNS_ROOT / stamp


def image_path(name: str) -> Path:
    return DEFAULT_IMAGE_ROOT / name


def default_inputs() -> dict[str, Any]:
    return {
        "pet_a_id": "e2e-wolf",
        "pet_a_name": "E2E Wolf",
        "pet_a_refs": [
            image_path("180&tit=Wolf.00854.jpg"),
            image_path("180&tit=Wolf.00855.jpg"),
        ],
        "pet_a_queries": [
            image_path("180&tit=Wolf.01229.jpg"),
            image_path("180&tit=Wolf.01230.jpg"),
        ],
        "pet_b_id": "e2e-dorl",
        "pet_b_name": "E2E Dorl",
        "pet_b_refs": [
            image_path("231&tit=Dorl.01084.jpg"),
            image_path("231&tit=Dorl.01086.jpg"),
        ],
        "pet_b_queries": [
            image_path("231&tit=Dorl.01087.jpg"),
            image_path("231&tit=Dorl.01088.jpg"),
        ],
    }


def _json_or_text(response: requests.Response, limit: int = 2500) -> Any:
    try:
        return response.json()
    except ValueError:
        text = response.text
        return text if len(text) <= limit else text[:limit] + "…"


def _finite_positive_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0.0
            for item in value
        )
    )


class LiveStackSmoke:
    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        self.args = args
        self.output = output
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.steps: list[dict[str, Any]] = []
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "test": "live-stack-http-e2e",
            "started_at_utc": utc_now(),
            "base_url": args.base_url,
            "python_url": args.python_url,
            "frontend_url": args.frontend_url,
            "expected_provider": args.expected_provider,
            "expected_fusion": args.expected_fusion,
            "run_dir": str(args.run_dir),
            "gallery_dir": str(args.gallery_dir),
            "steps": self.steps,
            "artifacts": {},
        }

    def record_check(
        self, label: str, passed: bool, details: Any | None = None
    ) -> None:
        entry: dict[str, Any] = {"label": label, "kind": "assertion", "ok": passed}
        if details is not None:
            entry["details"] = details
        self.steps.append(entry)
        if not passed:
            raise SmokeFailure(f"{label} assertion failed: {details!r}")

    def request(
        self,
        method: str,
        path_or_url: str,
        *,
        label: str,
        expected: int | Sequence[int] = 200,
        timeout: float | tuple[float, float] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        if isinstance(expected, int):
            expected_codes = {expected}
        else:
            expected_codes = set(expected)
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else self.args.base_url + path_or_url
        )
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "label": label,
            "kind": "http",
            "method": method.upper(),
            "url": url,
        }
        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                timeout=timeout if timeout is not None else self.args.timeout,
                **kwargs,
            )
        except requests.RequestException as error:
            entry.update(
                {
                    "ok": False,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            self.steps.append(entry)
            raise SmokeFailure(f"{label} could not reach {url}: {error}") from error

        entry.update(
            {
                "status": response.status_code,
                "ok": response.status_code in expected_codes,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            }
        )
        if response.status_code not in expected_codes:
            entry["body"] = _json_or_text(response)
            self.steps.append(entry)
            raise SmokeFailure(
                f"{label} returned HTTP {response.status_code}; "
                f"expected {sorted(expected_codes)}: {entry['body']!r}"
            )
        self.steps.append(entry)
        return response

    def json_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        response = self.request(*args, **kwargs)
        try:
            payload = response.json()
        except ValueError as error:
            raise SmokeFailure(
                f"{kwargs.get('label', args[1] if len(args) > 1 else 'request')} "
                "returned a non-JSON response"
            ) from error
        if not isinstance(payload, dict):
            raise SmokeFailure("expected a JSON object response")
        return payload

    @staticmethod
    def multipart_files(
        paths: Iterable[Path], field: str
    ) -> tuple[ExitStack, list[tuple[str, tuple[str, Any, str]]]]:
        stack = ExitStack()
        parts: list[tuple[str, tuple[str, Any, str]]] = []
        try:
            for path in paths:
                handle = stack.enter_context(path.open("rb"))
                content_type = (
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                )
                parts.append((field, (path.name, handle, content_type)))
        except Exception:
            stack.close()
            raise
        return stack, parts

    def save_report(self, *, passed: bool, error: str | None = None) -> None:
        self.report["finished_at_utc"] = utc_now()
        self.report["passed"] = passed
        if error:
            self.report["error"] = error
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self) -> None:
        args = self.args
        inputs = args.inputs
        self._validate_scope()
        self._check_inputs(inputs)

        python_health = self.json_request(
            "GET",
            args.python_url + "/health",
            label="python health",
            timeout=args.timeout,
        )
        self.record_check(
            "python health status",
            python_health.get("status") == "ok",
            python_health.get("status"),
        )
        self.request(
            "GET",
            args.python_url + "/v1/pets",
            label="Python API rejects a missing API key",
            expected=401,
            timeout=args.timeout,
        )

        frontend = self.request(
            "GET",
            args.frontend_url + "/",
            label="frontend entry",
            timeout=args.timeout,
        )
        self.record_check(
            "frontend has content",
            bool(frontend.content),
            {
                "bytes": len(frontend.content),
                "content_type": frontend.headers.get("content-type"),
            },
        )

        gateway_health = self.json_request(
            "GET", "/v1/upstream-health", label="Java gateway health"
        )
        backend = gateway_health.get("backend") or {}
        self.record_check(
            "gateway reports expected ONNX provider",
            backend.get("provider") == args.expected_provider,
            backend.get("provider"),
        )
        self.record_check(
            "gateway reports expected fusion",
            backend.get("fusion_mode") == args.expected_fusion,
            {
                "fusion_mode": backend.get("fusion_mode"),
                "embedding_dim": backend.get("embedding_dim"),
            },
        )
        self.record_check(
            "gateway reports 512d embeddings",
            backend.get("embedding_dim") == 512,
            backend.get("embedding_dim"),
        )
        if args.expected_agent:
            agent = backend.get("agent") or {}
            experts = backend.get("experts") or {}
            self.record_check(
                "gateway reports expected evidence Agent",
                agent.get("version") == args.expected_agent
                and "megadescriptor_b224" in experts
                and experts["megadescriptor_b224"].get("feature_dim") == 1024,
                {"agent": agent, "experts": experts},
            )
        if args.expected_fusion == "semantic_residual_v3+bifor_lowrank_v1":
            self.record_check(
                "gateway reports active BIFOR body runtime",
                backend.get("backend") == "onnxruntime-bifor"
                and float(backend.get("body_weight", 0.0)) > 0.0
                and bool(backend.get("body_detector")),
                {
                    "backend": backend.get("backend"),
                    "body_weight": backend.get("body_weight"),
                    "body_detector": backend.get("body_detector"),
                },
            )
        self.report["health"] = {
            "python": python_health,
            "gateway": gateway_health,
        }

        admin_key = self._read_admin_key()
        self.request(
            "GET",
            "/v1/admin/access",
            label="admin rejects missing key",
            expected=(401, 403),
        )
        self.request(
            "GET",
            "/v1/admin/access",
            label="admin rejects an invalid key",
            expected=(401, 403),
            headers={"X-Admin-Key": "live-stack-smoke-invalid"},
        )
        access = self.json_request(
            "GET",
            "/v1/admin/access",
            label="admin accepts quick-start key",
            headers={"X-Admin-Key": admin_key},
        )
        self.record_check(
            "admin access response", access.get("authorized") is True, access
        )

        enrollment: dict[str, Any] = {}
        for pet_id, display_name, refs in (
            (inputs["pet_a_id"], inputs["pet_a_name"], inputs["pet_a_refs"]),
            (inputs["pet_b_id"], inputs["pet_b_name"], inputs["pet_b_refs"]),
        ):
            stack, files = self.multipart_files(refs, "files")
            try:
                response = self.request(
                    "POST",
                    f"/v1/pets/{pet_id}/images",
                    label=f"enroll {pet_id}",
                    files=files,
                    data={"display_name": display_name},
                    timeout=args.inference_timeout,
                    expected=201,
                )
                payload = response.json()
            finally:
                stack.close()
            enrollment[pet_id] = payload
            pet = payload.get("pet") or {}
            self.record_check(
                f"{pet_id} enrollment identity",
                pet.get("pet_id") == pet_id
                and pet.get("reference_count", 0) >= len(refs),
                {
                    "reference_count": pet.get("reference_count"),
                    "added": len(payload.get("added_image_ids") or []),
                    "duplicates": len(payload.get("duplicate_image_ids") or []),
                },
            )
            self.record_check(
                f"{pet_id} enrollment accounts for every upload",
                len(payload.get("added_image_ids") or [])
                + len(payload.get("duplicate_image_ids") or [])
                == len(refs),
                payload,
            )

        duplicate_stack, duplicate_files = self.multipart_files(
            [inputs["pet_a_refs"][0]], "files"
        )
        try:
            duplicate_payload = self.json_request(
                "POST",
                f"/v1/pets/{inputs['pet_a_id']}/images",
                label="duplicate enrollment is idempotent",
                files=duplicate_files,
                timeout=args.inference_timeout,
                expected=201,
            )
        finally:
            duplicate_stack.close()
        self.record_check(
            "duplicate image is reported",
            bool(duplicate_payload.get("duplicate_image_ids")),
            duplicate_payload,
        )
        self.report["enrollment"] = enrollment

        pets = self.json_request("GET", "/v1/pets", label="list enrolled pets")
        pet_ids = {item.get("pet_id") for item in pets.get("pets", [])}
        self.record_check(
            "enrolled pets are visible through Java",
            inputs["pet_a_id"] in pet_ids and inputs["pet_b_id"] in pet_ids,
            {"pet_ids": sorted(item for item in pet_ids if item)},
        )
        details = self.json_request(
            "GET",
            f"/v1/pets/{inputs['pet_a_id']}",
            label="get enrolled pet details",
        )
        self.record_check(
            "pet details contain references",
            len(details.get("images") or []) >= len(inputs["pet_a_refs"]),
            {
                "reference_count": details.get("reference_count"),
                "images": len(details.get("images") or []),
            },
        )

        identification = self._identify(
            inputs["pet_a_queries"][0],
            label="identify known pet through Java",
            top_k=5,
        )
        self._assert_identification(
            identification,
            inputs["pet_a_id"],
            require_dual=not args.allow_single_branch,
        )
        self.report["identification"] = identification
        history_id = identification.get("history_id")
        self.record_check(
            "identification persists history id", bool(history_id), history_id
        )

        history = self.json_request(
            "GET",
            "/v1/history?page=1&page_size=100",
            label="list comparison history",
        )
        matching_history = [
            item
            for item in history.get("items", [])
            if item.get("history_id") == history_id
        ]
        self.record_check(
            "identified result is in history", bool(matching_history), matching_history
        )
        history_detail = self.json_request(
            "GET", f"/v1/history/{history_id}", label="get history detail"
        )
        self.record_check(
            "history detail matches identification",
            history_detail.get("history_id") == history_id
            and history_detail.get("predicted_pet_id") == inputs["pet_a_id"],
            history_detail,
        )
        history_image = self.request(
            "GET",
            f"/v1/history/{history_id}/image",
            label="download persisted query image",
        )
        self.record_check(
            "history image is downloadable",
            bool(history_image.content)
            and history_image.headers.get("content-type", "").startswith("image/"),
            {
                "bytes": len(history_image.content),
                "content_type": history_image.headers.get("content-type"),
            },
        )
        reviewed = self.json_request(
            "PATCH",
            f"/v1/history/{history_id}/review",
            label="review history result",
            json={"status": "correct", "note": "live-stack smoke"},
        )
        self.record_check(
            "history review is persisted",
            reviewed.get("review_status") == "correct",
            {
                "review_status": reviewed.get("review_status"),
                "review_note": reviewed.get("review_note"),
            },
        )
        reviewed_history = self.json_request(
            "GET",
            "/v1/history?page=1&page_size=100&review_status=correct",
            label="filter reviewed history",
        )
        self.record_check(
            "reviewed history filter contains result",
            any(
                item.get("history_id") == history_id
                for item in reviewed_history.get("items", [])
            ),
            {"total": reviewed_history.get("total")},
        )

        hard_case = self._identify(
            inputs["pet_a_queries"][0],
            label="force a rejected hard case",
            top_k=2,
            params={"minimum_margin": "2.0"},
        )
        self.record_check(
            "strict margin produces a rejection",
            hard_case.get("accepted") is False
            and bool(hard_case.get("hard_case_reasons"))
            and bool(hard_case.get("history_id")),
            {
                "accepted": hard_case.get("accepted"),
                "reasons": hard_case.get("hard_case_reasons"),
            },
        )
        hard_cases = self.json_request(
            "GET",
            "/v1/admin/hard-cases?page=1&page_size=200",
            label="list administrator hard cases",
            headers={"X-Admin-Key": admin_key},
        )
        self.record_check(
            "rejected result appears in hard cases",
            any(
                item.get("history_id") == hard_case.get("history_id")
                for item in hard_cases.get("items", [])
            ),
            {
                "total": hard_cases.get("total"),
                "history_id": hard_case.get("history_id"),
            },
        )

        batch_paths = [
            inputs["pet_a_queries"][0],
            inputs["pet_a_queries"][1],
            inputs["pet_b_queries"][0],
            inputs["pet_b_queries"][1],
        ]
        batch_labels = [
            inputs["pet_a_id"],
            inputs["pet_a_id"],
            inputs["pet_b_id"],
            inputs["pet_b_id"],
        ]
        batch_data: list[tuple[str, str]] = [("name", args.batch_name)]
        batch_data.extend(("expected_pet_ids", label) for label in batch_labels)
        batch_stack, batch_files = self.multipart_files(batch_paths, "files")
        try:
            batch_response = self.request(
                "POST",
                "/v1/admin/batches?top_k=5",
                label="create administrator batch",
                headers={"X-Admin-Key": admin_key},
                files=batch_files,
                data=batch_data,
                timeout=args.timeout,
                expected=202,
            )
            batch = batch_response.json()
        finally:
            batch_stack.close()
        batch_id = batch.get("batch_id")
        self.record_check("batch has an id", bool(batch_id), batch)
        completed_batch = self._wait_for_batch(batch_id, admin_key)
        # Keep the response in the report even if a semantic batch assertion
        # fails; this is the most useful payload when diagnosing a regression.
        self.report["batch"] = completed_batch
        self._assert_batch(completed_batch, batch_labels)

        csv_response = self.request(
            "GET",
            f"/v1/admin/batches/{batch_id}/results.csv",
            label="download batch CSV",
            headers={"X-Admin-Key": admin_key},
        )
        csv_path = args.run_dir / "batch-results.csv"
        csv_path.write_bytes(csv_response.content)
        self.report["artifacts"]["batch_csv"] = str(csv_path)
        csv_text = csv_response.content.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        required_columns = {
            "history_id",
            "expected_pet_id",
            "predicted_pet_id",
            "accepted",
            "hard_case_reasons",
        }
        self.record_check(
            "batch CSV contains all result rows and columns",
            len(rows) == len(batch_labels)
            and required_columns.issubset(rows[0].keys() if rows else set()),
            {"rows": len(rows), "columns": list(rows[0].keys()) if rows else []},
        )
        batch_list = self.json_request(
            "GET",
            "/v1/admin/batches?page=1&page_size=100",
            label="list administrator batches",
            headers={"X-Admin-Key": admin_key},
        )
        self.record_check(
            "completed batch appears in batch list",
            any(
                item.get("batch_id") == batch_id for item in batch_list.get("items", [])
            ),
            {"total": batch_list.get("total")},
        )

        backup_response = self.request(
            "GET",
            "/v1/admin/gallery/backup",
            label="download gallery backup",
            headers={"X-Admin-Key": admin_key},
            timeout=args.timeout,
        )
        backup_path = args.run_dir / "gallery-backup.zip"
        backup_path.write_bytes(backup_response.content)
        self.report["artifacts"]["gallery_backup"] = str(backup_path)
        self._assert_backup(backup_response.content, gateway_health)
        restore_stack, restore_files = self.multipart_files([backup_path], "file")
        try:
            restored = self.json_request(
                "POST",
                "/v1/admin/gallery/restore",
                label="merge-restore gallery backup",
                headers={"X-Admin-Key": admin_key},
                files=restore_files,
                timeout=args.inference_timeout,
            )
        finally:
            restore_stack.close()
        self.record_check(
            "restore is non-destructive merge",
            restored.get("mode") == "merge"
            and int(restored.get("duplicate_images", 0)) > 0
            and int(restored.get("added_images", 0)) == 0,
            restored,
        )
        self.report["restore"] = restored

        final_health = self.json_request(
            "GET", "/v1/upstream-health", label="final gateway health"
        )
        self.record_check(
            "final gallery still contains enrolled pets",
            int((final_health.get("gallery") or {}).get("pets", 0)) >= 2,
            final_health.get("gallery"),
        )

    def _validate_scope(self) -> None:
        run_dir = self.args.run_dir
        try:
            relative = run_dir.relative_to(LIVE_RUNS_ROOT)
        except ValueError as error:
            raise SmokeFailure(
                f"--run-dir must be under {LIVE_RUNS_ROOT}; refusing a non-isolated run"
            ) from error
        if len(relative.parts) != 1:
            raise SmokeFailure(
                f"--run-dir must be one named child of {LIVE_RUNS_ROOT}; "
                "refusing a broad or nested run directory"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        expected_gallery = (run_dir / "gallery").resolve()
        if self.args.gallery_dir.resolve() != expected_gallery:
            raise SmokeFailure(
                "--gallery-dir must be exactly <run-dir>\\gallery so the smoke "
                "cannot accidentally exercise the production gallery"
            )
        runtime_file = self.args.runtime_file
        if runtime_file.is_file():
            try:
                state = json.loads(runtime_file.read_text(encoding="utf-8-sig"))
                active_gallery = Path(state.get("gallery_dir", "")).resolve()
            except (OSError, json.JSONDecodeError) as error:
                raise SmokeFailure(
                    f"cannot read quick-start runtime state: {error}"
                ) from error
            self.record_check(
                "quick-start points at isolated gallery",
                active_gallery == expected_gallery,
                {"active": str(active_gallery), "expected": str(expected_gallery)},
            )
        else:
            self.record_check(
                "quick-start runtime state exists",
                False,
                f"missing {runtime_file}; start with scripts/pet-reid-stack.ps1",
            )

    def _check_inputs(self, inputs: dict[str, Any]) -> None:
        paths: list[Path] = []
        for key in ("pet_a_refs", "pet_a_queries", "pet_b_refs", "pet_b_queries"):
            paths.extend(inputs[key])
        missing = [str(path) for path in paths if not path.is_file()]
        self.record_check("all smoke images exist", not missing, missing)
        self.report["inputs"] = {
            key: [str(path) for path in value] if isinstance(value, list) else value
            for key, value in inputs.items()
        }

    def _read_admin_key(self) -> str:
        try:
            key = self.args.admin_key_file.read_text(encoding="ascii").strip()
        except OSError as error:
            raise SmokeFailure(
                f"cannot read admin key file {self.args.admin_key_file}: {error}"
            ) from error
        self.record_check(
            "quick-start admin key is present", bool(key), "key is empty or missing"
        )
        return key

    def _identify(
        self,
        path: Path,
        *,
        label: str,
        top_k: int,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        stack, files = self.multipart_files([path], "file")
        try:
            query = {"top_k": str(top_k)}
            if params:
                query.update(params)
            return self.json_request(
                "POST",
                "/v1/identify",
                label=label,
                files=files,
                params=query,
                timeout=self.args.inference_timeout,
            )
        finally:
            stack.close()

    def _assert_identification(
        self, result: dict[str, Any], expected_pet_id: str, *, require_dual: bool
    ) -> None:
        self.record_check(
            "known query is accepted and correctly identified",
            result.get("accepted") is True
            and result.get("predicted_pet_id") == expected_pet_id
            and bool(result.get("candidates")),
            {
                "accepted": result.get("accepted"),
                "predicted_pet_id": result.get("predicted_pet_id"),
                "top1_score": result.get("top1_score"),
            },
        )
        score = result.get("top1_score")
        self.record_check(
            "identification score and latency are finite",
            isinstance(score, (int, float))
            and math.isfinite(float(score))
            and isinstance(result.get("latency_ms"), (int, float))
            and math.isfinite(float(result["latency_ms"])),
            {"score": score, "latency_ms": result.get("latency_ms")},
        )
        descriptor = ((result.get("query") or {}).get("inference") or {}).get(
            "descriptor"
        ) or {}
        dual = (
            descriptor.get("branch_available") == [True, True]
            and isinstance(descriptor.get("detection"), dict)
            and _finite_positive_pair(descriptor.get("fusion_weights"))
        )
        self.record_check(
            "query used both fusion branches"
            if require_dual
            else "query inference metadata is present",
            dual if require_dual else isinstance(descriptor, dict),
            {
                "branch_available": descriptor.get("branch_available"),
                "fusion_weights": descriptor.get("fusion_weights"),
                "has_detection": isinstance(descriptor.get("detection"), dict),
            },
        )
        if self.args.expected_fusion == "semantic_residual_v3+bifor_lowrank_v1":
            body = (descriptor.get("runtime_diagnostics") or {}).get("body") or {}
            body_score = body.get("score")
            body_box = body.get("bbox_xyxy")
            self.record_check(
                "known query used a detected dog body",
                body.get("detected") is True
                and isinstance(body_score, (int, float))
                and math.isfinite(float(body_score))
                and isinstance(body_box, list)
                and len(body_box) == 4
                and all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in body_box
                ),
                body,
            )
        if self.args.expected_agent:
            agent = result.get("agent") or {}
            expert_weights = agent.get("expert_weights") or {}
            candidate_scores = (result.get("candidates") or [{}])[0].get(
                "expert_scores"
            ) or {}
            self.record_check(
                "known query includes independent Agent evidence",
                result.get("decision") == "agent_evidence_v1"
                and agent.get("decision") == "matched"
                and agent.get("expert_agreement") is True
                and set(expert_weights) == {"bifor", "megadescriptor_b224"}
                and set(candidate_scores) == {"bifor", "megadescriptor_b224"},
                {
                    "decision": result.get("decision"),
                    "agent": agent,
                    "candidate_expert_scores": candidate_scores,
                },
            )

    def _wait_for_batch(self, batch_id: str, admin_key: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.batch_timeout
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline:
            latest = self.json_request(
                "GET",
                f"/v1/admin/batches/{batch_id}",
                label=f"poll batch {batch_id[:8]}",
                headers={"X-Admin-Key": admin_key},
                timeout=self.args.timeout,
            )
            status = latest.get("status")
            if status in {"completed", "failed", "cancelled"}:
                return latest
            time.sleep(self.args.poll_interval)
        raise SmokeFailure(
            f"batch {batch_id} did not finish within {self.args.batch_timeout}s; "
            f"last status={latest.get('status')!r}"
        )

    def _assert_batch(self, batch: dict[str, Any], labels: Sequence[str]) -> None:
        results = batch.get("results") or []
        top1: list[str | None] = []
        for item in results:
            # Persisted HistoryItem summaries expose prediction at the top
            # level and may omit the original inference payload (result=null).
            predicted = item.get("predicted_pet_id")
            if predicted is None:
                nested = item.get("result") or {}
                candidates = nested.get("candidates") or []
                predicted = candidates[0].get("pet_id") if candidates else None
            top1.append(predicted)
        metrics = batch.get("metrics") or {}
        self.record_check(
            "batch completes without failed images",
            batch.get("status") == "completed"
            and int(batch.get("total", -1)) == len(labels)
            and int(batch.get("completed", -1)) == len(labels)
            and int(batch.get("succeeded", -1)) == len(labels)
            and int(batch.get("failed", -1)) == 0,
            {
                "status": batch.get("status"),
                "total": batch.get("total"),
                "completed": batch.get("completed"),
                "succeeded": batch.get("succeeded"),
                "failed": batch.get("failed"),
            },
        )
        self.record_check(
            "batch labels and Top-1 predictions agree",
            len(results) == len(labels)
            and all(
                item.get("expected_pet_id") == expected and predicted == expected
                for item, expected, predicted in zip(results, labels, top1)
            ),
            {"expected": list(labels), "top1": top1, "metrics": metrics},
        )
        self.record_check(
            "batch metrics include labelled accuracy",
            metrics.get("labelled") == len(labels)
            and float(metrics.get("top1_accuracy", 0.0)) >= 0.999,
            metrics,
        )

    def _assert_backup(self, data: bytes, health: dict[str, Any]) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                names = archive.namelist()
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            raise SmokeFailure(
                f"gallery backup is not a valid model-bound ZIP: {error}"
            ) from error
        self.record_check(
            "gallery backup contains a valid manifest",
            manifest.get("format") == "pet-reid-gallery-backup"
            and manifest.get("version") == 1
            and bool(manifest.get("model_fingerprint"))
            and "manifest.json" in names,
            {"files": len(names), "pets": len(manifest.get("pets", []))},
        )
        self.record_check(
            "gallery backup fingerprint matches serving model",
            manifest.get("model_fingerprint") == health.get("model_fingerprint"),
            {
                "backup": manifest.get("model_fingerprint"),
                "service": health.get("model_fingerprint"),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise the live Java gateway -> Python ONNX stack in an isolated gallery"
    )
    parser.add_argument("--run-dir", type=resolve_path, default=None)
    parser.add_argument("--gallery-dir", type=resolve_path, default=None)
    parser.add_argument(
        "--admin-key-file", type=resolve_path, default=DEFAULT_ADMIN_KEY_FILE
    )
    parser.add_argument(
        "--runtime-file", type=resolve_path, default=DEFAULT_RUNTIME_FILE
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--python-url", default="http://127.0.0.1:8000")
    # Vinext may resolve localhost to IPv6 (::1), so use the launcher's URL.
    parser.add_argument("--frontend-url", default="http://localhost:3000")
    parser.add_argument(
        "--expected-provider",
        choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    parser.add_argument(
        "--expected-fusion",
        choices=("semantic_residual_v3", "semantic_residual_v3+bifor_lowrank_v1"),
        default="semantic_residual_v3",
    )
    parser.add_argument("--expected-agent", default="")
    parser.add_argument("--batch-name", default="live-stack smoke")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--inference-timeout", type=float, default=180.0)
    parser.add_argument("--batch-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--allow-single-branch",
        action="store_true",
        help="do not fail when the selected known query lacks a real nose+face pair",
    )
    parser.add_argument("--pet-a-id", default="e2e-wolf")
    parser.add_argument("--pet-a-name", default="E2E Wolf")
    parser.add_argument("--pet-b-id", default="e2e-dorl")
    parser.add_argument("--pet-b-name", default="E2E Dorl")
    parser.add_argument("--output", type=resolve_path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir or default_run_dir()
    args.run_dir = resolve_path(args.run_dir)
    args.gallery_dir = resolve_path(args.gallery_dir or args.run_dir / "gallery")
    args.output = resolve_path(args.output or args.run_dir / "live-stack-smoke.json")
    args.admin_key_file = resolve_path(args.admin_key_file)
    args.runtime_file = resolve_path(args.runtime_file)
    if args.timeout <= 0 or args.inference_timeout <= 0 or args.batch_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.poll_interval < 0:
        parser.error("--poll-interval cannot be negative")

    inputs = default_inputs()
    inputs["pet_a_id"] = args.pet_a_id
    inputs["pet_a_name"] = args.pet_a_name
    inputs["pet_b_id"] = args.pet_b_id
    inputs["pet_b_name"] = args.pet_b_name
    args.inputs = inputs

    smoke = LiveStackSmoke(args, args.output)
    try:
        smoke.run()
    except (SmokeFailure, OSError, ValueError) as error:
        smoke.save_report(passed=False, error=str(error))
        print(
            json.dumps(
                {"passed": False, "error": str(error), "report": str(args.output)},
                ensure_ascii=False,
            )
        )
        return 1
    except KeyboardInterrupt:
        smoke.save_report(passed=False, error="interrupted")
        print(
            json.dumps(
                {"passed": False, "error": "interrupted", "report": str(args.output)},
                ensure_ascii=False,
            )
        )
        return 130
    else:
        smoke.save_report(passed=True)
        print(
            json.dumps(
                {
                    "passed": True,
                    "report": str(args.output),
                    "run_dir": str(args.run_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        smoke.session.close()


if __name__ == "__main__":
    sys.exit(main())
