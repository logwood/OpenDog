"""Persistent comparison history and batch-job storage for the local workspace."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any


REVIEW_STATES = {"unreviewed", "correct", "incorrect", "uncertain"}
BATCH_TERMINAL_STATES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class WorkspaceStore:
    """Additive SQLite schema sharing the gallery database and filesystem root."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.database_path = self.root / "gallery.sqlite3"
        self.history_root = self.root / "history_images"
        self.history_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS comparison_history (
                    history_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    batch_id TEXT,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    content_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    byte_size INTEGER NOT NULL,
                    stored_path TEXT,
                    expected_pet_id TEXT,
                    accepted INTEGER,
                    predicted_pet_id TEXT,
                    predicted_display_name TEXT,
                    top1_score REAL,
                    margin REAL,
                    match_threshold REAL,
                    minimum_margin REAL,
                    latency_ms REAL,
                    model_fingerprint TEXT NOT NULL,
                    gallery_pets INTEGER NOT NULL,
                    gallery_references INTEGER NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'unreviewed',
                    review_note TEXT,
                    reviewed_at TEXT,
                    hard_case_json TEXT NOT NULL DEFAULT '[]',
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_history_created
                    ON comparison_history(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_history_batch
                    ON comparison_history(batch_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_history_prediction
                    ON comparison_history(predicted_pet_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_history_review
                    ON comparison_history(review_status, created_at DESC);

                CREATE TABLE IF NOT EXISTS batch_jobs (
                    batch_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    total INTEGER NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    succeeded INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    model_fingerprint TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_batch_created
                    ON batch_jobs(created_at DESC);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO service_metadata(key, value) VALUES (?, ?)",
                ("workspace_schema_version", str(self.SCHEMA_VERSION)),
            )

    def _write_history_image(
        self,
        *,
        data: bytes,
        sha256: str,
        suffix: str,
    ) -> str:
        relative = Path("history_images") / sha256[:2] / f"{sha256}{suffix}"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual != sha256:
                raise RuntimeError("stored comparison image hash mismatch")
            return relative.as_posix()
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return relative.as_posix()

    def record_success(
        self,
        *,
        upload,
        result: dict[str, Any],
        source: str,
        batch_id: str | None,
        expected_pet_id: str | None,
        latency_ms: float,
        model_fingerprint: str,
        gallery: dict[str, int],
    ) -> str:
        history_id = uuid.uuid4().hex
        stored_path = self._write_history_image(
            data=upload.data,
            sha256=upload.sha256,
            suffix=upload.suffix,
        )
        hard_reasons = list(result.get("hard_case_reasons") or [])
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO comparison_history(
                    history_id, created_at, source, batch_id, status, filename,
                    sha256, content_type, width, height, byte_size, stored_path,
                    expected_pet_id, accepted, predicted_pet_id,
                    predicted_display_name, top1_score, margin, match_threshold,
                    minimum_margin, latency_ms, model_fingerprint, gallery_pets,
                    gallery_references, hard_case_json, result_json
                ) VALUES (?, ?, ?, ?, 'succeeded', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    history_id,
                    utc_now(),
                    source,
                    batch_id,
                    upload.filename,
                    upload.sha256,
                    upload.content_type,
                    upload.width,
                    upload.height,
                    len(upload.data),
                    stored_path,
                    expected_pet_id,
                    int(bool(result["accepted"])),
                    result.get("predicted_pet_id"),
                    result.get("predicted_display_name"),
                    result.get("top1_score"),
                    result.get("margin"),
                    result.get("match_threshold"),
                    result.get("minimum_margin"),
                    float(latency_ms),
                    model_fingerprint,
                    int(gallery["pets"]),
                    int(gallery["reference_images"]),
                    json.dumps(hard_reasons, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        return history_id

    def record_failure(
        self,
        *,
        payload,
        source: str,
        batch_id: str | None,
        expected_pet_id: str | None,
        latency_ms: float,
        model_fingerprint: str,
        gallery: dict[str, int],
        error_code: str,
        error_message: str,
    ) -> str:
        history_id = uuid.uuid4().hex
        digest = hashlib.sha256(payload.data).hexdigest()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO comparison_history(
                    history_id, created_at, source, batch_id, status, filename,
                    sha256, byte_size, expected_pet_id, latency_ms,
                    model_fingerprint, gallery_pets, gallery_references,
                    hard_case_json, error_code, error_message
                ) VALUES (?, ?, ?, ?, 'failed', ?, ?, ?, ?, ?, ?, ?, ?, '["processing_error"]', ?, ?)
                """,
                (
                    history_id,
                    utc_now(),
                    source,
                    batch_id,
                    payload.filename,
                    digest,
                    len(payload.data),
                    expected_pet_id,
                    float(latency_ms),
                    model_fingerprint,
                    int(gallery["pets"]),
                    int(gallery["reference_images"]),
                    error_code,
                    error_message,
                ),
            )
        return history_id

    @staticmethod
    def _history_dict(row: sqlite3.Row, *, include_result: bool) -> dict[str, Any]:
        result = {
            "history_id": row["history_id"],
            "created_at": row["created_at"],
            "source": row["source"],
            "batch_id": row["batch_id"],
            "status": row["status"],
            "filename": row["filename"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
            "byte_size": row["byte_size"],
            "image_available": bool(row["stored_path"]),
            "expected_pet_id": row["expected_pet_id"],
            "accepted": None if row["accepted"] is None else bool(row["accepted"]),
            "predicted_pet_id": row["predicted_pet_id"],
            "predicted_display_name": row["predicted_display_name"],
            "top1_score": row["top1_score"],
            "margin": row["margin"],
            "match_threshold": row["match_threshold"],
            "minimum_margin": row["minimum_margin"],
            "latency_ms": row["latency_ms"],
            "model_fingerprint": row["model_fingerprint"],
            "gallery_snapshot": {
                "pets": row["gallery_pets"],
                "reference_images": row["gallery_references"],
            },
            "review_status": row["review_status"],
            "review_note": row["review_note"],
            "reviewed_at": row["reviewed_at"],
            "hard_case_reasons": json.loads(row["hard_case_json"] or "[]"),
            "error": None
            if not row["error_code"]
            else {"code": row["error_code"], "message": row["error_message"]},
        }
        if include_result:
            result["result"] = (
                None if not row["result_json"] else json.loads(row["result_json"])
            )
        return result

    def list_history(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        source: str | None = None,
        accepted: bool | None = None,
        review_status: str | None = None,
        pet_id: str | None = None,
        hard_only: bool = False,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        page = max(int(page), 1)
        page_size = min(max(int(page_size), 1), 200)
        clauses: list[str] = []
        values: list[Any] = []
        if source:
            clauses.append("source = ?")
            values.append(source)
        if accepted is not None:
            clauses.append("accepted = ?")
            values.append(int(accepted))
        if review_status:
            if review_status not in REVIEW_STATES:
                raise ValueError("invalid review_status")
            clauses.append("review_status = ?")
            values.append(review_status)
        if pet_id:
            clauses.append("(predicted_pet_id = ? OR expected_pet_id = ?)")
            values.extend((pet_id, pet_id))
        if hard_only:
            clauses.append(
                "(hard_case_json <> '[]' OR review_status IN ('incorrect','uncertain'))"
            )
        if batch_id:
            clauses.append("batch_id = ?")
            values.append(batch_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock, self._connect() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM comparison_history" + where, values
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT * FROM comparison_history"
                + where
                + " ORDER BY created_at DESC, history_id DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        return {
            "items": [self._history_dict(row, include_result=False) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_history(self, history_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM comparison_history WHERE history_id = ?", (history_id,)
            ).fetchone()
        if row is None:
            raise KeyError(history_id)
        return self._history_dict(row, include_result=True)

    def history_image(self, history_id: str) -> tuple[Path, str, str]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT stored_path, content_type, filename FROM comparison_history WHERE history_id = ?",
                (history_id,),
            ).fetchone()
        if row is None or not row["stored_path"]:
            raise KeyError(history_id)
        path = (self.root / row["stored_path"]).resolve()
        path.relative_to(self.root)
        if not path.is_file():
            raise KeyError(history_id)
        return path, str(row["content_type"] or "application/octet-stream"), str(row["filename"])

    def review_history(
        self,
        history_id: str,
        *,
        status: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        if status not in REVIEW_STATES:
            raise ValueError("invalid review status")
        normalized_note = None if note is None else note.strip()[:1000] or None
        reviewed_at = None if status == "unreviewed" else utc_now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE comparison_history SET review_status = ?, review_note = ?, reviewed_at = ? WHERE history_id = ?",
                (status, normalized_note, reviewed_at, history_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(history_id)
        return self.get_history(history_id)

    def delete_history(self, history_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT stored_path FROM comparison_history WHERE history_id = ?",
                (history_id,),
            ).fetchone()
            if row is None:
                raise KeyError(history_id)
            connection.execute(
                "DELETE FROM comparison_history WHERE history_id = ?", (history_id,)
            )
            remaining = 0
            if row["stored_path"]:
                remaining = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM comparison_history WHERE stored_path = ?",
                        (row["stored_path"],),
                    ).fetchone()[0]
                )
        if row["stored_path"] and remaining == 0:
            (self.root / row["stored_path"]).unlink(missing_ok=True)
        return {"deleted_history_id": history_id}

    def create_batch(
        self,
        *,
        name: str,
        total: int,
        model_fingerprint: str,
        parameters: dict[str, Any],
    ) -> str:
        batch_id = uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO batch_jobs(
                    batch_id, name, status, created_at, total,
                    model_fingerprint, parameters_json, metrics_json
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, '{}')
                """,
                (
                    batch_id,
                    name,
                    utc_now(),
                    int(total),
                    model_fingerprint,
                    json.dumps(parameters, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        return batch_id

    def update_batch(
        self,
        batch_id: str,
        *,
        status: str | None = None,
        completed: int | None = None,
        succeeded: int | None = None,
        failed: int | None = None,
        metrics: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[Any] = []
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
            if status == "running":
                assignments.append("started_at = COALESCE(started_at, ?)")
                values.append(utc_now())
            if status in BATCH_TERMINAL_STATES:
                assignments.append("finished_at = ?")
                values.append(utc_now())
        for column, value in (
            ("completed", completed),
            ("succeeded", succeeded),
            ("failed", failed),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(int(value))
        if metrics is not None:
            assignments.append("metrics_json = ?")
            values.append(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")))
        if error_message is not None:
            assignments.append("error_message = ?")
            values.append(error_message[:2000])
        if not assignments:
            return
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE batch_jobs SET " + ", ".join(assignments) + " WHERE batch_id = ?",
                [*values, batch_id],
            )
            if cursor.rowcount == 0:
                raise KeyError(batch_id)

    @staticmethod
    def _batch_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "batch_id": row["batch_id"],
            "name": row["name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "total": row["total"],
            "completed": row["completed"],
            "succeeded": row["succeeded"],
            "failed": row["failed"],
            "cancel_requested": bool(row["cancel_requested"]),
            "model_fingerprint": row["model_fingerprint"],
            "parameters": json.loads(row["parameters_json"] or "{}"),
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "error_message": row["error_message"],
        }

    def get_batch(self, batch_id: str, *, include_results: bool = True) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM batch_jobs WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        result = self._batch_dict(row)
        if include_results:
            result["results"] = self.batch_history(batch_id)
        return result

    def batch_history(self, batch_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM comparison_history
                WHERE batch_id = ? ORDER BY created_at, history_id
                """,
                (batch_id,),
            ).fetchall()
        return [self._history_dict(row, include_result=False) for row in rows]

    def list_batches(self, *, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        page = max(int(page), 1)
        page_size = min(max(int(page_size), 1), 100)
        with self._lock, self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM batch_jobs").fetchone()[0])
            rows = connection.execute(
                "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, (page - 1) * page_size),
            ).fetchall()
        return {
            "items": [self._batch_dict(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def request_batch_cancel(self, batch_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE batch_jobs SET cancel_requested = 1 WHERE batch_id = ? AND status NOT IN ('completed','failed','cancelled')",
                (batch_id,),
            )
            exists = connection.execute(
                "SELECT COUNT(*) FROM batch_jobs WHERE batch_id = ?", (batch_id,)
            ).fetchone()[0]
        if not exists:
            raise KeyError(batch_id)
        return self.get_batch(batch_id, include_results=False)

    def batch_cancel_requested(self, batch_id: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM batch_jobs WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def batch_csv(self, batch_id: str) -> str:
        self.get_batch(batch_id, include_results=False)
        rows = self.batch_history(batch_id)
        output = StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            [
                "history_id", "filename", "expected_pet_id", "predicted_pet_id",
                "accepted", "top1_score", "margin", "latency_ms", "status",
                "review_status", "hard_case_reasons", "error_code",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["history_id"], row["filename"], row["expected_pet_id"],
                    row["predicted_pet_id"], row["accepted"], row["top1_score"],
                    row["margin"], row["latency_ms"], row["status"],
                    row["review_status"], "|".join(row["hard_case_reasons"]),
                    (row["error"] or {}).get("code"),
                ]
            )
        return output.getvalue()

    def summary(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            history = int(
                connection.execute("SELECT COUNT(*) FROM comparison_history").fetchone()[0]
            )
            hard_cases = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM comparison_history
                    WHERE hard_case_json <> '[]'
                       OR review_status IN ('incorrect','uncertain')
                    """
                ).fetchone()[0]
            )
            batches = int(connection.execute("SELECT COUNT(*) FROM batch_jobs").fetchone()[0])
        return {"history": history, "hard_cases": hard_cases, "batches": batches}
