"""SQLite persistence for tasks and their stage artifacts.

Layout of the tasks root::

    <root>/<task_id>/
        task.sqlite          # tasks + artifacts tables
        experiment.json      # stage "configure" export (runner consumes it)
        simulation.sqlite    # stage "run" output (studio attaches it)
        artifacts/
            manifest.json    # stage "cast" export

Artifacts are JSON payloads keyed by stage, so every stage can be edited and
re-run independently.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import STAGE_ORDER, SearchConfig, TaskRecord

STAGE_ARTIFACT_EXPORTS = {
    "cast": "manifest.json",
    "configure": "experiment.json",
}


class TaskError(RuntimeError):
    """A task-level domain error (unknown task, invalid stage, ...)."""


class TaskStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.root / "tasks.sqlite")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                seed_material TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL,
                budget REAL NOT NULL DEFAULT 0.0,
                search_config TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                stage TEXT NOT NULL DEFAULT 'decompose',
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_balance TEXT,
                finished_balance TEXT,
                spent TEXT,
                usage TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                task_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                edited INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (task_id, stage)
            );
            """
        )
        self._connection.commit()

    # ------------------------------------------------------------------
    # Task rows
    # ------------------------------------------------------------------

    def create_task(
        self,
        *,
        goal: str,
        seed_material: str = "",
        model: str = "deepseek-v4-flash",
        budget: float = 0.0,
        search_config: SearchConfig | None = None,
    ) -> TaskRecord:
        goal = goal.strip()
        if not goal:
            raise TaskError("goal must not be empty")
        task_id = uuid.uuid4().hex[:12]
        now = time.time()
        record = TaskRecord(
            task_id=task_id,
            goal=goal,
            seed_material=seed_material,
            model=model,
            budget=max(0.0, float(budget)),
            search_config=search_config or SearchConfig(),
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO tasks(task_id, goal, seed_material, model, budget,
                    search_config, status, stage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, goal, seed_material, model, record.budget,
                    json.dumps(record.search_config.to_mapping()),
                    record.status, record.stage, now, now,
                ),
            )
        return record

    def get_task(self, task_id: str) -> TaskRecord:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise TaskError(f"unknown task: {task_id}")
        return TaskRecord.from_mapping(dict(row))

    def list_tasks(self) -> list[TaskRecord]:
        rows = self._connection.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC"
        ).fetchall()
        return [TaskRecord.from_mapping(dict(row)) for row in rows]

    def update_task(self, record: TaskRecord) -> None:
        now = time.time()
        with self._connection:
            self._connection.execute(
                """
                UPDATE tasks SET goal=?, seed_material=?, model=?, budget=?,
                    search_config=?, status=?, stage=?, error=?,
                    updated_at=?, started_balance=?, finished_balance=?,
                    spent=?, usage=?
                WHERE task_id=?
                """,
                (
                    record.goal, record.seed_material, record.model,
                    record.budget, json.dumps(record.search_config.to_mapping()),
                    record.status, record.stage, record.error, now,
                    record.started_balance, record.finished_balance,
                    record.spent, json.dumps(record.usage),
                    record.task_id,
                ),
            )

    def set_status(
        self,
        task_id: str,
        status: str,
        *,
        error: object = "__keep__",
    ) -> TaskRecord:
        record = self.get_task(task_id)
        if error != "__keep__":
            record = replace(record, error=error)
        record = replace(record, status=status)
        self.update_task(record)
        return record

    def set_stage(self, task_id: str, stage: str) -> TaskRecord:
        if stage not in STAGE_ORDER:
            raise TaskError(f"invalid stage: {stage}")
        record = self.get_task(task_id)
        record = replace(record, stage=stage)
        self.update_task(record)
        return record

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def get_artifact(self, task_id: str, stage: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json, edited FROM artifacts WHERE task_id=? AND stage=?",
            (task_id, stage),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        return {**payload, "_edited": bool(row["edited"])}

    def set_artifact(
        self, task_id: str, stage: str, payload: dict[str, Any], *, edited: bool = False
    ) -> None:
        now = time.time()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO artifacts(task_id, stage, payload_json, edited, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id, stage) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    edited=excluded.edited,
                    updated_at=excluded.updated_at
                """,
                (task_id, stage, json.dumps(payload, ensure_ascii=False),
                 int(edited), now),
            )

    def mark_edited(self, task_id: str, stage: str) -> None:
        artifact = self.get_artifact(task_id, stage)
        if artifact is None:
            raise TaskError(f"no artifact for stage {stage}")
        self.set_artifact(task_id, stage, artifact, edited=True)

    # ------------------------------------------------------------------
    # Task directory helpers
    # ------------------------------------------------------------------

    def task_dir(self, task_id: str) -> Path:
        path = self.root / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts_dir(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def simulation_db(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "simulation.sqlite"

    def experiment_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "experiment.json"

    def export_artifact(self, task_id: str, stage: str) -> Path | None:
        """Write the stage artifact to its exported file, if it has one."""
        filename = STAGE_ARTIFACT_EXPORTS.get(stage)
        if filename is None:
            return None
        artifact = self.get_artifact(task_id, stage)
        if artifact is None:
            return None
        payload = {key: value for key, value in artifact.items() if key != "_edited"}
        path = self.task_dir(task_id) / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "TaskStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
