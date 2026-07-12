from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from icframe.domain.run import RunSummary, StudySummary, TrialRecord


class Catalog:
    """Rebuildable SQLite index over authoritative artifact manifests."""

    def __init__(self, root: str | Path = ".artifacts/icframe") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.sqlite3"
        self._initialize()

    def upsert_run(
        self,
        summary: RunSummary,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> None:
        if _connection is None:
            with self._connect() as connection:
                self.upsert_run(summary, _connection=connection)
            return
        connection = _connection
        connection.execute(
                """
                INSERT INTO runs (
                    id, pack_id, status, seed, retention, steps, feasible,
                    parameters_json, objectives_json, created_at, completed_at,
                    artifact_path, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    feasible=excluded.feasible,
                    parameters_json=excluded.parameters_json,
                    objectives_json=excluded.objectives_json,
                    completed_at=excluded.completed_at,
                    artifact_path=excluded.artifact_path,
                    summary_json=excluded.summary_json
                """,
                (
                    summary.run_id,
                    summary.pack_id,
                    summary.status.value,
                    summary.seed,
                    summary.retention.value,
                    summary.steps_completed,
                    int(summary.feasible),
                    json.dumps(summary.parameters, sort_keys=True),
                    json.dumps(summary.objectives, sort_keys=True),
                    _created_at(summary.artifacts.get("manifest")),
                    _manifest_value(summary.artifacts.get("manifest"), "completed_at"),
                    str(self.root / "runs" / summary.run_id),
                    summary.model_dump_json(),
                ),
            )
        connection.execute(
            "DELETE FROM metrics WHERE owner_type='run' AND owner_id=?",
            (summary.run_id,),
        )
        connection.executemany(
            "INSERT INTO metrics (owner_type, owner_id, name, value) VALUES ('run', ?, ?, ?)",
            [(summary.run_id, name, value) for name, value in summary.metrics.items()],
        )

    def upsert_study(
        self,
        summary: StudySummary,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> None:
        if _connection is None:
            with self._connect() as connection:
                self.upsert_study(summary, _connection=connection)
            return
        connection = _connection
        connection.execute(
                """
                INSERT INTO studies (
                    id, pack_id, status, mode, trial_count, created_at,
                    completed_at, parameters_json, objectives_json,
                    artifact_path, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    trial_count=excluded.trial_count,
                    completed_at=excluded.completed_at,
                    parameters_json=excluded.parameters_json,
                    objectives_json=excluded.objectives_json,
                    artifact_path=excluded.artifact_path,
                    summary_json=excluded.summary_json
                """,
                (
                    summary.study_id,
                    summary.pack_id,
                    summary.status.value,
                    summary.mode.value,
                    summary.trial_count,
                    _created_at(summary.artifacts.get("manifest")),
                    _manifest_value(summary.artifacts.get("manifest"), "completed_at"),
                    json.dumps(summary.parameters, sort_keys=True),
                    json.dumps(summary.objectives, sort_keys=True),
                    str(self.root / "studies" / summary.study_id),
                    summary.model_dump_json(),
                ),
            )

    def replace_trials(
        self,
        study_id: str,
        trials,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> None:
        if _connection is None:
            with self._connect() as connection:
                self.replace_trials(study_id, trials, _connection=connection)
            return
        connection = _connection
        connection.execute("DELETE FROM trials WHERE study_id=?", (study_id,))
        connection.executemany(
                """
                INSERT INTO trials (
                    study_id, number, feasible, parameters_json,
                    objectives_json, record_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        study_id,
                        trial.number,
                        int(trial.feasible),
                        json.dumps(trial.parameters, sort_keys=True),
                        json.dumps(trial.objective_values, sort_keys=True),
                        trial.model_dump_json(),
                    )
                    for trial in trials
                ],
            )

    def list_runs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, pack_id, status, seed, retention, steps, feasible,
                       parameters_json, objectives_json, created_at, completed_at
                FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [
            {
                "id": row[0],
                "pack_id": row[1],
                "status": row[2],
                "seed": row[3],
                "retention": row[4],
                "steps_completed": row[5],
                "feasible": bool(row[6]),
                "parameters": json.loads(row[7]),
                "objectives": json.loads(row[8]),
                "created_at": row[9],
                "completed_at": row[10],
            }
            for row in rows
        ]

    def list_studies(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, pack_id, status, mode, trial_count, parameters_json,
                       objectives_json, created_at, completed_at
                FROM studies ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [
            {
                "id": row[0],
                "pack_id": row[1],
                "status": row[2],
                "mode": row[3],
                "trial_count": row[4],
                "parameters": json.loads(row[5]),
                "objectives": json.loads(row[6]),
                "created_at": row[7],
                "completed_at": row[8],
            }
            for row in rows
        ]

    def count_runs(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def count_studies(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM studies").fetchone()[0])

    def get_run(self, run_id: str) -> RunSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        return RunSummary.model_validate_json(row[0]) if row else None

    def get_study(self, study_id: str) -> StudySummary | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM studies WHERE id=?", (study_id,)
            ).fetchone()
        return StudySummary.model_validate_json(row[0]) if row else None

    def get_trial(self, study_id: str, number: int) -> TrialRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM trials WHERE study_id=? AND number=?",
                (study_id, number),
            ).fetchone()
        return TrialRecord.model_validate_json(row[0]) if row else None

    def list_trials(
        self,
        study_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[TrialRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM trials
                WHERE study_id=? ORDER BY number LIMIT ? OFFSET ?
                """,
                (study_id, limit, offset),
            ).fetchall()
        return [TrialRecord.model_validate_json(row[0]) for row in rows]

    def count_trials(self, study_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM trials WHERE study_id=?", (study_id,)
                ).fetchone()[0]
            )

    def rebuild(self) -> dict[str, int]:
        # Parse every authoritative artifact before modifying the existing
        # catalog. A single malformed artifact must not destroy a usable index.
        runs = [
            RunSummary.model_validate_json(path.read_text())
            for path in sorted((self.root / "runs").glob("*/summary.json"))
        ]
        studies: list[tuple[StudySummary, list[TrialRecord]]] = []
        for path in sorted((self.root / "studies").glob("*/summary.json")):
            summary = StudySummary.model_validate_json(path.read_text())
            trial_path = path.with_name("trials.jsonl")
            trials = (
                [
                    TrialRecord.model_validate_json(line)
                    for line in trial_path.read_text().splitlines()
                    if line.strip()
                ]
                if trial_path.exists()
                else []
            )
            studies.append((summary, trials))
        with self._connect() as connection:
            connection.execute("DELETE FROM metrics")
            connection.execute("DELETE FROM trials")
            connection.execute("DELETE FROM runs")
            connection.execute("DELETE FROM studies")
            for summary in runs:
                self.upsert_run(summary, _connection=connection)
            for summary, trials in studies:
                self.upsert_study(summary, _connection=connection)
                if trials:
                    self.replace_trials(summary.study_id, trials, _connection=connection)
        return {"runs": len(runs), "studies": len(studies)}

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    pack_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    retention TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    feasible INTEGER NOT NULL,
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    objectives_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT '',
                    artifact_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    pack_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    trial_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT '',
                    parameters_json TEXT NOT NULL DEFAULT '[]',
                    objectives_json TEXT NOT NULL DEFAULT '[]',
                    artifact_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    study_id TEXT NOT NULL,
                    number INTEGER NOT NULL,
                    feasible INTEGER NOT NULL,
                    parameters_json TEXT NOT NULL,
                    objectives_json TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (study_id, number)
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (owner_type, owner_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_pack ON runs(pack_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_studies_pack ON studies(pack_id, created_at);
                """
            )
            _ensure_column(connection, "runs", "parameters_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(connection, "runs", "objectives_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(connection, "runs", "completed_at", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, "studies", "parameters_json", "TEXT NOT NULL DEFAULT '[]'")
            _ensure_column(connection, "studies", "objectives_json", "TEXT NOT NULL DEFAULT '[]'")
            _ensure_column(connection, "studies", "completed_at", "TEXT NOT NULL DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)


def _created_at(manifest_path: str | None) -> str:
    return _manifest_value(manifest_path, "started_at")


def _manifest_value(manifest_path: str | None, field: str) -> str:
    if not manifest_path:
        return ""
    path = Path(manifest_path)
    if not path.exists():
        return ""
    return str(json.loads(path.read_text()).get(field, "") or "")


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
