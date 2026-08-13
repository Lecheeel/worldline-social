"""LLM usage & cost tracking persisted alongside simulation data.

Every provider response carries token usage; DeepSeek additionally reports
cache hits (``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``).
Per-request records are appended to the experiment's SQLite database, and a
run-level cost row records the account balance before/after a run so the
difference is the money actually spent.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from worldline_social.providers.base import BalanceInfo

USAGE_METRICS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "total_tokens",
)

#: Callables that accept one UsageRecord (e.g. a store-backed recorder).
UsageRecorder = Callable[["UsageRecord"], None]


@dataclass(frozen=True)
class UsageRecord:
    """One model completion and its token usage."""

    simulation_id: str
    tick_id: int
    entity_id: str
    model: str
    phase: str = "run"
    provider_request_id: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


def aggregate_usage(records: list[UsageRecord]) -> dict[str, int]:
    """Sum the numeric metrics across a list of records."""
    totals: dict[str, int] = defaultdict(int)
    for record in records:
        for metric in USAGE_METRICS:
            totals[metric] += int(record.usage.get(metric, 0))
    return dict(totals)


class SQLiteUsageStore:
    """Persist per-request LLM usage and run-level cost into the experiment DB.

    Lives beside the engine-owned ``checkpoints`` / ``simulation_events``
    tables in the same SQLite file; the engine itself stays usage-agnostic.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                tick_id INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                model TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'run',
                provider_request_id TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                prompt_cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                recorded_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS llm_usage_lookup
            ON llm_usage(simulation_id, tick_id);
            CREATE TABLE IF NOT EXISTS run_cost (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                currency TEXT NOT NULL DEFAULT 'CNY',
                balance_before TEXT,
                balance_after TEXT,
                spent TEXT,
                started_at REAL,
                finished_at REAL
            );
            """
        )
        self._connection.commit()

    def record(self, record: UsageRecord) -> None:
        """Append one per-request usage row (best-effort, never fatal)."""
        usage = record.usage
        total = int(usage.get("total_tokens", 0)) or (
            int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO llm_usage(
                        simulation_id, tick_id, entity_id, model, phase,
                        provider_request_id, prompt_tokens, completion_tokens,
                        prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                        total_tokens, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.simulation_id,
                        record.tick_id,
                        record.entity_id,
                        record.model,
                        record.phase,
                        record.provider_request_id,
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                        int(usage.get("prompt_cache_hit_tokens", 0)),
                        int(usage.get("prompt_cache_miss_tokens", 0)),
                        total,
                        time.time(),
                    ),
                )
        except sqlite3.Error:
            # Stats are observational; a write failure must not break the run.
            return

    def record_run_cost(
        self,
        *,
        simulation_id: str,
        balance_before: BalanceLike | None,
        balance_after: BalanceLike | None,
        started_at: float,
        finished_at: float,
        currency: str = "CNY",
    ) -> None:
        """Record a run-level cost row; ``spent`` = before - after when both
        balances are present (any settlement lag is the provider's business)."""
        spent: str | None = None
        if balance_before is not None and balance_after is not None:
            try:
                spent = f"{float(balance_before) - float(balance_after):.4f}"
            except (TypeError, ValueError):
                spent = None
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO run_cost(
                        simulation_id, currency, balance_before, balance_after,
                        spent, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        simulation_id,
                        currency,
                        _fmt_balance(balance_before),
                        _fmt_balance(balance_after),
                        spent,
                        started_at,
                        finished_at,
                    ),
                )
        except sqlite3.Error:
            return

    def usage_rows(self, simulation_id: str) -> list[dict[str, Any]]:
        """All usage rows for a simulation, newest first."""
        try:
            rows = self._connection.execute(
                """
                SELECT tick_id, entity_id, model, phase, provider_request_id,
                       prompt_tokens, completion_tokens,
                       prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                       total_tokens, recorded_at
                FROM llm_usage WHERE simulation_id = ?
                ORDER BY usage_id DESC
                """,
                (simulation_id,),
            ).fetchall()
        except sqlite3.Error:
            return []
        columns = (
            "tick_id", "entity_id", "model", "phase", "provider_request_id",
            "prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens", "total_tokens", "recorded_at",
        )
        return [dict(zip(columns, row)) for row in rows]

    def totals(self, simulation_id: str) -> dict[str, int]:
        """Summed metrics across all recorded requests for a simulation."""
        totals: dict[str, int] = defaultdict(int)
        for row in self.usage_rows(simulation_id):
            for metric in USAGE_METRICS:
                totals[metric] += int(row.get(metric, 0) or 0)
        return dict(totals)

    def latest_run_cost(self, simulation_id: str) -> dict[str, Any] | None:
        """The most recent run-cost row for a simulation, if any."""
        try:
            row = self._connection.execute(
                """
                SELECT currency, balance_before, balance_after, spent,
                       started_at, finished_at
                FROM run_cost WHERE simulation_id = ?
                ORDER BY run_id DESC LIMIT 1
                """,
                (simulation_id,),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return {
            "currency": row[0],
            "balance_before": row[1],
            "balance_after": row[2],
            "spent": row[3],
            "started_at": row[4],
            "finished_at": row[5],
        }

    def close(self) -> None:
        self._connection.close()


BalanceLike = str | float | None


def _fmt_balance(value: BalanceLike) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "BalanceInfo",
    "SQLiteUsageStore",
    "UsageRecord",
    "UsageRecorder",
    "USAGE_METRICS",
    "aggregate_usage",
]
