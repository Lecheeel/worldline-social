"""Optional sqlite-vec indexing for canonical SQLite memory records.

This module is intentionally separate from the engine core. It stores only a
vector and a stable memory id; `SQLiteMemoryStore` remains the source of truth
for text and metadata.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class VectorExtensionUnavailable(RuntimeError):
    """Raised when the optional `sqlite-vec` extra is not installed."""


@dataclass(frozen=True)
class VectorMatch:
    memory_id: str
    distance: float


class SQLiteVecMemoryIndex:
    """A local KNN index colocated with the experiment SQLite database."""

    def __init__(
        self,
        path: str | Path,
        dimensions: int,
        table_name: str = "memory_vectors",
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if not _IDENTIFIER.fullmatch(table_name):
            raise ValueError("table_name must be a simple SQL identifier")
        try:
            import sqlite_vec
        except ImportError as error:
            raise VectorExtensionUnavailable(
                "Install the vector extra with: pip install 'worldline-social[vector]'"
            ) from error

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.table_name = table_name
        self._sqlite_vec = sqlite_vec
        self._connection = sqlite3.connect(self.path)
        self._connection.enable_load_extension(True)
        try:
            sqlite_vec.load(self._connection)
        finally:
            self._connection.enable_load_extension(False)
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}
            USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{self.dimensions}])
            """
        )
        self._connection.commit()

    def upsert(self, memory_id: str, embedding: Sequence[float]) -> None:
        packed = self._pack(embedding)
        with self._connection:
            self._connection.execute(
                f"DELETE FROM {self.table_name} WHERE memory_id = ?", (memory_id,)
            )
            self._connection.execute(
                f"INSERT INTO {self.table_name}(memory_id, embedding) VALUES (?, ?)",
                (memory_id, packed),
            )

    def search(self, embedding: Sequence[float], limit: int = 8) -> list[VectorMatch]:
        if limit < 1:
            raise ValueError("limit must be positive")
        packed = self._pack(embedding)
        rows = self._connection.execute(
            f"""
            SELECT memory_id, distance
            FROM {self.table_name}
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance ASC
            """,
            (packed, limit),
        ).fetchall()
        return [VectorMatch(memory_id=row[0], distance=float(row[1])) for row in rows]

    def count(self) -> int:
        row = self._connection.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connection.close()

    def _pack(self, embedding: Sequence[float]) -> bytes:
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"embedding must have {self.dimensions} dimensions, got {len(embedding)}"
            )
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise TypeError("embedding values must be numeric")
        return self._sqlite_vec.serialize_float32(list(embedding))
