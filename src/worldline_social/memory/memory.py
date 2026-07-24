"""Canonical memory retrieval and context assembly."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..providers.base import ModelMessage
from ..stores import MemoryRecord, SQLiteMemoryStore
from .vector import SQLiteVecMemoryIndex
from worldline_engine.protocols import TurnContext


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, text: str) -> Sequence[float]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding for tests and offline seed experiments.

    This is not a semantic model. It gives repeatable lexical similarity without
    downloading a model, making storage and retrieval tests self-contained.
    """

    _token_pattern = re.compile(r"[\w]+|[^\W\s]", re.UNICODE)

    def __init__(self, dimensions: int = 128, seed: str = "worldline-hash-v1") -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions
        self._seed = seed

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        tokens = self._token_pattern.findall(text.lower())
        features = list(tokens)
        features.extend(
            token[index : index + 3]
            for token in tokens
            for index in range(max(0, len(token) - 2))
        )
        for feature in features:
            digest = hashlib.sha256(f"{self._seed}:{feature}".encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)


class SentenceTransformerEmbeddingProvider:
    """Optional adapter for a real local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers to use this embedding provider"
            ) from error
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> tuple[float, ...]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class MemoryMatch:
    record: MemoryRecord
    distance: float


class SQLiteMemoryRecallRecorder:
    """Persists query-to-memory-id mappings for deterministic replay."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self._store = store

    def __call__(
        self,
        context: TurnContext,
        query: str,
        matches: Sequence[MemoryMatch],
    ) -> None:
        recall_id = hashlib.sha256(
            f"{context.simulation_id}:{context.turn_id}:{query}".encode("utf-8")
        ).hexdigest()[:32]
        self._store.record_recall(
            recall_id=recall_id,
            simulation_id=context.simulation_id,
            tick_id=context.tick_id,
            turn_id=context.turn_id,
            person_id=context.entity_id,
            query_text=query,
            memory_ids=[match.record.memory_id for match in matches],
        )


class MemoryProvider(Protocol):
    def add(self, record: MemoryRecord) -> None: ...

    def search(
        self,
        simulation_id: str,
        person_id: str,
        query: str,
        limit: int = 8,
    ) -> Sequence[MemoryMatch]: ...


class SQLiteMemoryProvider:
    """SQLite source of truth plus sqlite-vec candidate retrieval."""

    def __init__(
        self,
        memory_store: SQLiteMemoryStore,
        vector_index: SQLiteVecMemoryIndex,
        embedding_provider: EmbeddingProvider,
        candidate_multiplier: int = 8,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if embedding_provider.dimensions != vector_index.dimensions:
            raise ValueError("embedding and vector index dimensions must match")
        self.memory_store = memory_store
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.candidate_multiplier = candidate_multiplier

    def add(self, record: MemoryRecord) -> None:
        self.memory_store.add(record)
        self.vector_index.upsert(
            record.memory_id, self.embedding_provider.embed(record.content)
        )

    def search(
        self,
        simulation_id: str,
        person_id: str,
        query: str,
        limit: int = 8,
    ) -> list[MemoryMatch]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if self.vector_index.count() == 0:
            return []
        candidate_limit = max(limit * self.candidate_multiplier, 32)
        matches = self.vector_index.search(
            self.embedding_provider.embed(query), candidate_limit
        )
        records = self.memory_store.get_by_ids(
            [match.memory_id for match in matches], simulation_id, person_id
        )
        filtered = [
            MemoryMatch(record=records[match.memory_id], distance=match.distance)
            for match in matches
            if match.memory_id in records
        ]
        # Filtering on sqlite-vec auxiliary columns is intentionally avoided;
        # if the candidate window was dominated by other entities, expand to a
        # complete KNN result so person-scoped recall remains correct.
        if len(filtered) < limit:
            all_matches = self.vector_index.search(
                self.embedding_provider.embed(query), self.vector_index.count()
            )
            records = self.memory_store.get_by_ids(
                [match.memory_id for match in all_matches], simulation_id, person_id
            )
            filtered = [
                MemoryMatch(record=records[match.memory_id], distance=match.distance)
                for match in all_matches
                if match.memory_id in records
            ]
        return filtered[:limit]

    def close(self) -> None:
        self.memory_store.close()
        self.vector_index.close()


PromptBuilder = Callable[
    [TurnContext], Sequence[ModelMessage] | Awaitable[Sequence[ModelMessage]]
]


class MemoryContextBuilder:
    """Adds recalled canonical memories to an existing prompt builder."""

    def __init__(
        self,
        memory_provider: MemoryProvider,
        base_builder: PromptBuilder,
        limit: int = 5,
        query_builder: Callable[[TurnContext], str] | None = None,
        recall_recorder: Callable[
            [TurnContext, str, Sequence[MemoryMatch]], None | Awaitable[None]
        ] | None = None,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self._memory_provider = memory_provider
        self._base_builder = base_builder
        self._limit = limit
        self._query_builder = query_builder or self._default_query
        self._recall_recorder = recall_recorder
        self.last_matches: tuple[MemoryMatch, ...] = ()

    async def __call__(self, context: TurnContext) -> Sequence[ModelMessage]:
        messages = self._base_builder(context)
        if inspect.isawaitable(messages):
            messages = await messages
        query = self._query_builder(context)
        matches = tuple(
            self._memory_provider.search(
                context.simulation_id, context.entity_id, query, self._limit
            )
        )
        self.last_matches = matches
        if self._recall_recorder is not None:
            recorded = self._recall_recorder(context, query, matches)
            if inspect.isawaitable(recorded):
                await recorded
        if not matches:
            return tuple(messages)
        memory_text = "\n".join(
            f"- [{match.record.kind}] {match.record.content}"
            for match in matches
        )
        return tuple(messages) + (
            ModelMessage(
                "user",
                "Relevant memories from your history:\n" + memory_text,
            ),
        )

    @staticmethod
    def _default_query(context: TurnContext) -> str:
        previous = None
        if context.previous_result is not None:
            previous = {
                "status": context.previous_result.status.value,
                "data": dict(context.previous_result.data),
            }
        return json.dumps(
            {"observation": context.observation, "previous_result": previous},
            ensure_ascii=False,
            default=str,
        )
