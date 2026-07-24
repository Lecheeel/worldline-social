from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from worldline_social.memory.memory import (
    HashEmbeddingProvider,
    MemoryContextBuilder,
    SQLiteMemoryRecallRecorder,
    SQLiteMemoryProvider,
)
from worldline_social.providers import ModelMessage
from worldline_engine.protocols import TurnContext
from worldline_social.stores import MemoryRecord, SQLiteMemoryStore
from worldline_social.memory.vector import SQLiteVecMemoryIndex


def make_context() -> TurnContext:
    return TurnContext(
        simulation_id="run",
        tick_id=10,
        turn_id="10:0:alice",
        entity_id="alice",
        observation={"topic": "solar energy"},
        available_actions=(),
        previous_result=None,
        remaining_actions=2,
        remaining_controller_calls=2,
    )


class MemoryProviderTests(unittest.TestCase):
    def test_empty_index_returns_no_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = SQLiteMemoryStore(path)
            index = SQLiteVecMemoryIndex(path, dimensions=32)
            provider = SQLiteMemoryProvider(store, index, HashEmbeddingProvider(32))
            try:
                matches = provider.search("run", "alice", "anything")
            finally:
                provider.close()
        self.assertEqual([], matches)

    def test_recall_is_person_scoped_and_uses_canonical_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = SQLiteMemoryStore(path)
            index = SQLiteVecMemoryIndex(path, dimensions=64)
            provider = SQLiteMemoryProvider(store, index, HashEmbeddingProvider(64))
            try:
                provider.add(MemoryRecord("alice-solar", "run", "alice", 1, "event", "Alice researched solar energy policy."))
                provider.add(MemoryRecord("alice-transit", "run", "alice", 2, "event", "Alice planned a public transit route."))
                provider.add(MemoryRecord("bob-solar", "run", "bob", 3, "event", "Bob researched solar energy policy."))
                matches = provider.search("run", "alice", "solar energy policy", limit=1)
            finally:
                provider.close()
        self.assertEqual(["alice-solar"], [match.record.memory_id for match in matches])
        self.assertIn("solar energy", matches[0].record.content)

    def test_context_builder_appends_recalled_memories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = SQLiteMemoryStore(path)
            index = SQLiteVecMemoryIndex(path, dimensions=32)
            provider = SQLiteMemoryProvider(store, index, HashEmbeddingProvider(32))
            provider.add(MemoryRecord("m1", "run", "alice", 1, "summary", "Alice prefers solar energy evidence."))
            builder = MemoryContextBuilder(
                provider,
                lambda _context: (ModelMessage("system", "Base"),),
                limit=1,
                query_builder=lambda _context: "solar energy evidence",
            )
            try:
                messages = asyncio.run(builder(make_context()))
                matches = builder.last_matches
            finally:
                provider.close()
        self.assertEqual(2, len(messages))
        self.assertIn("solar energy evidence", messages[-1].content)
        self.assertEqual(["m1"], [match.record.memory_id for match in matches])

    def test_recalled_context_can_feed_a_model_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = SQLiteMemoryStore(path)
            index = SQLiteVecMemoryIndex(path, dimensions=32)
            provider = SQLiteMemoryProvider(store, index, HashEmbeddingProvider(32))
            provider.add(MemoryRecord("m1", "run", "alice", 1, "summary", "Alice trusts solar energy evidence."))
            builder = MemoryContextBuilder(
                provider,
                lambda _context: (ModelMessage("system", "Use tools."),),
                query_builder=lambda _context: "solar energy evidence",
                recall_recorder=SQLiteMemoryRecallRecorder(store),
            )
            try:
                messages = asyncio.run(builder(make_context()))
            finally:
                provider.close()
            reopened = SQLiteMemoryStore(path)
            recalls = reopened.list_recalls("run", "alice")
            reopened.close()
        self.assertEqual("system", messages[0].role)
        self.assertEqual("user", messages[-1].role)
        self.assertIn("Alice trusts solar energy evidence", messages[-1].content)
        self.assertEqual(["m1"], recalls[0]["memory_ids"])

    def test_stress_recall_with_many_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = SQLiteMemoryStore(path)
            index = SQLiteVecMemoryIndex(path, dimensions=64)
            provider = SQLiteMemoryProvider(store, index, HashEmbeddingProvider(64))
            try:
                for index_number in range(300):
                    topic = "solar energy research" if index_number == 217 else "public transit planning"
                    provider.add(
                        MemoryRecord(
                            f"m-{index_number}",
                            "run",
                            "alice",
                            index_number,
                            "experience",
                            f"Alice discussed {topic} with evidence and tradeoffs.",
                        )
                    )
                matches = provider.search("run", "alice", "solar energy research", limit=3)
            finally:
                provider.close()
        self.assertTrue(matches)
        self.assertEqual("m-217", matches[0].record.memory_id)
