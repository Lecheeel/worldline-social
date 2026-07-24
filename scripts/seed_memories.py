"""Seed and query a local SQLite + sqlite-vec memory store.

Example:
    python scripts/seed_memories.py --database runs/memory-demo.sqlite \
        --count 1000 --query "Alice prefers solar energy research"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldline_social.memory.memory import HashEmbeddingProvider, SQLiteMemoryProvider
from worldline_social.stores import MemoryRecord, SQLiteMemoryStore
from worldline_social.memory.vector import SQLiteVecMemoryIndex


TOPICS = (
    "solar energy research",
    "local election volunteering",
    "public transit planning",
    "water conservation policy",
    "community garden coordination",
    "open source software maintenance",
    "neighborhood emergency preparation",
    "small business accounting",
)


def seed(provider: SQLiteMemoryProvider, simulation_id: str, count: int) -> None:
    for index in range(count):
        person_id = f"person-{index % 20:03d}"
        topic = TOPICS[index % len(TOPICS)]
        provider.add(
            MemoryRecord(
                memory_id=f"memory-{index:06d}",
                simulation_id=simulation_id,
                person_id=person_id,
                tick_id=index,
                kind="experience" if index % 3 else "summary",
                content=(
                    f"{person_id} discussed {topic}; event {index} emphasized "
                    f"evidence, practical tradeoffs, and long-term planning."
                ),
                importance=0.2 + (index % 8) / 10,
                source_action_id=f"seed-action-{index:06d}",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--simulation-id", default="memory-seed")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--query", default="solar energy research")
    parser.add_argument("--person-id", default="person-000")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=128)
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must not be negative")

    store = SQLiteMemoryStore(args.database)
    index = SQLiteVecMemoryIndex(args.database, dimensions=args.dimensions)
    provider = SQLiteMemoryProvider(store, index, HashEmbeddingProvider(args.dimensions))
    try:
        seed(provider, args.simulation_id, args.count)
        matches = provider.search(
            args.simulation_id, args.person_id, args.query, args.limit
        )
        print(f"seeded={args.count} database={args.database}")
        print(f"query={args.query!r} person={args.person_id!r}")
        for match in matches:
            print(
                f"distance={match.distance:.4f} "
                f"memory_id={match.record.memory_id} "
                f"content={match.record.content}"
            )
    finally:
        provider.close()


if __name__ == "__main__":
    main()
