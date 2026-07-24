"""Optional memory and vector integrations for Worldline Social."""

from .memory import (
    HashEmbeddingProvider,
    MemoryContextBuilder,
    MemoryMatch,
    SQLiteMemoryProvider,
    SQLiteMemoryRecallRecorder,
    SentenceTransformerEmbeddingProvider,
)
from .vector import SQLiteVecMemoryIndex, VectorExtensionUnavailable

__all__ = [
    "HashEmbeddingProvider",
    "MemoryContextBuilder",
    "MemoryMatch",
    "SQLiteMemoryProvider",
    "SQLiteMemoryRecallRecorder",
    "SentenceTransformerEmbeddingProvider",
    "SQLiteVecMemoryIndex",
    "VectorExtensionUnavailable",
]
