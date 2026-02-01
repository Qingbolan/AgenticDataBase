"""
Storage layer for AgenticDB.

Provides pluggable storage backends for persisting entities,
with the default in-memory implementation for development.
"""

from agenticdb.storage.engine import StorageEngine, InMemoryStorage
from agenticdb.storage.index import Index, InMemoryIndex

__all__ = [
    "StorageEngine",
    "InMemoryStorage",
    "Index",
    "InMemoryIndex",
]
