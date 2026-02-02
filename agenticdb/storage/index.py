# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Indexing for AgenticDB.

This module provides indexing capabilities for fast queries,
supporting various index types for different access patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from threading import RLock
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field


K = TypeVar("K")
V = TypeVar("V")


class IndexEntry(BaseModel):
    """An entry in an index."""

    key: Any = Field(..., description="Index key")
    entity_id: str = Field(..., description="Entity ID")
    branch_id: str = Field(..., description="Branch ID")
    version: int = Field(..., description="Version when indexed")
    indexed_at: datetime = Field(..., description="When this entry was created")

    model_config = {"extra": "forbid"}


class Index(ABC, Generic[K, V]):
    """
    Abstract base class for indexes.

    Indexes provide fast lookup of entities by various keys,
    supporting the query patterns needed for AgenticDB.
    """

    @abstractmethod
    def add(self, key: K, value: V, branch_id: str, version: int) -> None:
        """Add an entry to the index."""
        pass

    @abstractmethod
    def get(self, key: K, branch_id: str) -> list[V]:
        """Get all values for a key in a branch."""
        pass

    @abstractmethod
    def remove(self, key: K, value: V, branch_id: str) -> bool:
        """Remove an entry from the index."""
        pass

    @abstractmethod
    def clear(self, branch_id: Optional[str] = None) -> None:
        """Clear the index, optionally for a specific branch."""
        pass


class InMemoryIndex(Index[K, V]):
    """
    In-memory index implementation.

    Stores index entries in memory using dictionaries.
    Thread-safe using a reentrant lock.
    """

    def __init__(self, name: str):
        """
        Initialize the index.

        Args:
            name: Index name for identification
        """
        self._name = name
        self._lock = RLock()
        # branch_id -> key -> set of values
        self._data: dict[str, dict[K, set[V]]] = defaultdict(lambda: defaultdict(set))

    @property
    def name(self) -> str:
        """Get the index name."""
        return self._name

    def add(self, key: K, value: V, branch_id: str, version: int) -> None:
        """Add an entry to the index."""
        with self._lock:
            self._data[branch_id][key].add(value)

    def get(self, key: K, branch_id: str) -> list[V]:
        """Get all values for a key in a branch."""
        with self._lock:
            return list(self._data.get(branch_id, {}).get(key, set()))

    def remove(self, key: K, value: V, branch_id: str) -> bool:
        """Remove an entry from the index."""
        with self._lock:
            if branch_id in self._data and key in self._data[branch_id]:
                try:
                    self._data[branch_id][key].discard(value)
                    return True
                except KeyError:
                    return False
            return False

    def clear(self, branch_id: Optional[str] = None) -> None:
        """Clear the index."""
        with self._lock:
            if branch_id is not None:
                self._data.pop(branch_id, None)
            else:
                self._data.clear()

    def keys(self, branch_id: str) -> list[K]:
        """Get all keys in a branch."""
        with self._lock:
            return list(self._data.get(branch_id, {}).keys())

    def size(self, branch_id: Optional[str] = None) -> int:
        """Get the number of entries in the index."""
        with self._lock:
            if branch_id is not None:
                return sum(len(v) for v in self._data.get(branch_id, {}).values())
            return sum(
                sum(len(v) for v in branch_data.values())
                for branch_data in self._data.values()
            )


class CompositeIndex(Index[tuple, str]):
    """
    Composite index for multi-key lookups.

    Allows indexing by multiple keys, e.g., (entity_type, status).
    """

    def __init__(self, name: str, key_names: list[str]):
        """
        Initialize the composite index.

        Args:
            name: Index name
            key_names: Names of the keys in order
        """
        self._name = name
        self._key_names = key_names
        self._lock = RLock()
        self._data: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))

    @property
    def name(self) -> str:
        """Get the index name."""
        return self._name

    @property
    def key_names(self) -> list[str]:
        """Get the key names."""
        return self._key_names

    def add(self, key: tuple, value: str, branch_id: str, version: int) -> None:
        """Add an entry to the index."""
        if len(key) != len(self._key_names):
            raise ValueError(f"Key must have {len(self._key_names)} components")
        with self._lock:
            self._data[branch_id][key].add(value)

    def get(self, key: tuple, branch_id: str) -> list[str]:
        """Get all values for a key in a branch."""
        with self._lock:
            return list(self._data.get(branch_id, {}).get(key, set()))

    def get_partial(self, partial_key: dict[str, Any], branch_id: str) -> list[str]:
        """
        Get values matching a partial key.

        Args:
            partial_key: Dictionary of key_name -> value for filtering
            branch_id: Branch to search in

        Returns:
            Entity IDs matching the partial key
        """
        with self._lock:
            results: set[str] = set()
            for key, values in self._data.get(branch_id, {}).items():
                match = True
                for i, key_name in enumerate(self._key_names):
                    if key_name in partial_key and key[i] != partial_key[key_name]:
                        match = False
                        break
                if match:
                    results.update(values)
            return list(results)

    def remove(self, key: tuple, value: str, branch_id: str) -> bool:
        """Remove an entry from the index."""
        with self._lock:
            if branch_id in self._data and key in self._data[branch_id]:
                try:
                    self._data[branch_id][key].discard(value)
                    return True
                except KeyError:
                    return False
            return False

    def clear(self, branch_id: Optional[str] = None) -> None:
        """Clear the index."""
        with self._lock:
            if branch_id is not None:
                self._data.pop(branch_id, None)
            else:
                self._data.clear()


class SubjectIndex(InMemoryIndex[str, str]):
    """
    Specialized index for Claim subjects.

    Supports hierarchical subject lookups, e.g.:
    - "user.u123.risk_score" matches queries for "user.*", "user.u123.*"
    """

    def __init__(self):
        """Initialize the subject index."""
        super().__init__("subject")

    def get_by_prefix(self, prefix: str, branch_id: str) -> list[str]:
        """
        Get all entity IDs whose subject starts with a prefix.

        Args:
            prefix: Subject prefix (e.g., "user.u123")
            branch_id: Branch to search in

        Returns:
            Entity IDs with matching subject prefix
        """
        with self._lock:
            results: set[str] = set()
            for subject, entity_ids in self._data.get(branch_id, {}).items():
                if str(subject).startswith(prefix):
                    results.update(entity_ids)
            return list(results)
