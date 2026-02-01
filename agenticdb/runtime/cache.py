"""
Dependency-aware caching for AgenticDB.

This module provides a cache that automatically invalidates entries
when their dependencies change - one of the core features that
differentiates AgenticDB from traditional systems.

Traditional Cache:  TTL-based or manual invalidation
AgenticDB Cache:    Dependency-based automatic invalidation

Design Philosophy:
    When you cache a computed value, you're implicitly saying
    "this value is valid as long as its inputs don't change."
    AgenticDB makes this explicit by tracking dependencies and
    automatically invalidating when the dependency graph changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from agenticdb.core.dependency import DependencyGraph, EdgeType


T = TypeVar("T")


class CacheEntry(BaseModel, Generic[T]):
    """
    A cached value with its dependencies.

    The key innovation: each cache entry knows what it depends on.
    When any dependency changes, this entry is automatically invalidated.
    """

    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")

    # Dependency tracking
    depends_on: list[str] = Field(
        default_factory=list,
        description="Entity IDs this value depends on"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this entry was created"
    )
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last access time"
    )
    hit_count: int = Field(default=0, description="Number of cache hits")

    # Version tracking
    computed_at_version: int = Field(
        default=0,
        description="Branch version when this was computed"
    )

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    def touch(self) -> None:
        """Update access time and hit count."""
        self.accessed_at = datetime.now(timezone.utc)
        self.hit_count += 1


class CacheStats(BaseModel):
    """Statistics for the cache."""

    hits: int = Field(default=0, description="Cache hits")
    misses: int = Field(default=0, description="Cache misses")
    invalidations: int = Field(default=0, description="Invalidations")
    size: int = Field(default=0, description="Current cache size")

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class DependencyAwareCache:
    """
    A cache that automatically invalidates based on dependency changes.

    This is a key differentiator for AgenticDB. Instead of:
    - TTL-based expiration (guessing how long data is valid)
    - Manual invalidation (error-prone and scattered)

    We use:
    - Dependency tracking (explicit knowledge of what affects what)
    - Automatic invalidation (when dependencies change, cache updates)

    Usage:
        ```python
        cache = DependencyAwareCache(graph)

        # Cache a computed value with its dependencies
        cache.set(
            key="user_risk_assessment",
            value=computed_risk,
            depends_on=[user_event_id, risk_model_claim_id]
        )

        # Get the cached value
        risk = cache.get("user_risk_assessment")

        # When a dependency changes, notify the cache
        cache.invalidate_dependents(risk_model_claim_id)
        # Now the cached risk assessment is gone
        ```
    """

    def __init__(
        self,
        graph: DependencyGraph,
        max_size: int = 10000,
    ):
        """
        Initialize the cache.

        Args:
            graph: Dependency graph for tracking relationships
            max_size: Maximum number of entries
        """
        self._graph = graph
        self._max_size = max_size
        self._lock = RLock()

        # Storage
        self._entries: dict[str, CacheEntry] = {}

        # Reverse index: entity_id -> set of cache keys that depend on it
        self._dependency_index: dict[str, set[str]] = {}

        # Statistics
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._entries)
            return self._stats.model_copy()

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if present and valid, None otherwise
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            entry.touch()
            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        depends_on: list[str],
        version: int = 0,
    ) -> CacheEntry:
        """
        Store a value in the cache with its dependencies.

        Args:
            key: Cache key
            value: Value to cache
            depends_on: Entity IDs this value depends on
            version: Current branch version

        Returns:
            Created cache entry
        """
        with self._lock:
            # Evict if at capacity
            if len(self._entries) >= self._max_size:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                depends_on=depends_on,
                computed_at_version=version,
            )

            # Store entry
            self._entries[key] = entry

            # Update dependency index
            for entity_id in depends_on:
                if entity_id not in self._dependency_index:
                    self._dependency_index[entity_id] = set()
                self._dependency_index[entity_id].add(key)

            return entry

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry existed and was invalidated
        """
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return False

            # Clean up dependency index
            for entity_id in entry.depends_on:
                if entity_id in self._dependency_index:
                    self._dependency_index[entity_id].discard(key)

            self._stats.invalidations += 1
            return True

    def invalidate_dependents(self, entity_id: str) -> list[str]:
        """
        Invalidate all cache entries that depend on an entity.

        This is the core auto-invalidation mechanism. When an entity
        changes, all cached values that depend on it are invalidated.

        Args:
            entity_id: Entity ID that changed

        Returns:
            List of invalidated cache keys
        """
        with self._lock:
            # Get directly dependent cache keys
            direct_keys = self._dependency_index.get(entity_id, set()).copy()

            # Also get transitively affected entities from the graph
            impact = self._graph.impact(entity_id)
            for affected_id in impact.entities:
                affected_keys = self._dependency_index.get(affected_id, set())
                direct_keys.update(affected_keys)

            # Invalidate all affected entries
            invalidated = []
            for key in direct_keys:
                if self.invalidate(key):
                    invalidated.append(key)

            return invalidated

    def compute_if_absent(
        self,
        key: str,
        compute_fn: Callable[[], tuple[Any, list[str]]],
        version: int = 0,
    ) -> Any:
        """
        Get a cached value or compute it if not present.

        The compute function must return both the value and its dependencies.

        Args:
            key: Cache key
            compute_fn: Function that returns (value, dependencies)
            version: Current branch version

        Returns:
            Cached or computed value
        """
        # Try cache first
        value = self.get(key)
        if value is not None:
            return value

        # Compute and cache
        value, depends_on = compute_fn()
        self.set(key, value, depends_on, version)
        return value

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._dependency_index.clear()
            return count

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return

        # Find LRU entry
        lru_key = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].accessed_at
        )
        self.invalidate(lru_key)

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get the full cache entry (for inspection)."""
        with self._lock:
            return self._entries.get(key)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._entries.keys())

    def dependencies_of(self, key: str) -> list[str]:
        """Get dependencies of a cache entry."""
        with self._lock:
            entry = self._entries.get(key)
            return list(entry.depends_on) if entry else []

    def dependents_of(self, entity_id: str) -> list[str]:
        """Get cache keys that depend on an entity."""
        with self._lock:
            return list(self._dependency_index.get(entity_id, set()))
