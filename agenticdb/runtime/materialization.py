"""
Materialization layer for AgenticDB.

This module provides automatic recomputation of derived values when their
dependencies change - a core differentiator from traditional caching systems.

Design Philosophy:
    Traditional caches only provide invalidation. When a dependency changes,
    the cached value is simply removed, requiring manual recomputation.

    AgenticDB's materialization layer goes further:
    - Automatically tracks what each derived value depends on
    - When dependencies change, can automatically recompute (eager mode)
    - Or mark as stale for lazy recomputation on next access
    - Supports transitive dependency tracking
    - Thread-safe for concurrent access

Key Concepts:
    - MaterializedView: A cached computed value with its dependencies
    - MaterializationManager: Orchestrates views and handles change propagation
    - RecomputeMode: EAGER (immediate recompute) or LAZY (on-access recompute)

Usage:
    ```python
    manager = MaterializationManager(graph, default_mode=RecomputeMode.LAZY)

    # Register a materialized view
    manager.register(
        key="user_risk_level",
        compute_fn=lambda deps: "HIGH" if deps["risk_score"] > 0.7 else "LOW",
        depends_on=["risk_score"],
    )

    # Compute initial value
    manager.compute("user_risk_level", {"risk_score": 0.3})

    # When risk_score changes, the view automatically updates
    manager.on_entity_changed("risk_score")
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from agenticdb.core.dependency import DependencyGraph, EdgeType


T = TypeVar("T")


class RecomputeMode(str, Enum):
    """
    Mode for handling dependency changes.

    EAGER: Recompute immediately when any dependency changes.
           Best for frequently accessed, critical values.

    LAZY: Mark as stale, recompute on next access.
          Best for expensive computations or rarely accessed values.
    """
    EAGER = "eager"
    LAZY = "lazy"


class MaterializationStats(BaseModel):
    """
    Statistics for monitoring materialization performance.

    Useful for:
    - Identifying expensive computations
    - Detecting excessive invalidations
    - Tuning recompute modes
    """

    total_computations: int = Field(
        default=0,
        description="Total number of computations performed"
    )
    total_invalidations: int = Field(
        default=0,
        description="Total number of invalidations triggered"
    )
    total_computation_time_ms: float = Field(
        default=0.0,
        description="Total time spent in computations (milliseconds)"
    )
    eager_recomputes: int = Field(
        default=0,
        description="Number of eager recomputations"
    )
    lazy_recomputes: int = Field(
        default=0,
        description="Number of lazy recomputations (on access)"
    )

    model_config = {"extra": "forbid"}


class MaterializedView(Generic[T]):
    """
    A materialized view representing a computed value with dependency tracking.

    The view maintains:
    - The computed value
    - The dependencies used to compute it
    - Whether it's stale (needs recomputation)
    - The version at which it was computed

    Thread Safety:
        Individual views are thread-safe for reading. Modifications
        should be coordinated through MaterializationManager.

    Attributes:
        key: Unique identifier for this view
        compute_fn: Function to compute the value from dependencies
        depends_on: Entity IDs this view depends on
        value: The cached computed value
        is_stale: Whether the value needs recomputation
        computed_at_version: Branch version when last computed

    Example:
        ```python
        view = MaterializedView(
            key="total_price",
            compute_fn=lambda deps: deps["price"] * deps["quantity"],
            depends_on=["price", "quantity"],
        )

        result = view.compute({"price": 10, "quantity": 5})  # 50
        view.mark_stale()
        result = view.compute({"price": 20, "quantity": 5})  # 100
        ```
    """

    def __init__(
        self,
        key: str,
        compute_fn: Callable[[dict[str, Any]], T],
        depends_on: list[str],
        mode: RecomputeMode = RecomputeMode.LAZY,
        value_resolver: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize a materialized view.

        Args:
            key: Unique identifier for this view
            compute_fn: Function that takes dependency values and returns computed result
            depends_on: List of entity IDs this view depends on
            mode: Recomputation mode (EAGER or LAZY)
            value_resolver: Optional function to resolve entity ID to current value
                           Required for eager mode to fetch updated values
        """
        self._key = key
        self._compute_fn = compute_fn
        self._depends_on = list(depends_on)
        self._mode = mode
        self._value_resolver = value_resolver

        self._lock = RLock()
        self._value: Optional[T] = None
        self._is_stale = True  # Initially stale, not computed yet
        self._computed_at_version = 0
        self._computation_count = 0
        self._total_computation_time_ms = 0.0
        self._created_at = datetime.now(timezone.utc)
        self._last_computed_at: Optional[datetime] = None

    @property
    def key(self) -> str:
        """Get the view key."""
        return self._key

    @property
    def compute_fn(self) -> Callable[[dict[str, Any]], T]:
        """Get the compute function."""
        return self._compute_fn

    @property
    def depends_on(self) -> list[str]:
        """Get the list of dependencies."""
        return list(self._depends_on)

    @property
    def mode(self) -> RecomputeMode:
        """Get the recomputation mode."""
        return self._mode

    @property
    def value(self) -> Optional[T]:
        """Get the cached value."""
        with self._lock:
            return self._value

    @property
    def is_stale(self) -> bool:
        """Check if the view needs recomputation."""
        with self._lock:
            return self._is_stale

    @property
    def computed_at_version(self) -> int:
        """Get the version at which the value was computed."""
        with self._lock:
            return self._computed_at_version

    @property
    def value_resolver(self) -> Optional[Callable[[str], Any]]:
        """Get the value resolver function."""
        return self._value_resolver

    def compute(
        self,
        dependency_values: dict[str, Any],
        version: int = 0,
    ) -> T:
        """
        Compute the value from dependency values.

        Args:
            dependency_values: Map of entity_id -> value for dependencies
            version: Current branch version

        Returns:
            The computed value
        """
        with self._lock:
            start_time = time.time()

            # Execute computation
            result = self._compute_fn(dependency_values)

            # Update state
            self._value = result
            self._is_stale = False
            self._computed_at_version = version
            self._computation_count += 1
            self._last_computed_at = datetime.now(timezone.utc)

            # Track timing
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_computation_time_ms += elapsed_ms

            return result

    def mark_stale(self) -> None:
        """Mark this view as stale, requiring recomputation."""
        with self._lock:
            self._is_stale = True

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for this view."""
        with self._lock:
            return {
                "key": self._key,
                "computation_count": self._computation_count,
                "total_computation_time_ms": self._total_computation_time_ms,
                "avg_computation_time_ms": (
                    self._total_computation_time_ms / self._computation_count
                    if self._computation_count > 0 else 0
                ),
                "is_stale": self._is_stale,
                "created_at": self._created_at.isoformat(),
                "last_computed_at": (
                    self._last_computed_at.isoformat()
                    if self._last_computed_at else None
                ),
            }


class MaterializationManager:
    """
    Manager for coordinating materialized views.

    Handles:
    - Registering and retrieving views
    - Triggering recomputation when dependencies change
    - Propagating invalidation through dependency chains
    - Collecting statistics

    Thread Safety:
        All operations are thread-safe. Multiple threads can read views
        concurrently, and modifications are properly synchronized.

    Example:
        ```python
        manager = MaterializationManager(graph, default_mode=RecomputeMode.EAGER)

        # Register views
        manager.register(
            key="risk_level",
            compute_fn=lambda deps: "HIGH" if deps["score"] > 0.7 else "LOW",
            depends_on=["score"],
            value_resolver=lambda eid: get_current_value(eid),
        )

        # Initial computation
        manager.compute("risk_level", {"score": 0.3})

        # When score changes, view automatically updates
        update_score(0.9)
        manager.on_entity_changed("score")  # Triggers eager recompute

        assert manager.get_view("risk_level").value == "HIGH"
        ```
    """

    def __init__(
        self,
        graph: DependencyGraph,
        default_mode: RecomputeMode = RecomputeMode.LAZY,
    ):
        """
        Initialize the materialization manager.

        Args:
            graph: Dependency graph for tracking relationships
            default_mode: Default recomputation mode for new views
        """
        self._graph = graph
        self._default_mode = default_mode
        self._lock = RLock()

        # View storage: key -> MaterializedView
        self._views: dict[str, MaterializedView] = {}

        # Reverse index: entity_id -> set of view keys that depend on it
        self._dependency_index: dict[str, set[str]] = {}

        # Statistics
        self._stats = MaterializationStats()

    def register(
        self,
        key: str,
        compute_fn: Callable[[dict[str, Any]], Any],
        depends_on: list[str],
        mode: Optional[RecomputeMode] = None,
        value_resolver: Optional[Callable[[str], Any]] = None,
    ) -> MaterializedView:
        """
        Register a new materialized view.

        Args:
            key: Unique identifier for the view
            compute_fn: Function to compute value from dependencies
            depends_on: List of entity IDs this view depends on
            mode: Recomputation mode (defaults to manager's default)
            value_resolver: Function to resolve entity ID to current value
                           Required for eager mode automatic recomputation

        Returns:
            The created MaterializedView

        Raises:
            ValueError: If a view with this key already exists
        """
        with self._lock:
            if key in self._views:
                raise ValueError(f"View already exists: {key}")

            view = MaterializedView(
                key=key,
                compute_fn=compute_fn,
                depends_on=depends_on,
                mode=mode or self._default_mode,
                value_resolver=value_resolver,
            )

            # Store view
            self._views[key] = view

            # Build dependency index
            for entity_id in depends_on:
                if entity_id not in self._dependency_index:
                    self._dependency_index[entity_id] = set()
                self._dependency_index[entity_id].add(key)

            return view

    def get_view(self, key: str) -> Optional[MaterializedView]:
        """
        Get a registered view by key.

        Args:
            key: View identifier

        Returns:
            MaterializedView if found, None otherwise
        """
        with self._lock:
            return self._views.get(key)

    def compute(
        self,
        key: str,
        dependency_values: dict[str, Any],
        version: int = 0,
    ) -> Any:
        """
        Compute or retrieve a materialized value.

        If the view is stale, recomputes using provided dependency values.
        If the view is fresh, returns cached value.

        Args:
            key: View identifier
            dependency_values: Map of entity_id -> value for dependencies
            version: Current branch version

        Returns:
            The computed or cached value

        Raises:
            ValueError: If view is not registered
        """
        with self._lock:
            view = self._views.get(key)
            if view is None:
                raise ValueError(f"View not found: {key}")

            # Recompute if stale
            if view.is_stale:
                result = view.compute(dependency_values, version)
                self._stats.total_computations += 1
                self._stats.total_computation_time_ms += (
                    view.get_stats()["total_computation_time_ms"]
                    - (self._stats.total_computation_time_ms if self._stats.total_computations > 1 else 0)
                )

                if view.mode == RecomputeMode.LAZY:
                    self._stats.lazy_recomputes += 1

                return result

            return view.value

    def get_value_if_valid(
        self,
        key: str,
        version: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Get value only if it's valid (not stale and computed at/after version).

        Args:
            key: View identifier
            version: Minimum required computation version

        Returns:
            Cached value if valid, None if stale or outdated
        """
        with self._lock:
            view = self._views.get(key)
            if view is None:
                return None

            if view.is_stale:
                return None

            if version is not None and view.computed_at_version < version:
                return None

            return view.value

    def on_entity_changed(self, entity_id: str) -> list[str]:
        """
        Handle an entity change by invalidating dependent views.

        For eager mode views with value resolvers, also triggers
        immediate recomputation.

        Args:
            entity_id: ID of the changed entity

        Returns:
            List of view keys that were invalidated
        """
        with self._lock:
            # Find all views that depend on this entity
            dependent_keys = self._get_all_dependent_keys(entity_id)

            invalidated = []
            for key in dependent_keys:
                view = self._views.get(key)
                if view is None:
                    continue

                # Mark as stale
                view.mark_stale()
                invalidated.append(key)
                self._stats.total_invalidations += 1

                # For eager mode with value resolver, recompute immediately
                if (
                    view.mode == RecomputeMode.EAGER
                    and view.value_resolver is not None
                ):
                    # Resolve current dependency values
                    dep_values = {}
                    for dep_id in view.depends_on:
                        dep_values[dep_id] = view.value_resolver(dep_id)

                    # Recompute
                    view.compute(dep_values)
                    self._stats.total_computations += 1
                    self._stats.eager_recomputes += 1

            return invalidated

    def _get_all_dependent_keys(self, entity_id: str) -> set[str]:
        """
        Get all view keys that depend on an entity (including transitively).

        Uses the dependency graph to find transitive dependents.

        Args:
            entity_id: Entity ID to check

        Returns:
            Set of view keys that depend on this entity
        """
        dependent_keys = set()

        # Direct dependents
        direct_keys = self._dependency_index.get(entity_id, set())
        dependent_keys.update(direct_keys)

        # Transitive dependents through the graph
        try:
            impact = self._graph.impact(entity_id)
            for affected_id in impact.entities:
                affected_keys = self._dependency_index.get(affected_id, set())
                dependent_keys.update(affected_keys)
        except Exception:
            # Graph might not have this entity
            pass

        # Also check if any view keys are themselves dependencies
        for key in list(dependent_keys):
            transitive_keys = self._dependency_index.get(key, set())
            dependent_keys.update(transitive_keys)

        return dependent_keys

    def unregister(self, key: str) -> bool:
        """
        Unregister a materialized view.

        Args:
            key: View identifier

        Returns:
            True if view was removed, False if not found
        """
        with self._lock:
            view = self._views.pop(key, None)
            if view is None:
                return False

            # Clean up dependency index
            for entity_id in view.depends_on:
                if entity_id in self._dependency_index:
                    self._dependency_index[entity_id].discard(key)

            return True

    def clear(self) -> int:
        """
        Clear all registered views.

        Returns:
            Number of views cleared
        """
        with self._lock:
            count = len(self._views)
            self._views.clear()
            self._dependency_index.clear()
            return count

    def get_stats(self) -> MaterializationStats:
        """Get materialization statistics."""
        with self._lock:
            # Update computation time from all views
            total_time = sum(
                v.get_stats()["total_computation_time_ms"]
                for v in self._views.values()
            )
            self._stats.total_computation_time_ms = total_time
            return self._stats.model_copy()

    def list_views(self) -> list[str]:
        """Get all registered view keys."""
        with self._lock:
            return list(self._views.keys())

    def get_dependencies_of(self, key: str) -> list[str]:
        """Get dependencies of a view."""
        with self._lock:
            view = self._views.get(key)
            return list(view.depends_on) if view else []

    def get_dependents_of(self, entity_id: str) -> list[str]:
        """Get view keys that depend on an entity."""
        with self._lock:
            return list(self._dependency_index.get(entity_id, set()))
