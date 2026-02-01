"""
Query engine for AgenticDB.

This module provides the main query execution engine that coordinates
between storage, indexing, and the dependency graph to answer queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from agenticdb.core.dependency import DependencyGraph, EdgeType
from agenticdb.core.models import Entity, EntityType, Event, Claim, Action
from agenticdb.core.version import Snapshot
from agenticdb.query.operators import (
    WhyQuery,
    ImpactQuery,
    TraceQuery,
    CausalChain,
    ImpactResult,
)
from agenticdb.storage.engine import StorageEngine


class QueryEngine:
    """
    Main query engine for AgenticDB.

    Provides a unified interface for all query operations:
    - Basic entity queries (get, list, filter)
    - Causal queries (why, impact, trace)
    - Time-travel queries (at_version, history)

    Usage:
        ```python
        engine = QueryEngine(storage, graph, "main")

        # Basic queries
        entity = engine.get("entity_id")
        events = list(engine.events())

        # Causal queries
        chain = engine.why("action_id")
        affected = engine.impact("claim_id")

        # Time travel
        snapshot = engine.at_version(5)
        history = engine.history("entity_id")
        ```
    """

    def __init__(
        self,
        storage: StorageEngine,
        graph: DependencyGraph,
        branch_id: str
    ):
        """
        Initialize the query engine.

        Args:
            storage: Storage backend
            graph: Dependency graph
            branch_id: Branch to query
        """
        self._storage = storage
        self._graph = graph
        self._branch_id = branch_id

        # Initialize query operators
        self._why_query = WhyQuery(storage, graph, branch_id)
        self._impact_query = ImpactQuery(storage, graph, branch_id)
        self._trace_query = TraceQuery(storage, graph, branch_id)

    @property
    def branch_id(self) -> str:
        """Get the current branch ID."""
        return self._branch_id

    # =========================================================================
    # Basic Entity Queries
    # =========================================================================

    def get(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity if found, None otherwise
        """
        return self._storage.get(entity_id, self._branch_id)

    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        return self.get(entity_id) is not None

    def events(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Event]:
        """
        Query events.

        Args:
            event_type: Filter by event type
            limit: Maximum results

        Yields:
            Matching events
        """
        filters = {}
        if event_type is not None:
            filters["event_type"] = event_type

        for entity in self._storage.query(
            self._branch_id,
            entity_type=EntityType.EVENT,
            filters=filters if filters else None,
            limit=limit,
        ):
            if isinstance(entity, Event):
                yield entity

    def claims(
        self,
        subject: Optional[str] = None,
        source: Optional[str] = None,
        active_only: bool = True,
        limit: Optional[int] = None,
    ) -> Iterator[Claim]:
        """
        Query claims.

        Args:
            subject: Filter by subject
            source: Filter by source
            active_only: Only return active claims
            limit: Maximum results

        Yields:
            Matching claims
        """
        filters = {}
        if subject is not None:
            filters["subject"] = subject
        if source is not None:
            filters["source"] = source

        for entity in self._storage.query(
            self._branch_id,
            entity_type=EntityType.CLAIM,
            filters=filters if filters else None,
            limit=limit,
        ):
            if isinstance(entity, Claim):
                if active_only and not entity.is_active():
                    continue
                yield entity

    def actions(
        self,
        action_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Action]:
        """
        Query actions.

        Args:
            action_type: Filter by action type
            agent_id: Filter by agent ID
            limit: Maximum results

        Yields:
            Matching actions
        """
        filters = {}
        if action_type is not None:
            filters["action_type"] = action_type
        if agent_id is not None:
            filters["agent_id"] = agent_id

        for entity in self._storage.query(
            self._branch_id,
            entity_type=EntityType.ACTION,
            filters=filters if filters else None,
            limit=limit,
        ):
            if isinstance(entity, Action):
                yield entity

    # =========================================================================
    # Causal Queries - THE DIFFERENTIATOR
    # =========================================================================

    def why(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        include_entity_data: bool = False,
    ) -> CausalChain:
        """
        Trace the causal chain that led to an entity.

        This answers: "Why did this happen? What caused this state?"

        Args:
            entity_id: Entity to trace
            max_depth: Maximum traversal depth
            include_entity_data: Include full entity data in results

        Returns:
            CausalChain showing the path from root causes

        Example:
            ```python
            # Why was this order approved?
            chain = engine.why(action_id)
            print(chain.to_tree_string())
            # Output:
            # Query: why(action_123)
            # └─ [event] UserRegistered
            # └─ [claim] risk_score = 0.3
            # └─ [action] ApproveOrder
            ```
        """
        return self._why_query.execute(
            entity_id,
            max_depth=max_depth,
            include_entity_data=include_entity_data,
        )

    def impact(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        auto_invalidate: bool = False,
    ) -> ImpactResult:
        """
        Find all entities affected by a change to this entity.

        This answers: "What breaks if this changes?"

        Args:
            entity_id: Entity to analyze
            max_depth: Maximum traversal depth
            auto_invalidate: Mark affected claims as invalidated

        Returns:
            ImpactResult showing all downstream dependencies

        Example:
            ```python
            # What depends on the risk model?
            result = engine.impact(model_id)
            print(f"Affected claims: {result.affected_claims}")
            print(f"Affected actions: {result.affected_actions}")
            ```
        """
        return self._impact_query.execute(
            entity_id,
            max_depth=max_depth,
            auto_invalidate=auto_invalidate,
        )

    def trace(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Get a complete trace of an entity (both causes and effects).

        Args:
            entity_id: Entity to trace
            max_depth: Maximum traversal depth

        Returns:
            Dictionary with 'why' and 'impact' results
        """
        return self._trace_query.execute(entity_id, max_depth)

    # =========================================================================
    # Time-Travel Queries
    # =========================================================================

    def at_version(self, version: int) -> Snapshot:
        """
        Get a snapshot of the state at a specific version.

        This enables time-travel debugging: "What was the state when
        this decision was made?"

        Args:
            version: Version number

        Returns:
            Snapshot of state at that version
        """
        return self._storage.create_snapshot(self._branch_id, version)

    def get_at_version(self, entity_id: str, version: int) -> Optional[Entity]:
        """
        Get an entity as it was at a specific version.

        Args:
            entity_id: Entity identifier
            version: Version number

        Returns:
            Entity as it was at that version
        """
        return self._storage.get_at_version(entity_id, self._branch_id, version)

    def history(
        self,
        entity_id: str,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
    ) -> list[Entity]:
        """
        Get the history of an entity across versions.

        Args:
            entity_id: Entity identifier
            from_version: Starting version
            to_version: Ending version

        Returns:
            List of entity states across versions
        """
        branch = self._storage.get_branch(self._branch_id)
        if branch is None:
            return []

        start = from_version or 1
        end = to_version or branch.head_version

        history = []
        for v in range(start, end + 1):
            entity = self.get_at_version(entity_id, v)
            if entity is not None:
                history.append(entity)

        return history

    # =========================================================================
    # Dependency Graph Queries
    # =========================================================================

    def dependencies_of(self, entity_id: str) -> list[str]:
        """
        Get the direct dependencies of an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            List of entity IDs this entity depends on
        """
        result = self._graph.why(entity_id, max_depth=1)
        return result.entities

    def dependents_of(self, entity_id: str) -> list[str]:
        """
        Get the direct dependents of an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            List of entity IDs that depend on this entity
        """
        result = self._graph.impact(entity_id, max_depth=1)
        return result.entities

    def path_between(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[list[str]]:
        """
        Find a dependency path between two entities.

        Args:
            source_id: Starting entity
            target_id: Target entity

        Returns:
            Path as list of entity IDs, or None if no path exists
        """
        return self._graph.find_path(source_id, target_id)

    def roots(self) -> list[str]:
        """
        Find all root entities (no dependencies).

        These are typically Events - the ground truth facts.

        Returns:
            List of entity IDs with no dependencies
        """
        return self._graph.get_roots()

    def leaves(self) -> list[str]:
        """
        Find all leaf entities (no dependents).

        These are typically final outputs or terminal actions.

        Returns:
            List of entity IDs with no dependents
        """
        return self._graph.get_leaves()
