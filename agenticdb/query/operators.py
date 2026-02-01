"""
Query operators for AgenticDB.

This module defines the query operators that differentiate AgenticDB
from traditional databases:

Traditional DB:  "What is the value of X?"
AgenticDB:       "Why did X become this value?"
                 "What will break if X changes?"

These operators traverse the dependency graph to answer causal questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from agenticdb.core.dependency import DependencyGraph, EdgeType, TraversalResult
from agenticdb.core.models import Entity, EntityType

if TYPE_CHECKING:
    from agenticdb.storage.engine import StorageEngine


class QueryResultType(str, Enum):
    """Type of query result."""

    WHY = "why"
    IMPACT = "impact"
    TRACE = "trace"
    HISTORY = "history"


class CausalStep(BaseModel):
    """A single step in a causal chain."""

    entity_id: str = Field(..., description="Entity ID")
    entity_type: EntityType = Field(..., description="Type of entity")
    relationship: str = Field(..., description="How this relates to the next step")
    depth: int = Field(..., description="Depth from the query root")
    summary: str = Field(default="", description="Human-readable summary")

    # Optional full entity data
    entity_data: Optional[dict[str, Any]] = Field(default=None, description="Full entity data")

    model_config = {"extra": "forbid"}


class CausalChain(BaseModel):
    """
    A causal chain explaining why something happened.

    This is the result of a why() query - a sequence of steps
    showing the causal path from root causes to the queried entity.
    """

    query_type: QueryResultType = Field(..., description="Type of query")
    root_id: str = Field(..., description="Entity ID that was queried")
    steps: list[CausalStep] = Field(default_factory=list, description="Steps in the causal chain")
    total_depth: int = Field(default=0, description="Maximum depth of the chain")

    # Metadata
    executed_at: datetime = Field(..., description="When the query was executed")
    execution_time_ms: float = Field(default=0.0, description="Query execution time")

    model_config = {"extra": "forbid"}

    def to_tree_string(self, indent: str = "  ") -> str:
        """
        Format the causal chain as a tree string.

        Returns:
            Human-readable tree representation
        """
        lines = [f"Query: {self.query_type.value}({self.root_id})"]

        for step in self.steps:
            prefix = indent * step.depth
            lines.append(f"{prefix}└─ [{step.entity_type.value}] {step.entity_id}")
            if step.summary:
                lines.append(f"{prefix}   {step.summary}")

        return "\n".join(lines)


class ImpactResult(BaseModel):
    """
    Result of an impact() query.

    Shows all entities that depend on the queried entity,
    organized by type and depth.
    """

    query_type: QueryResultType = Field(default=QueryResultType.IMPACT)
    root_id: str = Field(..., description="Entity ID that was queried")

    # Affected entities by type
    affected_events: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    affected_actions: list[str] = Field(default_factory=list)

    # Total counts
    total_affected: int = Field(default=0, description="Total affected entities")
    max_depth: int = Field(default=0, description="Maximum dependency depth")

    # Invalidation info
    invalidated_ids: list[str] = Field(
        default_factory=list,
        description="Entity IDs that were marked as invalidated"
    )

    model_config = {"extra": "forbid"}


class WhyQuery:
    """
    Query operator for tracing causal chains.

    Usage:
        ```python
        query = WhyQuery(storage, graph)
        chain = query.execute(entity_id)
        print(chain.to_tree_string())
        ```
    """

    def __init__(
        self,
        storage: StorageEngine,
        graph: DependencyGraph,
        branch_id: str
    ):
        """
        Initialize the query operator.

        Args:
            storage: Storage engine for retrieving entities
            graph: Dependency graph for traversal
            branch_id: Branch to query
        """
        self._storage = storage
        self._graph = graph
        self._branch_id = branch_id

    def execute(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        include_entity_data: bool = False,
    ) -> CausalChain:
        """
        Execute the why query.

        Traces upstream through the dependency graph to find all
        entities that contributed to the target entity.

        Args:
            entity_id: Entity to trace
            max_depth: Maximum depth to traverse
            include_entity_data: Whether to include full entity data

        Returns:
            CausalChain showing the causal path
        """
        start_time = datetime.now()

        # Traverse upstream
        traversal = self._graph.why(entity_id, max_depth=max_depth)

        # Build causal steps
        steps: list[CausalStep] = []
        visited_depths: dict[str, int] = {entity_id: 0}

        # BFS to assign depths
        queue = [(entity_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            for edge in traversal.edges:
                if edge.target_id == current_id and edge.source_id not in visited_depths:
                    visited_depths[edge.source_id] = depth + 1
                    queue.append((edge.source_id, depth + 1))

        # Build steps from traversal
        for eid in traversal.entities:
            entity = self._storage.get(eid, self._branch_id)
            if entity is None:
                continue

            step = CausalStep(
                entity_id=eid,
                entity_type=entity.entity_type,
                relationship=self._get_relationship(eid, traversal),
                depth=visited_depths.get(eid, 0),
                summary=self._summarize_entity(entity),
                entity_data=entity.model_dump() if include_entity_data else None,
            )
            steps.append(step)

        # Sort by depth
        steps.sort(key=lambda s: s.depth)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000

        return CausalChain(
            query_type=QueryResultType.WHY,
            root_id=entity_id,
            steps=steps,
            total_depth=traversal.depth,
            executed_at=start_time,
            execution_time_ms=execution_time,
        )

    def _get_relationship(self, entity_id: str, traversal: TraversalResult) -> str:
        """Get the relationship type for an entity in the traversal."""
        for edge in traversal.edges:
            if edge.source_id == entity_id:
                return edge.edge_type.value
        return "unknown"

    def _summarize_entity(self, entity: Entity) -> str:
        """Create a human-readable summary of an entity."""
        if entity.entity_type == EntityType.EVENT:
            from agenticdb.core.models import Event
            if isinstance(entity, Event):
                return f"{entity.event_type}: {entity.data}"
        elif entity.entity_type == EntityType.CLAIM:
            from agenticdb.core.models import Claim
            if isinstance(entity, Claim):
                return f"{entity.subject} = {entity.value} (from {entity.source})"
        elif entity.entity_type == EntityType.ACTION:
            from agenticdb.core.models import Action
            if isinstance(entity, Action):
                return f"{entity.action_type} by {entity.agent_id}"
        return ""


class ImpactQuery:
    """
    Query operator for finding downstream dependencies.

    Usage:
        ```python
        query = ImpactQuery(storage, graph)
        result = query.execute(entity_id)
        print(f"Affected: {result.total_affected}")
        ```
    """

    def __init__(
        self,
        storage: StorageEngine,
        graph: DependencyGraph,
        branch_id: str
    ):
        """
        Initialize the query operator.

        Args:
            storage: Storage engine for retrieving entities
            graph: Dependency graph for traversal
            branch_id: Branch to query
        """
        self._storage = storage
        self._graph = graph
        self._branch_id = branch_id

    def execute(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        auto_invalidate: bool = False,
    ) -> ImpactResult:
        """
        Execute the impact query.

        Finds all entities that depend on the target entity.
        Optionally marks them as invalidated.

        Args:
            entity_id: Entity to analyze
            max_depth: Maximum depth to traverse
            auto_invalidate: Whether to mark affected entities as invalidated

        Returns:
            ImpactResult showing all affected entities
        """
        # Traverse downstream
        traversal = self._graph.impact(entity_id, max_depth=max_depth)

        # Categorize affected entities
        affected_events: list[str] = []
        affected_claims: list[str] = []
        affected_actions: list[str] = []
        invalidated_ids: list[str] = []

        for eid in traversal.entities:
            entity = self._storage.get(eid, self._branch_id)
            if entity is None:
                continue

            if entity.entity_type == EntityType.EVENT:
                affected_events.append(eid)
            elif entity.entity_type == EntityType.CLAIM:
                affected_claims.append(eid)
                if auto_invalidate:
                    from agenticdb.core.models import EntityStatus
                    entity.status = EntityStatus.INVALIDATED
                    invalidated_ids.append(eid)
            elif entity.entity_type == EntityType.ACTION:
                affected_actions.append(eid)

        return ImpactResult(
            root_id=entity_id,
            affected_events=affected_events,
            affected_claims=affected_claims,
            affected_actions=affected_actions,
            total_affected=len(traversal.entities),
            max_depth=traversal.depth,
            invalidated_ids=invalidated_ids,
        )


class TraceQuery:
    """
    Query operator for tracing a complete execution path.

    Combines why() and impact() to show the full context
    of an entity: both what caused it and what depends on it.
    """

    def __init__(
        self,
        storage: StorageEngine,
        graph: DependencyGraph,
        branch_id: str
    ):
        """
        Initialize the query operator.

        Args:
            storage: Storage engine for retrieving entities
            graph: Dependency graph for traversal
            branch_id: Branch to query
        """
        self._storage = storage
        self._graph = graph
        self._branch_id = branch_id
        self._why_query = WhyQuery(storage, graph, branch_id)
        self._impact_query = ImpactQuery(storage, graph, branch_id)

    def execute(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Execute a full trace query.

        Args:
            entity_id: Entity to trace
            max_depth: Maximum depth in each direction

        Returns:
            Dictionary with 'why' and 'impact' results
        """
        return {
            "entity_id": entity_id,
            "why": self._why_query.execute(entity_id, max_depth),
            "impact": self._impact_query.execute(entity_id, max_depth),
        }
