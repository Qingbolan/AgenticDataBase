"""
Dependency graph for AgenticDB.

This module provides causal tracking between entities, enabling
the core queries that differentiate AgenticDB from traditional databases:

- why(x): Trace the causal chain that led to X
- impact(x): Find everything that depends on X

Design Philosophy:
    Traditional databases answer "what is the data?"
    AgenticDB answers "how did state become this way?"

    The dependency graph is not an afterthought or a log - it's a
    first-class data structure that captures the causal relationships
    between all entities in the system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator, Optional

import networkx as nx
from pydantic import BaseModel, Field

from agenticdb.core.models import generate_id


class EdgeType(str, Enum):
    """Types of dependency relationships."""

    DEPENDS_ON = "depends_on"  # A requires B to exist/execute
    PRODUCES = "produces"  # A creates B
    INVALIDATES = "invalidates"  # A makes B stale
    DERIVED_FROM = "derived_from"  # A was computed from B
    SUPERSEDES = "supersedes"  # A replaces B (new version)
    CAUSED_BY = "caused_by"  # A happened because of B


class DependencyEdge(BaseModel):
    """
    A directed edge in the dependency graph.

    Edges capture causal relationships between entities:
    - source → target means "source relates to target"
    - edge_type specifies the nature of the relationship

    Examples:
        - Action "ApproveOrder" DEPENDS_ON Claim "risk_score"
        - Action "TrainModel" PRODUCES Event "ModelTrained"
        - Claim "new_score" SUPERSEDES Claim "old_score"
        - Event "DataUpdated" INVALIDATES Claim "cached_result"
    """

    id: str = Field(default_factory=generate_id, description="Edge identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    edge_type: EdgeType = Field(..., description="Type of relationship")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this edge was created"
    )
    branch_id: Optional[str] = Field(default=None, description="Branch this edge belongs to")
    version: Optional[int] = Field(default=None, description="Version when this edge was created")

    # Additional context
    weight: float = Field(default=1.0, description="Edge weight for graph algorithms")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")

    model_config = {"frozen": True, "extra": "forbid"}


class TraversalResult(BaseModel):
    """Result of a graph traversal operation."""

    root_id: str = Field(..., description="Starting entity ID")
    direction: str = Field(..., description="Traversal direction: upstream or downstream")
    entities: list[str] = Field(default_factory=list, description="Entity IDs in traversal order")
    edges: list[DependencyEdge] = Field(default_factory=list, description="Edges traversed")
    depth: int = Field(default=0, description="Maximum depth reached")

    model_config = {"extra": "forbid"}


class DependencyGraph:
    """
    A directed graph tracking causal relationships between entities.

    This is the core data structure that enables AgenticDB's killer queries:

    1. why(entity_id) - Traverse upstream to find all causes
    2. impact(entity_id) - Traverse downstream to find all effects
    3. invalidate(entity_id) - Mark downstream entities as stale

    Implementation uses NetworkX for graph algorithms, providing efficient
    traversal for even large dependency graphs.

    Thread Safety:
        This class is NOT thread-safe. Use appropriate synchronization
        if accessing from multiple threads.
    """

    def __init__(self, branch_id: Optional[str] = None):
        """
        Initialize an empty dependency graph.

        Args:
            branch_id: Optional branch ID for scoping
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._branch_id = branch_id
        self._edges: dict[str, DependencyEdge] = {}

    @property
    def branch_id(self) -> Optional[str]:
        """Get the branch ID this graph belongs to."""
        return self._branch_id

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self._graph.number_of_edges()

    def add_entity(self, entity_id: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """
        Add an entity node to the graph.

        Args:
            entity_id: Unique entity identifier
            metadata: Optional node metadata
        """
        self._graph.add_node(entity_id, **(metadata or {}))

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        version: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DependencyEdge:
        """
        Add a dependency edge between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Type of dependency relationship
            version: Optional version number
            metadata: Optional edge metadata

        Returns:
            The created DependencyEdge
        """
        # Ensure nodes exist
        if source_id not in self._graph:
            self.add_entity(source_id)
        if target_id not in self._graph:
            self.add_entity(target_id)

        # Create edge model
        edge = DependencyEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            branch_id=self._branch_id,
            version=version,
            metadata=metadata or {},
        )

        # Add to NetworkX graph
        self._graph.add_edge(
            source_id,
            target_id,
            edge_id=edge.id,
            edge_type=edge_type.value,
            weight=edge.weight,
        )

        # Store edge model
        self._edges[edge.id] = edge

        return edge

    def remove_entity(self, entity_id: str) -> list[DependencyEdge]:
        """
        Remove an entity and all its edges from the graph.

        Args:
            entity_id: Entity ID to remove

        Returns:
            List of removed edges
        """
        if entity_id not in self._graph:
            return []

        # Collect edges to remove
        removed_edges = []
        for edge_id, edge in list(self._edges.items()):
            if edge.source_id == entity_id or edge.target_id == entity_id:
                removed_edges.append(edge)
                del self._edges[edge_id]

        # Remove from graph
        self._graph.remove_node(entity_id)

        return removed_edges

    def get_edge(self, edge_id: str) -> Optional[DependencyEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def get_edges_between(
        self,
        source_id: str,
        target_id: str
    ) -> list[DependencyEdge]:
        """Get all edges between two entities."""
        return [
            edge for edge in self._edges.values()
            if edge.source_id == source_id and edge.target_id == target_id
        ]

    def has_entity(self, entity_id: str) -> bool:
        """Check if an entity exists in the graph."""
        return entity_id in self._graph

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if an edge exists between two entities."""
        return self._graph.has_edge(source_id, target_id)

    # =========================================================================
    # CORE QUERIES - The heart of AgenticDB
    # =========================================================================

    def why(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        edge_types: Optional[list[EdgeType]] = None,
    ) -> TraversalResult:
        """
        Trace the causal chain that led to an entity.

        This is the "why" query - traverse upstream through the dependency
        graph to find all entities that contributed to the target.

        Args:
            entity_id: The entity to trace
            max_depth: Maximum traversal depth (None = unlimited)
            edge_types: Filter to specific edge types (None = all)

        Returns:
            TraversalResult with all upstream entities and edges

        Example:
            ```python
            # Why was this order approved?
            result = graph.why(action_id)
            # Returns: [Event(UserRegistered), Claim(risk_score), ...]
            ```
        """
        return self._traverse(
            entity_id,
            direction="upstream",
            max_depth=max_depth,
            edge_types=edge_types,
        )

    def impact(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        edge_types: Optional[list[EdgeType]] = None,
    ) -> TraversalResult:
        """
        Find all entities affected by a change to this entity.

        This is the "impact" query - traverse downstream through the
        dependency graph to find everything that depends on the source.

        Args:
            entity_id: The entity to analyze
            max_depth: Maximum traversal depth (None = unlimited)
            edge_types: Filter to specific edge types (None = all)

        Returns:
            TraversalResult with all downstream entities and edges

        Example:
            ```python
            # What breaks if the risk model changes?
            result = graph.impact(model_claim_id)
            # Returns: [Action(ApproveOrder), Claim(derived_score), ...]
            ```
        """
        return self._traverse(
            entity_id,
            direction="downstream",
            max_depth=max_depth,
            edge_types=edge_types,
        )

    def _traverse(
        self,
        entity_id: str,
        direction: str,
        max_depth: Optional[int] = None,
        edge_types: Optional[list[EdgeType]] = None,
    ) -> TraversalResult:
        """
        Internal traversal implementation.

        Args:
            entity_id: Starting entity
            direction: "upstream" (predecessors) or "downstream" (successors)
            max_depth: Maximum depth
            edge_types: Edge type filter

        Returns:
            TraversalResult
        """
        if entity_id not in self._graph:
            return TraversalResult(
                root_id=entity_id,
                direction=direction,
                entities=[],
                edges=[],
                depth=0,
            )

        # Get traversal function based on direction
        if direction == "upstream":
            get_neighbors = self._graph.predecessors
            get_edge_data = lambda s, t: self._graph.get_edge_data(t, s)
        else:
            get_neighbors = self._graph.successors
            get_edge_data = lambda s, t: self._graph.get_edge_data(s, t)

        # BFS traversal
        visited: set[str] = set()
        result_entities: list[str] = []
        result_edges: list[DependencyEdge] = []
        queue: list[tuple[str, int]] = [(entity_id, 0)]
        max_depth_reached = 0

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue

            visited.add(current_id)
            if current_id != entity_id:
                result_entities.append(current_id)
            max_depth_reached = max(max_depth_reached, depth)

            # Explore neighbors
            for neighbor_id in get_neighbors(current_id):
                edge_data = get_edge_data(current_id, neighbor_id)
                if edge_data is None:
                    continue

                # Filter by edge type if specified
                if edge_types is not None:
                    edge_type_str = edge_data.get("edge_type")
                    if edge_type_str not in [et.value for et in edge_types]:
                        continue

                # Get the edge model
                edge_id = edge_data.get("edge_id")
                if edge_id and edge_id in self._edges:
                    result_edges.append(self._edges[edge_id])

                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))

        return TraversalResult(
            root_id=entity_id,
            direction=direction,
            entities=result_entities,
            edges=result_edges,
            depth=max_depth_reached,
        )

    def find_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[list[str]]:
        """
        Find a path between two entities.

        Args:
            source_id: Starting entity
            target_id: Target entity

        Returns:
            List of entity IDs forming the path, or None if no path exists
        """
        if source_id not in self._graph or target_id not in self._graph:
            return None

        try:
            return nx.shortest_path(self._graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None

    def detect_cycles(self) -> list[list[str]]:
        """
        Detect cycles in the dependency graph.

        Cycles indicate circular dependencies which can cause issues
        with invalidation and reproducibility.

        Returns:
            List of cycles (each cycle is a list of entity IDs)
        """
        try:
            cycles = list(nx.simple_cycles(self._graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []

    def topological_sort(self) -> list[str]:
        """
        Return entities in topological order (dependencies first).

        Useful for replaying state changes in correct order.

        Returns:
            List of entity IDs in dependency order

        Raises:
            ValueError: If the graph contains cycles
        """
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("Graph contains cycles, cannot perform topological sort")
        return list(nx.topological_sort(self._graph))

    def get_roots(self) -> list[str]:
        """
        Find all root entities (no incoming edges).

        Roots are typically Events - the ground truth facts
        that everything else depends on.

        Returns:
            List of entity IDs with no dependencies
        """
        return [
            node for node in self._graph.nodes()
            if self._graph.in_degree(node) == 0
        ]

    def get_leaves(self) -> list[str]:
        """
        Find all leaf entities (no outgoing edges).

        Leaves are typically the final outputs or actions
        at the end of dependency chains.

        Returns:
            List of entity IDs with no dependents
        """
        return [
            node for node in self._graph.nodes()
            if self._graph.out_degree(node) == 0
        ]

    def subgraph(self, entity_ids: list[str]) -> DependencyGraph:
        """
        Create a subgraph containing only the specified entities.

        Args:
            entity_ids: Entity IDs to include

        Returns:
            New DependencyGraph with only the specified entities
        """
        new_graph = DependencyGraph(branch_id=self._branch_id)

        for entity_id in entity_ids:
            if entity_id in self._graph:
                new_graph.add_entity(entity_id)

        for edge in self._edges.values():
            if edge.source_id in entity_ids and edge.target_id in entity_ids:
                new_graph._graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    edge_id=edge.id,
                    edge_type=edge.edge_type.value,
                    weight=edge.weight,
                )
                new_graph._edges[edge.id] = edge

        return new_graph

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the graph to a dictionary.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "branch_id": self._branch_id,
            "nodes": list(self._graph.nodes()),
            "edges": [edge.model_dump() for edge in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DependencyGraph:
        """
        Deserialize a graph from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed DependencyGraph
        """
        graph = cls(branch_id=data.get("branch_id"))

        for node in data.get("nodes", []):
            graph.add_entity(node)

        for edge_data in data.get("edges", []):
            edge = DependencyEdge(**edge_data)
            graph._graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_id=edge.id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
            )
            graph._edges[edge.id] = edge

        return graph
