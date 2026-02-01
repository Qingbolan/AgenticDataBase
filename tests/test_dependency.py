"""
Tests for AgenticDB dependency graph.
"""

import pytest

from agenticdb.core.dependency import (
    DependencyGraph,
    DependencyEdge,
    EdgeType,
    TraversalResult,
)


class TestDependencyGraph:
    """Tests for the dependency graph."""

    def test_add_entity(self):
        """Should add entities to graph."""
        graph = DependencyGraph()
        graph.add_entity("entity-1")
        graph.add_entity("entity-2")

        assert graph.has_entity("entity-1")
        assert graph.has_entity("entity-2")
        assert not graph.has_entity("entity-3")
        assert graph.node_count == 2

    def test_add_edge(self):
        """Should add edges between entities."""
        graph = DependencyGraph()
        edge = graph.add_edge("a", "b", EdgeType.DEPENDS_ON)

        assert graph.has_edge("a", "b")
        assert not graph.has_edge("b", "a")  # Directed
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.edge_type == EdgeType.DEPENDS_ON

    def test_auto_create_nodes(self):
        """Should auto-create nodes when adding edges."""
        graph = DependencyGraph()
        graph.add_edge("new-a", "new-b", EdgeType.DEPENDS_ON)

        assert graph.has_entity("new-a")
        assert graph.has_entity("new-b")

    def test_remove_entity(self):
        """Should remove entity and its edges."""
        graph = DependencyGraph()
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "c", EdgeType.DEPENDS_ON)

        removed = graph.remove_entity("b")
        assert not graph.has_entity("b")
        assert len(removed) == 2  # Both edges involving b


class TestWhyQuery:
    """Tests for the why() traversal."""

    def test_simple_chain(self):
        """Should trace simple dependency chain."""
        graph = DependencyGraph()
        # C depends on B, B depends on A
        graph.add_edge("c", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)

        result = graph.why("c")

        assert "b" in result.entities
        assert "a" in result.entities
        assert result.depth >= 1

    def test_multiple_dependencies(self):
        """Should find all upstream dependencies."""
        graph = DependencyGraph()
        # D depends on B and C, both depend on A
        graph.add_edge("d", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("d", "c", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "a", EdgeType.DEPENDS_ON)

        result = graph.why("d")

        assert "b" in result.entities
        assert "c" in result.entities
        assert "a" in result.entities

    def test_max_depth(self):
        """Should respect max depth."""
        graph = DependencyGraph()
        graph.add_edge("d", "c", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)

        result = graph.why("d", max_depth=1)

        assert "c" in result.entities
        assert "b" not in result.entities
        assert "a" not in result.entities

    def test_nonexistent_entity(self):
        """Should handle nonexistent entity gracefully."""
        graph = DependencyGraph()
        result = graph.why("nonexistent")

        assert result.entities == []
        assert result.depth == 0


class TestImpactQuery:
    """Tests for the impact() traversal."""

    def test_simple_downstream(self):
        """Should find downstream dependents."""
        graph = DependencyGraph()
        # B and C depend on A
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "a", EdgeType.DEPENDS_ON)

        result = graph.impact("a")

        assert "b" in result.entities
        assert "c" in result.entities

    def test_transitive_impact(self):
        """Should find transitive dependents."""
        graph = DependencyGraph()
        # D depends on C, C depends on B, B depends on A
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("d", "c", EdgeType.DEPENDS_ON)

        result = graph.impact("a")

        assert "b" in result.entities
        assert "c" in result.entities
        assert "d" in result.entities

    def test_max_depth_impact(self):
        """Should respect max depth in impact."""
        graph = DependencyGraph()
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("d", "c", EdgeType.DEPENDS_ON)

        result = graph.impact("a", max_depth=1)

        assert "b" in result.entities
        assert "c" not in result.entities


class TestGraphAnalysis:
    """Tests for graph analysis functions."""

    def test_find_path(self):
        """Should find path between entities."""
        graph = DependencyGraph()
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "c", EdgeType.DEPENDS_ON)

        path = graph.find_path("a", "c")
        assert path == ["a", "b", "c"]

        no_path = graph.find_path("c", "a")  # Wrong direction
        assert no_path is None

    def test_detect_cycles(self):
        """Should detect cycles in graph."""
        graph = DependencyGraph()
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "c", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "a", EdgeType.DEPENDS_ON)  # Cycle!

        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_topological_sort(self):
        """Should sort in dependency order."""
        graph = DependencyGraph()
        graph.add_edge("c", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)

        sorted_nodes = graph.topological_sort()

        # a should come before b, b before c
        assert sorted_nodes.index("a") < sorted_nodes.index("b")
        assert sorted_nodes.index("b") < sorted_nodes.index("c")

    def test_topological_sort_with_cycle(self):
        """Should raise error for cyclic graph."""
        graph = DependencyGraph()
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)

        with pytest.raises(ValueError):
            graph.topological_sort()

    def test_get_roots(self):
        """Should find root nodes (no dependencies)."""
        graph = DependencyGraph()
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "a", EdgeType.DEPENDS_ON)
        graph.add_entity("d")  # Isolated node is also a root

        roots = graph.get_roots()
        assert "a" in roots
        assert "d" in roots
        assert "b" not in roots

    def test_get_leaves(self):
        """Should find leaf nodes (no dependents)."""
        graph = DependencyGraph()
        graph.add_edge("b", "a", EdgeType.DEPENDS_ON)
        graph.add_edge("c", "b", EdgeType.DEPENDS_ON)

        leaves = graph.get_leaves()
        assert "c" in leaves
        assert "a" not in leaves

    def test_subgraph(self):
        """Should create subgraph with specified entities."""
        graph = DependencyGraph()
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)
        graph.add_edge("b", "c", EdgeType.DEPENDS_ON)
        graph.add_edge("d", "e", EdgeType.DEPENDS_ON)

        sub = graph.subgraph(["a", "b", "c"])

        assert sub.has_entity("a")
        assert sub.has_entity("b")
        assert not sub.has_entity("d")
        assert sub.has_edge("a", "b")


class TestSerialization:
    """Tests for graph serialization."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        graph = DependencyGraph(branch_id="test-branch")
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)

        data = graph.to_dict()

        assert data["branch_id"] == "test-branch"
        assert "a" in data["nodes"]
        assert "b" in data["nodes"]
        assert len(data["edges"]) == 1

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        graph = DependencyGraph(branch_id="test-branch")
        graph.add_edge("a", "b", EdgeType.DEPENDS_ON)
        data = graph.to_dict()

        restored = DependencyGraph.from_dict(data)

        assert restored.branch_id == "test-branch"
        assert restored.has_entity("a")
        assert restored.has_edge("a", "b")
