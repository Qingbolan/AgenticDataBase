"""
Tests for AgenticDB Materialization layer.

This module tests the automatic recomputation feature that distinguishes
AgenticDB from traditional caching systems. Materialized views automatically
recompute when their dependencies change, ensuring derived values are
always consistent with their sources.

Test Coverage:
    - MaterializedView creation and registration
    - Automatic recomputation on dependency change
    - Lazy vs Eager recomputation modes
    - Dependency tracking and version awareness
    - Integration with DependencyGraph
    - Concurrent access safety
"""

import pytest
import time
from datetime import datetime, timezone
from typing import Any, Callable
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from agenticdb.core.models import Event, Claim, Action, EntityType
from agenticdb.core.dependency import DependencyGraph, EdgeType


class TestMaterializedView:
    """Tests for MaterializedView basic functionality."""

    def test_create_materialized_view(self):
        """Should create a materialized view with compute function."""
        from agenticdb.runtime.materialization import MaterializedView

        def compute_fn(deps: dict[str, Any]) -> int:
            return sum(deps.values())

        view = MaterializedView(
            key="total_score",
            compute_fn=compute_fn,
            depends_on=["score_a", "score_b"],
        )

        assert view.key == "total_score"
        assert view.depends_on == ["score_a", "score_b"]
        assert view.compute_fn is compute_fn
        assert view.is_stale == True  # Initially stale, not computed yet

    def test_compute_value(self):
        """Should compute and cache the value."""
        from agenticdb.runtime.materialization import MaterializedView

        call_count = 0

        def compute_fn(deps: dict[str, Any]) -> int:
            nonlocal call_count
            call_count += 1
            return deps.get("a", 0) + deps.get("b", 0)

        view = MaterializedView(
            key="sum",
            compute_fn=compute_fn,
            depends_on=["a", "b"],
        )

        # Compute with dependencies
        result = view.compute({"a": 10, "b": 20})
        assert result == 30
        assert view.value == 30
        assert view.is_stale == False
        assert call_count == 1

        # Accessing cached value should not recompute
        _ = view.value
        assert call_count == 1

    def test_mark_stale(self):
        """Should mark view as stale when dependency changes."""
        from agenticdb.runtime.materialization import MaterializedView

        view = MaterializedView(
            key="test",
            compute_fn=lambda deps: sum(deps.values()),
            depends_on=["a"],
        )

        view.compute({"a": 10})
        assert view.is_stale == False

        view.mark_stale()
        assert view.is_stale == True

    def test_recompute_when_stale(self):
        """Should recompute when accessed and stale."""
        from agenticdb.runtime.materialization import MaterializedView

        compute_count = 0
        current_value = 10

        def compute_fn(deps: dict[str, Any]) -> int:
            nonlocal compute_count
            compute_count += 1
            return deps.get("value", current_value)

        view = MaterializedView(
            key="dynamic",
            compute_fn=compute_fn,
            depends_on=["value"],
        )

        # First computation
        result = view.compute({"value": 10})
        assert result == 10
        assert compute_count == 1

        # Mark stale and recompute
        view.mark_stale()
        result = view.compute({"value": 20})
        assert result == 20
        assert compute_count == 2


class TestMaterializationManager:
    """Tests for MaterializationManager orchestration."""

    def test_register_materialized_view(self):
        """Should register and retrieve materialized views."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        # Register a view
        view = manager.register(
            key="user_score",
            compute_fn=lambda deps: deps.get("raw_score", 0) * 2,
            depends_on=["raw_score"],
        )

        assert view.key == "user_score"
        assert manager.get_view("user_score") is view
        assert manager.get_view("nonexistent") is None

    def test_compute_value(self):
        """Should compute materialized value with resolved dependencies."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        # Add source entities to graph
        graph.add_entity("event_1")
        graph.add_entity("claim_1")

        # Register view
        manager.register(
            key="aggregated",
            compute_fn=lambda deps: deps.get("event_1", 0) + deps.get("claim_1", 0),
            depends_on=["event_1", "claim_1"],
        )

        # Provide dependency values and compute
        result = manager.compute(
            "aggregated",
            dependency_values={"event_1": 100, "claim_1": 50}
        )
        assert result == 150

    def test_auto_invalidate_on_dependency_change(self):
        """Should automatically invalidate views when dependencies change."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        compute_count = 0

        def tracked_compute(deps: dict[str, Any]) -> int:
            nonlocal compute_count
            compute_count += 1
            return deps.get("source", 0) * 10

        # Register view
        manager.register(
            key="derived",
            compute_fn=tracked_compute,
            depends_on=["source"],
        )

        # Initial computation
        manager.compute("derived", {"source": 5})
        assert compute_count == 1

        # Simulate dependency change
        manager.on_entity_changed("source")

        # View should be marked stale
        view = manager.get_view("derived")
        assert view.is_stale == True

    def test_eager_recomputation_mode(self):
        """Should recompute immediately when dependency changes (eager mode)."""
        from agenticdb.runtime.materialization import (
            MaterializationManager,
            RecomputeMode,
        )
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph, default_mode=RecomputeMode.EAGER)

        compute_count = 0
        current_value = {"source": 10}

        def compute_fn(deps: dict[str, Any]) -> int:
            nonlocal compute_count
            compute_count += 1
            return deps.get("source", 0) * 2

        # Register view with value resolver
        manager.register(
            key="eager_view",
            compute_fn=compute_fn,
            depends_on=["source"],
            value_resolver=lambda entity_id: current_value.get(entity_id, 0),
        )

        # Initial computation
        manager.compute("eager_view", {"source": 10})
        assert compute_count == 1

        # Change dependency value and notify
        current_value["source"] = 20
        manager.on_entity_changed("source")

        # Eager mode should have recomputed
        assert compute_count == 2
        view = manager.get_view("eager_view")
        assert view.value == 40  # 20 * 2

    def test_lazy_recomputation_mode(self):
        """Should recompute only when accessed (lazy mode)."""
        from agenticdb.runtime.materialization import (
            MaterializationManager,
            RecomputeMode,
        )
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph, default_mode=RecomputeMode.LAZY)

        compute_count = 0

        def compute_fn(deps: dict[str, Any]) -> int:
            nonlocal compute_count
            compute_count += 1
            return deps.get("source", 0) * 2

        # Register view
        manager.register(
            key="lazy_view",
            compute_fn=compute_fn,
            depends_on=["source"],
        )

        # Initial computation
        manager.compute("lazy_view", {"source": 10})
        assert compute_count == 1

        # Change dependency
        manager.on_entity_changed("source")

        # Lazy mode should NOT have recomputed yet
        assert compute_count == 1

        # Now access the value - should trigger recomputation
        manager.compute("lazy_view", {"source": 20})
        assert compute_count == 2


class TestTransitiveDependencies:
    """Tests for transitive dependency invalidation."""

    def test_cascade_invalidation(self):
        """Should invalidate downstream views when upstream changes."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        # Create a chain: raw_data -> processed -> final

        # Level 1: raw_data (source entity)
        graph.add_entity("raw_data")

        # Level 2: processed depends on raw_data
        manager.register(
            key="processed",
            compute_fn=lambda deps: deps.get("raw_data", 0) + 100,
            depends_on=["raw_data"],
        )

        # Level 3: final depends on processed
        manager.register(
            key="final",
            compute_fn=lambda deps: deps.get("processed", 0) * 2,
            depends_on=["processed"],
        )

        # Compute the chain
        manager.compute("processed", {"raw_data": 10})  # 10 + 100 = 110
        manager.compute("final", {"processed": 110})    # 110 * 2 = 220

        # Verify initial values
        assert manager.get_view("processed").value == 110
        assert manager.get_view("final").value == 220

        # Change raw_data - should invalidate both processed and final
        manager.on_entity_changed("raw_data")

        assert manager.get_view("processed").is_stale == True
        assert manager.get_view("final").is_stale == True

    def test_multiple_dependencies(self):
        """Should handle views with multiple dependencies."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        # Add source entities
        graph.add_entity("price")
        graph.add_entity("quantity")
        graph.add_entity("discount")

        # Register view depending on all three
        manager.register(
            key="total",
            compute_fn=lambda deps: (
                deps.get("price", 0) *
                deps.get("quantity", 0) *
                (1 - deps.get("discount", 0))
            ),
            depends_on=["price", "quantity", "discount"],
        )

        # Compute
        manager.compute("total", {"price": 100, "quantity": 5, "discount": 0.1})
        assert manager.get_view("total").value == 450  # 100 * 5 * 0.9

        # Changing any dependency should invalidate
        manager.on_entity_changed("discount")
        assert manager.get_view("total").is_stale == True


class TestVersionAwareness:
    """Tests for version-aware materialization."""

    def test_track_computation_version(self):
        """Should track at which version the value was computed."""
        from agenticdb.runtime.materialization import MaterializedView

        view = MaterializedView(
            key="versioned",
            compute_fn=lambda deps: deps.get("value", 0),
            depends_on=["value"],
        )

        # Compute at version 1
        view.compute({"value": 10}, version=1)
        assert view.computed_at_version == 1

        # Compute at version 5
        view.mark_stale()
        view.compute({"value": 20}, version=5)
        assert view.computed_at_version == 5

    def test_skip_recompute_if_version_unchanged(self):
        """Should skip recomputation if dependency version hasn't changed."""
        from agenticdb.runtime.materialization import (
            MaterializationManager,
            RecomputeMode,
        )
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        compute_count = 0

        def tracked_compute(deps: dict[str, Any]) -> int:
            nonlocal compute_count
            compute_count += 1
            return deps.get("source", 0)

        manager.register(
            key="version_aware",
            compute_fn=tracked_compute,
            depends_on=["source"],
        )

        # Compute at version 1
        manager.compute("version_aware", {"source": 10}, version=1)
        assert compute_count == 1

        # Request computation at same version - should use cached
        result = manager.get_value_if_valid("version_aware", version=1)
        assert result == 10
        assert compute_count == 1  # No recomputation


class TestMaterializationIntegration:
    """Integration tests with AgenticDB components."""

    def test_integrate_with_branch_handle(self):
        """Should integrate with BranchHandle for automatic recomputation."""
        from agenticdb.interface.client import AgenticDB
        from agenticdb.core.models import Event, Claim
        from agenticdb.runtime.materialization import (
            MaterializationManager,
            RecomputeMode,
        )

        db = AgenticDB()
        branch = db.create_branch("test-materialization")

        # Create materialization manager for this branch
        manager = MaterializationManager(
            branch._graph,
            default_mode=RecomputeMode.LAZY,
        )

        # Record some events
        event1 = branch.record(Event(
            event_type="OrderPlaced",
            data={"amount": 100}
        ))
        event2 = branch.record(Event(
            event_type="OrderPlaced",
            data={"amount": 200}
        ))

        # Register a materialized view for total orders
        def compute_total(deps: dict[str, Any]) -> float:
            total = 0
            for entity_id, data in deps.items():
                if isinstance(data, dict):
                    total += data.get("amount", 0)
            return total

        manager.register(
            key="total_orders",
            compute_fn=compute_total,
            depends_on=[event1.id, event2.id],
        )

        # Compute total
        result = manager.compute("total_orders", {
            event1.id: event1.data,
            event2.id: event2.data,
        })

        assert result == 300

    def test_materialized_claim_auto_update(self):
        """Should auto-update materialized claims when sources change."""
        from agenticdb.interface.client import AgenticDB
        from agenticdb.core.models import Event, Claim
        from agenticdb.runtime.materialization import (
            MaterializationManager,
            RecomputeMode,
        )

        db = AgenticDB()
        branch = db.create_branch("test-claim-materialization")

        manager = MaterializationManager(
            branch._graph,
            default_mode=RecomputeMode.EAGER,
        )

        # Record a claim with a score
        risk_claim = branch.record(Claim(
            subject="user.risk_score",
            predicate="equals",
            value=0.3,
            source="risk_model_v1",
        ))

        # Store the claim value for resolution
        claim_values = {risk_claim.id: risk_claim.value}

        # Register a derived decision
        manager.register(
            key="approval_decision",
            compute_fn=lambda deps: "APPROVED" if deps.get(risk_claim.id, 1.0) < 0.5 else "REJECTED",
            depends_on=[risk_claim.id],
            value_resolver=lambda eid: claim_values.get(eid, 1.0),
        )

        # Initial computation
        manager.compute("approval_decision", {risk_claim.id: 0.3})
        assert manager.get_view("approval_decision").value == "APPROVED"

        # Update claim value (simulating new risk assessment)
        claim_values[risk_claim.id] = 0.7
        manager.on_entity_changed(risk_claim.id)

        # Eager mode should have recomputed
        assert manager.get_view("approval_decision").value == "REJECTED"


class TestConcurrency:
    """Tests for thread-safe materialization."""

    def test_concurrent_reads(self):
        """Should handle concurrent reads safely."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        manager.register(
            key="concurrent",
            compute_fn=lambda deps: sum(deps.values()),
            depends_on=["a", "b"],
        )

        manager.compute("concurrent", {"a": 10, "b": 20})

        results = []

        def read_value():
            for _ in range(100):
                view = manager.get_view("concurrent")
                if view:
                    results.append(view.value)

        threads = [Thread(target=read_value) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should return the same value
        assert all(v == 30 for v in results)

    def test_concurrent_invalidation_and_compute(self):
        """Should handle concurrent invalidation and computation."""
        from agenticdb.runtime.materialization import (
            MaterializationManager,
            RecomputeMode,
        )
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph, default_mode=RecomputeMode.LAZY)

        compute_count = 0

        def tracked_compute(deps: dict[str, Any]) -> int:
            nonlocal compute_count
            compute_count += 1
            time.sleep(0.001)  # Simulate computation time
            return deps.get("source", 0)

        manager.register(
            key="contested",
            compute_fn=tracked_compute,
            depends_on=["source"],
        )

        manager.compute("contested", {"source": 1})

        errors = []

        def invalidate_loop():
            try:
                for i in range(50):
                    manager.on_entity_changed("source")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def compute_loop():
            try:
                for i in range(50):
                    manager.compute("contested", {"source": i})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(invalidate_loop),
                executor.submit(compute_loop),
                executor.submit(invalidate_loop),
                executor.submit(compute_loop),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestMaterializationStats:
    """Tests for materialization statistics and monitoring."""

    def test_track_computation_stats(self):
        """Should track computation statistics."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        manager.register(
            key="stats_test",
            compute_fn=lambda deps: sum(deps.values()),
            depends_on=["a"],
        )

        # Perform computations
        manager.compute("stats_test", {"a": 10})
        manager.on_entity_changed("a")
        manager.compute("stats_test", {"a": 20})
        manager.on_entity_changed("a")
        manager.compute("stats_test", {"a": 30})

        stats = manager.get_stats()
        assert stats.total_computations == 3
        assert stats.total_invalidations == 2

    def test_track_computation_time(self):
        """Should track computation time."""
        from agenticdb.runtime.materialization import MaterializationManager
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph(branch_id="test")
        manager = MaterializationManager(graph)

        def slow_compute(deps: dict[str, Any]) -> int:
            time.sleep(0.05)  # 50ms
            return sum(deps.values())

        manager.register(
            key="slow",
            compute_fn=slow_compute,
            depends_on=["a"],
        )

        manager.compute("slow", {"a": 10})

        stats = manager.get_stats()
        assert stats.total_computation_time_ms >= 50
