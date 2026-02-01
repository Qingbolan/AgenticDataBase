"""
Integration tests for AgenticDB.

These tests verify the full system works together correctly.
"""

import pytest

from agenticdb import AgenticDB, Event, Claim, Action
from agenticdb.core.models import EntityType


class TestAgenticDBIntegration:
    """Integration tests for the full AgenticDB system."""

    @pytest.fixture
    def db(self):
        """Create a fresh database instance."""
        return AgenticDB()

    def test_basic_workflow(self, db):
        """Test basic event-claim-action workflow."""
        branch = db.create_branch("test-workflow")

        # Record an event
        event = branch.record(Event(
            event_type="UserCreated",
            data={"user_id": "u1"}
        ))
        assert event.id is not None
        assert event.branch_id == branch.id

        # Store a claim
        claim = branch.record(Claim(
            subject="user.u1.verified",
            value=True,
            source="verification-service",
            derived_from=[event.id]
        ))
        assert claim.id is not None

        # Execute an action
        action = branch.execute(Action(
            action_type="ActivateUser",
            agent_id="activation-agent",
            depends_on=[event.id, claim.id]
        ))
        assert action.id is not None
        assert len(action.depends_on) == 2

    def test_why_query(self, db):
        """Test why() query returns correct causal chain."""
        branch = db.create_branch("test-why")

        # Build a dependency chain
        event = branch.record(Event(event_type="E1", data={}))
        claim = branch.record(Claim(
            subject="c1",
            value=1,
            source="s1",
            derived_from=[event.id]
        ))
        action = branch.execute(Action(
            action_type="A1",
            agent_id="agent",
            depends_on=[claim.id]
        ))

        # Query why
        chain = branch.why(action.id)

        # Should include the claim in the chain
        entity_ids = [step.entity_id for step in chain.steps]
        assert claim.id in entity_ids

    def test_impact_query(self, db):
        """Test impact() query finds downstream dependencies."""
        branch = db.create_branch("test-impact")

        # Build dependencies
        event = branch.record(Event(event_type="E1", data={}))
        claim1 = branch.record(Claim(
            subject="c1",
            value=1,
            source="s1",
            derived_from=[event.id]
        ))
        claim2 = branch.record(Claim(
            subject="c2",
            value=2,
            source="s2",
            derived_from=[claim1.id]
        ))
        action = branch.execute(Action(
            action_type="A1",
            agent_id="agent",
            depends_on=[claim2.id]
        ))

        # Query impact of event
        impact = branch.impact(event.id)

        # Should find all downstream entities
        assert impact.total_affected >= 1

    def test_version_tracking(self, db):
        """Test version numbers increment correctly."""
        branch = db.create_branch("test-version")

        assert branch.version == 0

        e1 = branch.record(Event(event_type="E1", data={}))
        assert branch.version == 1

        e2 = branch.record(Event(event_type="E2", data={}))
        assert branch.version == 2

    def test_time_travel(self, db):
        """Test time travel snapshot."""
        branch = db.create_branch("test-timetravel")

        # Record some events
        e1 = branch.record(Event(event_type="E1", data={"v": 1}))
        e2 = branch.record(Event(event_type="E2", data={"v": 2}))
        e3 = branch.record(Event(event_type="E3", data={"v": 3}))

        # Get snapshot at version 2
        snapshot = branch.at(version=2)

        # Should have 2 events
        assert snapshot.version == 2
        assert len(snapshot.events) == 2

    def test_entity_retrieval(self, db):
        """Test entity retrieval by ID."""
        branch = db.create_branch("test-retrieval")

        event = branch.record(Event(event_type="Test", data={"key": "value"}))

        # Retrieve by ID
        retrieved = branch.get(event.id)
        assert retrieved is not None
        assert retrieved.id == event.id
        assert retrieved.entity_type == EntityType.EVENT

    def test_query_by_type(self, db):
        """Test querying entities by type."""
        branch = db.create_branch("test-query-type")

        # Create multiple entities
        branch.record(Event(event_type="E1", data={}))
        branch.record(Event(event_type="E2", data={}))
        branch.record(Claim(subject="c1", value=1, source="s"))
        branch.execute(Action(action_type="A1", agent_id="agent"))

        # Query by type
        events = list(branch.events())
        claims = list(branch.claims())
        actions = list(branch.actions())

        assert len(events) == 2
        assert len(claims) == 1
        assert len(actions) == 1

    def test_query_with_filters(self, db):
        """Test querying with filters."""
        branch = db.create_branch("test-filters")

        branch.record(Event(event_type="TypeA", data={}))
        branch.record(Event(event_type="TypeA", data={}))
        branch.record(Event(event_type="TypeB", data={}))

        # Filter by event_type
        type_a_events = list(branch.events(event_type="TypeA"))
        assert len(type_a_events) == 2

    def test_multiple_branches(self, db):
        """Test working with multiple branches."""
        branch1 = db.create_branch("branch-1")
        branch2 = db.create_branch("branch-2")

        # Add to branch 1
        e1 = branch1.record(Event(event_type="E1", data={}))

        # Add to branch 2
        e2 = branch2.record(Event(event_type="E2", data={}))

        # Verify isolation
        assert branch1.get(e1.id) is not None
        assert branch1.get(e2.id) is None
        assert branch2.get(e2.id) is not None
        assert branch2.get(e1.id) is None

    def test_list_branches(self, db):
        """Test listing all branches."""
        db.create_branch("b1")
        db.create_branch("b2")

        branches = db.list_branches()
        names = [b.name for b in branches]

        assert "main" in names
        assert "b1" in names
        assert "b2" in names


class TestCacheIntegration:
    """Integration tests for dependency-aware caching."""

    @pytest.fixture
    def db(self):
        """Create a fresh database instance."""
        return AgenticDB()

    def test_cache_invalidation_on_dependency_change(self, db):
        """Test that cache invalidates when dependencies change."""
        from agenticdb.runtime.cache import DependencyAwareCache
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph()
        cache = DependencyAwareCache(graph)

        # Add entities to graph
        graph.add_entity("source")
        graph.add_entity("derived")
        from agenticdb.core.dependency import EdgeType
        graph.add_edge("derived", "source", EdgeType.DEPENDS_ON)

        # Cache a value that depends on source
        cache.set("my_cache_key", "cached_value", depends_on=["source"])

        # Verify it's cached
        assert cache.get("my_cache_key") == "cached_value"

        # Invalidate based on source change
        invalidated = cache.invalidate_dependents("source")

        # Cache should be invalidated
        assert "my_cache_key" in invalidated
        assert cache.get("my_cache_key") is None

    def test_cache_stats(self, db):
        """Test cache statistics tracking."""
        from agenticdb.runtime.cache import DependencyAwareCache
        from agenticdb.core.dependency import DependencyGraph

        graph = DependencyGraph()
        cache = DependencyAwareCache(graph)

        # Miss
        cache.get("nonexistent")

        # Set and hit
        cache.set("key", "value", depends_on=[])
        cache.get("key")
        cache.get("key")

        stats = cache.stats
        assert stats.misses == 1
        assert stats.hits == 2


class TestSubscriptionIntegration:
    """Integration tests for subscriptions."""

    @pytest.fixture
    def db(self):
        """Create a fresh database instance."""
        return AgenticDB()

    def test_subscribe_to_entity(self, db):
        """Test subscribing to entity changes."""
        branch = db.create_branch("test-sub")

        events_received = []

        def callback(event):
            events_received.append(event)

        # Subscribe to all events
        db.subscribe_type(EntityType.EVENT, callback, branch.id)

        # Create an event
        branch.record(Event(event_type="Test", data={}))

        # Should have received notification
        assert len(events_received) == 1
        assert events_received[0].event_type == "create"
