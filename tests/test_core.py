"""
Tests for AgenticDB core models.
"""

import pytest
from datetime import datetime, timezone

from agenticdb.core.models import (
    Entity,
    EntityType,
    EntityStatus,
    Event,
    Claim,
    Action,
    ActionStatus,
    generate_id,
    compute_content_hash,
)


class TestGenerateId:
    """Tests for ID generation."""

    def test_generates_unique_ids(self):
        """IDs should be unique."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_ids_are_sortable(self):
        """IDs should be sortable by time (ULID property)."""
        id1 = generate_id()
        id2 = generate_id()
        # ULID: later IDs should be greater
        assert id2 >= id1


class TestContentHash:
    """Tests for content hashing."""

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        data = {"key": "value", "number": 42}
        hash1 = compute_content_hash(data)
        hash2 = compute_content_hash(data)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        hash1 = compute_content_hash({"key": "value1"})
        hash2 = compute_content_hash({"key": "value2"})
        assert hash1 != hash2

    def test_order_independent(self):
        """Hash should be independent of key order."""
        hash1 = compute_content_hash({"a": 1, "b": 2})
        hash2 = compute_content_hash({"b": 2, "a": 1})
        assert hash1 == hash2


class TestEvent:
    """Tests for Event entity."""

    def test_create_event(self):
        """Should create an event with required fields."""
        event = Event(
            event_type="UserRegistered",
            data={"user_id": "u123"}
        )
        assert event.event_type == "UserRegistered"
        assert event.data == {"user_id": "u123"}
        assert event.entity_type == EntityType.EVENT
        assert event.id is not None
        assert event.content_hash is not None

    def test_event_immutability_marker(self):
        """Events should be marked as immutable type."""
        event = Event(event_type="Test", data={})
        assert event.entity_type == EntityType.EVENT

    def test_event_with_source(self):
        """Should track event source."""
        event = Event(
            event_type="Test",
            data={},
            source_agent="test-agent",
            source_system="test-system"
        )
        assert event.source_agent == "test-agent"
        assert event.source_system == "test-system"

    def test_event_correlation(self):
        """Should support correlation and causation IDs."""
        event = Event(
            event_type="Test",
            data={},
            correlation_id="corr-123",
            causation_id="cause-456"
        )
        assert event.correlation_id == "corr-123"
        assert event.causation_id == "cause-456"

    def test_event_type_validation(self):
        """Should reject empty event type."""
        with pytest.raises(ValueError):
            Event(event_type="", data={})

        with pytest.raises(ValueError):
            Event(event_type="   ", data={})


class TestClaim:
    """Tests for Claim entity."""

    def test_create_claim(self):
        """Should create a claim with required fields."""
        claim = Claim(
            subject="user.u123.risk_score",
            value=0.5,
            source="risk-model"
        )
        assert claim.subject == "user.u123.risk_score"
        assert claim.value == 0.5
        assert claim.source == "risk-model"
        assert claim.entity_type == EntityType.CLAIM

    def test_claim_confidence(self):
        """Should validate confidence bounds."""
        # Valid confidence
        claim = Claim(subject="test", value=1, source="test", confidence=0.5)
        assert claim.confidence == 0.5

        # Out of bounds
        with pytest.raises(ValueError):
            Claim(subject="test", value=1, source="test", confidence=1.5)

        with pytest.raises(ValueError):
            Claim(subject="test", value=1, source="test", confidence=-0.1)

    def test_claim_validity(self):
        """Should check claim validity at timestamp."""
        claim = Claim(
            subject="test",
            value=1,
            source="test",
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 12, 31, tzinfo=timezone.utc)
        )

        # Valid in range
        assert claim.is_valid_at(datetime(2024, 6, 15, tzinfo=timezone.utc))

        # Before valid_from
        assert not claim.is_valid_at(datetime(2023, 12, 31, tzinfo=timezone.utc))

        # After valid_until
        assert not claim.is_valid_at(datetime(2025, 1, 1, tzinfo=timezone.utc))

    def test_claim_conflict_detection(self):
        """Should detect conflicting claims."""
        claim1 = Claim(subject="user.risk", value=0.5, source="model-a")
        claim2 = Claim(subject="user.risk", value=0.8, source="model-b")
        claim3 = Claim(subject="user.trust", value=0.5, source="model-a")

        assert claim1.conflicts_with(claim2)  # Same subject, different value
        assert not claim1.conflicts_with(claim3)  # Different subject

    def test_claim_derived_from(self):
        """Should track derivation sources."""
        claim = Claim(
            subject="test",
            value=1,
            source="test",
            derived_from=["entity-1", "entity-2"]
        )
        assert claim.derived_from == ["entity-1", "entity-2"]


class TestAction:
    """Tests for Action entity."""

    def test_create_action(self):
        """Should create an action with required fields."""
        action = Action(
            action_type="ApproveOrder",
            agent_id="approval-agent"
        )
        assert action.action_type == "ApproveOrder"
        assert action.agent_id == "approval-agent"
        assert action.entity_type == EntityType.ACTION
        assert action.action_status == ActionStatus.PENDING

    def test_action_lifecycle(self):
        """Should track action lifecycle."""
        action = Action(action_type="Test", agent_id="agent")

        assert action.action_status == ActionStatus.PENDING
        assert action.started_at is None

        action.start()
        assert action.action_status == ActionStatus.RUNNING
        assert action.started_at is not None

        action.complete({"result": "success"})
        assert action.action_status == ActionStatus.COMPLETED
        assert action.completed_at is not None
        assert action.outputs == {"result": "success"}

    def test_action_failure(self):
        """Should track action failure."""
        action = Action(action_type="Test", agent_id="agent")
        action.start()
        action.fail("Something went wrong")

        assert action.action_status == ActionStatus.FAILED
        assert action.error == "Something went wrong"

    def test_action_dependencies(self):
        """Should track dependencies."""
        action = Action(
            action_type="Test",
            agent_id="agent",
            depends_on=["entity-1", "entity-2"]
        )
        assert action.depends_on == ["entity-1", "entity-2"]

        action.add_dependency("entity-3")
        assert "entity-3" in action.depends_on

        # Should not duplicate
        action.add_dependency("entity-1")
        assert action.depends_on.count("entity-1") == 1

    def test_action_reasoning(self):
        """Should store reasoning for audit."""
        action = Action(
            action_type="Approve",
            agent_id="agent",
            reasoning="User meets all criteria",
            reasoning_tokens=150
        )
        assert action.reasoning == "User meets all criteria"
        assert action.reasoning_tokens == 150
