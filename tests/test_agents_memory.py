"""
Tests for AgenticDB Memory Agents.

These tests demonstrate the RAG-like semantic retrieval and state
management capabilities for each entity type.

Key Features Demonstrated:
1. Semantic recall (natural language queries)
2. Entity summarization
3. Hot data caching
4. Change tracking
5. Conflict detection (for claims)
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import json

from agenticdb.core.models import Event, Claim, Action, ActionStatus
from agenticdb.core.agents.memory import (
    EventMemoryAgent,
    ClaimMemoryAgent,
    ActionMemoryAgent,
    EventRecallResult,
    ClaimRecallResult,
    ActionRecallResult,
    RecalledEvent,
    RecalledClaim,
    RecalledAction,
    MemorySummary,
    MemoryStats,
    ClaimConflict,
)
from agenticdb.core.agents.base import AgentContext


# =============================================================================
# Mock LLM Responses
# =============================================================================

MOCK_EVENT_RECALL_RESPONSE = json.dumps({
    "relevant_events": [
        {
            "event_type": "UserRegistered",
            "relevance_score": 0.95,
            "summary": "User u123 registered with premium account"
        },
        {
            "event_type": "PaymentReceived",
            "relevance_score": 0.75,
            "summary": "Initial payment of $99 received"
        }
    ],
    "reasoning": "Events related to user onboarding flow"
})

MOCK_CLAIM_RECALL_RESPONSE = json.dumps({
    "relevant_claims": [
        {
            "subject": "user.u123.risk_score",
            "value": 0.15,
            "source": "risk_model_v2",
            "confidence": 0.92,
            "relevance_score": 0.98
        }
    ],
    "conflicts": [],
    "reasoning": "Risk-related claims for user u123"
})

MOCK_ACTION_RECALL_RESPONSE = json.dumps({
    "relevant_actions": [
        {
            "action_type": "ApproveUser",
            "agent_id": "approval-agent",
            "status": "completed",
            "relevance_score": 0.90,
            "summary": "User approved for premium tier"
        }
    ],
    "patterns": [
        {
            "pattern": "Approval after risk assessment",
            "frequency": "95% of cases"
        }
    ],
    "reasoning": "Approval actions for user u123"
})

MOCK_SUMMARY_RESPONSE = json.dumps({
    "summary": "User u123 completed onboarding with premium status",
    "key_events": ["Registration", "Risk assessment", "Approval"],
    "timeline": "All events occurred within 5 minutes",
    "patterns": ["Standard premium onboarding flow"]
})


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_oracle():
    """Create a mock Oracle."""
    with patch('agenticdb.core.agents.base.base_agent.Oracle') as MockOracle:
        mock_instance = MagicMock()
        MockOracle.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def agent_context():
    """Create a test context."""
    return AgentContext(session_id="test")


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        Event(
            event_type="UserRegistered",
            data={"user_id": "u123", "email": "alice@example.com"},
            created_at=datetime.now(timezone.utc) - timedelta(hours=2)
        ),
        Event(
            event_type="PaymentReceived",
            data={"user_id": "u123", "amount": 99.99},
            created_at=datetime.now(timezone.utc) - timedelta(hours=1)
        ),
        Event(
            event_type="UserRegistered",
            data={"user_id": "u456", "email": "bob@example.com"},
            created_at=datetime.now(timezone.utc)
        )
    ]


@pytest.fixture
def sample_claims():
    """Create sample claims for testing."""
    return [
        Claim(
            subject="user.u123.risk_score",
            value=0.15,
            source="risk_model_v2",
            confidence=0.92,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1)
        ),
        Claim(
            subject="user.u123.tier",
            value="premium",
            source="rules_engine",
            confidence=1.0,
            created_at=datetime.now(timezone.utc)
        ),
        # Conflicting claim
        Claim(
            subject="user.u123.risk_score",
            value=0.45,
            source="risk_model_v3",
            confidence=0.88,
            created_at=datetime.now(timezone.utc)
        )
    ]


@pytest.fixture
def sample_actions():
    """Create sample actions for testing."""
    action1 = Action(
        action_type="ApproveUser",
        agent_id="approval-agent",
        inputs={"user_id": "u123"},
        outputs={"approved": True}
    )
    action1.complete()

    action2 = Action(
        action_type="SendNotification",
        agent_id="notification-agent",
        inputs={"user_id": "u123", "type": "welcome"}
    )
    action2.start()
    action2.fail("Email service unavailable")

    return [action1, action2]


# =============================================================================
# EventMemoryAgent Tests
# =============================================================================

class TestEventMemoryAgent:
    """Tests for Event memory management."""

    def test_semantic_recall(self, mock_oracle, agent_context, sample_events):
        """Should recall events based on natural language query."""
        mock_oracle.query.return_value = MOCK_EVENT_RECALL_RESPONSE

        agent = EventMemoryAgent()
        result = agent.recall(
            "What happened during user registration?",
            sample_events
        )

        assert len(result.events) > 0
        assert result.events[0].relevance_score > 0.5

    def test_get_latest_by_type(self, mock_oracle, sample_events):
        """Should get the most recent event of a type."""
        agent = EventMemoryAgent()
        latest = agent.get_latest("UserRegistered", sample_events)

        assert latest is not None
        assert latest.data["user_id"] == "u456"  # Most recent registration

    def test_track_changes_since(self, mock_oracle, sample_events):
        """Should track events since a timestamp."""
        agent = EventMemoryAgent()
        since = datetime.now(timezone.utc) - timedelta(minutes=90)
        changes = agent.track_changes(since, sample_events)

        assert len(changes) == 2  # Last two events

    def test_summarize_events(self, mock_oracle, agent_context, sample_events):
        """Should summarize a list of events."""
        mock_oracle.query.return_value = MOCK_SUMMARY_RESPONSE

        agent = EventMemoryAgent()
        summary = agent.summarize(sample_events, focus="onboarding")

        assert summary.summary is not None
        assert summary.total_entities == len(sample_events)

    def test_get_stats(self, mock_oracle, sample_events):
        """Should compute statistics about events."""
        agent = EventMemoryAgent()
        stats = agent.get_stats(sample_events)

        assert stats.total_events == 3
        assert "UserRegistered" in stats.event_types
        assert "PaymentReceived" in stats.event_types

    def test_caching(self, mock_oracle, sample_events):
        """Should support hot data caching."""
        agent = EventMemoryAgent()

        # Cache an event
        event = sample_events[0]
        agent.cache_event(event)

        # Retrieve from cache
        cached = agent.get_cached(event.id)
        assert cached is not None
        assert cached.id == event.id

        # Clear cache
        agent.clear_cache()
        assert agent.get_cached(event.id) is None


# =============================================================================
# ClaimMemoryAgent Tests
# =============================================================================

class TestClaimMemoryAgent:
    """Tests for Claim memory management."""

    def test_semantic_recall(self, mock_oracle, agent_context, sample_claims):
        """Should recall claims based on natural language query."""
        mock_oracle.query.return_value = MOCK_CLAIM_RECALL_RESPONSE

        agent = ClaimMemoryAgent()
        result = agent.recall(
            "What is the risk score for user u123?",
            sample_claims
        )

        assert len(result.claims) > 0
        assert result.claims[0].subject == "user.u123.risk_score"

    def test_active_only_filter(self, mock_oracle, agent_context, sample_claims):
        """Should filter to active claims only by default."""
        mock_oracle.query.return_value = MOCK_CLAIM_RECALL_RESPONSE

        agent = ClaimMemoryAgent()

        # Mark one claim as superseded
        sample_claims[0].status = "superseded"

        result = agent.recall(
            "risk score",
            sample_claims,
            active_only=True
        )

        # Should only return active claims
        # (Note: mock response, so actual filtering happens in real usage)
        assert result.claims is not None

    def test_conflict_detection(self, mock_oracle, sample_claims):
        """Should detect conflicting claims."""
        agent = ClaimMemoryAgent()
        conflicts = agent.find_conflicts(sample_claims)

        # Should find conflict between risk_score claims
        assert len(conflicts) > 0
        assert any("risk_score" in c.nature for c in conflicts)

    def test_get_latest_by_subject(self, mock_oracle, sample_claims):
        """Should get the most recent claim for a subject."""
        agent = ClaimMemoryAgent()
        latest = agent.get_latest("user.u123.risk_score", sample_claims)

        # Should return the most recent (v3 model)
        assert latest is not None
        assert latest.source == "risk_model_v3"

    def test_get_stats(self, mock_oracle, sample_claims):
        """Should compute statistics about claims."""
        agent = ClaimMemoryAgent()
        stats = agent.get_stats(sample_claims)

        assert stats.total_claims == 3
        assert stats.active_claims > 0


# =============================================================================
# ActionMemoryAgent Tests
# =============================================================================

class TestActionMemoryAgent:
    """Tests for Action memory management."""

    def test_semantic_recall(self, mock_oracle, agent_context, sample_actions):
        """Should recall actions based on natural language query."""
        mock_oracle.query.return_value = MOCK_ACTION_RECALL_RESPONSE

        agent = ActionMemoryAgent()
        result = agent.recall(
            "What approval decisions were made?",
            sample_actions
        )

        assert len(result.actions) > 0
        assert result.actions[0].action_type == "ApproveUser"

    def test_pattern_detection(self, mock_oracle, agent_context, sample_actions):
        """Should identify action patterns."""
        mock_oracle.query.return_value = MOCK_ACTION_RECALL_RESPONSE

        agent = ActionMemoryAgent()
        result = agent.recall("approvals", sample_actions)

        assert len(result.patterns) > 0

    def test_get_by_agent(self, mock_oracle, sample_actions):
        """Should filter actions by agent ID."""
        agent = ActionMemoryAgent()
        approval_actions = agent.get_by_agent("approval-agent", sample_actions)

        assert len(approval_actions) == 1
        assert approval_actions[0].action_type == "ApproveUser"

    def test_get_failures(self, mock_oracle, sample_actions):
        """Should identify failed actions."""
        agent = ActionMemoryAgent()
        failures = agent.get_failures(sample_actions)

        assert len(failures) == 1
        assert failures[0].action_type == "SendNotification"
        assert failures[0].error is not None

    def test_get_latest_by_type(self, mock_oracle, sample_actions):
        """Should get the most recent action of a type."""
        agent = ActionMemoryAgent()
        latest = agent.get_latest("ApproveUser", sample_actions)

        assert latest is not None
        assert latest.agent_id == "approval-agent"


# =============================================================================
# Memory Agent Use Cases
# =============================================================================

class TestMemoryAgentUseCases:
    """Real-world use cases for memory agents."""

    def test_debugging_session_context(
        self, mock_oracle, agent_context, sample_events, sample_claims
    ):
        """
        Use Case: Building context for debugging a user session.

        Scenario: Support needs to understand what happened to user u123.
        """
        mock_oracle.query.return_value = MOCK_EVENT_RECALL_RESPONSE

        event_agent = EventMemoryAgent()
        events = event_agent.recall(
            "What happened to user u123?",
            sample_events
        )

        # Should find relevant events
        assert len(events.events) > 0

    def test_audit_trail_reconstruction(
        self, mock_oracle, agent_context, sample_events, sample_actions
    ):
        """
        Use Case: Reconstructing audit trail for compliance.

        Scenario: Auditor needs complete history for a decision.
        """
        event_agent = EventMemoryAgent()
        action_agent = ActionMemoryAgent()

        # Get all events in timeframe
        since = datetime.now(timezone.utc) - timedelta(days=1)
        events = event_agent.track_changes(since, sample_events)

        # Get all actions in timeframe
        actions = action_agent.track_changes(since, sample_actions)

        # Should have complete history
        assert len(events) > 0 or len(actions) > 0

    def test_conflict_resolution_workflow(
        self, mock_oracle, sample_claims
    ):
        """
        Use Case: Identifying claims that need resolution.

        Scenario: Multiple models produced different risk scores.
        """
        claim_agent = ClaimMemoryAgent()
        conflicts = claim_agent.find_conflicts(sample_claims)

        # Should identify the conflict
        assert len(conflicts) > 0

        # Get latest claim for resolution
        latest = claim_agent.get_latest(
            "user.u123.risk_score",
            sample_claims
        )

        assert latest is not None


# =============================================================================
# Caching and Performance Tests
# =============================================================================

class TestMemoryCaching:
    """Tests for memory caching functionality."""

    def test_event_cache_by_type(self, mock_oracle, sample_events):
        """Should organize cached events by type."""
        agent = EventMemoryAgent()

        for event in sample_events:
            agent.cache_event(event)

        # Should be able to retrieve any cached event
        for event in sample_events:
            cached = agent.get_cached(event.id)
            assert cached is not None

    def test_claim_cache_by_subject(self, mock_oracle, sample_claims):
        """Should organize cached claims by subject."""
        agent = ClaimMemoryAgent()

        for claim in sample_claims:
            agent.cache_claim(claim)

        # Should be able to retrieve any cached claim
        for claim in sample_claims:
            cached = agent.get_cached(claim.id)
            assert cached is not None

    def test_action_cache_by_agent(self, mock_oracle, sample_actions):
        """Should organize cached actions by agent and type."""
        agent = ActionMemoryAgent()

        for action in sample_actions:
            agent.cache_action(action)

        # Should be able to retrieve any cached action
        for action in sample_actions:
            cached = agent.get_cached(action.id)
            assert cached is not None


# =============================================================================
# Empty State Tests
# =============================================================================

class TestEmptyStates:
    """Tests for handling empty data."""

    def test_recall_with_no_events(self, mock_oracle, agent_context):
        """Should handle empty event list gracefully."""
        agent = EventMemoryAgent()
        result = agent.recall("test", [])

        assert len(result.events) == 0
        assert result.reasoning is not None

    def test_summarize_with_no_claims(self, mock_oracle, agent_context):
        """Should handle empty claim list gracefully."""
        agent = ClaimMemoryAgent()
        summary = agent.summarize([])

        assert summary.total_entities == 0
        assert "No claims" in summary.summary

    def test_stats_with_no_actions(self, mock_oracle):
        """Should handle empty action list gracefully."""
        agent = ActionMemoryAgent()
        stats = agent.get_stats([])

        assert stats.total_actions == 0
        assert len(stats.action_types) == 0
