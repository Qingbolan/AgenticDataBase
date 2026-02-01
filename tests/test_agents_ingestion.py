"""
Tests for AgenticDB Ingestion Agents.

These tests demonstrate the advanced capabilities of the agent-driven
architecture for semantic text processing.

Key Features Demonstrated:
1. LLM-powered entity extraction (Events, Claims, Actions)
2. Automatic dependency inference
3. Confidence scoring
4. Structured output parsing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from agenticdb.core.agents.ingestion import (
    IngestionCoordinator,
    EventExtractorAgent,
    ClaimExtractorAgent,
    ActionExtractorAgent,
    DependencyInferenceAgent,
    ExtractedEvent,
    ExtractedClaim,
    ExtractedAction,
    InferredEdge,
    EdgeType,
    IngestionResult,
)
from agenticdb.core.agents.base import AgentContext


# =============================================================================
# Mock LLM Responses for Testing
# =============================================================================

MOCK_EVENT_RESPONSE = json.dumps({
    "events": [
        {
            "event_type": "UserRegistered",
            "data": {"user_id": "u123", "email": "alice@example.com"},
            "source_system": "auth-service"
        },
        {
            "event_type": "PaymentReceived",
            "data": {"amount": 99.99, "currency": "USD"},
            "source_system": "payment-service"
        }
    ],
    "confidence": 0.95,
    "reasoning": "Clear registration and payment events identified"
})

MOCK_CLAIM_RESPONSE = json.dumps({
    "claims": [
        {
            "subject": "user.u123.risk_score",
            "value": 0.15,
            "source": "risk_model_v2",
            "confidence": 0.92
        },
        {
            "subject": "user.u123.tier",
            "value": "premium",
            "source": "rules_engine",
            "confidence": 1.0
        }
    ],
    "confidence": 0.90,
    "reasoning": "Risk score and tier assignment extracted"
})

MOCK_ACTION_RESPONSE = json.dumps({
    "actions": [
        {
            "action_type": "ApproveUser",
            "agent_id": "approval-agent",
            "agent_type": "rule-based",
            "inputs": {"user_id": "u123", "risk_score": 0.15},
            "outputs": {"approved": True},
            "depends_on_refs": ["risk_score_claim"],
            "reasoning": "Risk below threshold"
        }
    ],
    "confidence": 0.88,
    "reasoning": "Approval action identified with dependency"
})

MOCK_DEPENDENCY_RESPONSE = json.dumps({
    "edges": [
        {
            "from_ref": "action_0",
            "to_ref": "claim_0",
            "edge_type": "DEPENDS_ON",
            "reasoning": "Approval depends on risk score"
        },
        {
            "from_ref": "claim_0",
            "to_ref": "event_0",
            "edge_type": "DERIVED_FROM",
            "reasoning": "Risk computed from user registration"
        }
    ],
    "confidence": 0.85,
    "reasoning": "Causal chain identified"
})


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_oracle():
    """Create a mock Oracle for testing without real LLM calls."""
    with patch('agenticdb.core.agents.base.base_agent.Oracle') as MockOracle:
        mock_instance = MagicMock()
        MockOracle.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def agent_context():
    """Create a test agent context."""
    return AgentContext(
        session_id="test-session",
        branch_id="test-branch",
        language="en"
    )


# =============================================================================
# EventExtractorAgent Tests
# =============================================================================

class TestEventExtractorAgent:
    """Tests for Event extraction capabilities."""

    def test_extracts_events_from_text(self, mock_oracle, agent_context):
        """Should extract events with proper structure."""
        mock_oracle.query.return_value = MOCK_EVENT_RESPONSE

        agent = EventExtractorAgent()
        result = agent.run(agent_context, "User registered and made payment")

        assert len(result.events) == 2
        assert result.events[0].event_type == "UserRegistered"
        assert result.events[0].data["user_id"] == "u123"
        assert result.events[1].event_type == "PaymentReceived"
        assert result.confidence == 0.95

    def test_assigns_reference_ids(self, mock_oracle, agent_context):
        """Should assign unique ref_ids for dependency linking."""
        mock_oracle.query.return_value = MOCK_EVENT_RESPONSE

        agent = EventExtractorAgent()
        result = agent.run(agent_context, "test text")

        assert result.events[0].ref_id == "event_0"
        assert result.events[1].ref_id == "event_1"

    def test_handles_empty_response(self, mock_oracle, agent_context):
        """Should handle cases with no events found."""
        mock_oracle.query.return_value = json.dumps({
            "events": [],
            "confidence": 0.5,
            "reasoning": "No clear events identified"
        })

        agent = EventExtractorAgent()
        result = agent.run(agent_context, "random text")

        assert len(result.events) == 0
        assert result.confidence == 0.5

    def test_extracts_source_information(self, mock_oracle, agent_context):
        """Should preserve source agent and system info."""
        mock_oracle.query.return_value = MOCK_EVENT_RESPONSE

        agent = EventExtractorAgent()
        result = agent.run(agent_context, "test")

        assert result.events[0].source_system == "auth-service"


# =============================================================================
# ClaimExtractorAgent Tests
# =============================================================================

class TestClaimExtractorAgent:
    """Tests for Claim extraction capabilities."""

    def test_extracts_claims_with_provenance(self, mock_oracle, agent_context):
        """Should extract claims with source attribution."""
        mock_oracle.query.return_value = MOCK_CLAIM_RESPONSE

        agent = ClaimExtractorAgent()
        result = agent.run(agent_context, "Risk score computed")

        assert len(result.claims) == 2
        assert result.claims[0].subject == "user.u123.risk_score"
        assert result.claims[0].value == 0.15
        assert result.claims[0].source == "risk_model_v2"
        assert result.claims[0].confidence == 0.92

    def test_extracts_different_value_types(self, mock_oracle, agent_context):
        """Should handle numeric, string, and boolean values."""
        mock_oracle.query.return_value = MOCK_CLAIM_RESPONSE

        agent = ClaimExtractorAgent()
        result = agent.run(agent_context, "test")

        # Numeric value
        assert isinstance(result.claims[0].value, float)
        # String value
        assert isinstance(result.claims[1].value, str)

    def test_preserves_confidence_scores(self, mock_oracle, agent_context):
        """Should preserve per-claim confidence scores."""
        mock_oracle.query.return_value = MOCK_CLAIM_RESPONSE

        agent = ClaimExtractorAgent()
        result = agent.run(agent_context, "test")

        assert result.claims[0].confidence == 0.92
        assert result.claims[1].confidence == 1.0


# =============================================================================
# ActionExtractorAgent Tests
# =============================================================================

class TestActionExtractorAgent:
    """Tests for Action extraction capabilities."""

    def test_extracts_actions_with_dependencies(self, mock_oracle, agent_context):
        """Should extract actions with dependency references."""
        mock_oracle.query.return_value = MOCK_ACTION_RESPONSE

        agent = ActionExtractorAgent()
        result = agent.run(agent_context, "User approved")

        assert len(result.actions) == 1
        assert result.actions[0].action_type == "ApproveUser"
        assert result.actions[0].agent_id == "approval-agent"
        assert "risk_score_claim" in result.actions[0].depends_on_refs

    def test_extracts_reasoning(self, mock_oracle, agent_context):
        """Should capture agent reasoning for audit trail."""
        mock_oracle.query.return_value = MOCK_ACTION_RESPONSE

        agent = ActionExtractorAgent()
        result = agent.run(agent_context, "test")

        assert result.actions[0].reasoning == "Risk below threshold"

    def test_extracts_inputs_outputs(self, mock_oracle, agent_context):
        """Should capture action inputs and outputs."""
        mock_oracle.query.return_value = MOCK_ACTION_RESPONSE

        agent = ActionExtractorAgent()
        result = agent.run(agent_context, "test")

        assert result.actions[0].inputs["user_id"] == "u123"
        assert result.actions[0].outputs["approved"] is True


# =============================================================================
# DependencyInferenceAgent Tests
# =============================================================================

class TestDependencyInferenceAgent:
    """Tests for dependency inference capabilities."""

    def test_infers_depends_on_edges(self, mock_oracle, agent_context):
        """Should infer DEPENDS_ON relationships."""
        mock_oracle.query.return_value = MOCK_DEPENDENCY_RESPONSE

        agent = DependencyInferenceAgent()
        events = [ExtractedEvent("UserRegistered", {}, ref_id="event_0")]
        claims = [ExtractedClaim("risk", 0.15, "model", ref_id="claim_0")]
        actions = [ExtractedAction("Approve", "agent", ref_id="action_0")]

        result = agent.run(agent_context, events, claims, actions)

        depends_on = [e for e in result.edges if e.edge_type == EdgeType.DEPENDS_ON]
        assert len(depends_on) == 1
        assert depends_on[0].from_ref == "action_0"
        assert depends_on[0].to_ref == "claim_0"

    def test_infers_derived_from_edges(self, mock_oracle, agent_context):
        """Should infer DERIVED_FROM relationships."""
        mock_oracle.query.return_value = MOCK_DEPENDENCY_RESPONSE

        agent = DependencyInferenceAgent()
        result = agent.run(agent_context, [], [], [])

        derived = [e for e in result.edges if e.edge_type == EdgeType.DERIVED_FROM]
        assert len(derived) == 1

    def test_includes_reasoning(self, mock_oracle, agent_context):
        """Should include reasoning for each edge."""
        mock_oracle.query.return_value = MOCK_DEPENDENCY_RESPONSE

        agent = DependencyInferenceAgent()
        result = agent.run(agent_context, [], [], [])

        for edge in result.edges:
            assert edge.reasoning is not None


# =============================================================================
# IngestionCoordinator Tests
# =============================================================================

class TestIngestionCoordinator:
    """Tests for the full ingestion pipeline."""

    def test_full_pipeline_integration(self, mock_oracle, agent_context):
        """Should coordinate all agents in the correct order."""
        # Configure mock to return different responses based on call order
        mock_oracle.query.side_effect = [
            MOCK_EVENT_RESPONSE,
            MOCK_CLAIM_RESPONSE,
            MOCK_ACTION_RESPONSE,
            MOCK_DEPENDENCY_RESPONSE,
        ]

        coordinator = IngestionCoordinator()
        result = coordinator.ingest(
            "User registered, risk computed, approved",
            ctx=agent_context
        )

        assert len(result.events) == 2
        assert len(result.claims) == 2
        assert len(result.actions) == 1
        assert len(result.edges) == 2

    def test_calculates_metrics(self, mock_oracle, agent_context):
        """Should calculate overall metrics."""
        mock_oracle.query.side_effect = [
            MOCK_EVENT_RESPONSE,
            MOCK_CLAIM_RESPONSE,
            MOCK_ACTION_RESPONSE,
            MOCK_DEPENDENCY_RESPONSE,
        ]

        coordinator = IngestionCoordinator()
        result = coordinator.ingest("test", ctx=agent_context)

        assert result.total_entities == 5  # 2 events + 2 claims + 1 action
        assert result.average_confidence > 0

    def test_selective_extraction(self, mock_oracle, agent_context):
        """Should support extracting only specific entity types."""
        mock_oracle.query.return_value = MOCK_EVENT_RESPONSE

        coordinator = IngestionCoordinator()
        result = coordinator.extract_events("test", ctx=agent_context)

        assert len(result.events) == 2
        assert len(result.claims) == 0
        assert len(result.actions) == 0

    def test_skip_dependency_inference(self, mock_oracle, agent_context):
        """Should allow skipping dependency inference."""
        mock_oracle.query.side_effect = [
            MOCK_EVENT_RESPONSE,
            MOCK_CLAIM_RESPONSE,
            MOCK_ACTION_RESPONSE,
        ]

        coordinator = IngestionCoordinator()
        result = coordinator.ingest(
            "test",
            ctx=agent_context,
            infer_dependencies=False
        )

        assert len(result.edges) == 0


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_malformed_json(self, mock_oracle, agent_context):
        """Should handle non-JSON responses gracefully."""
        mock_oracle.query.return_value = "Not valid JSON"

        agent = EventExtractorAgent()
        with pytest.raises(ValueError):
            agent.run(agent_context, "test")

    def test_handles_missing_fields(self, mock_oracle, agent_context):
        """Should handle responses with missing optional fields."""
        mock_oracle.query.return_value = json.dumps({
            "events": [{"event_type": "Test", "data": {}}],
            "confidence": 0.8
            # Missing "reasoning" field
        })

        agent = EventExtractorAgent()
        result = agent.run(agent_context, "test")

        assert len(result.events) == 1
        assert result.reasoning is None or result.reasoning == ""

    def test_handles_empty_input(self, mock_oracle, agent_context):
        """Should handle empty input text."""
        mock_oracle.query.return_value = json.dumps({
            "events": [],
            "confidence": 0.0,
            "reasoning": "Empty input"
        })

        agent = EventExtractorAgent()
        result = agent.run(agent_context, "")

        assert len(result.events) == 0


# =============================================================================
# IngestionResult Tests
# =============================================================================

class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_total_entities_calculation(self):
        """Should correctly calculate total entities."""
        result = IngestionResult(
            events=[ExtractedEvent("E1", {}), ExtractedEvent("E2", {})],
            claims=[ExtractedClaim("s", 1, "src")],
            actions=[ExtractedAction("A1", "agent")]
        )

        assert result.total_entities == 4

    def test_average_confidence_calculation(self):
        """Should correctly calculate average confidence."""
        result = IngestionResult(
            event_confidence=0.8,
            claim_confidence=0.9,
            action_confidence=0.7,
            edge_confidence=0.6
        )

        expected = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert abs(result.average_confidence - expected) < 0.001

    def test_average_confidence_with_zeros(self):
        """Should exclude zero confidences from average."""
        result = IngestionResult(
            event_confidence=0.8,
            claim_confidence=0.0,  # Should be excluded
            action_confidence=0.6,
            edge_confidence=0.0   # Should be excluded
        )

        expected = (0.8 + 0.6) / 2
        assert abs(result.average_confidence - expected) < 0.001
