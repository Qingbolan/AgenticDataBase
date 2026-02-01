"""
Tests for AgenticDB Query Agents.

These tests demonstrate the causal reasoning and impact analysis
capabilities that make AgenticDB a "killer app" for agent debugging.

Key Features Demonstrated:
1. Natural language explanations of causal chains (why(x))
2. Impact analysis with severity assessment
3. Actionable recommendations
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from agenticdb.core.agents.query import (
    CausalReasoningAgent,
    ImpactAnalysisAgent,
    CausalExplanation,
    ImpactExplanation,
    KeyFactor,
    CriticalImpact,
    AffectedCount,
    Severity,
)
from agenticdb.core.agents.base import AgentContext


# =============================================================================
# Mock LLM Responses
# =============================================================================

MOCK_CAUSAL_RESPONSE = json.dumps({
    "summary": "User was approved because the risk score was below threshold.",
    "explanation": """
The approval decision followed a clear causal chain:

1. **User Registration**: The process started when user u123 registered with
   their email alice@example.com. This event triggered the risk assessment.

2. **Risk Assessment**: The risk model v2.1 analyzed the user's data and
   computed a risk score of 0.15, which is considered low risk (below the
   0.3 threshold). The model had 92% confidence in this assessment.

3. **Approval Decision**: Given the low risk score, the approval agent
   automatically approved the user for premium onboarding.
""",
    "key_factors": [
        {
            "factor": "Low risk score (0.15)",
            "role": "Primary decision factor - below 0.3 threshold"
        },
        {
            "factor": "Risk model confidence (92%)",
            "role": "High confidence supported automated decision"
        }
    ],
    "causal_depth": 3,
    "confidence": 0.92
})

MOCK_IMPACT_RESPONSE = json.dumps({
    "summary": "Changing the risk model would affect 5 pending approvals and 2 active policies.",
    "affected_count": {
        "events": 0,
        "claims": 5,
        "actions": 2
    },
    "critical_impacts": [
        {
            "entity_type": "claim",
            "entity_ref": "risk_score_claim_1",
            "impact": "Risk scores would need recalculation",
            "severity": "high"
        },
        {
            "entity_type": "action",
            "entity_ref": "pending_approval_1",
            "impact": "Approval decision may change",
            "severity": "high"
        }
    ],
    "cascade_effects": [
        "All downstream claims derived from risk scores would be invalidated",
        "Pending approval decisions would need review",
        "Notification workflows may trigger"
    ],
    "recommended_actions": [
        "Run risk model comparison on affected users before deployment",
        "Implement gradual rollout with monitoring",
        "Prepare rollback procedure"
    ],
    "confidence": 0.88
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
def sample_causal_chain():
    """Create a sample causal chain for testing."""
    return [
        {
            "id": "action_1",
            "type": "Action",
            "action_type": "ApproveUser",
            "agent_id": "approval-agent"
        },
        {
            "id": "claim_1",
            "type": "Claim",
            "subject": "user.u123.risk_score",
            "value": 0.15,
            "source": "risk_model_v2"
        },
        {
            "id": "event_1",
            "type": "Event",
            "event_type": "UserRegistered",
            "data": {"user_id": "u123"}
        }
    ]


# =============================================================================
# CausalReasoningAgent Tests - The "Why" Query
# =============================================================================

class TestCausalReasoningAgent:
    """
    Tests for causal reasoning - the key differentiator.

    Traditional databases: "What is the data?"
    AgenticDB: "Why did this happen?"
    """

    def test_generates_natural_language_explanation(
        self, mock_oracle, agent_context, sample_causal_chain
    ):
        """Should generate human-readable explanations."""
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()
        result = agent.run(
            agent_context,
            target_entity=sample_causal_chain[0],
            causal_chain=sample_causal_chain,
            edges=[]
        )

        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert "approved" in result.summary.lower()

    def test_identifies_key_factors(
        self, mock_oracle, agent_context, sample_causal_chain
    ):
        """Should identify the most important factors in the causal chain."""
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()
        result = agent.run(
            agent_context,
            target_entity=sample_causal_chain[0],
            causal_chain=sample_causal_chain,
            edges=[]
        )

        assert len(result.key_factors) >= 2
        assert any("risk" in f.factor.lower() for f in result.key_factors)

    def test_tracks_causal_depth(
        self, mock_oracle, agent_context, sample_causal_chain
    ):
        """Should report the depth of the causal chain."""
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()
        result = agent.run(
            agent_context,
            target_entity=sample_causal_chain[0],
            causal_chain=sample_causal_chain,
            edges=[]
        )

        assert result.causal_depth == 3

    def test_reports_confidence(
        self, mock_oracle, agent_context, sample_causal_chain
    ):
        """Should report confidence in the explanation."""
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()
        result = agent.run(
            agent_context,
            target_entity=sample_causal_chain[0],
            causal_chain=sample_causal_chain,
            edges=[]
        )

        assert 0 <= result.confidence <= 1.0

    def test_simple_explanation_mode(self, mock_oracle, agent_context):
        """Should support simplified text-based input."""
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()
        result = agent.explain_simple(
            agent_context,
            target_description="User was approved",
            causes=["Low risk score", "Complete profile"]
        )

        assert result.summary is not None
        assert result.explanation is not None


# =============================================================================
# ImpactAnalysisAgent Tests - The "What If" Query
# =============================================================================

class TestImpactAnalysisAgent:
    """
    Tests for impact analysis - enabling safe changes.

    Before changing anything in production:
    "What would break if I change this?"
    """

    def test_counts_affected_entities(self, mock_oracle, agent_context):
        """Should count affected entities by type."""
        mock_oracle.query.return_value = MOCK_IMPACT_RESPONSE

        agent = ImpactAnalysisAgent()
        result = agent.run(
            agent_context,
            source_entity={"type": "model", "name": "risk_model_v2"},
            affected_entities=[],
            edges=[]
        )

        assert result.affected_count.claims == 5
        assert result.affected_count.actions == 2
        assert result.affected_count.total == 7

    def test_identifies_critical_impacts(self, mock_oracle, agent_context):
        """Should identify high-severity impacts."""
        mock_oracle.query.return_value = MOCK_IMPACT_RESPONSE

        agent = ImpactAnalysisAgent()
        result = agent.run(
            agent_context,
            source_entity={},
            affected_entities=[],
            edges=[]
        )

        high_severity = [
            i for i in result.critical_impacts
            if i.severity == Severity.HIGH
        ]
        assert len(high_severity) >= 2

    def test_describes_cascade_effects(self, mock_oracle, agent_context):
        """Should describe cascading effects."""
        mock_oracle.query.return_value = MOCK_IMPACT_RESPONSE

        agent = ImpactAnalysisAgent()
        result = agent.run(
            agent_context,
            source_entity={},
            affected_entities=[],
            edges=[]
        )

        assert len(result.cascade_effects) > 0
        assert any("invalidate" in e.lower() for e in result.cascade_effects)

    def test_provides_recommendations(self, mock_oracle, agent_context):
        """Should provide actionable recommendations."""
        mock_oracle.query.return_value = MOCK_IMPACT_RESPONSE

        agent = ImpactAnalysisAgent()
        result = agent.run(
            agent_context,
            source_entity={},
            affected_entities=[],
            edges=[]
        )

        assert len(result.recommended_actions) > 0
        # Should include practical advice
        assert any(
            "rollout" in r.lower() or "rollback" in r.lower()
            for r in result.recommended_actions
        )

    def test_simple_analysis_mode(self, mock_oracle, agent_context):
        """Should support simplified text-based input."""
        mock_oracle.query.return_value = MOCK_IMPACT_RESPONSE

        agent = ImpactAnalysisAgent()
        result = agent.analyze_simple(
            agent_context,
            source_description="Risk model v2",
            affected_descriptions=[
                "User risk scores",
                "Approval decisions"
            ]
        )

        assert result.summary is not None
        assert result.affected_count.total > 0


# =============================================================================
# Integration Scenarios - Real-World Use Cases
# =============================================================================

class TestQueryAgentUseCases:
    """
    Real-world use cases demonstrating the value of query agents.
    """

    def test_debug_unexpected_approval(self, mock_oracle, agent_context):
        """
        Use Case: Debugging why a user was unexpectedly approved.

        Scenario: A user with suspicious behavior got approved.
        Question: "Why did this happen?"
        """
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()
        result = agent.explain_simple(
            agent_context,
            target_description="User u123 approved for premium account",
            causes=[
                "Risk score was 0.15 (low)",
                "Profile was complete",
                "No fraud flags"
            ]
        )

        # Should provide actionable insight
        assert result.summary is not None
        assert result.key_factors is not None

    def test_assess_model_update_risk(self, mock_oracle, agent_context):
        """
        Use Case: Assessing risk before deploying a model update.

        Scenario: Team wants to update the risk model.
        Question: "What would break if we deploy this?"
        """
        mock_oracle.query.return_value = MOCK_IMPACT_RESPONSE

        agent = ImpactAnalysisAgent()
        result = agent.analyze_simple(
            agent_context,
            source_description="Risk model v2 -> v3 update",
            affected_descriptions=[
                "All cached risk scores",
                "Pending approval decisions",
                "Active monitoring rules"
            ]
        )

        # Should quantify the blast radius
        assert result.affected_count.total > 0
        # Should provide mitigation steps
        assert len(result.recommended_actions) > 0

    def test_trace_data_lineage(self, mock_oracle, agent_context):
        """
        Use Case: Understanding data lineage for compliance.

        Scenario: Auditor asks "Where did this decision come from?"
        """
        mock_oracle.query.return_value = MOCK_CAUSAL_RESPONSE

        agent = CausalReasoningAgent()

        # Full chain from decision back to source
        chain = [
            {"type": "Action", "description": "Policy approved"},
            {"type": "Claim", "description": "Credit score = 750"},
            {"type": "Event", "description": "Credit check completed"},
            {"type": "Event", "description": "Application submitted"}
        ]

        result = agent.run(
            agent_context,
            target_entity=chain[0],
            causal_chain=chain,
            edges=[]
        )

        # Should provide audit-ready explanation
        assert result.explanation is not None
        assert result.causal_depth > 0


# =============================================================================
# Type Tests
# =============================================================================

class TestQueryTypes:
    """Tests for query result types."""

    def test_key_factor_structure(self):
        """KeyFactor should have factor and role."""
        kf = KeyFactor(factor="Risk score", role="Primary input")
        assert kf.factor == "Risk score"
        assert kf.role == "Primary input"

    def test_critical_impact_severity(self):
        """CriticalImpact should support severity levels."""
        impact = CriticalImpact(
            entity_type="claim",
            entity_ref="claim_1",
            impact="Would be invalidated",
            severity=Severity.HIGH
        )
        assert impact.severity == Severity.HIGH

    def test_affected_count_total(self):
        """AffectedCount should calculate total correctly."""
        count = AffectedCount(events=1, claims=2, actions=3)
        assert count.total == 6

    def test_causal_explanation_completeness(self):
        """CausalExplanation should have all required fields."""
        explanation = CausalExplanation(
            summary="Test summary",
            explanation="Full explanation",
            key_factors=[KeyFactor("f1", "r1")],
            causal_depth=2,
            confidence=0.9
        )
        assert explanation.summary == "Test summary"
        assert len(explanation.key_factors) == 1
