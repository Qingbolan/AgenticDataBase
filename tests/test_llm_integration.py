"""
Real LLM integration tests for AgenticDB agents.

These tests actually call the OpenAI API and require:
1. OPENAI_API_KEY environment variable set
2. Run with: uv run pytest tests/test_llm_integration.py -v -m llm

Skip these tests in CI with: pytest -m "not llm"
"""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agenticdb.core.agents.base import AgentContext
from agenticdb.core.agents.ingestion import (
    EventExtractorAgent,
    ClaimExtractorAgent,
    ActionExtractorAgent,
    DependencyInferenceAgent,
    IngestionCoordinator,
)
from agenticdb.core.agents.query import (
    CausalReasoningAgent,
    ImpactAnalysisAgent,
)
from agenticdb.core.agents.memory import (
    EventMemoryAgent,
    ClaimMemoryAgent,
    ActionMemoryAgent,
)
from agenticdb.core.models import Event, Claim, Action
from agenticdb.core.dependency import EdgeType

# Skip all tests in this module if no API key
pytestmark = pytest.mark.llm


def has_api_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def agent_context():
    """Create a test agent context."""
    return AgentContext(
        session_id="llm-test-session",
        branch_id="llm-test-branch",
        language="en",
    )


@pytest.fixture
def sample_text():
    """Sample text for extraction tests."""
    return """
    User u123 registered on the platform with email alice@example.com.
    The risk assessment model v2.1 computed a risk score of 0.15 with 92% confidence.
    Based on the low risk score (below 0.3 threshold), the approval agent
    automatically approved the user for premium tier access.
    """


class TestRealEventExtraction:
    """Tests that actually call the LLM for event extraction."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_extracts_real_events_from_text(self, agent_context, sample_text):
        """Should extract events using real LLM."""
        agent = EventExtractorAgent()
        result = agent.run(agent_context, sample_text)

        # Should extract at least the registration event
        assert len(result.events) >= 1
        assert result.confidence > 0.5

        # Check event structure
        event_types = [e.event_type for e in result.events]
        # Should identify some form of registration
        assert any("register" in et.lower() or "user" in et.lower() for et in event_types)

        print(f"\n✅ Extracted {len(result.events)} events:")
        for e in result.events:
            print(f"   - {e.event_type}: {e.data}")


class TestRealClaimExtraction:
    """Tests that actually call the LLM for claim extraction."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_extracts_real_claims_from_text(self, agent_context, sample_text):
        """Should extract claims using real LLM."""
        agent = ClaimExtractorAgent()
        result = agent.run(agent_context, sample_text)

        # Should extract at least the risk score claim
        assert len(result.claims) >= 1
        assert result.confidence > 0.5

        print(f"\n✅ Extracted {len(result.claims)} claims:")
        for c in result.claims:
            print(f"   - {c.subject}: {c.value} (source: {c.source}, conf: {c.confidence})")


class TestRealActionExtraction:
    """Tests that actually call the LLM for action extraction."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_extracts_real_actions_from_text(self, agent_context, sample_text):
        """Should extract actions using real LLM."""
        agent = ActionExtractorAgent()
        result = agent.run(agent_context, sample_text)

        # Should extract the approval action
        assert len(result.actions) >= 1
        assert result.confidence > 0.5

        print(f"\n✅ Extracted {len(result.actions)} actions:")
        for a in result.actions:
            print(f"   - {a.action_type} by {a.agent_id}")
            print(f"     Reasoning: {a.reasoning}")
            print(f"     Depends on: {a.depends_on_refs}")


class TestRealIngestionPipeline:
    """Tests the full ingestion pipeline with real LLM."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_full_pipeline_with_real_llm(self, agent_context, sample_text):
        """Should run full ingestion pipeline with real LLM."""
        coordinator = IngestionCoordinator()
        result = coordinator.ingest(
            text=sample_text,
            ctx=agent_context,
            infer_dependencies=True,
        )

        # Should extract entities
        assert result.total_entities > 0
        print(f"\n✅ Full pipeline results:")
        print(f"   Events: {len(result.events)}")
        print(f"   Claims: {len(result.claims)}")
        print(f"   Actions: {len(result.actions)}")
        print(f"   Edges: {len(result.edges)}")
        print(f"   Average confidence: {result.average_confidence:.2f}")

        # Print extracted entities
        print("\n   Events:")
        for e in result.events:
            print(f"     [{e.ref_id}] {e.event_type}")

        print("\n   Claims:")
        for c in result.claims:
            print(f"     [{c.ref_id}] {c.subject} = {c.value}")

        print("\n   Actions:")
        for a in result.actions:
            print(f"     [{a.ref_id}] {a.action_type}")

        print("\n   Dependencies:")
        for edge in result.edges:
            print(f"     {edge.from_ref} --[{edge.edge_type}]--> {edge.to_ref}")


class TestRealCausalReasoning:
    """Tests causal reasoning with real LLM."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_explains_causal_chain(self, agent_context):
        """Should generate natural language causal explanation."""
        # Create sample causal chain
        target = Action(
            action_type="ApproveUser",
            agent_id="approval-agent",
            inputs={"user_id": "u123"},
        )
        target.complete({"approved": True, "tier": "premium"})

        causal_chain = [
            Event(
                event_type="UserRegistered",
                data={"user_id": "u123", "email": "alice@example.com"},
            ),
            Claim(
                subject="user.u123.risk_score",
                value=0.15,
                source="risk_model_v2",
                confidence=0.92,
            ),
        ]

        edges = [
            {"source": "claim_risk", "target": "event_register", "type": EdgeType.DERIVED_FROM},
            {"source": "action_approve", "target": "claim_risk", "type": EdgeType.DEPENDS_ON},
        ]

        agent = CausalReasoningAgent()
        result = agent.run(agent_context, target, causal_chain, edges)

        assert result.summary
        assert len(result.key_factors) > 0
        assert result.confidence > 0

        print(f"\n✅ Causal Reasoning Result:")
        print(f"   Summary: {result.summary}")
        print(f"   Explanation: {result.explanation[:200]}...")
        print(f"   Key factors: {[f.factor for f in result.key_factors]}")
        print(f"   Confidence: {result.confidence:.2f}")


class TestRealImpactAnalysis:
    """Tests impact analysis with real LLM."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_analyzes_downstream_impact(self, agent_context):
        """Should analyze impact of changing an entity."""
        # Source entity being changed
        source = Claim(
            subject="user.u123.risk_score",
            value=0.15,
            source="risk_model_v2",
        )

        # Entities that depend on it
        affected = [
            Action(
                action_type="ApproveUser",
                agent_id="approval-agent",
                inputs={"user_id": "u123", "risk_score": 0.15},
            ),
            Claim(
                subject="user.u123.tier",
                value="premium",
                source="rules_engine",
            ),
        ]

        edges = [
            {"source": "action_approve", "target": "claim_risk", "type": EdgeType.DEPENDS_ON},
            {"source": "claim_tier", "target": "claim_risk", "type": EdgeType.DERIVED_FROM},
        ]

        agent = ImpactAnalysisAgent()
        result = agent.run(agent_context, source, affected, edges)

        assert result.summary
        assert result.affected_count.total > 0

        print(f"\n✅ Impact Analysis Result:")
        print(f"   Summary: {result.summary}")
        print(f"   Affected: {result.affected_count.total} entities")
        print(f"   Critical impacts: {len(result.critical_impacts)}")
        print(f"   Recommendations:")
        for rec in result.recommended_actions:
            print(f"     - {rec}")


class TestRealMemoryAgents:
    """Tests memory agents with real LLM."""

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_event_semantic_recall(self, agent_context):
        """Should recall events semantically."""
        events = [
            Event(event_type="UserRegistered", data={"user_id": "u123"}),
            Event(event_type="PaymentReceived", data={"amount": 99.99}),
            Event(event_type="OrderPlaced", data={"product": "premium"}),
        ]

        agent = EventMemoryAgent()
        result = agent.recall(
            query="What happened during user onboarding?",
            events=events,
        )

        assert len(result.events) > 0
        print(f"\n✅ Event Memory Recall:")
        print(f"   Query: 'What happened during user onboarding?'")
        print(f"   Found: {len(result.events)} relevant events")
        print(f"   Reasoning: {result.reasoning}")

    @pytest.mark.skipif(not has_api_key(), reason="No OPENAI_API_KEY set")
    def test_claim_summarization(self, agent_context):
        """Should summarize claims about an entity."""
        claims = [
            Claim(subject="user.u123.risk_score", value=0.15, source="model_v2"),
            Claim(subject="user.u123.tier", value="premium", source="rules"),
            Claim(subject="user.u123.ltv", value=2500, source="analytics"),
        ]

        agent = ClaimMemoryAgent()
        result = agent.summarize(
            claims=claims,
            focus="user profile",
        )

        assert result.summary
        print(f"\n✅ Claim Memory Summary:")
        print(f"   Focus: 'user profile'")
        print(f"   Summary: {result.summary}")
        print(f"   Key points: {result.key_points}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_llm_integration.py -v -s
    pytest.main([__file__, "-v", "-s", "-m", "llm"])
