"""
Pytest configuration and shared fixtures for AgenticDB tests.

This module provides common fixtures used across test modules,
including mock LLM responses and sample data generators.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import json

from agenticdb.core.models import Event, Claim, Action
from agenticdb.core.agents.base import AgentContext


# =============================================================================
# LLM Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_oracle():
    """
    Create a mock Oracle for testing without real LLM calls.

    This fixture patches the Oracle class so that all agents
    use a mock instead of making actual API calls.
    """
    with patch('agenticdb.core.agents.base.base_agent.Oracle') as MockOracle:
        mock_instance = MagicMock()
        MockOracle.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_oracle_with_error():
    """Create a mock Oracle that raises errors."""
    with patch('agenticdb.core.agents.base.base_agent.Oracle') as MockOracle:
        mock_instance = MagicMock()
        mock_instance.query.side_effect = Exception("API Error")
        MockOracle.return_value = mock_instance
        yield mock_instance


# =============================================================================
# Context Fixtures
# =============================================================================

@pytest.fixture
def agent_context():
    """Create a standard test agent context."""
    return AgentContext(
        session_id="test-session-001",
        branch_id="test-branch",
        language="en",
        extra={"environment": "test"}
    )


@pytest.fixture
def agent_context_zh():
    """Create a Chinese language agent context."""
    return AgentContext(
        session_id="test-session-002",
        branch_id="test-branch",
        language="zh"
    )


# =============================================================================
# Sample Entity Fixtures
# =============================================================================

@pytest.fixture
def sample_event():
    """Create a single sample event."""
    return Event(
        event_type="UserRegistered",
        data={
            "user_id": "u123",
            "email": "alice@example.com",
            "registration_source": "web"
        },
        source_agent="registration-service",
        source_system="auth"
    )


@pytest.fixture
def sample_events():
    """Create a list of sample events representing a user journey."""
    now = datetime.now(timezone.utc)
    return [
        Event(
            event_type="UserRegistered",
            data={"user_id": "u123", "email": "alice@example.com"},
            source_system="auth-service",
            created_at=now - timedelta(hours=2)
        ),
        Event(
            event_type="ProfileCompleted",
            data={"user_id": "u123", "completion_rate": 100},
            source_system="profile-service",
            created_at=now - timedelta(hours=1, minutes=30)
        ),
        Event(
            event_type="PaymentMethodAdded",
            data={"user_id": "u123", "type": "credit_card"},
            source_system="payment-service",
            created_at=now - timedelta(hours=1)
        ),
        Event(
            event_type="FirstPurchase",
            data={"user_id": "u123", "amount": 49.99, "product": "premium_plan"},
            source_system="commerce-service",
            created_at=now - timedelta(minutes=30)
        )
    ]


@pytest.fixture
def sample_claim():
    """Create a single sample claim."""
    return Claim(
        subject="user.u123.risk_score",
        value=0.15,
        source="risk_model_v2",
        source_version="2.1.0",
        confidence=0.92
    )


@pytest.fixture
def sample_claims():
    """Create a list of sample claims."""
    now = datetime.now(timezone.utc)
    return [
        Claim(
            subject="user.u123.risk_score",
            value=0.15,
            source="risk_model_v2",
            confidence=0.92,
            created_at=now - timedelta(hours=1)
        ),
        Claim(
            subject="user.u123.tier",
            value="premium",
            source="rules_engine",
            confidence=1.0,
            created_at=now - timedelta(minutes=45)
        ),
        Claim(
            subject="user.u123.lifetime_value",
            value=2500.0,
            source="analytics_model",
            confidence=0.85,
            created_at=now - timedelta(minutes=30)
        )
    ]


@pytest.fixture
def conflicting_claims():
    """Create claims that conflict with each other."""
    return [
        Claim(
            subject="user.u123.risk_score",
            value=0.15,
            source="risk_model_v2",
            confidence=0.92
        ),
        Claim(
            subject="user.u123.risk_score",
            value=0.45,
            source="risk_model_v3",
            confidence=0.88
        )
    ]


@pytest.fixture
def sample_action():
    """Create a single sample action."""
    action = Action(
        action_type="ApproveUser",
        agent_id="approval-agent",
        agent_type="rule-based",
        inputs={"user_id": "u123", "risk_score": 0.15},
        depends_on=["claim_risk_score"],
        reasoning="Risk score below threshold (0.3)"
    )
    action.complete({"approved": True, "tier": "premium"})
    return action


@pytest.fixture
def sample_actions():
    """Create a list of sample actions."""
    action1 = Action(
        action_type="ComputeRiskScore",
        agent_id="risk-agent",
        inputs={"user_id": "u123"}
    )
    action1.complete({"risk_score": 0.15})

    action2 = Action(
        action_type="ApproveUser",
        agent_id="approval-agent",
        inputs={"user_id": "u123", "risk_score": 0.15},
        depends_on=[action1.id]
    )
    action2.complete({"approved": True})

    action3 = Action(
        action_type="SendWelcomeEmail",
        agent_id="notification-agent",
        inputs={"user_id": "u123", "email": "alice@example.com"},
        depends_on=[action2.id]
    )
    action3.start()
    action3.fail("SMTP connection failed")

    return [action1, action2, action3]


# =============================================================================
# Mock Response Fixtures
# =============================================================================

@pytest.fixture
def mock_event_extraction_response():
    """Standard mock response for event extraction."""
    return json.dumps({
        "events": [
            {
                "event_type": "UserRegistered",
                "data": {"user_id": "u123", "email": "alice@example.com"},
                "source_system": "auth-service"
            }
        ],
        "confidence": 0.95,
        "reasoning": "Clear registration event identified"
    })


@pytest.fixture
def mock_claim_extraction_response():
    """Standard mock response for claim extraction."""
    return json.dumps({
        "claims": [
            {
                "subject": "user.u123.risk_score",
                "value": 0.15,
                "source": "risk_model_v2",
                "confidence": 0.92
            }
        ],
        "confidence": 0.90,
        "reasoning": "Risk score claim extracted"
    })


@pytest.fixture
def mock_action_extraction_response():
    """Standard mock response for action extraction."""
    return json.dumps({
        "actions": [
            {
                "action_type": "ApproveUser",
                "agent_id": "approval-agent",
                "inputs": {"user_id": "u123"},
                "outputs": {"approved": True},
                "depends_on_refs": ["risk_score_claim"],
                "reasoning": "Risk below threshold"
            }
        ],
        "confidence": 0.88,
        "reasoning": "Approval action identified"
    })


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def empty_db():
    """Create an empty AgenticDB instance."""
    from agenticdb import AgenticDB
    db = AgenticDB()
    yield db
    db.clear()


@pytest.fixture
def populated_db(sample_events, sample_claims, sample_actions):
    """Create an AgenticDB with sample data."""
    from agenticdb import AgenticDB
    db = AgenticDB()
    branch = db.create_branch("test-branch")

    for event in sample_events:
        branch.record(event)

    for claim in sample_claims:
        branch.record(claim)

    for action in sample_actions:
        branch.execute(action)

    yield db, branch
    db.clear()


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM API (deselect with '-m \"not llm\"')"
    )
