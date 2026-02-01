"""
AgenticDB Agent-Driven Ingestion Example

This example demonstrates the new agent-driven architecture where
specialized LLM agents handle text → entity extraction:

- EventExtractorAgent: Identifies immutable events
- ClaimExtractorAgent: Identifies sourced assertions
- ActionExtractorAgent: Identifies agent behaviors
- DependencyInferenceAgent: Infers relationships

Usage:
    python examples/agent_driven_ingestion.py

Note: Requires OPENAI_API_KEY environment variable.
"""

from agenticdb.core.agents.ingestion import (
    IngestionCoordinator,
    EventExtractorAgent,
    ClaimExtractorAgent,
    ActionExtractorAgent,
)
from agenticdb.core.agents.base import AgentContext


def demo_individual_agents():
    """Demonstrate using individual extraction agents."""
    print("=" * 70)
    print("Demo 1: Individual Extraction Agents")
    print("=" * 70)

    ctx = AgentContext()

    # Sample text describing an agent workflow
    text = """
    User u123 registered with email alice@example.com on January 15, 2024.
    The risk model v2.1 computed a risk score of 0.15 with 92% confidence.
    Based on the low risk score (below 0.3 threshold), the approval agent
    decided to approve the user for premium onboarding.
    """

    print(f"\nInput text:\n{text}")
    print("-" * 50)

    # Event Extraction
    print("\n[EventExtractorAgent]")
    event_agent = EventExtractorAgent()
    event_result = event_agent.run(ctx, text)

    print(f"Confidence: {event_result.confidence:.2f}")
    for event in event_result.events:
        print(f"  - {event.event_type}")
        print(f"    Data: {event.data}")

    # Claim Extraction
    print("\n[ClaimExtractorAgent]")
    claim_agent = ClaimExtractorAgent()
    claim_result = claim_agent.run(ctx, text)

    print(f"Confidence: {claim_result.confidence:.2f}")
    for claim in claim_result.claims:
        print(f"  - {claim.subject} = {claim.value}")
        print(f"    Source: {claim.source}, Confidence: {claim.confidence}")

    # Action Extraction
    print("\n[ActionExtractorAgent]")
    action_agent = ActionExtractorAgent()
    action_result = action_agent.run(ctx, text)

    print(f"Confidence: {action_result.confidence:.2f}")
    for action in action_result.actions:
        print(f"  - {action.action_type} by {action.agent_id}")
        print(f"    Depends on: {action.depends_on_refs}")


def demo_coordinator():
    """Demonstrate using the IngestionCoordinator."""
    print("\n" + "=" * 70)
    print("Demo 2: IngestionCoordinator (Full Pipeline)")
    print("=" * 70)

    coordinator = IngestionCoordinator()

    text = """
    The fraud detection system flagged transaction tx-456 as suspicious.
    ML model v3.2 computed fraud_probability = 0.78.
    Alert agent notified the compliance team about the high-risk transaction.
    Transaction was placed on hold pending manual review.
    """

    print(f"\nInput text:\n{text}")
    print("-" * 50)

    result = coordinator.ingest(text)

    print(f"\nExtraction Summary:")
    print(f"  Total entities: {result.total_entities}")
    print(f"  Average confidence: {result.average_confidence:.2f}")

    print(f"\nEvents ({len(result.events)}):")
    for event in result.events:
        print(f"  - [{event.ref_id}] {event.event_type}")

    print(f"\nClaims ({len(result.claims)}):")
    for claim in result.claims:
        print(f"  - [{claim.ref_id}] {claim.subject} = {claim.value}")

    print(f"\nActions ({len(result.actions)}):")
    for action in result.actions:
        print(f"  - [{action.ref_id}] {action.action_type}")

    print(f"\nInferred Dependencies ({len(result.edges)}):")
    for edge in result.edges:
        print(f"  - {edge.from_ref} --{edge.edge_type.value}--> {edge.to_ref}")
        if edge.reasoning:
            print(f"    Reason: {edge.reasoning}")


def demo_selective_extraction():
    """Demonstrate selective extraction modes."""
    print("\n" + "=" * 70)
    print("Demo 3: Selective Extraction")
    print("=" * 70)

    coordinator = IngestionCoordinator()

    text = """
    Order #789 was placed by customer c-101 for $299.99.
    Inventory system confirmed stock availability.
    Payment processor charged the credit card successfully.
    """

    print(f"\nInput text:\n{text}")
    print("-" * 50)

    # Extract only events
    print("\n[Events Only]")
    events_only = coordinator.extract_events(text)
    for event in events_only.events:
        print(f"  - {event.event_type}: {event.data}")

    # Extract only claims
    print("\n[Claims Only]")
    claims_only = coordinator.extract_claims(text)
    for claim in claims_only.claims:
        print(f"  - {claim.subject} = {claim.value}")


def main():
    print("AgenticDB Agent-Driven Architecture Demo")
    print("=" * 70)
    print("""
This demo shows how specialized LLM agents extract semantic entities
from natural language text. Each agent focuses on one entity type:

- EventExtractorAgent → Events (immutable facts)
- ClaimExtractorAgent → Claims (sourced assertions)
- ActionExtractorAgent → Actions (agent behaviors)
- DependencyInferenceAgent → Relationships between entities

The IngestionCoordinator orchestrates all agents together.
""")

    try:
        demo_individual_agents()
        demo_coordinator()
        demo_selective_extraction()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure OPENAI_API_KEY is set and dependencies are installed:")
        print("  pip install openai")


if __name__ == "__main__":
    main()
