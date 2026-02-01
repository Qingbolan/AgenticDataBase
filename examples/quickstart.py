"""
AgenticDB Quickstart Example

This example demonstrates the core concepts that make AgenticDB different
from traditional databases:

1. Events, Claims, and Actions as first-class entities
2. Automatic dependency tracking
3. Causal queries: why(x) and impact(x)
4. Time-travel debugging
"""

from agenticdb import AgenticDB, Event, Claim, Action


def main():
    # ==========================================================================
    # Initialize AgenticDB
    # ==========================================================================
    print("=" * 60)
    print("AgenticDB Quickstart")
    print("=" * 60)

    db = AgenticDB()

    # Create a branch for our workflow
    branch = db.create_branch("user-onboarding", description="User signup workflow")
    print(f"\nCreated branch: {branch.name} (id: {branch.id})")

    # ==========================================================================
    # Record Events (Immutable Facts)
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Recording Events")
    print("-" * 40)

    # Events are immutable facts that happened
    user_registered = branch.record(Event(
        event_type="UserRegistered",
        data={
            "user_id": "u123",
            "email": "alice@example.com",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        source_agent="signup-service"
    ))
    print(f"Recorded: {user_registered.event_type} (id: {user_registered.id})")

    email_verified = branch.record(Event(
        event_type="EmailVerified",
        data={"user_id": "u123", "verified_at": "2024-01-15T10:35:00Z"},
        source_agent="email-service",
        causation_id=user_registered.id  # This event was caused by registration
    ))
    print(f"Recorded: {email_verified.event_type} (id: {email_verified.id})")

    # ==========================================================================
    # Store Claims (Structured Assertions)
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Storing Claims")
    print("-" * 40)

    # Claims are computed values with provenance
    risk_score = branch.record(Claim(
        subject="user.u123.risk_score",
        value=0.15,
        source="risk_model_v2",
        source_version="2.1.0",
        confidence=0.92,
        derived_from=[user_registered.id, email_verified.id]  # Explicit dependencies
    ))
    print(f"Stored claim: {risk_score.subject} = {risk_score.value}")
    print(f"  Source: {risk_score.source}, Confidence: {risk_score.confidence}")

    trust_level = branch.record(Claim(
        subject="user.u123.trust_level",
        value="standard",
        source="trust_classifier",
        confidence=0.88,
        derived_from=[risk_score.id]  # This depends on risk_score
    ))
    print(f"Stored claim: {trust_level.subject} = {trust_level.value}")

    # ==========================================================================
    # Execute Actions (Agent Behaviors)
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Executing Actions")
    print("-" * 40)

    # Actions are agent behaviors with explicit dependencies
    approve_action = branch.execute(Action(
        action_type="ApproveUser",
        agent_id="approval-agent",
        agent_type="rule-based",
        inputs={"user_id": "u123", "threshold": 0.3},
        depends_on=[
            user_registered.id,
            email_verified.id,
            risk_score.id,
            trust_level.id
        ],
        reasoning="User risk score (0.15) is below threshold (0.3), email verified, approving."
    ))
    print(f"Executed: {approve_action.action_type} by {approve_action.agent_id}")
    print(f"  Status: {approve_action.action_status}")
    print(f"  Dependencies: {len(approve_action.depends_on)} entities")

    # ==========================================================================
    # Causal Query: WHY
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Step 4: Causal Query - why()")
    print("-" * 40)
    print("Question: Why was this user approved?")

    # The killer feature: trace the causal chain
    why_result = branch.why(approve_action.id)
    print(f"\nCausal chain (depth: {why_result.total_depth}):")
    print(why_result.to_tree_string())

    # ==========================================================================
    # Impact Query: WHAT IF
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Impact Query - impact()")
    print("-" * 40)
    print("Question: What would be affected if the risk model changes?")

    # Find everything that depends on the risk score
    impact_result = branch.impact(risk_score.id)
    print(f"\nImpact analysis:")
    print(f"  Affected claims: {len(impact_result.affected_claims)}")
    print(f"  Affected actions: {len(impact_result.affected_actions)}")
    print(f"  Total affected: {impact_result.total_affected}")
    print(f"  Max depth: {impact_result.max_depth}")

    # ==========================================================================
    # Time Travel
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Step 6: Time Travel")
    print("-" * 40)

    print(f"Current version: {branch.version}")

    # Get a snapshot at version 2 (after first two events, before claims)
    snapshot = branch.at(version=2)
    print(f"\nSnapshot at version 2:")
    print(f"  Events: {len(snapshot.events)}")
    print(f"  Claims: {len(snapshot.claims)}")
    print(f"  Actions: {len(snapshot.actions)}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Summary: What makes this different?")
    print("=" * 60)
    print("""
Traditional DB:  "What is user.u123.risk_score?"  → 0.15

AgenticDB:       "Why is user.u123.risk_score 0.15?"
                 → Derived from UserRegistered + EmailVerified
                 → Computed by risk_model_v2 (confidence: 0.92)

                 "What breaks if risk_model_v2 changes?"
                 → trust_level claim
                 → ApproveUser action
                 → (auto-invalidation available)
""")


if __name__ == "__main__":
    main()
