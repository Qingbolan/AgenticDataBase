"""
AgenticDB Semantic Ingestion Example

This example demonstrates the core thesis of AgenticDB:
Text in → Evolvable state out → Full causal history

The semantic ingestion layer compiles unstructured text into
structured Event/Claim/Action objects with automatic dependency
inference and schema evolution proposals.
"""

from agenticdb import AgenticDB


def main():
    print("=" * 70)
    print("AgenticDB Semantic Ingestion: Text → Evolvable State")
    print("=" * 70)

    db = AgenticDB()
    branch = db.create_branch("semantic-demo")

    # =========================================================================
    # Example 1: Basic Text Ingestion
    # =========================================================================
    print("\n" + "-" * 50)
    print("Example 1: Ingesting Agent Trace Text")
    print("-" * 50)

    trace_text = """
    User u123 registered with email alice@example.com at 2024-01-15T10:30:00Z.
    Risk model v2 computed risk_score = 0.15 with confidence 0.92.
    Since risk_score < 0.3 threshold, user was approved for onboarding.
    """

    print(f"Input text:\n{trace_text}")
    print("\nIngesting...")

    result = branch.ingest(trace_text, mode="agent_trace")

    print(f"\nExtracted {result.compilation.entity_count} entities:")
    print(f"  Events: {len(result.events)}")
    for event in result.events:
        print(f"    - {event.event_type}: {event.data}")

    print(f"  Claims: {len(result.claims)}")
    for claim in result.claims:
        print(f"    - {claim.subject} = {claim.value} (source: {claim.source})")

    print(f"  Actions: {len(result.actions)}")
    for action in result.actions:
        print(f"    - {action.action_type} by {action.agent_id}")

    # =========================================================================
    # Example 2: Querying the Compiled State
    # =========================================================================
    print("\n" + "-" * 50)
    print("Example 2: Querying Compiled State")
    print("-" * 50)

    if result.actions:
        action = result.actions[0]
        print(f"\nWhy was {action.action_type} executed?")

        chain = branch.why(action.id)
        print(chain.to_tree_string())

    # =========================================================================
    # Example 3: Schema Evolution
    # =========================================================================
    print("\n" + "-" * 50)
    print("Example 3: Schema Evolution")
    print("-" * 50)

    new_trace = """
    User u123 achieved loyalty_tier = gold based on purchase_history.
    Customer lifetime value estimated at $2,500.
    """

    print(f"Input with new concepts:\n{new_trace}")
    print("\nIngesting...")

    result2 = branch.ingest(new_trace, mode="description")

    if result2.schema_proposal and result2.schema_proposal.changes:
        print("\nSchema changes proposed:")
        print(result2.schema_proposal.summary())
    else:
        print("\nNo new schema changes detected (extractor may need enhancement)")

    # =========================================================================
    # Example 4: Comparing High-Level vs Low-Level API
    # =========================================================================
    print("\n" + "-" * 50)
    print("Example 4: High-Level vs Low-Level API")
    print("-" * 50)

    print("""
HIGH-LEVEL (Semantic Ingestion):
    result = branch.ingest("User registered, risk score 0.2, approved.")
    # System extracts entities, builds dependencies, proposes schema

LOW-LEVEL (Structured API):
    event = branch.record(Event(event_type="UserRegistered", data={...}))
    claim = branch.record(Claim(subject="risk_score", value=0.2, ...))
    action = branch.execute(Action(..., depends_on=[event.id, claim.id]))
    # You control every detail explicitly

Both compile down to the same semantic model.
High-level is for convenience.
Low-level is for determinism and testing.
""")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("The Thesis")
    print("=" * 70)
    print("""
AgenticDB isn't a database you query with English.
It's a database that COMPILES English into evolvable state.

Traditional:  "SELECT * FROM users WHERE risk_score < 0.3"
              → Returns data rows

AgenticDB:    "User registered, risk computed, approved because low risk"
              → Creates Event, Claim, Action
              → Builds dependency graph
              → Enables why(x), impact(x), replay()
              → Proposes schema evolution

This is TEXT-TO-STATE, not CHAT-TO-SQL.
""")


if __name__ == "__main__":
    main()
