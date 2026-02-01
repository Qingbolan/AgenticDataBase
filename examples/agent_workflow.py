"""
AgenticDB Agent Workflow Example

This example demonstrates how AgenticDB enables:
1. Multi-agent coordination with explicit dependencies
2. Audit trails for agent decisions
3. Automatic cache invalidation when data changes
4. Reproducible agent workflows
"""

from agenticdb import AgenticDB, Event, Claim, Action
from agenticdb.core.models import EntityType


def main():
    print("=" * 60)
    print("Multi-Agent Order Processing Workflow")
    print("=" * 60)

    db = AgenticDB()
    branch = db.create_branch("order-processing")

    # ==========================================================================
    # Scenario: E-commerce order with multiple agents
    # ==========================================================================

    # Step 1: Order placed (Event from external system)
    order_placed = branch.record(Event(
        event_type="OrderPlaced",
        data={
            "order_id": "ord_789",
            "customer_id": "cust_456",
            "items": [
                {"sku": "WIDGET-A", "quantity": 2, "price": 29.99},
                {"sku": "GADGET-B", "quantity": 1, "price": 149.99}
            ],
            "total": 209.97
        },
        source_system="ecommerce-api"
    ))
    print(f"\n1. Order placed: {order_placed.data['order_id']}")

    # Step 2: Inventory Agent checks stock
    inventory_check = branch.execute(Action(
        action_type="CheckInventory",
        agent_id="inventory-agent",
        agent_type="deterministic",
        inputs={"order_id": "ord_789"},
        depends_on=[order_placed.id],
        outputs={"in_stock": True, "warehouse": "WH-EAST"}
    ))
    print(f"2. Inventory checked by {inventory_check.agent_id}")

    # Store inventory claim
    inventory_claim = branch.record(Claim(
        subject="order.ord_789.inventory_status",
        value={"in_stock": True, "warehouse": "WH-EAST"},
        source="inventory-agent",
        confidence=1.0,
        derived_from=[inventory_check.id]
    ))

    # Step 3: Fraud Detection Agent
    fraud_check = branch.execute(Action(
        action_type="FraudAnalysis",
        agent_id="fraud-detection-agent",
        agent_type="ml-model",
        inputs={"order_id": "ord_789", "customer_id": "cust_456"},
        depends_on=[order_placed.id],
        reasoning="Customer has good history, order value normal, no red flags."
    ))
    print(f"3. Fraud analysis by {fraud_check.agent_id}")

    fraud_score = branch.record(Claim(
        subject="order.ord_789.fraud_score",
        value=0.05,
        source="fraud-model-v3",
        confidence=0.94,
        derived_from=[fraud_check.id]
    ))

    # Step 4: Pricing Agent validates totals
    price_validation = branch.execute(Action(
        action_type="ValidatePricing",
        agent_id="pricing-agent",
        agent_type="rule-based",
        inputs={"order_id": "ord_789"},
        depends_on=[order_placed.id],
        outputs={"valid": True, "discounts_applied": []}
    ))
    print(f"4. Pricing validated by {price_validation.agent_id}")

    # Step 5: Approval Agent makes final decision
    # This agent depends on ALL previous checks
    approval = branch.execute(Action(
        action_type="ApproveOrder",
        agent_id="approval-agent",
        agent_type="orchestrator",
        inputs={"order_id": "ord_789"},
        depends_on=[
            order_placed.id,
            inventory_check.id,
            inventory_claim.id,
            fraud_check.id,
            fraud_score.id,
            price_validation.id
        ],
        reasoning="""
        Decision factors:
        - Inventory: In stock at WH-EAST
        - Fraud score: 0.05 (below 0.3 threshold)
        - Pricing: Valid, no issues
        Conclusion: Approve order for fulfillment.
        """.strip()
    ))
    print(f"5. Order approved by {approval.agent_id}")

    # ==========================================================================
    # Query: Full decision audit trail
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Audit Trail: Why was this order approved?")
    print("-" * 40)

    why_result = branch.why(approval.id)
    print(why_result.to_tree_string())

    # ==========================================================================
    # Scenario: Fraud model update - what's affected?
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Impact Analysis: What if fraud model changes?")
    print("-" * 40)

    impact = branch.impact(fraud_score.id)
    print(f"If fraud_score changes:")
    print(f"  - {len(impact.affected_actions)} actions would need re-evaluation")
    print(f"  - Max dependency depth: {impact.max_depth}")

    # ==========================================================================
    # Demonstrate dependency-aware caching
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Dependency-Aware Caching")
    print("-" * 40)

    # Simulate caching a computed value
    from agenticdb.runtime.cache import DependencyAwareCache
    from agenticdb.core.dependency import DependencyGraph

    graph = DependencyGraph()
    cache = DependencyAwareCache(graph)

    # Cache the order approval decision with its dependencies
    cache.set(
        key="order_approval_ord_789",
        value={"approved": True, "reason": "All checks passed"},
        depends_on=[fraud_score.id, inventory_claim.id],
        version=branch.version
    )

    print(f"Cached: order_approval_ord_789")
    print(f"  Dependencies: fraud_score, inventory_claim")

    # Get from cache
    cached = cache.get("order_approval_ord_789")
    print(f"  Retrieved: {cached}")

    # Simulate fraud model update - this would invalidate the cache
    print(f"\nSimulating fraud model update...")
    # In production, this happens automatically when fraud_score is updated
    # invalidated = cache.invalidate_dependents(fraud_score.id)
    # print(f"  Invalidated {len(invalidated)} cache entries")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. EXPLICIT DEPENDENCIES
   Each action declares what it depends on.
   No hidden assumptions, no "magic" data flows.

2. FULL AUDIT TRAIL
   why(approval) shows every factor in the decision.
   Regulators and auditors can trace any decision.

3. AUTOMATIC IMPACT ANALYSIS
   impact(fraud_model) shows what needs re-evaluation.
   No manual tracking of downstream effects.

4. DEPENDENCY-AWARE CACHING
   Cache entries know their dependencies.
   Updates automatically invalidate stale values.

This is what "agent-native" means:
The database understands agent behavior, not just data storage.
""")


if __name__ == "__main__":
    main()
