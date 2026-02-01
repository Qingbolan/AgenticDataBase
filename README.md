# AgenticDB

<div align="center">

**The database that turns text into evolvable state.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*AgenticDB isn't a database you query with English.*
*It's a database that compiles English into structured state, with dependency-aware history, replay, and invalidation.*

</div>

---

## The Thesis

Every "AI + Database" project today does one of two things:

1. **Chat-to-SQL**: Translate natural language into queries against existing schemas
2. **Agent Wrappers**: Let agents call database APIs, logging interactions externally

**Neither changes what the database fundamentally understands.**

AgenticDB proposes a different model:

```
┌─────────────────────────────────────────────────────────────────┐
│                         TEXT INPUT                              │
│  "User u123 registered. Risk score 0.3 from model_v2.          │
│   Approved because score < threshold."                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓ compile
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC LAYER                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐    │
│  │ Event   │  │ Claim   │  │ Action  │  │ Schema Proposal │    │
│  │ (fact)  │  │ (belief)│  │ (act)   │  │ (type evolution)│    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘    │
│       └────────────┼────────────┴────────────────┘              │
│                    ↓                                            │
│          ┌─────────────────────┐                                │
│          │  Dependency Graph   │                                │
│          │  + Version History  │                                │
│          └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓ query
┌─────────────────────────────────────────────────────────────────┐
│  why(x)  →  causal chain                                        │
│  impact(x)  →  downstream dependencies + auto-invalidation      │
│  replay(trace_id)  →  deterministic reconstruction              │
└─────────────────────────────────────────────────────────────────┘
```

**The key innovation:**
- Input is text (agent traces, logs, descriptions, structured hints)
- System **compiles** into Events/Claims/Actions
- System **proposes and evolves** schemas with versioned history
- Queries answer "why" and "what-if", not just "what"

---

## Two Layers

AgenticDB has two interfaces. **Both are intentional.**

| Layer | Input | Purpose | Status |
|-------|-------|---------|--------|
| **Semantic Ingestion** | Text, agent traces, descriptions | Product interface — what agents and systems talk to | 🚧 Building |
| **Structured API** | Event/Claim/Action objects | Engineering interface — deterministic, testable, auditable | ✅ Implemented |

The **Semantic Ingestion** layer is the product vision.
The **Structured API** is the foundation that makes it reproducible.

---

## Quick Start: The Vision

```python
from agenticdb import AgenticDB

db = AgenticDB()
branch = db.create_branch("user-onboarding")

# HIGH-LEVEL: Ingest text, system compiles to semantic objects
trace = branch.ingest("""
User u123 registered with email alice@example.com at 2024-01-15T10:30:00Z.
Risk model v2 computed risk_score = 0.15 with confidence 0.92.
Since risk_score < 0.3 threshold, user was approved for onboarding.
""", mode="agent_trace")

# System automatically:
# 1. Extracts Event(UserRegistered), Claim(risk_score=0.15), Action(ApproveUser)
# 2. Builds dependency graph: Action depends on Claim depends on Event
# 3. Proposes schema if new entity types detected

# Query the compiled state
print(branch.why(trace.actions[0].id))
# → CausalChain: Event(UserRegistered) → Claim(risk_score) → Action(Approve)

print(branch.impact("risk_model_v2"))
# → All claims and actions that would need re-evaluation if model changes
```

### When schema evolution is needed:

```python
# New concept appears in trace
trace = branch.ingest("""
User u123 received a loyalty_tier upgrade to "gold" based on purchase_history.
""")

# System detects unknown concept, proposes schema change
proposal = trace.schema_proposal
print(proposal)
# → SchemaProposal:
# →   + ClaimType("user.loyalty_tier", values=["bronze", "silver", "gold"])
# →   + DerivedFrom("purchase_history")

# Review and commit (or auto-commit with policy)
branch.schema.apply(proposal, auto_migrate=True)
```

---

## Quick Start: The Foundation (Available Now)

The low-level API provides deterministic control:

```python
from agenticdb import AgenticDB, Event, Claim, Action

db = AgenticDB()
branch = db.create_branch("user-onboarding")

# Explicitly construct semantic objects
event = branch.record(Event(
    event_type="UserRegistered",
    data={"user_id": "u123", "email": "alice@example.com"},
    source_agent="signup-service"
))

claim = branch.record(Claim(
    subject="user.u123.risk_score",
    value=0.15,
    source="risk_model_v2",
    confidence=0.92,
    derived_from=[event.id]  # Explicit dependency
))

action = branch.execute(Action(
    action_type="ApproveUser",
    agent_id="approval-agent",
    inputs={"user_id": "u123"},
    depends_on=[event.id, claim.id],  # Explicit dependencies
    reasoning="Risk score 0.15 < threshold 0.3"
))

# Query: Why was this user approved?
chain = branch.why(action.id)
print(chain.to_tree_string())
# Query: why(01JAXYZ...)
# └─ [event] UserRegistered
# └─ [claim] risk_score = 0.15
# └─ [action] ApproveUser

# Query: What depends on the risk model?
impact = branch.impact(claim.id)
print(f"Affected: {impact.total_affected} entities")

# Time travel
snapshot = branch.at(version=2)
```

**Why both layers?**
- **Semantic Ingestion**: For agents, workflows, real-time traces — convenience
- **Structured API**: For testing, auditing, replay — determinism

The Structured API is what Semantic Ingestion compiles down to.

---

## Core Concepts

### Three Primitives

| Primitive | What it is | Immutability |
|-----------|------------|--------------|
| **Event** | Fact that happened | Immutable forever |
| **Claim** | Assertion with source & confidence | Can be superseded |
| **Action** | Agent behavior with dependencies | Immutable, but can be replayed |

### Two Killer Queries

```python
# WHY: Trace the causal chain
chain = db.why(entity_id)
# Returns: all upstream entities that contributed to this state

# IMPACT: Find downstream effects
affected = db.impact(entity_id, auto_invalidate=True)
# Returns: all entities that depend on this, optionally marks them stale
```

### Schema Evolution

```python
# Schemas are not fixed — they evolve with the data
proposal = SchemaProposal(
    add_entity_type="LoyaltyTier",
    add_fields={"tier": "enum(bronze,silver,gold)", "since": "datetime"},
    derived_from=["purchase_history", "account_age"]
)

# Review diff before applying
print(branch.schema.diff(proposal))

# Apply with version tracking
branch.schema.apply(proposal)  # Creates SchemaCommit with version
```

---

## Architecture

```
agenticdb/
├── ingestion/              # 🚧 Semantic Ingestion Layer
│   ├── compiler.py         # Text → Event/Claim/Action
│   ├── extractor.py        # Entity/relation extraction
│   └── schema_proposer.py  # Schema evolution proposals
│
├── core/                   # ✅ Core Semantic Model
│   ├── models.py           # Event, Claim, Action
│   ├── version.py          # Branch, Version, Snapshot
│   ├── dependency.py       # DependencyGraph, why(), impact()
│   └── schema.py           # 🚧 Schema registry & evolution
│
├── storage/                # ✅ Storage Layer
│   ├── engine.py           # Pluggable backends
│   └── index.py            # Fast lookups
│
├── query/                  # ✅ Query Engine
│   ├── engine.py           # Query execution
│   └── operators.py        # why(), impact(), trace()
│
├── runtime/                # ✅ Execution Runtime
│   ├── cache.py            # Dependency-aware caching
│   └── subscription.py     # Reactive subscriptions
│
└── interface/              # ✅ SDK
    └── client.py           # AgenticDB entry point
```

---

## What This Is NOT

| What people might think | What AgenticDB actually is |
|------------------------|---------------------------|
| "Chat-to-SQL wrapper" | No — we don't translate queries, we compile state |
| "Just event sourcing" | No — we have semantic types (Event/Claim/Action) + schema evolution |
| "LLM-dependent system" | No — LLM is pluggable extractor, core is deterministic |
| "Another agent framework" | No — we're the state layer agents write to, not the orchestrator |

---

## Comparison

| Feature | AgenticDB | Firebase | Datomic | LangChain/LangGraph |
|---------|-----------|----------|---------|---------------------|
| Text-to-state ingestion | ✅ (building) | ❌ | ❌ | ❌ |
| Schema evolution | ✅ (building) | ❌ | Partial | ❌ |
| `why(x)` causal query | ✅ | ❌ | ❌ | ❌ |
| `impact(x)` + invalidation | ✅ | ❌ | ❌ | ❌ |
| Agent actions as first-class | ✅ | ❌ | ❌ | Partial |
| Version/branch native | ✅ | ❌ | ✅ | ❌ |
| Dependency-aware cache | ✅ | ❌ | ❌ | ❌ |

---

## Design Philosophy

> **"AgenticDB absorbs core backend work — state transitions, dependency coordination, cache invalidation — into the data system. It doesn't replace all backends; it turns them into thin I/O execution boundaries."**

What stays **inside** AgenticDB:
- State management, versioning, branching
- Dependency tracking and causal queries
- Schema evolution and migration
- Cache invalidation based on dependencies

What stays **outside**:
- External I/O (payments, emails, 3rd-party APIs)
- Rate limiting, circuit breakers
- UI rendering, A/B testing

---

## Roadmap

### ✅ Implemented
- Core primitives: Event, Claim, Action
- Dependency graph with `why()` and `impact()`
- Branch/version support
- Dependency-aware caching
- Reactive subscriptions

### 🚧 In Progress
- **Semantic Ingestion**: Text → structured state compilation
- **Schema Evolution**: Propose, diff, apply schema changes
- **Extractor Plugins**: Pluggable LLM/rule-based extractors

### 📋 Planned
- Persistent storage backends (SQLite, PostgreSQL)
- Distributed mode
- Policy engine for declarative rules
- REST/gRPC interface
- Visual trace explorer

---

## Contributing

This is an early-stage research project exploring agent-native data systems.

The core thesis: **Text in, evolvable state out, with full causal history.**

Contributions, feedback, and discussions are welcome.

---

## License

MIT

---

<div align="center">

**AgenticDB** — The state runtime for the agentic era.

*Where text becomes state, and state remembers why.*

</div>
