# AgenticDB Architecture

This document provides detailed architecture documentation for AgenticDB, including system components, layer responsibilities, and implementation details.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 AgenticDB                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Pattern Cache (Workload Memoization)                 │ │
│  │    Extracted patterns → bound templates → direct SQL execution          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                │ cache miss                                  │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   Intent-Aware Transaction Pipeline                     │ │
│  │                                                                          │ │
│  │   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐ │ │
│  │   │   Intent   │───►│  Binding   │───►│ Validation │───►│   Schema   │ │ │
│  │   │   Parser   │    │  Resolver  │    │   Engine   │    │  Resolver  │ │ │
│  │   └────────────┘    └────────────┘    └────────────┘    └────────────┘ │ │
│  │         │                 │                                     │       │ │
│  │         │ PENDING_BINDING │ PENDING_CONFIRMATION               │       │ │
│  │         ▼                 ▼                                     ▼       │ │
│  │   [Transaction State Register]                           ┌───────────┐ │ │
│  │                                                          │  Query    │ │ │
│  │                                                          │  Builder  │ │ │
│  │                                                          └───────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Dependency Graph                                 │ │
│  │     Entity versioning • Edge tracking • Transitive invalidation         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Materialization Manager                              │ │
│  │     View registration • Lazy/Eager recomputation • Version tracking     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   Storage Layer (SQLite / PostgreSQL)                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          MCP Interface                                  │ │
│  │            Dynamic tool generation from current schema state            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Architecture

AgenticDB follows a layered architecture separating concerns across interface, domain, application, and infrastructure layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Interface Layer                                 │
│  agenticdb/interface/                                                        │
│    └── client.py              AgenticDB, BranchHandle - primary SDK entry   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Core Domain Layer                                  │
│  agenticdb/core/                                                             │
│    ├── models.py              Entity primitives: Event, Claim, Action       │
│    ├── dependency.py          DependencyGraph, EdgeType, why()/impact()     │
│    └── version.py             Branch, Version, Snapshot - temporal model    │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Application Services Layer                           │
│  agenticdb/core/agents/       AI-powered semantic processing agents         │
│    ├── base/                  BaseAgent, AgentContext                       │
│    ├── ingestion/             Text → Entity extraction pipeline             │
│    │   ├── coordinator.py     IngestionCoordinator orchestration            │
│    │   ├── event_extractor_agent.py                                         │
│    │   ├── claim_extractor_agent.py                                         │
│    │   ├── action_extractor_agent.py                                        │
│    │   └── dependency_inference_agent.py                                    │
│    ├── schema/                Schema evolution agents                       │
│    │   ├── schema_detector_agent.py                                         │
│    │   └── schema_proposal_agent.py                                         │
│    ├── query/                 Causal reasoning agents                       │
│    │   ├── causal_reasoning_agent.py    Explains why(x)                     │
│    │   └── impact_analysis_agent.py     Explains impact(x)                  │
│    └── memory/                Entity memory management                      │
│        ├── event_memory_agent.py                                            │
│        ├── claim_memory_agent.py                                            │
│        └── action_memory_agent.py                                           │
│                                                                              │
│  agenticdb/query/             Query execution engine                        │
│    ├── engine.py              QueryEngine - unified query interface         │
│    └── operators.py           WhyQuery, ImpactQuery, CausalChain            │
│                                                                              │
│  agenticdb/ingestion/         Text-to-semantic compilation                  │
│    ├── compiler.py            TraceCompiler, IngestResult                   │
│    ├── extractor.py           Extraction orchestration                      │
│    └── schema_proposer.py     Schema change proposals                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Infrastructure Layer                                │
│  agenticdb/storage/           Persistence backends                          │
│    ├── engine.py              StorageEngine (abstract), InMemoryStorage     │
│    ├── sqlite.py              SQLiteStorage implementation                  │
│    └── index.py               Entity indexing                               │
│                                                                              │
│  agenticdb/runtime/           Runtime services                              │
│    ├── materialization.py     MaterializationManager, MaterializedView      │
│    ├── cache.py               DependencyAwareCache                          │
│    └── subscription.py        SubscriptionManager, reactive updates         │
│                                                                              │
│  agenticdb/mcp/               External interface                            │
│    ├── server.py              MCP server implementation                     │
│    └── tools.py               Dynamic tool generation from schema           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Key Abstractions |
|-------|----------------|------------------|
| **Interface** | SDK entry points, API surface | `AgenticDB`, `BranchHandle` |
| **Core Domain** | Entity primitives, causal graph, versioning | `Event`, `Claim`, `Action`, `DependencyGraph` |
| **Application** | Semantic processing, query execution, ingestion | `*Agent`, `QueryEngine`, `TraceCompiler` |
| **Infrastructure** | Persistence, caching, materialization, MCP | `StorageEngine`, `MaterializationManager` |

---

## Component Details

### Intent Parser

Extracts structured Intent from natural language input. The Intent is an intermediate representation, not a semantic interpretation.

### Binding Resolver

Resolves `Pending` parameter slots through in-transaction binding. This is the core mechanism that distinguishes AgenticDB from traditional databases.

### Validation Engine

Enforces safety constraints before execution. Operations that fail validation enter `REJECTED` state. Destructive operations enter `PENDING_CONFIRMATION` state.

### Schema Resolver

Evolves schema from workload patterns. Schema is treated as mutable physical state, not a fixed logical contract.

### Dependency Graph

Tracks causal relationships between entities using NetworkX. Supports:
- `why(x)`: Upstream traversal to find causes
- `impact(x)`: Downstream traversal to find effects
- Transitive invalidation propagation

### Materialization Manager

Maintains derived views with automatic invalidation:

| Mode | Behavior |
|------|----------|
| `LAZY` | Recompute on next access |
| `EAGER` | Recompute immediately on invalidation |

### Pattern Cache

Memoizes query patterns for fast execution without LLM interpretation:

```
Pattern Extraction:
  "show orders from last month"  →  Pattern: "show {E:entity} from {T:temporal}"
  "show users from last week"    →  Matches existing pattern, skips LLM
```

### MCP Interface

Exposes schema-aware tools via Model Context Protocol for external AI applications.

---

## Core Primitives

AgenticDB introduces three **first-class entity types** that make agent behavior queryable:

| Primitive | Semantics | Mutability | Example |
|-----------|-----------|------------|---------|
| **Event** | Immutable fact that occurred | Append-only | `UserRegistered`, `PaymentReceived` |
| **Claim** | Assertion with provenance and confidence | Supersedable | `risk_score = 0.3 (source: model_v2)` |
| **Action** | Agent behavior with explicit dependencies | Trackable | `ApproveOrder(depends_on: [event, claim])` |

### Design Philosophy

> *Traditional databases store "what data is". AgenticDB stores "how state became this way".*

The dependency graph is not an afterthought—it is a first-class data structure that captures causal relationships between all entities.

---

## Transaction State Machine

```
                         ┌─────────────────────────────────────┐
                         │                                     │
                         ▼                                     │
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ RECEIVED │───►│  PARSED  │───►│  BOUND   │───►│VALIDATED │──┐
    └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
                         │               │               │        │
                         │ partial       │ unsafe        │ invalid│
                         ▼               ▼               ▼        │
                   ┌───────────┐  ┌────────────┐  ┌──────────┐   │
                   │  PENDING  │  │  PENDING   │  │ REJECTED │   │
                   │  BINDING  │  │CONFIRMATION│  │          │   │
                   └───────────┘  └────────────┘  └──────────┘   │
                         │               │                        │
                         │ resolved      │ confirmed              │
                         └───────────────┴────────────────────────┘
                                         │
                                         ▼
                                  ┌────────────┐
                                  │  EXECUTED  │
                                  └────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ COMPLETED │ FAILED  │
                              └─────────────────────┘
```

---

## Workload-Driven Schema Evolution

Schema is treated as mutable physical state inferred from workload patterns:

```python
# Schema emerges from first write
db.store("OrderPlaced", {"user_id": "u1", "amount": 100})
# → Infers schema: orders(id, user_id, amount, timestamp)

# Schema evolves on workload change
db.store("OrderPlaced", {"user_id": "u2", "amount": 200, "currency": "USD"})
# → Evolves schema: orders(id, user_id, amount, currency, timestamp)
```

Schema evolution is **constrained**: type-incompatible changes, destructive alterations, and semantic conflicts require explicit confirmation via `PENDING_CONFIRMATION` state.

---

## Dependency-Aware Materialization

```
Dependency Graph:
    Event(e1) ──────┐
                    ├──► MaterializedView(total_orders)
    Event(e2) ──────┘              │
                                   ▼
                         MaterializedView(daily_summary)
```

When a source entity changes, downstream materialized views are invalidated automatically.

```python
# Register materialized view
manager.register(
    key="total_orders",
    compute_fn=lambda deps: sum(d["amount"] for d in deps.values()),
    depends_on=["order_event_1", "order_event_2"],
)

# Automatic invalidation on source change
manager.on_entity_changed("order_event_1")
# → total_orders marked stale
# → Downstream views (daily_summary) also invalidated
```

This enables **provenance-aware queries**:
- `why(x)`: Trace the derivation of a materialized value
- `impact(x)`: Identify all views affected by a source change
