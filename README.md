# AgenticDB

<div align="center">

**Intent-Aware Transactions for Agent-Driven Workloads**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Thesis

Traditional database transactions assume complete, deterministic, trusted queries. Agent-driven workloads violate all three assumptions. AgenticDB introduces **Intent-Aware Transactions (IAT)**—a transaction model where incomplete intent is a valid state, binding occurs within the transaction boundary, and safety is enforced before execution.

> **Agent memory remembers what an agent did.**
> **AgenticDB decides whether an agent is *allowed* to do it.**

---

## The Problem

When the caller is a generative agent, four assumptions of conventional databases fail:

| Assumption | Traditional | Agent Workload | Failure Mode |
|------------|-------------|----------------|--------------|
| **Complete Parameters** | All params known at parse time | Params may be unspecified | Guess or fail |
| **Deterministic Queries** | Same input → same plan | Generative, variable | Unpredictable |
| **Trusted Caller** | Backend validates | DB faces agent directly | Safety violations |
| **Stateless Semantics** | Query is self-contained | Requires refinement | Context lost |

**Result**: Traditional databases either execute incorrectly or fail entirely. There is no mechanism to represent "I understood your intent but it's incomplete."

---

## The Contribution: Intent-Aware Transactions

AgenticDB extends transaction semantics with three mechanisms:

### 1. Intent as Intermediate Representation

An **Intent** is a formal transaction object that may contain unbound parameters:

```
Intent := {
    operation   : QUERY | STORE | UPDATE | DELETE
    bindings    : Map<Slot, Value | Pending>
    constraints : Set<SafetyConstraint>
    state       : COMPLETE | PARTIAL | INVALID
}
```

When `state = PARTIAL`, the transaction enters `PENDING_BINDING`—it does not fail.

### 2. In-Transaction Binding

Binding resolution occurs *within* the transaction boundary:

```
RECEIVED → PARSED → [PENDING_BINDING] → BOUND → VALIDATED → EXECUTED → COMPLETED
                          ↑                         ↓
                        bind()              PENDING_CONFIRMATION
                                                    ↓
                                                REJECTED
```

**Key distinction**: Client-side binding has no transaction guarantees. In-transaction binding is commit-protected.

### 3. Pre-Execution Safety

Validation happens **before** execution, not after failure:
- Unsafe operations → `PENDING_CONFIRMATION`
- Invalid operations → `REJECTED`
- Safe, complete operations → `EXECUTED`

---

## Formal Properties

IAT provides three guarantees that traditional transactions cannot:

**Property 1: Binding Monotonicity**
```
Once bound, a slot cannot be unbound within the same transaction.
∀T, s, v: bind(T, s, v) → T.bindings[s] = v at all subsequent states
```

**Property 2: Safety-Preserving Refinement**
```
Clarification cannot introduce safety violations.
initial_valid(I) ∧ I' = refine(I) → final_valid(I')
```

**Property 3: Deterministic Resolution**
```
Same intent + same binding sequence → same final intent.
Enables replay and audit.
```

See **[docs/theory.md](docs/theory.md)** for complete formalization.

---

## System Overview

The core contribution is the **Intent-Aware Transaction Pipeline**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Intent-Aware Transaction Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   RECEIVED ──► PARSED ──► BOUND ──► VALIDATED ──► EXECUTED          │
│                  │          │           │                            │
│                  │ partial  │ unsafe    │ invalid                    │
│                  ▼          ▼           ▼                            │
│            ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│            │ PENDING  │ │ PENDING  │ │ REJECTED │                   │
│            │ BINDING  │ │ CONFIRM  │ │          │                   │
│            └────┬─────┘ └────┬─────┘ └──────────┘                   │
│                 │            │                                       │
│                 │ bind()     │ confirm()                             │
│                 └────────────┴──────────────────► EXECUTED           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: `PENDING_BINDING` and `PENDING_CONFIRMATION` are *transaction states*, not errors. Clarification happens inside the transaction boundary.

See **[docs/architecture.md](docs/architecture.md)** for implementation details (storage, dependency tracking, materialization).

---

## Usage

```python
from agenticdb import AgenticDB

db = AgenticDB()
branch = db.create_branch("main")

# Complete intent → executes directly
result = branch.query("orders for user Alice")

# Partial intent → enters PENDING_BINDING (not failure)
result = branch.query("records from last month")
# → {"status": "pending_binding", "unbound": ["target_entity"], "txn_id": "..."}

# Binding occurs WITHIN the same transaction
result = branch.bind(target_entity="orders")
# → Transaction proceeds to VALIDATED → EXECUTED

# Unsafe operation → enters PENDING_CONFIRMATION (not failure)
result = branch.execute("delete all orders older than 30 days")
# → {"status": "pending_confirmation", "operation": "DELETE", "affected_rows": 1523}

result = branch.confirm()
# → Transaction proceeds to EXECUTED
```

**The key difference**: Traditional databases would either execute the delete immediately or reject the ambiguous query. IAT provides a *transaction-protected* path for clarification and confirmation.

---

## Positioning

| System | Central Question | What It Is |
|--------|-----------------|------------|
| NL2SQL | "Convert question to SQL" | Query translation |
| Semantic Layer | "What does 'revenue' mean?" | Metric definitions |
| Agent Memory | "What did the agent do?" | Experience storage |
| **AgenticDB** | "Is clarification needed before execution?" | **Execution governance** |

> **AgenticDB is a transaction system where clarification is safer than execution.**

Traditional databases assume: if you can parse it, execute it. IAT assumes: if it's ambiguous or unsafe, *ask first*—without losing transaction context.

See **[docs/comparison.md](docs/comparison.md)** for detailed analysis.

---

## Documentation

| Document | Content |
|----------|---------|
| **[docs/theory.md](docs/theory.md)** | Formal IAT model, properties, proofs |
| **[docs/comparison.md](docs/comparison.md)** | vs. NL2SQL, semantic layers, agent memory |
| **[docs/architecture.md](docs/architecture.md)** | System components, layers, implementation |
| **[docs/vision.md](docs/vision.md)** | Killer applications, use cases |

---

## Installation

```bash
git clone https://github.com/Qingbolan/AgenticDataBase.git
cd AgenticDataBase
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Tests

```bash
uv run pytest tests/ -v
```

---

## References

- Gray & Reuter. *Transaction Processing: Concepts and Techniques*. 1992.
- Hellerstein et al. *Architecture of a Database System*. 2007.
- Buneman et al. *Why and Where: A Characterization of Data Provenance*. ICDT 2001.

---

## License

MIT

---

<div align="center">

**AgenticDB** — A transaction model where clarification is safer than execution.

</div>
