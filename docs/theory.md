# Intent-Aware Transaction Model: Formal Foundations

This document provides the formal theoretical foundation for AgenticDB's transaction model. It defines the key abstractions, properties, and guarantees that distinguish Intent-Aware Transactions (IAT) from traditional database transactions.

---

## 1. Motivation: Why Traditional Transactions Fail

Traditional database transactions assume:

1. **Parameter Completeness**: All query parameters are known at parse time
2. **Deterministic Execution**: Same query always produces same execution plan
3. **Trusted Caller**: The client is a program that won't submit invalid requests
4. **Atomic Intent**: A single request maps to a single intent

Agent-driven workloads violate all four assumptions:

| Assumption | Traditional | Agent Workload | Failure Mode |
|------------|-------------|----------------|--------------|
| Parameter Completeness | ✓ | ✗ | Query fails or executes wrong |
| Deterministic Execution | ✓ | ✗ | Unpredictable behavior |
| Trusted Caller | ✓ | ✗ | Safety violations |
| Atomic Intent | ✓ | ✗ | Semantic mismatch |

**The core problem**: Traditional transactions have no mechanism to represent, refine, or validate *incomplete intent*. They either execute (possibly incorrectly) or fail (losing context).

---

## 2. Intent-Aware Transaction Model

### 2.1 Intent Definition

An **Intent** is a formal intermediate representation that captures both the desired operation and its binding state:

```
Intent I := {
    id          : TransactionId
    operation   : Op ∈ {QUERY, STORE, UPDATE, DELETE}
    target      : EntityRef | ⊥ (unbound)
    predicates  : List<Predicate>
    bindings    : Map<Slot, Value | ⊥>
    constraints : Set<Constraint>
    state       : State ∈ {PARTIAL, COMPLETE, INVALID}
}

where:
    Slot      := (name: String, type: SlotType, required: Bool)
    SlotType  := ENTITY | TEMPORAL | NUMERIC | STRING | BOOLEAN
    Predicate := (slot: Slot, op: CompareOp, value: Value | ⊥)
    Constraint := SafetyConstraint | SchemaConstraint | BusinessConstraint
```

### 2.2 Intent State

```
state(I) =
    INVALID   if ∃c ∈ I.constraints : ¬satisfiable(c, I.bindings)
    PARTIAL   if ∃s ∈ required_slots(I) : I.bindings[s] = ⊥
    COMPLETE  otherwise
```

### 2.3 Binding Operation

The **bind** operation resolves an unbound slot:

```
bind : Intent × Slot × Value → Intent

bind(I, s, v) = I' where:
    I'.bindings = I.bindings[s ↦ v]
    I'.state = recompute_state(I')
    I'.id = I.id  (same transaction)
```

**Critical Property**: Binding occurs *within* the transaction boundary. The transaction `I.id` persists across binding operations.

---

## 3. Transaction Lifecycle

### 3.1 State Machine

```
                    ┌────────────────────────────────────┐
                    │                                    │
                    ▼                                    │
RECEIVED ──► PARSED ──► BOUND ──► VALIDATED ──► EXECUTED ──► COMPLETED
               │          │           │
               │          │           └──► REJECTED
               │          │
               │          └──► PENDING_CONFIRMATION
               │
               └──► PENDING_BINDING
                          │
                          │ bind(s, v)
                          └──────────────────────────────┘
```

### 3.2 Formal State Definitions

```
TransactionState := {
    RECEIVED            : Intent received, not yet parsed
    PARSED              : Intent parsed, bindings extracted
    PENDING_BINDING     : state(I) = PARTIAL, awaiting bind()
    BOUND               : state(I) = COMPLETE
    PENDING_CONFIRMATION: Destructive operation, awaiting confirm()
    VALIDATED           : All constraints satisfied
    REJECTED            : state(I) = INVALID or constraint violation
    EXECUTED            : Query/mutation executed
    COMPLETED           : Transaction committed
    FAILED              : Execution error
}
```

### 3.3 Transition Rules

```
RECEIVED → PARSED           : parse(input) succeeds
PARSED → PENDING_BINDING    : state(I) = PARTIAL
PARSED → BOUND              : state(I) = COMPLETE
PENDING_BINDING → BOUND     : bind(s, v) makes state(I) = COMPLETE
BOUND → PENDING_CONFIRMATION: is_destructive(I) ∧ requires_confirmation(I)
BOUND → VALIDATED           : ∀c ∈ I.constraints : satisfied(c, I)
BOUND → REJECTED            : ∃c ∈ I.constraints : ¬satisfied(c, I)
VALIDATED → EXECUTED        : execute(I) succeeds
VALIDATED → FAILED          : execute(I) fails
EXECUTED → COMPLETED        : commit succeeds
```

---

## 4. Formal Properties

AgenticDB guarantees three critical properties that traditional databases cannot provide for agent workloads:

### 4.1 Binding Monotonicity

> Once a slot is bound within a transaction, it cannot be unbound.

```
Property (Binding Monotonicity):
    ∀I, s, v, v' :
        bind(I, s, v) = I' ∧ I'.bindings[s] = v
        → ∀I'' reachable from I' : I''.bindings[s] = v

Proof sketch:
    The bind operation only adds mappings to I.bindings.
    No operation removes or modifies existing bindings.
    Transaction identity I.id is preserved across bindings.
    □
```

**Why this matters**: Client-side binding has no such guarantee. A session variable can be modified at any time. In-transaction binding is commit-protected.

### 4.2 Safety-Preserving Refinement

> If a partial intent passes initial validation, any valid completion also passes validation.

```
Property (Safety-Preserving Refinement):
    ∀I, I' :
        state(I) = PARTIAL ∧
        initial_validation(I) = PASS ∧
        I' = complete(I)  (I' is a valid completion of I)
        → full_validation(I') = PASS

Formal definition of complete():
    complete(I) = I' where:
        ∀s ∈ required_slots(I) : I'.bindings[s] ≠ ⊥
        ∀s : I.bindings[s] ≠ ⊥ → I'.bindings[s] = I.bindings[s]
```

**Why this matters**: Agent clarification cannot introduce safety violations. The system guarantees that refining an intent only narrows the possible outcomes, never expands them to include unsafe operations.

### 4.3 Deterministic Intent Resolution

> Given the same partial intent and the same binding sequence, the system produces the same final intent.

```
Property (Deterministic Resolution):
    ∀I, [(s₁,v₁), (s₂,v₂), ..., (sₙ,vₙ)] :
        bind(bind(...bind(I, s₁, v₁), s₂, v₂)..., sₙ, vₙ) = I'
        is deterministic

Corollary:
    Intent resolution is replayable. Given a transaction log,
    the system can reconstruct the exact intent that was executed.
```

**Why this matters**: Auditability. You can always answer "what did the agent actually ask for?" by replaying the binding sequence.

---

## 5. Comparison with Traditional Isolation Levels

Intent-Aware Transactions introduce a new dimension orthogonal to traditional isolation:

| Property | READ COMMITTED | SERIALIZABLE | IAT |
|----------|---------------|--------------|-----|
| Phantom reads | Possible | Prevented | Prevented |
| Non-repeatable reads | Possible | Prevented | Prevented |
| Incomplete parameters | Error | Error | **PENDING_BINDING** |
| Unsafe operations | Execute/Error | Execute/Error | **PENDING_CONFIRMATION** |
| Intent auditability | No | No | **Yes** |

### 5.1 Interaction with Isolation Levels

IAT is composable with standard isolation levels:

```
IAT + READ COMMITTED:
    Binding resolution sees committed state at each bind() call
    Final execution uses READ COMMITTED semantics

IAT + SERIALIZABLE:
    Binding resolution is serialized with other transactions
    Final execution uses SERIALIZABLE semantics
```

---

## 6. Dependency Graph Formalization

### 6.1 Entity Model

AgenticDB stores three entity types with distinct semantics:

```
Entity := Event | Claim | Action

Event := {
    id: EntityId
    type: EventType
    data: JSON
    timestamp: Timestamp
    immutable: true
}

Claim := {
    id: EntityId
    subject: String
    predicate: String
    value: Any
    source: String
    confidence: [0, 1]
    derived_from: Set<EntityId>
    superseded_by: EntityId | ⊥
}

Action := {
    id: EntityId
    type: ActionType
    agent_id: AgentId
    inputs: JSON
    outputs: JSON
    depends_on: Set<EntityId>
    produces: Set<EntityId>
    invalidates: Set<EntityId>
}
```

### 6.2 Dependency Graph

```
DependencyGraph G := (V, E) where:
    V = Set<EntityId>
    E ⊆ V × V × EdgeType
    EdgeType := DEPENDS_ON | PRODUCES | INVALIDATES | DERIVED_FROM | SUPERSEDES
```

### 6.3 Causal Queries

```
why(x) := transitive_closure(G, x, direction=UPSTREAM)
    Returns all entities that x transitively depends on

impact(x) := transitive_closure(G, x, direction=DOWNSTREAM)
    Returns all entities that transitively depend on x
```

### 6.4 Automatic Invalidation

```
invalidate(x) :=
    for each y ∈ impact(x):
        if y is Claim:
            y.status ← INVALIDATED
        if y is MaterializedView:
            y.is_stale ← true

    propagate(invalidate, impact(x))
```

---

## 7. Materialization Semantics

### 7.1 Materialized View Definition

```
MaterializedView := {
    key: String
    compute_fn: (Map<EntityId, Value>) → Value
    depends_on: Set<EntityId>
    value: Value | ⊥
    is_stale: Boolean
    computed_at_version: Version
}
```

### 7.2 Recomputation Modes

```
LAZY:
    Access to stale view triggers recomputation
    get(view) := if view.is_stale then recompute(view) else view.value

EAGER:
    Invalidation triggers immediate recomputation
    on_invalidate(view) := recompute(view)
```

### 7.3 Consistency Guarantees

```
Property (View Consistency):
    ∀ view V, ∀ version n:
        V.computed_at_version = n
        → V.value = V.compute_fn(snapshot(V.depends_on, n))

Property (Transitive Invalidation):
    ∀ entity e, ∀ view V:
        e ∈ transitive_deps(V) ∧ changed(e)
        → V.is_stale = true
```

---

## 8. Theoretical Contributions

AgenticDB makes three contributions to database theory:

### 8.1 Intent as Transaction Primitive

Traditional transactions operate on complete, validated queries. IAT introduces **partial intents** as first-class transaction objects, with well-defined state, transitions, and properties.

### 8.2 Binding as Transaction Operation

Traditional binding happens client-side before transaction begins. IAT elevates binding to a **transaction-protected operation** with monotonicity and safety guarantees.

### 8.3 Causal Dependency as Query Primitive

Traditional databases answer "what is the data?" AgenticDB answers **"how did the data become this way?"** through native causal queries (why, impact) over a first-class dependency graph.

---

## References

1. Gray, J., & Reuter, A. (1992). *Transaction Processing: Concepts and Techniques*. Morgan Kaufmann.
2. Hellerstein, J. M., Stonebraker, M., & Hamilton, J. (2007). Architecture of a Database System. *Foundations and Trends in Databases*, 1(2).
3. Abiteboul, S., Hull, R., & Vianu, V. (1995). *Foundations of Databases*. Addison-Wesley.
4. Buneman, P., Khanna, S., & Tan, W. C. (2001). Why and Where: A Characterization of Data Provenance. *ICDT*.
5. Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance Semirings. *PODS*.
