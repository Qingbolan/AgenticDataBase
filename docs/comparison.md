# AgenticDB vs. The Alternatives: A Rigorous Comparison

This document provides a systematic comparison between AgenticDB and existing approaches to connecting language models with databases. We argue that these approaches solve fundamentally different problems, and that AgenticDB addresses a gap none of them fill.

---

## Executive Summary

| Approach                 | Core Abstraction   | Central Question             | What It Cannot Do                      |
| ------------------------ | ------------------ | ---------------------------- | -------------------------------------- |
| **NL2SQL**         | Query translation  | "Convert my question to SQL" | Handle ambiguity, ensure safety        |
| **Semantic Layer** | Metric definitions | "What does 'revenue' mean?"  | Dynamic queries, agent workflows       |
| **LLM-as-Parser**  | Intent extraction  | "What does the user want?"   | Transaction state, dependency tracking |
| **Agent Memory**   | Experience storage | "What did the agent do/see?" | Execution control, safety constraints  |
| **AgenticDB**      | Execution control  | "Is this operation allowed?" | High-throughput OLTP                   |

**The fundamental distinction**:

> **Agent memory remembers what an agent did.**
> **AgenticDB decides whether an agent is *allowed* to do it.**

AgenticDB is not a storage system. It is an **execution control plane** for agent operations.

---

## 1. NL2SQL: Query Translation

### What It Is

NL2SQL systems translate natural language questions into SQL queries:

```
Input:  "Show me sales from last month"
Output: SELECT * FROM sales WHERE date >= '2024-01-01' AND date < '2024-02-01'
```

Representative systems: Text2SQL, NSQL, DIN-SQL, C3SQL, DAIL-SQL

### What It Does Well

- Converts well-formed questions to executable SQL
- Handles schema mapping (table/column names)
- Improves with few-shot examples and schema context

### What It Cannot Do

#### 1.1 Handle Ambiguity

```
User: "Show me the data"
NL2SQL: SELECT * FROM ??? WHERE ???

The system must either:
- Guess (dangerous)
- Fail (loses context)
- Ask (but where does clarification state live?)
```

NL2SQL has no mechanism to represent "I parsed your intent but it's incomplete."

#### 1.2 Maintain Transaction State

```
User: "Show me sales"
NL2SQL: SELECT * FROM sales
User: "Filter by region"
NL2SQL: SELECT * FROM sales WHERE region = ???

The second query has no connection to the first.
NL2SQL is stateless by design.
```

#### 1.3 Validate Safety

```
User: "Delete all orders"
NL2SQL: DELETE FROM orders  -- Executes immediately

NL2SQL translates, it doesn't validate.
Safety must be bolted on externally.
```

#### 1.4 Track Provenance

```
User: "What's the total revenue?"
NL2SQL: SELECT SUM(amount) FROM orders

Q: "Why is this number $1.2M?"
A: ¯\_(ツ)_/¯
```

NL2SQL produces answers, not explanations.

### AgenticDB Difference

| Aspect                | NL2SQL        | AgenticDB                        |
| --------------------- | ------------- | -------------------------------- |
| Ambiguous input       | Fail or guess | `PENDING_BINDING` state        |
| Multi-turn refinement | Stateless     | Transaction-protected            |
| Safety validation     | External      | Built into transaction lifecycle |
| Provenance            | None          | Native `why(x)` query          |

---

## 2. Semantic Layer: Metric Definitions

### What It Is

Semantic layers define business metrics in a declarative way, providing a single source of truth for "what does X mean?":

```yaml
metrics:
  - name: revenue
    sql: SUM(orders.amount)
    filters:
      - status = 'completed'
    dimensions:
      - region
      - product_category
```

Representative systems: dbt Semantic Layer, Cube.js, Looker, Metriql, AtScale

### What It Does Well

- Defines canonical metric calculations
- Ensures consistency across reports
- Supports dimension slicing

### What It Cannot Do

#### 2.1 Handle Dynamic Queries

```
Agent: "Find users who did X, then Y, within 24 hours"

Semantic layers define metrics, not ad-hoc query patterns.
This requires composing entities that aren't predefined.
```

#### 2.2 Support Schema Evolution

```
Day 1: orders(id, amount, user_id)
Day 2: orders(id, amount, user_id, currency)

Semantic layer: Requires manual YAML update
AgenticDB: Schema evolves from workload automatically
```

#### 2.3 Track Agent Behavior

```
Agent makes a decision based on risk_score.
risk_score later found to be wrong.

Semantic layer: No concept of "decisions" or "dependencies"
Cannot answer: "What decisions used the wrong risk_score?"
```

### AgenticDB Difference

| Aspect         | Semantic Layer     | AgenticDB                        |
| -------------- | ------------------ | -------------------------------- |
| Schema         | Fixed, declarative | Workload-driven, emergent        |
| Query types    | Predefined metrics | Arbitrary intent                 |
| Entity types   | Tables/columns     | Event, Claim, Action             |
| Causal queries | No                 | Native `why(x)`, `impact(x)` |

---

## 3. LLM-as-Parser: Intent Extraction

### What It Is

Using an LLM to extract structured intent from natural language, then executing against the database:

```python
# LLM extracts intent
intent = llm.parse("Show me Alice's orders")
# → {"action": "query", "entity": "orders", "filter": {"user": "Alice"}}

# Application code executes
result = db.query(intent["entity"]).filter(intent["filter"])
```

Representative patterns: LangChain SQL Agent, LlamaIndex Query Engine, function calling

### What It Does Well

- Extracts structured data from unstructured input
- Flexible intent representation
- Easy to integrate with existing databases

### What It Cannot Do

#### 3.1 Handle Incomplete Intent

```python
intent = llm.parse("Show me the data")
# → {"action": "query", "entity": ???, "filter": ???}

# Application must handle this somehow.
# But the LLM has no memory of this failed parse.
# Re-prompting loses context.
```

LLM-as-parser is a single-shot operation. It has no concept of "pending" intent.

#### 3.2 Provide Transaction Guarantees

```python
intent1 = llm.parse("Start transferring $100 from A to B")
# Application begins transaction

intent2 = llm.parse("Actually, make it $200")
# Is this part of the same transaction?
# The LLM doesn't know. The application must track this.
```

The LLM is stateless. Transaction state must be managed externally.

#### 3.3 Guarantee Monotonic Refinement

```python
intent1 = llm.parse("Show me sales")
# → {"entity": "sales"}

intent2 = llm.parse("Show me sales from last month")
# → {"entity": "sales", "filter": {"date": "last_month"}}

# But what if the LLM produces:
intent2 = llm.parse("Show me sales from last month")
# → {"entity": "orders", "filter": {"date": "last_month"}}
# The entity changed! This violates monotonicity.
```

LLM outputs are not guaranteed to refine consistently.

#### 3.4 Support Native Dependency Tracking

```python
# Agent makes decision based on LLM-parsed intent
decision = agent.decide(intent)

# Later: "Why did the agent make this decision?"
# The intent was ephemeral. No causal record exists.
```

### AgenticDB Difference

| Aspect                | LLM-as-Parser       | AgenticDB                   |
| --------------------- | ------------------- | --------------------------- |
| Intent state          | Ephemeral           | Transaction-protected       |
| Multi-turn refinement | Application-managed | Native `bind()` operation |
| Binding monotonicity  | Not guaranteed      | Formally guaranteed         |
| Causal tracking       | None                | Native dependency graph     |

---

## 4. Agent Memory: Experience Storage

### What It Is

Agent memory systems store and retrieve agent experiences, enabling long-term context and learning:

```python
# Store a memory
memory.add("User prefers email communication")
memory.add("User purchased iPhone last month")

# Retrieve relevant memories
context = memory.retrieve("What does the user want?")
# → Returns semantically similar memories
```

Representative systems: MemGPT, LangChain Memory, Zep, Mem0, Letta

### What It Does Well

- Stores agent experiences persistently
- Retrieves relevant context via semantic search
- Compresses and summarizes long conversations
- Maintains conversation history

### What It Cannot Do

#### 4.1 Control Execution

```python
# Agent memory stores what happened
memory.add("Agent transferred $10,000 to account X")

# But it cannot:
# - Prevent the transfer if it was unauthorized
# - Require confirmation before execution
# - Validate that the operation is safe
```

Agent memory is **post-hoc**. It records what happened. It doesn't control whether something *should* happen.

#### 4.2 Enforce Safety Constraints

```python
# Memory stores experiences, not constraints
memory.add("User has $500 balance")
memory.add("User requested $1000 withdrawal")

# Agent memory cannot:
# - Block the withdrawal
# - Require confirmation
# - Enter a "pending" state

# It just remembers what was requested
```

#### 4.3 Track Causal Dependencies

```python
# Memory stores items, not relationships
memory.add("Risk score is 0.3")
memory.add("User was approved")

# Agent memory cannot answer:
# - "Was the approval BECAUSE of the risk score?"
# - "If risk score changes, is approval still valid?"
# - "What else depends on this risk score?"
```

Agent memory stores **facts**, not **causal relationships**.

#### 4.4 Handle Incomplete Intent at Execution Time

```python
# Agent wants to "transfer money"
# But hasn't specified: to whom? how much?

# Agent memory doesn't help here:
# - It can retrieve past transfers
# - It cannot represent "pending transfer with missing params"
# - It has no concept of "bind the missing parameter"
```

### The Core Distinction

| Aspect                         | Agent Memory             | AgenticDB           |
| ------------------------------ | ------------------------ | ------------------- |
| **Central object**       | Memory item / experience | Operation / intent  |
| **Central question**     | "What do you remember?"  | "Can you do this?"  |
| **Uncertainty handling** | At retrieval time        | At execution time   |
| **Core value**           | Data structure           | Execution semantics |
| **Safety**               | Not addressed            | First-class concern |
| **Dependencies**         | Not tracked              | Native graph        |

### Why This Matters

Agent memory and AgenticDB solve **orthogonal problems**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Agent Memory                          AgenticDB                       │
│   ────────────                          ─────────                       │
│                                                                         │
│   "Remember that the user                "Before executing this         │
│    prefers email"                         operation, verify:            │
│                                           - Is intent complete?         │
│   "Recall what happened                   - Is it safe?                 │
│    last session"                          - Does user confirm?"         │
│                                                                         │
│   ↓                                       ↓                             │
│   Retrieval problem                      Execution control problem      │
│                                                                         │
│   ↓                                       ↓                             │
│   Solved by: embeddings,                 Solved by: transaction states, │
│   vector search, summarization           binding, validation, audit     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**You might use both**:

- Agent memory to give the agent context about past interactions
- AgenticDB to control what the agent is allowed to do with that context

---

## 5. Problem Space Comparison

The fundamental issue is that these approaches solve different problems:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Problem Space                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NL2SQL                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ "I have a well-formed question, translate it to SQL"             │   │
│  │                                                                  │   │
│  │ Assumption: User knows exactly what they want                    │   │
│  │ Assumption: Single-turn interaction                              │   │
│  │ Assumption: Schema is stable                                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Semantic Layer                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ "Define business metrics so everyone uses the same calculation"  │   │
│  │                                                                  │   │
│  │ Assumption: Metrics are predefined                               │   │
│  │ Assumption: Human analysts are the primary users                 │   │
│  │ Assumption: Schema is designed by humans                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LLM-as-Parser                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ "Extract structured intent from natural language"                │   │
│  │                                                                  │   │
│  │ Assumption: Intent can be extracted in one shot                  │   │
│  │ Assumption: Application handles state management                 │   │
│  │ Assumption: No need for causal tracking                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  AgenticDB                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ "Execute safely when intent is incomplete, generative, unsafe"   │   │
│  │                                                                  │   │
│  │ Assumption: Caller may not know exactly what they want           │   │
│  │ Assumption: Multi-turn refinement is the norm                    │   │
│  │ Assumption: Schema emerges from workload                         │   │
│  │ Assumption: Causal tracking is required                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. When to Use What

### Use NL2SQL When:

- Users ask well-formed questions
- Schema is stable and well-documented
- Single-turn interaction is sufficient
- You need fast query translation

### Use Semantic Layer When:

- Business metrics need canonical definitions
- Multiple teams need consistent calculations
- Human analysts are the primary users
- Governance and auditability of metrics is important

### Use LLM-as-Parser When:

- You need to extract structured data from text
- Application can manage state externally
- Causal tracking is not required
- You're building a traditional application with LLM features

### Use Agent Memory When:

- Agent needs to remember past interactions
- Semantic retrieval of context is sufficient
- You don't need execution control
- Safety is handled elsewhere in your stack

### Use AgenticDB When:

- Agent operations must be **controlled, not just recorded**
- Intent may be incomplete and require in-transaction refinement
- Safety validation is required **before** execution
- You need causal tracking: "why did this happen?" and "what breaks if this changes?"
- Operations must be auditable and replayable

---

## 7. Composition, Not Competition

AgenticDB is designed to compose with existing systems:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Composition Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Agent/LLM   │───►│  AgenticDB   │───►│  Underlying Database     │   │
│  │              │    │              │    │  (PostgreSQL, SQLite)    │   │
│  │  Generative  │    │  Intent-Aware│    │                          │   │
│  │  Caller      │    │  Transaction │    │  + Semantic Layer        │   │
│  │              │    │  Processing  │    │  + NL2SQL (optional)     │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│                      ┌──────────────┐                                   │
│                      │  Dependency  │                                   │
│                      │  Graph       │                                   │
│                      │  why(x)      │                                   │
│                      │  impact(x)   │                                   │
│                      └──────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

- AgenticDB can use **NL2SQL** internally for query generation
- AgenticDB can sit in front of databases with **semantic layers**
- AgenticDB replaces **LLM-as-parser** with transaction-protected intent

The unique value is the transaction model and dependency tracking, not the LLM integration.

---

## 8. Summary Table

| Capability                              | NL2SQL | Semantic Layer | LLM-as-Parser | Agent Memory | AgenticDB |
| --------------------------------------- | ------ | -------------- | ------------- | ------------ | --------- |
| Query translation                       | ✓     | Partial        | Via LLM       | ✗           | Via LLM   |
| Metric definitions                      | ✗     | ✓             | ✗            | ✗           | ✗        |
| Intent extraction                       | ✗     | ✗             | ✓            | ✗           | ✓        |
| Experience storage                      | ✗     | ✗             | ✗            | ✓           | ✓        |
| Semantic retrieval                      | ✗     | ✗             | ✗            | ✓           | Partial   |
| **Incomplete intent handling**    | ✗     | ✗             | ✗            | ✗           | ✓        |
| **Transaction-protected binding** | ✗     | ✗             | ✗            | ✗           | ✓        |
| **Pre-execution safety**          | ✗     | ✗             | ✗            | ✗           | ✓        |
| **Dependency tracking**           | ✗     | ✗             | ✗            | ✗           | ✓        |
| **Causal queries (why/impact)**   | ✗     | ✗             | ✗            | ✗           | ✓        |
| **Execution control**             | ✗     | ✗             | ✗            | ✗           | ✓        |

---

## 9. The Theoretical Distinction

The deepest distinction is at the transaction theory level:

**NL2SQL, Semantic Layer, LLM-as-Parser**: These are all **pre-transaction** technologies. They help you formulate a query *before* it enters the database.

**AgenticDB**: Introduces **intra-transaction** uncertainty handling. Incomplete intent is a *valid transaction state*, not an error.

This is analogous to the distinction between:

- **Compiled languages**: All types must be known at compile time
- **Gradual typing**: Types can be partially known, refined incrementally

AgenticDB brings gradual typing to database transactions.

---

## References

1. Katsogiannis-Meimarakis, G., & Koutrika, G. (2023). A Survey on Deep Learning Approaches for Text-to-SQL. *VLDB Journal*.
2. Pourreza, M., & Rafiei, D. (2023). DIN-SQL: Decomposed In-Context Learning of Text-to-SQL. *EMNLP*.
3. Looker. (2021). *The Semantic Layer: A Definitive Guide*.
4. dbt Labs. (2023). *dbt Semantic Layer Documentation*.
5. LangChain. (2023). *SQL Database Agent Documentation*.
