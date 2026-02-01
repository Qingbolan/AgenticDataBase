# AgenticDB

<div align="center">

**The Intelligent DataBase: Let Frontend Focus on Business Logic**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*AgenticDB = Intelligent Backend + Database*

*Frontend describes requirements in natural language, AgenticDB handles storage, queries, and schema evolution automatically.*

</div>

---

## The Problem

**Traditional Architecture:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │ ──► │   Backend   │ ──► │  Database   │
│  Must know: │     │  Must write:│     │ Must design:│
│  - API format│     │  - REST APIs│     │  - Schema   │
│  - Field names│    │  - Business │     │  - Indexes  │
│  - Data struct│    │    logic    │     │  - Migrations│
└─────────────┘     └─────────────┘     └─────────────┘
```

**AgenticDB Solution:**

```
┌─────────────────────────────────────┐
│            Frontend                  │
│                                      │
│  Only cares: User interaction +      │
│              Business logic          │
│                                      │
│  Doesn't need: Fields, schemas, APIs │
│                                      │
│  db.store("user signed up", {...})   │
│  db.query("what did this user buy")  │
│                                      │
└─────────────────────────────────────┘
                    ↕ Natural Language Dialog
┌─────────────────────────────────────┐
│           AgenticDB                  │
│                                      │
│  ✓ Understands semantics             │
│  ✓ Proactively clarifies ambiguity   │
│  ✓ Auto-manages schema & migrations  │
│  ✓ Learns query patterns for speed   │
│  ✓ Validates requests, rejects danger│
│  ✓ Exposes dynamic MCP interface     │
│                                      │
└─────────────────────────────────────┘
```

---

## Quick Start

```python
from agenticdb import AgenticDB

db = AgenticDB()

# Store - Just describe what happened
db.store("user signed up", {"name": "Alice", "email": "alice@example.com"})
# → Auto creates users table, inserts data

db.store("user placed order", {"user": "Alice", "product": "iPhone 15", "price": 999})
# → Auto creates orders table, links to user

# Query - Ask in natural language
result = db.query("what did Alice buy")
# → Returns: [{"product": "iPhone 15", "price": 999}]

result = db.query("total sales this month")
# → Returns: {"answer": "Total sales: $152,000 from 89 orders"}
```

---

## Core Capabilities

### 1. Conversational Interaction

AgenticDB proactively clarifies ambiguous requests:

```python
>>> db.query("show me last month's data")
{
    "needs_clarification": True,
    "question": "Which data do you want to see?",
    "options": ["users", "orders", "products"]
}

>>> db.clarify("orders")
{"data": [...], "summary": "523 orders last month"}
```

Dangerous operations require confirmation:

```python
>>> db.update("delete all orders")
{
    "needs_confirmation": True,
    "affected_rows": 5000,
    "question": "Are you sure you want to delete 5000 records?"
}

>>> db.confirm(yes=True)
{"deleted": 5000}
```

### 2. Automatic Schema Evolution

No manual table creation or alterations:

```python
# First store - auto create table
db.store("user signed up", {"name": "Alice", "email": "a@test.com"})
# → CREATE TABLE users (id, name, email, created_at)

# New field appears - auto add column
db.store("user signed up", {"name": "Bob", "email": "b@test.com", "phone": "138xxx"})
# → ALTER TABLE users ADD COLUMN phone
```

### 3. Query Pattern Caching

Learn repeated patterns, skip LLM for faster execution:

```python
# First query - LLM parses (~500ms)
db.query("show orders from last month")

# Similar query - pattern match (~10ms)
db.query("show orders from last week")
# → Matches pattern "show {entity} from {time}" → skips LLM
```

### 4. Request Validation

Rejects invalid requests with suggestions:

```python
>>> db.store("set price to -100", {"product_id": "p_001"})
{
    "rejected": True,
    "reason": "Price cannot be negative",
    "suggestion": "Did you mean to set a discount?"
}
```

### 5. Dynamic MCP Interface

AgenticDB exposes itself as an MCP (Model Context Protocol) server, allowing external AI applications to interact with it. **The interface dynamically updates based on current database schema.**

```python
# Start MCP server
agenticdb --mcp --port 3000
```

**Dynamic Tool Generation:**

When tables change, MCP tools automatically update:

```
Database State                    Generated MCP Tools
─────────────────────────────────────────────────────────────────
Empty database                 →  [query, store]

After "user signed up"         →  [query, store,
                                   get_users, create_user,
                                   update_user, delete_user]

After "user placed order"      →  [query, store,
                                   get_users, create_user, ...,
                                   get_orders, create_order, ...]
```

**MCP Tool Schema Example:**

```json
{
  "name": "get_users",
  "description": "Query users table. Fields: id, name, email, phone, created_at",
  "inputSchema": {
    "type": "object",
    "properties": {
      "filter": {"type": "string", "description": "Natural language filter, e.g. 'VIP users'"},
      "limit": {"type": "integer", "default": 100}
    }
  }
}
```

**Usage with Claude Desktop:**

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "agenticdb": {
      "command": "agenticdb",
      "args": ["--mcp"],
      "env": {
        "AGENTICDB_PATH": "/path/to/your/database.db"
      }
    }
  }
}
```

Then in Claude:

```
Human: Show me all users who signed up this week
Claude: [Calls get_users tool with filter="signed up this week"]
        Found 15 users who signed up this week...
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AgenticDB                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Pattern Cache                             │    │
│  │  Learned patterns → SQL templates → fast execution           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓ cache miss                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ IntentAgent  │→ │ClarifyAgent  │→ │ValidateAgent │              │
│  │  Parse intent│  │  Clarify     │  │  Validate    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                              ↓                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ SchemaAgent  │→ │ QueryBuilder │→ │  Executor    │              │
│  │ Auto schema  │  │  Text→SQL    │  │ Execute+fmt  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              SQLite / PostgreSQL / MySQL                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      MCP Server                              │    │
│  │  Dynamic tools based on schema: get_X, create_X, update_X   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │           External AI Applications           │
        │  Claude Desktop / Cursor / Custom Agents    │
        └─────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent                   | Responsibility             | Example                      |
| ----------------------- | -------------------------- | ---------------------------- |
| **IntentAgent**   | Parse user intent          | "show orders" → QUERY       |
| **ClarifyAgent**  | Clarify ambiguous requests | "Which time period?"         |
| **ValidateAgent** | Validate reasonableness    | "Price cannot be negative"   |
| **SchemaAgent**   | Manage table structure     | Auto create/alter tables     |
| **QueryBuilder**  | Generate SQL               | Natural language → SQL      |
| **Executor**      | Execute and format         | Return user-friendly results |

### Request Flow

1. **Pattern Cache Check** - If query matches a learned pattern, skip LLM and execute directly
2. **Intent Recognition** - LLM parses user intent (QUERY / STORE / UPDATE / DELETE)
3. **Clarification** - If ambiguous, ask user for clarification
4. **Validation** - Check if request is reasonable, reject dangerous operations
5. **Schema Evolution** - Auto create/alter tables if needed
6. **Query Building** - Generate SQL from natural language
7. **Execution** - Execute SQL, format results, learn new patterns

### Query Pattern Learning

```
User Query                    Learned Pattern              SQL Template
─────────────────────────────────────────────────────────────────────────
"show orders from last month" → "show {entity} from {time}" → SELECT * FROM {table}
                                                               WHERE created_at
                                                               BETWEEN {start} AND {end}

"how many users"              → "how many {entity}"         → SELECT COUNT(*) FROM {table}

"find orders where price > 100" → "find {entity} where {condition}" → SELECT * FROM {table}
                                                                        WHERE {condition}
```

### MCP Dynamic Interface

AgenticDB automatically generates MCP tools based on database schema:

```
Schema Change                  MCP Tools Update
─────────────────────────────────────────────────────────────────────────
CREATE TABLE users          →  + get_users(filter?, limit?)
                               + create_user(name, email, ...)
                               + update_user(id, fields...)
                               + delete_user(id)

ALTER TABLE users           →  Tool schemas update to include
ADD COLUMN phone               new 'phone' field

CREATE TABLE orders         →  + get_orders(filter?, limit?)
                               + create_order(user_id, product, ...)
                               + ...
```

---

## Use Cases

### Rapid Prototyping

```python
# No database design, no API writing, just start
from agenticdb import AgenticDB

db = AgenticDB()

@app.post("/api/action")
async def handle(request):
    return db.execute(request.natural_language_input)
```

### Internal Tools / Admin Dashboards

```python
# Operations staff query directly in natural language
db.query("find VIP users who haven't logged in for 7 days")
db.query("show conversion rate by channel")
db.query("mark these users as high risk")
```

### AI Agent State Storage

```python
# Memory and state management for AI agents
db.store("user said they want to buy a phone", {"user_id": "u123", "intent": "purchase"})
db.store("recommended iPhone 15", {"user_id": "u123", "recommendation": "..."})
db.query("what has this user chatted about before")
```

### MCP Backend for AI Applications

```python
# Run as MCP server for Claude Desktop, Cursor, etc.
agenticdb --mcp --port 3000

# External AI can now:
# - Query any table with natural language
# - Create/update/delete records
# - Get schema-aware tool suggestions
```

### Low-Code / No-Code Platforms

```python
# Business users configure, no developer intervention
db.store("create a new customer", form_data)
db.query("all orders for this customer")
db.update("upgrade customer to VIP", {"customer_id": "..."})
```

---

## Comparison

| Feature                    | AgenticDB | Supabase | Firebase | Traditional Backend |
| -------------------------- | --------- | -------- | -------- | ------------------- |
| Natural language interface | ✅        | ❌       | ❌       | ❌                  |
| Auto schema evolution      | ✅        | ❌       | Partial  | ❌                  |
| Proactive clarification    | ✅        | ❌       | ❌       | ❌                  |
| Query pattern learning     | ✅        | ❌       | ❌       | ❌                  |
| Dynamic MCP interface      | ✅        | ❌       | ❌       | ❌                  |
| Zero frontend config       | ✅        | ❌       | Partial  | ❌                  |

---

## Project Structure

```
agenticdb/
├── core/
│   ├── database.py           # Database connection (SQLite/PostgreSQL)
│   ├── schema.py             # Dynamic schema management
│   ├── types.py              # Core type definitions
│   └── session.py            # Conversation state management
├── agents/
│   ├── base/                 # LLM Agent base class
│   ├── intent.py             # Intent recognition
│   ├── clarify.py            # Ambiguity clarification
│   ├── validate.py           # Request validation
│   ├── query_builder.py      # Text → SQL generation
│   └── schema_evolver.py     # Schema evolution
├── patterns/
│   ├── cache.py              # Query pattern storage
│   ├── matcher.py            # Pattern matching engine
│   └── learner.py            # Auto-learn new patterns
├── executor/
│   ├── engine.py             # SQL execution
│   └── formatter.py          # Result formatting
├── mcp/
│   ├── server.py             # MCP server implementation
│   ├── tools.py              # Dynamic tool generation
│   └── schema_sync.py        # Schema → MCP tool sync
└── prompts/                  # LLM prompts
    ├── intent.md
    ├── clarify.md
    ├── query.md
    └── schema.md
```

---

## Installation

```bash
git clone https://github.com/Qingbolan/AgenticDataBase.git
cd AgenticDataBase
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure LLM
cp .env.example .env
# Add OPENAI_API_KEY to .env
```

## Running

```bash
# Python SDK
from agenticdb import AgenticDB
db = AgenticDB("./data.db")

# MCP Server mode
agenticdb --mcp --port 3000

# With specific database
agenticdb --mcp --db ./myapp.db
```

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## License

MIT

---

<div align="center">

**AgenticDB** — Natural Language In, Intelligent Backend Out

</div>
