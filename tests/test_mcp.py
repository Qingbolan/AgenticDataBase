"""
Tests for AgenticDB MCP (Model Context Protocol) Server.

This module tests the MCP server that exposes AgenticDB to external AI
applications like Claude Desktop, enabling dynamic tool generation based
on the current database schema.

Test Coverage:
    - MCPServer initialization and lifecycle
    - Dynamic tool generation from schema
    - Tool execution (get, create, update, delete)
    - Schema change triggers tool update
    - Natural language query/store tools
    - Tool schema validation
    - Error handling
"""

import pytest
import json
from typing import Any
from unittest.mock import Mock, AsyncMock, patch

from agenticdb.interface.client import AgenticDB
from agenticdb.core.models import Event, Claim, Action, EntityType


class TestMCPToolSchema:
    """Tests for MCP tool schema generation."""

    def test_create_base_tools(self):
        """Should create base query and store tools."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        generator = MCPToolGenerator(db)

        tools = generator.get_tools()
        tool_names = [t["name"] for t in tools]

        # Base tools should always exist
        assert "query" in tool_names
        assert "store" in tool_names

    def test_query_tool_schema(self):
        """Should have correct schema for query tool."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        generator = MCPToolGenerator(db)

        tools = generator.get_tools()
        query_tool = next(t for t in tools if t["name"] == "query")

        assert query_tool["description"] is not None
        assert "inputSchema" in query_tool
        assert query_tool["inputSchema"]["type"] == "object"
        assert "natural_language_query" in query_tool["inputSchema"]["properties"]

    def test_store_tool_schema(self):
        """Should have correct schema for store tool."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        generator = MCPToolGenerator(db)

        tools = generator.get_tools()
        store_tool = next(t for t in tools if t["name"] == "store")

        assert store_tool["description"] is not None
        assert "inputSchema" in store_tool
        props = store_tool["inputSchema"]["properties"]
        assert "description" in props
        assert "data" in props


class TestDynamicToolGeneration:
    """Tests for dynamic tool generation based on entity types."""

    def test_generate_event_tools(self):
        """Should generate tools for events when events exist."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        branch = db.create_branch("test")

        # Record an event to establish the entity type
        branch.record(Event(
            event_type="UserRegistered",
            data={"user_id": "u123", "email": "test@example.com"}
        ))

        generator = MCPToolGenerator(db, branch_id=branch.id)
        tools = generator.get_tools()
        tool_names = [t["name"] for t in tools]

        # Should have event-related tools
        assert "get_events" in tool_names
        assert "get_event" in tool_names

    def test_generate_claim_tools(self):
        """Should generate tools for claims when claims exist."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        branch = db.create_branch("test")

        # Record a claim
        branch.record(Claim(
            subject="user.risk_score",
            predicate="equals",
            value=0.5,
            source="risk_model"
        ))

        generator = MCPToolGenerator(db, branch_id=branch.id)
        tools = generator.get_tools()
        tool_names = [t["name"] for t in tools]

        # Should have claim-related tools
        assert "get_claims" in tool_names
        assert "get_claim" in tool_names
        assert "create_claim" in tool_names

    def test_generate_action_tools(self):
        """Should generate tools for actions when actions exist."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        branch = db.create_branch("test")

        # Execute an action
        branch.execute(Action(
            action_type="ApproveUser",
            agent_id="test-agent",
            inputs={"user_id": "u123"},
            depends_on=[]
        ))

        generator = MCPToolGenerator(db, branch_id=branch.id)
        tools = generator.get_tools()
        tool_names = [t["name"] for t in tools]

        # Should have action-related tools
        assert "get_actions" in tool_names
        assert "get_action" in tool_names

    def test_tools_update_on_schema_change(self):
        """Should update tools when new entity types are added."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        branch = db.create_branch("test")

        generator = MCPToolGenerator(db, branch_id=branch.id)

        # Initially just base tools
        initial_tools = generator.get_tools()
        initial_names = [t["name"] for t in initial_tools]
        assert "get_events" not in initial_names

        # Add an event
        branch.record(Event(
            event_type="OrderPlaced",
            data={"order_id": "o123"}
        ))

        # Refresh tools
        generator.refresh()
        updated_tools = generator.get_tools()
        updated_names = [t["name"] for t in updated_tools]

        # Now should have event tools
        assert "get_events" in updated_names


class TestMCPToolExecution:
    """Tests for executing MCP tools."""

    def test_execute_query_tool(self):
        """Should execute natural language query."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        # Add some data
        branch.record(Event(
            event_type="UserRegistered",
            data={"user_id": "u123", "name": "Alice"}
        ))

        server = MCPServer(db, branch_id=branch.id)

        # Execute query tool
        result = server.execute_tool("query", {
            "natural_language_query": "show all events"
        })

        assert result is not None
        assert "error" not in result or result.get("error") is None

    def test_execute_store_tool(self):
        """Should execute store tool to record an event."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")
        server = MCPServer(db, branch_id=branch.id)

        # Execute store tool
        result = server.execute_tool("store", {
            "description": "user registered",
            "data": {"user_id": "u456", "email": "bob@example.com"}
        })

        assert result is not None
        assert "entity_id" in result

    def test_execute_get_events_tool(self):
        """Should execute get_events tool."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        # Add events
        branch.record(Event(event_type="EventA", data={"id": 1}))
        branch.record(Event(event_type="EventB", data={"id": 2}))

        server = MCPServer(db, branch_id=branch.id)

        # Execute get_events
        result = server.execute_tool("get_events", {})

        assert result is not None
        assert "events" in result
        assert len(result["events"]) == 2

    def test_execute_get_event_by_id(self):
        """Should get a specific event by ID."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        event = branch.record(Event(
            event_type="TestEvent",
            data={"value": 42}
        ))

        server = MCPServer(db, branch_id=branch.id)

        result = server.execute_tool("get_event", {
            "entity_id": event.id
        })

        assert result is not None
        assert result.get("event_type") == "TestEvent"

    def test_execute_create_claim_tool(self):
        """Should create a claim via tool."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")
        server = MCPServer(db, branch_id=branch.id)

        result = server.execute_tool("create_claim", {
            "subject": "user.status",
            "predicate": "equals",
            "value": "active",
            "source": "status_checker"
        })

        assert result is not None
        assert "entity_id" in result

    def test_execute_get_claims_with_filter(self):
        """Should get claims with filter."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        branch.record(Claim(
            subject="user.risk",
            predicate="equals",
            value=0.3,
            source="model_a"
        ))
        branch.record(Claim(
            subject="user.score",
            predicate="equals",
            value=100,
            source="model_b"
        ))

        server = MCPServer(db, branch_id=branch.id)

        result = server.execute_tool("get_claims", {
            "subject": "user.risk"
        })

        assert result is not None
        assert "claims" in result
        assert len(result["claims"]) == 1

    def test_unknown_tool_error(self):
        """Should return error for unknown tool."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        server = MCPServer(db)

        result = server.execute_tool("nonexistent_tool", {})

        assert result is not None
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestMCPCausalQueries:
    """Tests for causal query tools in MCP."""

    def test_why_tool(self):
        """Should provide why() query via MCP tool."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        # Create a dependency chain
        event = branch.record(Event(
            event_type="UserSignup",
            data={"user_id": "u123"}
        ))

        claim = branch.record(Claim(
            subject="user.risk",
            predicate="equals",
            value=0.2,
            source="risk_model",
            derived_from=[event.id]
        ))

        action = branch.execute(Action(
            action_type="Approve",
            agent_id="agent-1",
            depends_on=[claim.id]
        ))

        server = MCPServer(db, branch_id=branch.id)

        # Query why
        result = server.execute_tool("why", {
            "entity_id": action.id
        })

        assert result is not None
        assert "causal_chain" in result

    def test_impact_tool(self):
        """Should provide impact() query via MCP tool."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        # Create a dependency chain
        event = branch.record(Event(
            event_type="DataChange",
            data={"field": "price"}
        ))

        claim = branch.record(Claim(
            subject="product.value",
            predicate="equals",
            value=100,
            source="calculator",
            derived_from=[event.id]
        ))

        server = MCPServer(db, branch_id=branch.id)

        # Query impact
        result = server.execute_tool("impact", {
            "entity_id": event.id
        })

        assert result is not None
        # ImpactResult has affected_events, affected_claims, affected_actions
        assert "affected_events" in result or "total_affected" in result


class TestMCPServerLifecycle:
    """Tests for MCP server lifecycle management."""

    def test_server_initialization(self):
        """Should initialize MCP server with AgenticDB."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        server = MCPServer(db)

        assert server is not None
        assert server.db is db

    def test_server_with_specific_branch(self):
        """Should initialize server with specific branch."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("custom-branch")

        server = MCPServer(db, branch_id=branch.id)

        assert server.branch_id == branch.id

    def test_list_tools(self):
        """Should list all available tools."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        server = MCPServer(db)

        tools = server.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all("name" in t for t in tools)
        assert all("description" in t for t in tools)

    def test_get_tool_schema(self):
        """Should get schema for a specific tool."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        server = MCPServer(db)

        schema = server.get_tool_schema("query")

        assert schema is not None
        assert "inputSchema" in schema

    def test_server_handles_concurrent_requests(self):
        """Should handle concurrent tool executions."""
        from agenticdb.mcp.server import MCPServer
        from concurrent.futures import ThreadPoolExecutor

        db = AgenticDB()
        branch = db.create_branch("concurrent-test")
        server = MCPServer(db, branch_id=branch.id)

        # Add initial data
        branch.record(Event(event_type="Init", data={}))

        errors = []
        results = []

        def execute_query():
            try:
                result = server.execute_tool("get_events", {})
                results.append(result)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_query) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(results) == 20


class TestMCPProtocolCompliance:
    """Tests for MCP protocol compliance."""

    def test_tool_schema_follows_json_schema(self):
        """Tool schemas should follow JSON Schema format."""
        from agenticdb.mcp.tools import MCPToolGenerator

        db = AgenticDB()
        generator = MCPToolGenerator(db)

        tools = generator.get_tools()

        for tool in tools:
            # Required fields
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

            # Schema should be valid JSON Schema
            schema = tool["inputSchema"]
            assert schema.get("type") == "object"
            assert "properties" in schema

    def test_tool_result_format(self):
        """Tool results should follow MCP format."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")
        server = MCPServer(db, branch_id=branch.id)

        # Execute a tool
        result = server.execute_tool("query", {
            "natural_language_query": "list events"
        })

        # Result should be JSON serializable
        assert result is not None
        json_str = json.dumps(result)
        assert json_str is not None

    def test_error_response_format(self):
        """Error responses should follow MCP format."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        server = MCPServer(db)

        # Execute invalid tool
        result = server.execute_tool("invalid_tool", {})

        assert "error" in result
        assert isinstance(result["error"], str)


class TestMCPTimeTravelTools:
    """Tests for time-travel tools in MCP."""

    def test_snapshot_tool(self):
        """Should get snapshot at specific version."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        # Create some history
        branch.record(Event(event_type="Event1", data={}))
        branch.record(Event(event_type="Event2", data={}))

        server = MCPServer(db, branch_id=branch.id)

        result = server.execute_tool("snapshot", {
            "version": 1
        })

        assert result is not None
        assert "snapshot" in result or "entities" in result

    def test_history_tool(self):
        """Should get history of an entity."""
        from agenticdb.mcp.server import MCPServer

        db = AgenticDB()
        branch = db.create_branch("test")

        event = branch.record(Event(
            event_type="TestEvent",
            data={"v": 1}
        ))

        server = MCPServer(db, branch_id=branch.id)

        result = server.execute_tool("history", {
            "entity_id": event.id
        })

        assert result is not None
