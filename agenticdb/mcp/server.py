"""
MCP Server implementation for AgenticDB.

This module provides the main MCP server that exposes AgenticDB functionality
to external AI applications through the Model Context Protocol.

Design Philosophy:
    The MCP server acts as a bridge between AI agents and AgenticDB,
    translating tool calls into database operations and returning
    structured results that agents can interpret.

Server Capabilities:
    - Tool listing and schema retrieval
    - Tool execution with parameter validation
    - Concurrent request handling (thread-safe)
    - Error handling with meaningful messages

Usage:
    ```python
    from agenticdb import AgenticDB
    from agenticdb.mcp import MCPServer

    db = AgenticDB()
    server = MCPServer(db)

    # List available tools
    tools = server.list_tools()

    # Execute a tool
    result = server.execute_tool("query", {
        "natural_language_query": "show all events"
    })
    ```
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Optional, TYPE_CHECKING

from agenticdb.core.models import Event, Claim, Action, EntityType
from agenticdb.mcp.tools import MCPToolGenerator

if TYPE_CHECKING:
    from agenticdb.interface.client import AgenticDB


class MCPServer:
    """
    MCP Server for AgenticDB.

    Provides a complete MCP-compliant server interface that external
    AI applications can use to interact with AgenticDB.

    Thread Safety:
        All operations are thread-safe for concurrent access.

    Attributes:
        db: The AgenticDB instance being served
        branch_id: The branch being served

    Example:
        ```python
        server = MCPServer(db, branch_id="main")

        # Get available tools
        for tool in server.list_tools():
            print(f"Tool: {tool['name']}")

        # Execute a query
        result = server.execute_tool("get_events", {"limit": 10})
        print(result)
        ```
    """

    def __init__(
        self,
        db: "AgenticDB",
        branch_id: Optional[str] = None,
    ):
        """
        Initialize the MCP server.

        Args:
            db: AgenticDB instance to serve
            branch_id: Branch to serve (uses main if not specified)
        """
        self._db = db
        self._branch_id = branch_id
        self._lock = RLock()

        # Initialize tool generator
        self._tool_generator = MCPToolGenerator(db, branch_id)

        # Tool execution handlers
        self._handlers: dict[str, callable] = {
            # Base tools
            "query": self._handle_query,
            "store": self._handle_store,

            # Event tools
            "get_events": self._handle_get_events,
            "get_event": self._handle_get_event,

            # Claim tools
            "get_claims": self._handle_get_claims,
            "get_claim": self._handle_get_claim,
            "create_claim": self._handle_create_claim,

            # Action tools
            "get_actions": self._handle_get_actions,
            "get_action": self._handle_get_action,

            # Causal tools
            "why": self._handle_why,
            "impact": self._handle_impact,

            # Time-travel tools
            "snapshot": self._handle_snapshot,
            "history": self._handle_history,
        }

    @property
    def db(self) -> "AgenticDB":
        """Get the AgenticDB instance."""
        return self._db

    @property
    def branch_id(self) -> Optional[str]:
        """Get the current branch ID."""
        return self._branch_id

    def list_tools(self) -> list[dict[str, Any]]:
        """
        List all available MCP tools.

        Returns:
            List of tool definitions in MCP format
        """
        with self._lock:
            # Refresh tools to pick up any schema changes
            self._tool_generator.refresh()
            return self._tool_generator.get_tools()

    def get_tool_schema(self, tool_name: str) -> Optional[dict[str, Any]]:
        """
        Get the schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool definition if found, None otherwise
        """
        with self._lock:
            tools = self._tool_generator.get_tools()
            for tool in tools:
                if tool["name"] == tool_name:
                    return tool
            return None

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute an MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result (JSON-serializable dict)
        """
        with self._lock:
            handler = self._handlers.get(tool_name)
            if handler is None:
                return {"error": f"Tool not found: {tool_name}"}

            try:
                return handler(arguments)
            except Exception as e:
                return {"error": str(e)}

    def _get_branch(self):
        """Get the branch handle."""
        return self._db.branch(self._branch_id)

    # =========================================================================
    # Base Tool Handlers
    # =========================================================================

    def _handle_query(self, args: dict[str, Any]) -> dict[str, Any]:
        """
        Handle natural language query.

        This is a simplified implementation - in production, this would
        use the IntentAgent and QueryBuilder to parse natural language.
        """
        query = args.get("natural_language_query", "")
        limit = args.get("limit", 100)

        branch = self._get_branch()

        # Simple keyword-based routing for demonstration
        query_lower = query.lower()

        results = []

        if "event" in query_lower:
            events = list(branch.events(limit=limit))
            results = [self._serialize_event(e) for e in events]
            return {"type": "events", "results": results, "count": len(results)}

        elif "claim" in query_lower:
            claims = list(branch.claims(limit=limit))
            results = [self._serialize_claim(c) for c in claims]
            return {"type": "claims", "results": results, "count": len(results)}

        elif "action" in query_lower:
            actions = list(branch.actions(limit=limit))
            results = [self._serialize_action(a) for a in actions]
            return {"type": "actions", "results": results, "count": len(results)}

        else:
            # Return all entity types
            events = list(branch.events(limit=limit))
            claims = list(branch.claims(limit=limit))
            actions = list(branch.actions(limit=limit))

            return {
                "events": [self._serialize_event(e) for e in events],
                "claims": [self._serialize_claim(c) for c in claims],
                "actions": [self._serialize_action(a) for a in actions],
                "total_count": len(events) + len(claims) + len(actions)
            }

    def _handle_store(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle store operation - creates an event from description."""
        description = args.get("description", "")
        data = args.get("data", {})

        branch = self._get_branch()

        # Parse description into event type (simplified)
        event_type = self._description_to_event_type(description)

        event = branch.record(Event(
            event_type=event_type,
            data=data,
        ))

        return {
            "entity_id": event.id,
            "entity_type": "event",
            "event_type": event.event_type,
            "message": f"Successfully stored event: {event_type}"
        }

    def _description_to_event_type(self, description: str) -> str:
        """Convert a description to an event type name."""
        # Simple conversion: "user signed up" -> "UserSignedUp"
        words = description.strip().split()
        return "".join(word.capitalize() for word in words)

    # =========================================================================
    # Event Tool Handlers
    # =========================================================================

    def _handle_get_events(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get events, optionally filtered."""
        event_type = args.get("event_type")
        limit = args.get("limit", 100)

        branch = self._get_branch()
        events = list(branch.events(event_type=event_type, limit=limit))

        return {
            "events": [self._serialize_event(e) for e in events],
            "count": len(events)
        }

    def _handle_get_event(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get a specific event by ID."""
        entity_id = args.get("entity_id")
        if not entity_id:
            return {"error": "entity_id is required"}

        branch = self._get_branch()
        entity = branch.get(entity_id)

        if entity is None:
            return {"error": f"Event not found: {entity_id}"}

        if not isinstance(entity, Event):
            return {"error": f"Entity is not an event: {entity_id}"}

        return self._serialize_event(entity)

    # =========================================================================
    # Claim Tool Handlers
    # =========================================================================

    def _handle_get_claims(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get claims, optionally filtered."""
        subject = args.get("subject")
        source = args.get("source")
        limit = args.get("limit", 100)

        branch = self._get_branch()
        claims = list(branch.claims(
            subject=subject,
            source=source,
            limit=limit
        ))

        return {
            "claims": [self._serialize_claim(c) for c in claims],
            "count": len(claims)
        }

    def _handle_get_claim(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get a specific claim by ID."""
        entity_id = args.get("entity_id")
        if not entity_id:
            return {"error": "entity_id is required"}

        branch = self._get_branch()
        entity = branch.get(entity_id)

        if entity is None:
            return {"error": f"Claim not found: {entity_id}"}

        if not isinstance(entity, Claim):
            return {"error": f"Entity is not a claim: {entity_id}"}

        return self._serialize_claim(entity)

    def _handle_create_claim(self, args: dict[str, Any]) -> dict[str, Any]:
        """Create a new claim."""
        subject = args.get("subject")
        predicate = args.get("predicate")
        value = args.get("value")
        source = args.get("source")
        confidence = args.get("confidence", 1.0)

        if not all([subject, predicate, source]):
            return {"error": "subject, predicate, and source are required"}

        branch = self._get_branch()

        claim = branch.record(Claim(
            subject=subject,
            predicate=predicate,
            value=value,
            source=source,
            confidence=confidence,
        ))

        return {
            "entity_id": claim.id,
            "entity_type": "claim",
            "subject": claim.subject,
            "message": f"Successfully created claim: {subject}"
        }

    # =========================================================================
    # Action Tool Handlers
    # =========================================================================

    def _handle_get_actions(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get actions, optionally filtered."""
        action_type = args.get("action_type")
        agent_id = args.get("agent_id")
        limit = args.get("limit", 100)

        branch = self._get_branch()
        actions = list(branch.actions(
            action_type=action_type,
            agent_id=agent_id,
            limit=limit
        ))

        return {
            "actions": [self._serialize_action(a) for a in actions],
            "count": len(actions)
        }

    def _handle_get_action(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get a specific action by ID."""
        entity_id = args.get("entity_id")
        if not entity_id:
            return {"error": "entity_id is required"}

        branch = self._get_branch()
        entity = branch.get(entity_id)

        if entity is None:
            return {"error": f"Action not found: {entity_id}"}

        if not isinstance(entity, Action):
            return {"error": f"Entity is not an action: {entity_id}"}

        return self._serialize_action(entity)

    # =========================================================================
    # Causal Tool Handlers
    # =========================================================================

    def _handle_why(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle why() query."""
        entity_id = args.get("entity_id")
        max_depth = args.get("max_depth")

        if not entity_id:
            return {"error": "entity_id is required"}

        branch = self._get_branch()
        result = branch.why(entity_id, max_depth=max_depth)

        return {
            "entity_id": entity_id,
            "causal_chain": {
                "steps": [
                    {
                        "entity_id": step.entity_id,
                        "entity_type": step.entity_type.value if hasattr(step.entity_type, 'value') else str(step.entity_type),
                        "relationship": step.relationship,
                        "depth": step.depth,
                        "summary": step.summary,
                    }
                    for step in result.steps
                ],
                "total_depth": result.total_depth,
                "execution_time_ms": result.execution_time_ms,
            }
        }

    def _handle_impact(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle impact() query."""
        entity_id = args.get("entity_id")
        max_depth = args.get("max_depth")

        if not entity_id:
            return {"error": "entity_id is required"}

        branch = self._get_branch()
        result = branch.impact(entity_id, max_depth=max_depth)

        return {
            "entity_id": entity_id,
            "affected_events": result.affected_events,
            "affected_claims": result.affected_claims,
            "affected_actions": result.affected_actions,
            "total_affected": result.total_affected,
            "max_depth": result.max_depth,
            "invalidated_ids": result.invalidated_ids,
        }

    # =========================================================================
    # Time-Travel Tool Handlers
    # =========================================================================

    def _handle_snapshot(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle snapshot query."""
        version = args.get("version")

        if version is None:
            return {"error": "version is required"}

        branch = self._get_branch()
        snapshot = branch.at(version)

        return {
            "snapshot": {
                "branch_id": snapshot.branch_id,
                "version": snapshot.version,
                "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
                "event_count": len(snapshot.events),
                "claim_count": len(snapshot.claims),
                "action_count": len(snapshot.actions),
                "total_entities": snapshot.entity_count,
            }
        }

    def _handle_history(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle history query."""
        entity_id = args.get("entity_id")

        if not entity_id:
            return {"error": "entity_id is required"}

        branch = self._get_branch()
        history = branch.history(entity_id)

        return {
            "entity_id": entity_id,
            "versions": [
                {
                    "version": e.version,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "status": e.status.value if hasattr(e.status, 'value') else str(e.status),
                }
                for e in history
            ],
            "total_versions": len(history)
        }

    # =========================================================================
    # Serialization Helpers
    # =========================================================================

    def _serialize_event(self, event: Event) -> dict[str, Any]:
        """Serialize an event to a dict."""
        return {
            "id": event.id,
            "entity_type": "event",
            "event_type": event.event_type,
            "data": event.data,
            "created_at": event.created_at.isoformat() if event.created_at else None,
            "source_agent": event.source_agent,
            "source_system": event.source_system,
        }

    def _serialize_claim(self, claim: Claim) -> dict[str, Any]:
        """Serialize a claim to a dict."""
        return {
            "id": claim.id,
            "entity_type": "claim",
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "source": claim.source,
            "confidence": claim.confidence,
            "created_at": claim.created_at.isoformat() if claim.created_at else None,
        }

    def _serialize_action(self, action: Action) -> dict[str, Any]:
        """Serialize an action to a dict."""
        return {
            "id": action.id,
            "entity_type": "action",
            "action_type": action.action_type,
            "agent_id": action.agent_id,
            "inputs": action.inputs,
            "outputs": action.outputs,
            "depends_on": action.depends_on,
            "status": action.action_status.value if hasattr(action.action_status, 'value') else str(action.action_status),
            "created_at": action.created_at.isoformat() if action.created_at else None,
        }
