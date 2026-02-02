"""
MCP Tool Generator for AgenticDB.

This module provides dynamic tool generation based on the current database
schema, enabling AI applications to interact with AgenticDB through
well-defined MCP tools.

Design Philosophy:
    Traditional APIs require manual endpoint definition. AgenticDB's MCP
    tools are generated automatically based on what entities exist in the
    database, providing a self-documenting, dynamic interface.

Tool Categories:
    1. Base Tools: Always available (query, store)
    2. Entity Tools: Generated per entity type (get_events, create_claim, etc.)
    3. Causal Tools: For dependency tracking (why, impact)
    4. Time-Travel Tools: For versioning (snapshot, history)

Tool Schema Format (MCP Standard):
    ```json
    {
        "name": "tool_name",
        "description": "What this tool does",
        "inputSchema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    ```
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from agenticdb.core.models import EntityType

if TYPE_CHECKING:
    from agenticdb.interface.client import AgenticDB


class MCPToolGenerator:
    """
    Generates MCP tools dynamically based on database schema.

    The generator creates tool definitions that follow the MCP specification,
    automatically updating when new entity types are introduced to the database.

    Attributes:
        db: AgenticDB instance to generate tools for
        branch_id: Optional branch to scope tools to

    Example:
        ```python
        generator = MCPToolGenerator(db, branch_id="main")
        tools = generator.get_tools()

        # After adding events:
        branch.record(Event(...))
        generator.refresh()
        # Now includes get_events, get_event tools
        ```
    """

    def __init__(
        self,
        db: "AgenticDB",
        branch_id: Optional[str] = None,
    ):
        """
        Initialize the tool generator.

        Args:
            db: AgenticDB instance
            branch_id: Optional branch ID (uses main if not specified)
        """
        self._db = db
        self._branch_id = branch_id
        self._tools: list[dict[str, Any]] = []
        self._entity_types_present: set[EntityType] = set()

        # Generate initial tools
        self._generate_tools()

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get all available MCP tools.

        Returns:
            List of tool definitions in MCP format
        """
        return list(self._tools)

    def refresh(self) -> None:
        """
        Refresh tools based on current database state.

        Call this after schema changes to update available tools.
        """
        self._generate_tools()

    def _generate_tools(self) -> None:
        """Generate all tool definitions."""
        self._tools = []

        # Always include base tools
        self._tools.extend(self._generate_base_tools())

        # Check what entity types exist in the branch
        self._scan_entity_types()

        # Generate entity-specific tools
        if EntityType.EVENT in self._entity_types_present:
            self._tools.extend(self._generate_event_tools())

        if EntityType.CLAIM in self._entity_types_present:
            self._tools.extend(self._generate_claim_tools())

        if EntityType.ACTION in self._entity_types_present:
            self._tools.extend(self._generate_action_tools())

        # Always include causal and time-travel tools
        self._tools.extend(self._generate_causal_tools())
        self._tools.extend(self._generate_time_travel_tools())

    def _scan_entity_types(self) -> None:
        """Scan the database to find which entity types exist."""
        self._entity_types_present = set()

        try:
            branch = self._db.branch(self._branch_id)

            # Check for events
            events = list(branch.events(limit=1))
            if events:
                self._entity_types_present.add(EntityType.EVENT)

            # Check for claims
            claims = list(branch.claims(limit=1))
            if claims:
                self._entity_types_present.add(EntityType.CLAIM)

            # Check for actions
            actions = list(branch.actions(limit=1))
            if actions:
                self._entity_types_present.add(EntityType.ACTION)
        except Exception:
            # If branch doesn't exist or has no data, no entity types present
            pass

    def _generate_base_tools(self) -> list[dict[str, Any]]:
        """Generate base tools that are always available."""
        return [
            {
                "name": "query",
                "description": (
                    "Query the database using natural language. "
                    "Supports questions like 'show all users', 'find orders from last week', "
                    "'how many events happened today?'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "natural_language_query": {
                            "type": "string",
                            "description": "Natural language query describing what you want to find"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 100
                        }
                    },
                    "required": ["natural_language_query"]
                }
            },
            {
                "name": "store",
                "description": (
                    "Store data in the database by describing what happened. "
                    "Examples: 'user signed up', 'order placed', 'payment received'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of what happened (e.g., 'user signed up')"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data associated with the event"
                        }
                    },
                    "required": ["description", "data"]
                }
            }
        ]

    def _generate_event_tools(self) -> list[dict[str, Any]]:
        """Generate event-related tools."""
        return [
            {
                "name": "get_events",
                "description": "Get all events from the database, optionally filtered by type",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "Filter by event type (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of events to return",
                            "default": 100
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_event",
                "description": "Get a specific event by its ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The event ID"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        ]

    def _generate_claim_tools(self) -> list[dict[str, Any]]:
        """Generate claim-related tools."""
        return [
            {
                "name": "get_claims",
                "description": "Get all claims from the database, optionally filtered",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Filter by claim subject (optional)"
                        },
                        "source": {
                            "type": "string",
                            "description": "Filter by claim source (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of claims to return",
                            "default": 100
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_claim",
                "description": "Get a specific claim by its ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The claim ID"
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "create_claim",
                "description": (
                    "Create a new claim (assertion) in the database. "
                    "Claims represent derived or asserted facts with provenance."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Subject of the claim (e.g., 'user.risk_score')"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Predicate (e.g., 'equals', 'greater_than')"
                        },
                        "value": {
                            "description": "Value of the claim"
                        },
                        "source": {
                            "type": "string",
                            "description": "Source of the claim (e.g., 'risk_model_v1')"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score (0-1)",
                            "default": 1.0
                        }
                    },
                    "required": ["subject", "predicate", "value", "source"]
                }
            }
        ]

    def _generate_action_tools(self) -> list[dict[str, Any]]:
        """Generate action-related tools."""
        return [
            {
                "name": "get_actions",
                "description": "Get all actions from the database, optionally filtered",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "description": "Filter by action type (optional)"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Filter by agent ID (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of actions to return",
                            "default": 100
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_action",
                "description": "Get a specific action by its ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The action ID"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        ]

    def _generate_causal_tools(self) -> list[dict[str, Any]]:
        """Generate causal query tools."""
        return [
            {
                "name": "why",
                "description": (
                    "Trace the causal chain that led to an entity. "
                    "Answers: 'Why did this happen? What caused this state?'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity to trace"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (optional)"
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "impact",
                "description": (
                    "Find all entities affected by a change to this entity. "
                    "Answers: 'What breaks if this changes?'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity to analyze"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (optional)"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        ]

    def _generate_time_travel_tools(self) -> list[dict[str, Any]]:
        """Generate time-travel tools."""
        return [
            {
                "name": "snapshot",
                "description": (
                    "Get a snapshot of the database state at a specific version. "
                    "Enables time-travel debugging."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "version": {
                            "type": "integer",
                            "description": "Version number to snapshot"
                        }
                    },
                    "required": ["version"]
                }
            },
            {
                "name": "history",
                "description": "Get the history of an entity across versions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity to trace"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        ]
