"""
MCP (Model Context Protocol) module for AgenticDB.

This module provides MCP server functionality that exposes AgenticDB
to external AI applications like Claude Desktop, Cursor, and custom agents.

Key Features:
    - Dynamic tool generation based on database schema
    - Natural language query and store interfaces
    - Causal query tools (why, impact)
    - Time-travel tools (snapshot, history)
    - Automatic tool refresh on schema changes

Usage:
    ```python
    from agenticdb import AgenticDB
    from agenticdb.mcp import MCPServer

    db = AgenticDB()
    server = MCPServer(db)

    # Get available tools
    tools = server.list_tools()

    # Execute a tool
    result = server.execute_tool("query", {
        "natural_language_query": "show all events"
    })
    ```

For Claude Desktop integration:
    ```json
    {
      "mcpServers": {
        "agenticdb": {
          "command": "agenticdb",
          "args": ["--mcp"],
          "env": {
            "AGENTICDB_PATH": "/path/to/database.db"
          }
        }
      }
    }
    ```
"""

from agenticdb.mcp.server import MCPServer
from agenticdb.mcp.tools import MCPToolGenerator

__all__ = [
    "MCPServer",
    "MCPToolGenerator",
]
