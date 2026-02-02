# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Session management for AgenticDB.

Provides multi-turn context for interactive queries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import AgenticDB
    from .query_result import QueryResult


@dataclass
class SessionState:
    """
    State tracked within a session.

    Attributes:
        recent_tables: Recently accessed tables
        recent_entities: Recently referenced entities
        pending_transaction_id: ID of pending transaction if any
        conversation_context: Context from conversation
        bindings: Accumulated bindings
    """

    recent_tables: List[str] = field(default_factory=list)
    recent_entities: Dict[str, Any] = field(default_factory=dict)
    pending_transaction_id: Optional[str] = None
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    bindings: Dict[str, Any] = field(default_factory=dict)


class Session:
    """
    Session for multi-turn interactions.

    A session maintains context across multiple queries, enabling
    natural conversational interactions with the database.

    Example:
        with db.session() as session:
            r1 = session.query("show me users")  # May enter PENDING_BINDING
            r2 = session.bind(target="customers")  # Resolves binding
            r3 = session.query("filter by active")  # Continues with context
    """

    def __init__(self, db: "AgenticDB"):
        """
        Initialize the session.

        Args:
            db: AgenticDB instance
        """
        self._db = db
        self._state = SessionState()
        self._started_at = datetime.now(timezone.utc)
        self._query_count = 0
        self._closed = False

    def query(
        self,
        query_text: str,
        bindings: Optional[Dict[str, Any]] = None,
    ) -> "QueryResult":
        """
        Execute a query within the session context.

        Args:
            query_text: Natural language query
            bindings: Optional explicit bindings

        Returns:
            QueryResult with operation outcome
        """
        self._check_closed()

        # Merge session bindings with provided bindings
        merged_bindings = {**self._state.bindings}
        if bindings:
            merged_bindings.update(bindings)

        # Add context hints
        if self._state.recent_tables:
            merged_bindings.setdefault("_context_tables", self._state.recent_tables)

        # Execute query
        result = self._db.query(query_text, merged_bindings)

        # Update session state
        self._query_count += 1

        if result.status == "pending_binding":
            self._state.pending_transaction_id = result.transaction_id
        elif result.status == "completed":
            self._state.pending_transaction_id = None

        # Track accessed tables
        if result.data and isinstance(result.data, list):
            # Try to infer table from result
            pass

        return result

    def bind(self, **bindings: Any) -> "QueryResult":
        """
        Provide bindings for a pending operation.

        Args:
            **bindings: Named bindings

        Returns:
            QueryResult with updated status
        """
        self._check_closed()

        # Update session bindings
        self._state.bindings.update(bindings)

        # If there's a pending transaction, resolve it
        if self._state.pending_transaction_id:
            result = self._db.bind(self._state.pending_transaction_id, **bindings)

            if result.status != "pending_binding":
                self._state.pending_transaction_id = None

            return result

        # Otherwise, just acknowledge the binding
        from .query_result import QueryResult
        return QueryResult.success(data={"bindings": bindings})

    def confirm(self, yes: bool = True) -> "QueryResult":
        """
        Confirm or reject a pending confirmation.

        Args:
            yes: True to confirm, False to reject

        Returns:
            QueryResult with final status
        """
        self._check_closed()

        if not self._state.pending_transaction_id:
            from .query_result import QueryResult
            return QueryResult.error("No pending transaction to confirm")

        result = self._db.confirm(self._state.pending_transaction_id, yes)
        self._state.pending_transaction_id = None
        return result

    def store(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> "QueryResult":
        """
        Store an event within the session.

        Args:
            event_type: Event type name
            data: Event data

        Returns:
            QueryResult with store outcome
        """
        self._check_closed()
        return self._db.store(event_type, data)

    def context(self, **values: Any) -> "Session":
        """
        Add context values to the session.

        Args:
            **values: Context key-value pairs

        Returns:
            self for chaining
        """
        self._state.conversation_context.update(values)
        return self

    def remember_table(self, table: str) -> "Session":
        """
        Remember a table for context.

        Args:
            table: Table name

        Returns:
            self for chaining
        """
        if table not in self._state.recent_tables:
            self._state.recent_tables.insert(0, table)
            # Keep only last 5 tables
            self._state.recent_tables = self._state.recent_tables[:5]
        return self

    def remember_entity(self, name: str, value: Any) -> "Session":
        """
        Remember an entity reference.

        Args:
            name: Entity name/key
            value: Entity value/reference

        Returns:
            self for chaining
        """
        self._state.recent_entities[name] = value
        return self

    def clear_context(self) -> "Session":
        """
        Clear session context.

        Returns:
            self for chaining
        """
        self._state = SessionState()
        return self

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.

        Returns:
            Dictionary with session stats
        """
        return {
            "started_at": self._started_at.isoformat(),
            "query_count": self._query_count,
            "pending_transaction": self._state.pending_transaction_id,
            "recent_tables": self._state.recent_tables,
            "bindings": list(self._state.bindings.keys()),
        }

    def close(self) -> None:
        """Close the session."""
        self._closed = True
        self._state = SessionState()

    def _check_closed(self) -> None:
        """Check if session is closed."""
        if self._closed:
            raise RuntimeError("Session is closed")

    def __enter__(self) -> "Session":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
