# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Fluent API builders for AgenticDB.

Provides builder patterns for constructing queries and operations.
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import AgenticDB
    from .query_result import QueryResult


class TransactionBuilder:
    """
    Fluent builder for transaction operations.

    Provides a chainable API for building and executing queries.

    Example:
        result = (
            db.transaction()
              .query("show orders from last week")
              .where(total__gt=100)
              .limit(10)
              .execute()
        )
    """

    def __init__(self, db: "AgenticDB"):
        """
        Initialize the builder.

        Args:
            db: AgenticDB instance
        """
        self._db = db
        self._query_text: Optional[str] = None
        self._bindings: Dict[str, Any] = {}
        self._filters: Dict[str, Any] = {}
        self._limit_value: Optional[int] = None
        self._order_by: Optional[str] = None
        self._on_binding: Optional[Callable] = None
        self._on_confirmation: Optional[Callable] = None

    def query(self, query_text: str) -> "TransactionBuilder":
        """
        Set the query text.

        Args:
            query_text: Natural language query

        Returns:
            self for chaining
        """
        self._query_text = query_text
        return self

    def where(self, **conditions: Any) -> "TransactionBuilder":
        """
        Add filter conditions.

        Supports Django-style lookups:
            - field=value → field = value
            - field__gt=value → field > value
            - field__gte=value → field >= value
            - field__lt=value → field < value
            - field__lte=value → field <= value
            - field__contains=value → field LIKE %value%
            - field__in=[values] → field IN (values)

        Args:
            **conditions: Filter conditions

        Returns:
            self for chaining
        """
        self._filters.update(conditions)
        return self

    def limit(self, n: int) -> "TransactionBuilder":
        """
        Set result limit.

        Args:
            n: Maximum number of results

        Returns:
            self for chaining
        """
        self._limit_value = n
        self._bindings["limit"] = n
        return self

    def order_by(self, field: str) -> "TransactionBuilder":
        """
        Set ordering.

        Args:
            field: Field to order by (prefix with - for descending)

        Returns:
            self for chaining
        """
        self._order_by = field
        return self

    def bind(self, **bindings: Any) -> "TransactionBuilder":
        """
        Add explicit bindings.

        Args:
            **bindings: Named bindings

        Returns:
            self for chaining
        """
        self._bindings.update(bindings)
        return self

    def on_pending_binding(
        self,
        handler: Callable[["QueryResult"], "QueryResult"],
    ) -> "TransactionBuilder":
        """
        Set handler for pending binding state.

        Args:
            handler: Function to handle pending binding

        Returns:
            self for chaining
        """
        self._on_binding = handler
        return self

    def on_pending_confirmation(
        self,
        handler: Callable[["QueryResult"], "QueryResult"],
    ) -> "TransactionBuilder":
        """
        Set handler for pending confirmation state.

        Args:
            handler: Function to handle pending confirmation

        Returns:
            self for chaining
        """
        self._on_confirmation = handler
        return self

    def execute(self) -> "QueryResult":
        """
        Execute the built query.

        Returns:
            QueryResult with operation outcome
        """
        from .query_result import QueryResult

        if not self._query_text:
            return QueryResult.error("No query specified")

        # Execute query
        result = self._db.query(self._query_text, self._bindings)

        # Handle pending states
        if result.status == "pending_binding" and self._on_binding:
            result = self._on_binding(result)

        if result.status == "pending_confirmation" and self._on_confirmation:
            result = self._on_confirmation(result)

        return result

    def dry_run(self) -> Dict[str, Any]:
        """
        Preview the query without executing.

        Returns:
            Dictionary with query details
        """
        return {
            "query": self._query_text,
            "bindings": self._bindings,
            "filters": self._filters,
            "limit": self._limit_value,
            "order_by": self._order_by,
        }


class StoreBuilder:
    """
    Fluent builder for store operations.

    Example:
        result = (
            db.store_builder()
              .event_type("UserRegistered")
              .data(name="Alice", email="alice@example.com")
              .execute()
        )
    """

    def __init__(self, db: "AgenticDB"):
        """
        Initialize the builder.

        Args:
            db: AgenticDB instance
        """
        self._db = db
        self._event_type: Optional[str] = None
        self._data: Dict[str, Any] = {}
        self._branch_id: Optional[str] = None

    def event_type(self, type_name: str) -> "StoreBuilder":
        """
        Set the event type.

        Args:
            type_name: Event type name

        Returns:
            self for chaining
        """
        self._event_type = type_name
        return self

    def data(self, **fields: Any) -> "StoreBuilder":
        """
        Add data fields.

        Args:
            **fields: Data fields

        Returns:
            self for chaining
        """
        self._data.update(fields)
        return self

    def in_branch(self, branch_id: str) -> "StoreBuilder":
        """
        Set the target branch.

        Args:
            branch_id: Branch ID

        Returns:
            self for chaining
        """
        self._branch_id = branch_id
        return self

    def execute(self) -> "QueryResult":
        """
        Execute the store operation.

        Returns:
            QueryResult with operation outcome
        """
        from .query_result import QueryResult

        if not self._event_type:
            return QueryResult.error("No event type specified")

        return self._db.store(
            self._event_type,
            self._data,
            self._branch_id,
        )


class FluentQueryBuilder:
    """
    Fluent query builder with SQL-like syntax.

    Example:
        result = (
            FluentQueryBuilder(db)
              .select("orders")
              .where(status="active", total__gt=100)
              .order_by("-created_at")
              .limit(10)
              .execute()
        )
    """

    def __init__(self, db: "AgenticDB"):
        """
        Initialize the builder.

        Args:
            db: AgenticDB instance
        """
        self._db = db
        self._table: Optional[str] = None
        self._conditions: Dict[str, Any] = {}
        self._limit_value: int = 100
        self._order_field: Optional[str] = None
        self._ascending: bool = True

    def select(self, table: str) -> "FluentQueryBuilder":
        """
        Set the target table.

        Args:
            table: Table name

        Returns:
            self for chaining
        """
        self._table = table
        return self

    def where(self, **conditions: Any) -> "FluentQueryBuilder":
        """
        Add filter conditions.

        Args:
            **conditions: Filter conditions

        Returns:
            self for chaining
        """
        self._conditions.update(conditions)
        return self

    def order_by(self, field: str) -> "FluentQueryBuilder":
        """
        Set ordering.

        Args:
            field: Field to order by (prefix with - for descending)

        Returns:
            self for chaining
        """
        if field.startswith("-"):
            self._order_field = field[1:]
            self._ascending = False
        else:
            self._order_field = field
            self._ascending = True
        return self

    def limit(self, n: int) -> "FluentQueryBuilder":
        """
        Set result limit.

        Args:
            n: Maximum number of results

        Returns:
            self for chaining
        """
        self._limit_value = n
        return self

    def execute(self) -> "QueryResult":
        """
        Execute the query.

        Returns:
            QueryResult with operation outcome
        """
        from .query_result import QueryResult

        if not self._table:
            return QueryResult.error("No table specified")

        # Build query text
        query_parts = [f"show {self._table}"]

        if self._conditions:
            conditions_str = ", ".join(
                f"{k}={v}" for k, v in self._conditions.items()
            )
            query_parts.append(f"where {conditions_str}")

        if self._order_field:
            direction = "ascending" if self._ascending else "descending"
            query_parts.append(f"order by {self._order_field} {direction}")

        query_parts.append(f"limit {self._limit_value}")

        query_text = " ".join(query_parts)

        return self._db.query(
            query_text,
            bindings={"target": self._table, "limit": self._limit_value},
        )

    def to_sql(self) -> str:
        """
        Preview the SQL that would be generated.

        Returns:
            SQL query string
        """
        parts = [f"SELECT * FROM {self._table or '?'}"]

        if self._conditions:
            where_parts = []
            for key, value in self._conditions.items():
                if "__" in key:
                    field, op = key.rsplit("__", 1)
                    op_map = {
                        "gt": ">",
                        "gte": ">=",
                        "lt": "<",
                        "lte": "<=",
                        "contains": "LIKE",
                    }
                    sql_op = op_map.get(op, "=")
                    if sql_op == "LIKE":
                        where_parts.append(f"{field} {sql_op} '%{value}%'")
                    else:
                        where_parts.append(f"{field} {sql_op} {value!r}")
                else:
                    where_parts.append(f"{key} = {value!r}")

            parts.append("WHERE " + " AND ".join(where_parts))

        if self._order_field:
            direction = "ASC" if self._ascending else "DESC"
            parts.append(f"ORDER BY {self._order_field} {direction}")

        parts.append(f"LIMIT {self._limit_value}")

        return " ".join(parts)
