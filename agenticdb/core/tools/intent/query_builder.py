"""
Query builder tool for Intent → SQL generation.

Pure computation - no LLM calls. Converts Intent IR to SQL queries
with proper parameterization to prevent SQL injection.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...models import (
    Intent,
    OperationType,
    Predicate,
    ParameterSlot,
)


@dataclass
class SQLQuery:
    """
    Generated SQL query with parameters.

    Attributes:
        sql: The SQL string with placeholders
        parameters: Parameter values for placeholders
        operation: The operation type
        table: Target table
        estimated_complexity: Estimated query complexity (1-10)
    """

    sql: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    operation: Optional[OperationType] = None
    table: Optional[str] = None
    estimated_complexity: int = 1

    def to_tuple(self) -> Tuple[str, Dict[str, Any]]:
        """Return (sql, parameters) tuple for execution."""
        return self.sql, self.parameters


@dataclass
class BuildResult:
    """
    Result of query building.

    Attributes:
        success: Whether building succeeded
        query: The generated SQLQuery
        error: Error message if building failed
        warnings: Non-fatal warnings
    """

    success: bool = True
    query: Optional[SQLQuery] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_query(cls, query: SQLQuery) -> "BuildResult":
        """Create a successful result."""
        return cls(success=True, query=query)

    @classmethod
    def from_error(cls, error: str) -> "BuildResult":
        """Create a failed result."""
        return cls(success=False, error=error)


class QueryBuilder:
    """
    Build SQL queries from Intent IR.

    This is a pure computation tool (no LLM) that generates
    parameterized SQL from Intent objects.

    Features:
        - Parameterized queries (prevents SQL injection)
        - Support for all operation types (QUERY, STORE, UPDATE, DELETE)
        - Predicate translation
        - Temporal expression handling

    Example:
        builder = QueryBuilder()
        result = builder.build(intent)
        # → BuildResult(query=SQLQuery(sql="SELECT * FROM orders WHERE ..."))
    """

    # Operator mapping
    OPERATOR_MAP = {
        "eq": "=",
        "ne": "!=",
        "neq": "!=",
        "gt": ">",
        "gte": ">=",
        "ge": ">=",
        "lt": "<",
        "lte": "<=",
        "le": "<=",
        "contains": "LIKE",
        "like": "LIKE",
        "in": "IN",
        "between": "BETWEEN",
        "is_null": "IS NULL",
        "is_not_null": "IS NOT NULL",
    }

    # Reserved SQL keywords (for escaping)
    RESERVED_WORDS = {
        "select", "from", "where", "and", "or", "not", "in", "like",
        "between", "is", "null", "true", "false", "order", "by",
        "group", "having", "limit", "offset", "join", "left", "right",
        "inner", "outer", "on", "as", "distinct", "count", "sum",
        "avg", "min", "max", "insert", "update", "delete", "create",
        "drop", "alter", "table", "index", "view", "trigger",
    }

    def __init__(
        self,
        default_limit: int = 100,
        quote_identifiers: bool = True,
        parameter_style: str = "named",  # "named" (:param) or "qmark" (?)
    ):
        """
        Initialize the query builder.

        Args:
            default_limit: Default LIMIT for SELECT queries
            quote_identifiers: Whether to quote table/column names
            parameter_style: Parameter placeholder style
        """
        self.default_limit = default_limit
        self.quote_identifiers = quote_identifiers
        self.parameter_style = parameter_style
        self._param_counter = 0

    def build(self, intent: Intent) -> BuildResult:
        """
        Build SQL from Intent.

        Args:
            intent: The Intent to convert

        Returns:
            BuildResult with the generated query
        """
        # Validate intent state
        if not intent.is_complete():
            unbound = intent.get_unbound_slot_names()
            return BuildResult.from_error(
                f"Cannot build SQL: Intent has unbound slots: {unbound}"
            )

        # Get target table
        table = intent.get_target_name()
        if not table:
            return BuildResult.from_error("Cannot build SQL: No target table")

        # Build based on operation type
        try:
            if intent.operation == OperationType.QUERY:
                return self._build_select(intent, table)
            elif intent.operation == OperationType.STORE:
                return self._build_insert(intent, table)
            elif intent.operation == OperationType.UPDATE:
                return self._build_update(intent, table)
            elif intent.operation == OperationType.DELETE:
                return self._build_delete(intent, table)
            else:
                return BuildResult.from_error(f"Unknown operation: {intent.operation}")
        except Exception as e:
            return BuildResult.from_error(f"Query build failed: {str(e)}")

    def build_from_pattern(
        self,
        template: str,
        parameters: Dict[str, Any],
    ) -> BuildResult:
        """
        Build SQL from a pattern template.

        Args:
            template: SQL template with {placeholders}
            parameters: Values for placeholders

        Returns:
            BuildResult with the generated query
        """
        try:
            # Convert template placeholders to parameter placeholders
            sql = template
            params = {}

            for key, value in parameters.items():
                placeholder = f"{{{key}}}"
                if placeholder in sql:
                    param_name = self._next_param_name(key)
                    if self.parameter_style == "named":
                        sql = sql.replace(placeholder, f":{param_name}")
                    else:
                        sql = sql.replace(placeholder, "?")
                    params[param_name] = value

            query = SQLQuery(
                sql=sql,
                parameters=params,
                estimated_complexity=self._estimate_complexity(sql),
            )
            return BuildResult.from_query(query)

        except Exception as e:
            return BuildResult.from_error(f"Template build failed: {str(e)}")

    def _build_select(self, intent: Intent, table: str) -> BuildResult:
        """Build SELECT query."""
        self._param_counter = 0
        params = {}
        warnings = []

        # Start building
        quoted_table = self._quote_identifier(table)
        sql_parts = [f"SELECT * FROM {quoted_table}"]

        # Add WHERE clause from predicates
        where_clause, where_params = self._build_where(intent.predicates)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
            params.update(where_params)

        # Add ORDER BY if specified in metadata
        order_by = intent.metadata.get("order_by")
        if order_by:
            sql_parts.append(f"ORDER BY {self._quote_identifier(order_by)}")

        # Add LIMIT
        limit = intent.bindings.get("limit", self.default_limit)
        sql_parts.append(f"LIMIT {int(limit)}")

        sql = " ".join(sql_parts)

        query = SQLQuery(
            sql=sql,
            parameters=params,
            operation=OperationType.QUERY,
            table=table,
            estimated_complexity=self._estimate_complexity(sql),
        )

        return BuildResult(success=True, query=query, warnings=warnings)

    def _build_insert(self, intent: Intent, table: str) -> BuildResult:
        """Build INSERT query."""
        self._param_counter = 0
        params = {}

        # Get data to insert from bindings
        data = intent.bindings.get("data", {})
        if not data:
            return BuildResult.from_error("INSERT requires data to insert")

        quoted_table = self._quote_identifier(table)
        columns = []
        placeholders = []

        for key, value in data.items():
            columns.append(self._quote_identifier(key))
            param_name = self._next_param_name(key)
            if self.parameter_style == "named":
                placeholders.append(f":{param_name}")
            else:
                placeholders.append("?")
            params[param_name] = value

        sql = f"INSERT INTO {quoted_table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        query = SQLQuery(
            sql=sql,
            parameters=params,
            operation=OperationType.STORE,
            table=table,
            estimated_complexity=1,
        )

        return BuildResult.from_query(query)

    def _build_update(self, intent: Intent, table: str) -> BuildResult:
        """Build UPDATE query."""
        self._param_counter = 0
        params = {}
        warnings = []

        # Get data to update from bindings
        data = intent.bindings.get("data", {})
        if not data:
            return BuildResult.from_error("UPDATE requires data to update")

        quoted_table = self._quote_identifier(table)

        # Build SET clause
        set_parts = []
        for key, value in data.items():
            param_name = self._next_param_name(key)
            if self.parameter_style == "named":
                set_parts.append(f"{self._quote_identifier(key)} = :{param_name}")
            else:
                set_parts.append(f"{self._quote_identifier(key)} = ?")
            params[param_name] = value

        sql_parts = [f"UPDATE {quoted_table} SET {', '.join(set_parts)}"]

        # Add WHERE clause (required for UPDATE)
        where_clause, where_params = self._build_where(intent.predicates)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
            params.update(where_params)
        else:
            warnings.append("UPDATE without WHERE clause - affects all rows")

        sql = " ".join(sql_parts)

        query = SQLQuery(
            sql=sql,
            parameters=params,
            operation=OperationType.UPDATE,
            table=table,
            estimated_complexity=self._estimate_complexity(sql),
        )

        return BuildResult(success=True, query=query, warnings=warnings)

    def _build_delete(self, intent: Intent, table: str) -> BuildResult:
        """Build DELETE query."""
        self._param_counter = 0
        params = {}
        warnings = []

        quoted_table = self._quote_identifier(table)
        sql_parts = [f"DELETE FROM {quoted_table}"]

        # Add WHERE clause (strongly recommended for DELETE)
        where_clause, where_params = self._build_where(intent.predicates)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
            params.update(where_params)
        else:
            warnings.append("DELETE without WHERE clause - deletes all rows")

        sql = " ".join(sql_parts)

        query = SQLQuery(
            sql=sql,
            parameters=params,
            operation=OperationType.DELETE,
            table=table,
            estimated_complexity=self._estimate_complexity(sql),
        )

        return BuildResult(success=True, query=query, warnings=warnings)

    def _build_where(
        self, predicates: List[Predicate]
    ) -> Tuple[str, Dict[str, Any]]:
        """Build WHERE clause from predicates."""
        if not predicates:
            return "", {}

        parts = []
        params = {}

        for pred in predicates:
            clause, pred_params = self._build_predicate(pred)
            if clause:
                parts.append(clause)
                params.update(pred_params)

        if not parts:
            return "", {}

        return " AND ".join(parts), params

    def _build_predicate(
        self, predicate: Predicate
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a single predicate clause."""
        field = self._quote_identifier(predicate.field)
        operator = self.OPERATOR_MAP.get(predicate.operator, predicate.operator)
        value = predicate.get_resolved_value()
        params = {}

        # Handle special operators
        if predicate.operator in ("is_null", "is_not_null"):
            clause = f"{field} {operator}"
        elif predicate.operator == "in":
            if not isinstance(value, (list, tuple)):
                value = [value]
            placeholders = []
            for i, v in enumerate(value):
                param_name = self._next_param_name(f"{predicate.field}_{i}")
                if self.parameter_style == "named":
                    placeholders.append(f":{param_name}")
                else:
                    placeholders.append("?")
                params[param_name] = v
            clause = f"{field} IN ({', '.join(placeholders)})"
        elif predicate.operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return "", {}
            param1 = self._next_param_name(f"{predicate.field}_start")
            param2 = self._next_param_name(f"{predicate.field}_end")
            if self.parameter_style == "named":
                clause = f"{field} BETWEEN :{param1} AND :{param2}"
            else:
                clause = f"{field} BETWEEN ? AND ?"
            params[param1] = value[0]
            params[param2] = value[1]
        elif predicate.operator in ("contains", "like"):
            param_name = self._next_param_name(predicate.field)
            if self.parameter_style == "named":
                clause = f"{field} LIKE :{param_name}"
            else:
                clause = f"{field} LIKE ?"
            params[param_name] = f"%{value}%"
        else:
            param_name = self._next_param_name(predicate.field)
            if self.parameter_style == "named":
                clause = f"{field} {operator} :{param_name}"
            else:
                clause = f"{field} {operator} ?"
            params[param_name] = value

        # Handle negation
        if predicate.negate:
            clause = f"NOT ({clause})"

        return clause, params

    def _quote_identifier(self, identifier: str) -> str:
        """Quote an identifier if needed."""
        if not self.quote_identifiers:
            return identifier

        # Don't quote if it's a simple alphanumeric name
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            if identifier.lower() not in self.RESERVED_WORDS:
                return identifier

        # Quote with double quotes (SQL standard)
        return f'"{identifier}"'

    def _next_param_name(self, base: str = "p") -> str:
        """Generate the next parameter name."""
        self._param_counter += 1
        # Clean the base name
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", base)
        return f"{clean}_{self._param_counter}"

    def _estimate_complexity(self, sql: str) -> int:
        """Estimate query complexity (1-10)."""
        complexity = 1

        # Count JOINs
        complexity += sql.upper().count("JOIN")

        # Count subqueries
        complexity += sql.count("SELECT") - 1

        # Count conditions
        complexity += sql.upper().count("AND") // 2
        complexity += sql.upper().count("OR") // 2

        # Cap at 10
        return min(complexity, 10)
