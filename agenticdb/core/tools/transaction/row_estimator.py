"""
Row estimation tool for transaction validation.

Pure computation - no LLM calls. Estimates the number of rows
that will be affected by a query/mutation.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from ...models import (
    Intent,
    OperationType,
    Predicate,
)


@dataclass
class RowEstimate:
    """
    Estimated row count for an operation.

    Attributes:
        count: Estimated row count
        confidence: Confidence in the estimate [0, 1]
        is_bounded: Whether the estimate is bounded (e.g., by LIMIT)
        lower_bound: Minimum possible rows
        upper_bound: Maximum possible rows
        method: How the estimate was computed
    """

    count: int
    confidence: float = 0.5
    is_bounded: bool = False
    lower_bound: int = 0
    upper_bound: Optional[int] = None
    method: str = "heuristic"

    @classmethod
    def exact(cls, count: int) -> "RowEstimate":
        """Create an exact estimate."""
        return cls(
            count=count,
            confidence=1.0,
            is_bounded=True,
            lower_bound=count,
            upper_bound=count,
            method="exact",
        )

    @classmethod
    def bounded(cls, count: int, upper: int) -> "RowEstimate":
        """Create a bounded estimate."""
        return cls(
            count=count,
            confidence=0.8,
            is_bounded=True,
            lower_bound=0,
            upper_bound=upper,
            method="bounded",
        )

    @classmethod
    def unbounded(cls, estimate: int) -> "RowEstimate":
        """Create an unbounded estimate."""
        return cls(
            count=estimate,
            confidence=0.3,
            is_bounded=False,
            lower_bound=0,
            upper_bound=None,
            method="unbounded",
        )


class RowEstimator:
    """
    Estimate the number of rows affected by operations.

    This is a pure computation tool (no LLM) that estimates
    row counts for safety validation.

    Estimation Methods:
        - Exact: From LIMIT clause or single-row operations
        - Statistical: From table statistics (if available)
        - Heuristic: Based on predicate analysis

    Example:
        estimator = RowEstimator(table_stats={"orders": 10000})
        estimate = estimator.estimate(intent)
        # → RowEstimate(count=100, confidence=0.7, ...)
    """

    # Default selectivity estimates for predicates
    DEFAULT_SELECTIVITY = {
        "eq": 0.01,       # = matches ~1% of rows
        "ne": 0.99,       # != matches ~99%
        "gt": 0.33,       # > matches ~33%
        "gte": 0.34,      # >= matches ~34%
        "lt": 0.33,       # < matches ~33%
        "lte": 0.34,      # <= matches ~34%
        "contains": 0.1,  # LIKE matches ~10%
        "like": 0.1,
        "in": 0.05,       # IN matches ~5%
        "between": 0.1,   # BETWEEN matches ~10%
        "is_null": 0.05,  # NULL is rare
        "is_not_null": 0.95,
    }

    def __init__(
        self,
        table_stats: Optional[Dict[str, int]] = None,
        default_table_size: int = 10000,
        selectivity_overrides: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the row estimator.

        Args:
            table_stats: Dict mapping table names to row counts
            default_table_size: Default row count for unknown tables
            selectivity_overrides: Custom selectivity for operators
        """
        self.table_stats = table_stats or {}
        self.default_table_size = default_table_size
        self.selectivity = {**self.DEFAULT_SELECTIVITY}
        if selectivity_overrides:
            self.selectivity.update(selectivity_overrides)

    def estimate(self, intent: Intent) -> RowEstimate:
        """
        Estimate rows affected by an Intent.

        Args:
            intent: The Intent to estimate

        Returns:
            RowEstimate with the estimate
        """
        # Get table size
        table = intent.get_target_name()
        table_size = self._get_table_size(table)

        # Handle different operation types
        if intent.operation == OperationType.QUERY:
            return self._estimate_query(intent, table_size)
        elif intent.operation == OperationType.STORE:
            # INSERT always affects 1 row
            return RowEstimate.exact(1)
        elif intent.operation == OperationType.UPDATE:
            return self._estimate_mutation(intent, table_size)
        elif intent.operation == OperationType.DELETE:
            return self._estimate_mutation(intent, table_size)
        else:
            return RowEstimate.unbounded(table_size)

    def estimate_from_sql(self, sql: str, params: Dict[str, Any]) -> RowEstimate:
        """
        Estimate rows from raw SQL.

        Args:
            sql: The SQL query
            params: Query parameters

        Returns:
            RowEstimate with the estimate
        """
        sql_upper = sql.upper()

        # Check for LIMIT clause
        limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
        if limit_match:
            limit = int(limit_match.group(1))
            return RowEstimate.bounded(limit, limit)

        # Check for single-row by ID
        if "WHERE" in sql_upper and ("= :" in sql or "= ?" in sql):
            # Single equality predicate - likely single row
            return RowEstimate.bounded(1, 10)

        # INSERT always 1 row
        if sql_upper.startswith("INSERT"):
            return RowEstimate.exact(1)

        # Count predicates for UPDATE/DELETE
        if sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE"):
            if "WHERE" not in sql_upper:
                # No WHERE clause - affects all rows
                return RowEstimate.unbounded(self.default_table_size)

            # Count AND clauses as rough filter estimate
            and_count = sql_upper.count(" AND ")
            selectivity = 0.33 ** (and_count + 1)
            estimate = int(self.default_table_size * selectivity)
            return RowEstimate(
                count=estimate,
                confidence=0.5,
                is_bounded=False,
                lower_bound=0,
                method="sql_analysis",
            )

        # Default for SELECT without LIMIT
        return RowEstimate.unbounded(self.default_table_size)

    def update_table_stats(self, table: str, row_count: int) -> None:
        """Update statistics for a table."""
        self.table_stats[table.lower()] = row_count

    def _get_table_size(self, table: Optional[str]) -> int:
        """Get the row count for a table."""
        if not table:
            return self.default_table_size

        return self.table_stats.get(table.lower(), self.default_table_size)

    def _estimate_query(self, intent: Intent, table_size: int) -> RowEstimate:
        """Estimate rows for a SELECT query."""
        # Check for explicit LIMIT in bindings
        limit = intent.bindings.get("limit")
        if limit is not None:
            return RowEstimate.bounded(int(limit), int(limit))

        # Apply predicate selectivity
        selectivity = self._compute_selectivity(intent.predicates)
        estimate = int(table_size * selectivity)

        return RowEstimate(
            count=estimate,
            confidence=0.6,
            is_bounded=False,
            lower_bound=0,
            upper_bound=table_size,
            method="selectivity",
        )

    def _estimate_mutation(self, intent: Intent, table_size: int) -> RowEstimate:
        """Estimate rows for UPDATE or DELETE."""
        # No predicates = all rows
        if not intent.predicates:
            return RowEstimate.unbounded(table_size)

        # Apply predicate selectivity
        selectivity = self._compute_selectivity(intent.predicates)
        estimate = int(table_size * selectivity)

        # Mutations without WHERE are dangerous
        confidence = 0.7 if intent.predicates else 0.3

        return RowEstimate(
            count=estimate,
            confidence=confidence,
            is_bounded=False,
            lower_bound=0,
            upper_bound=table_size,
            method="selectivity",
        )

    def _compute_selectivity(self, predicates: List[Predicate]) -> float:
        """
        Compute combined selectivity of predicates.

        Uses independence assumption: combined = product of individual
        """
        if not predicates:
            return 1.0

        selectivity = 1.0
        for pred in predicates:
            op_selectivity = self.selectivity.get(pred.operator, 0.5)
            selectivity *= op_selectivity

        return max(selectivity, 0.001)  # Floor at 0.1%

    def is_dangerous(
        self,
        estimate: RowEstimate,
        threshold: int = 1000,
    ) -> bool:
        """
        Check if an operation is dangerous based on row estimate.

        Args:
            estimate: The row estimate
            threshold: Danger threshold

        Returns:
            True if operation is considered dangerous
        """
        # Unbounded estimates over threshold are dangerous
        if not estimate.is_bounded and estimate.count > threshold:
            return True

        # Even bounded, if upper bound is high, it's risky
        if estimate.upper_bound and estimate.upper_bound > threshold:
            return True

        return False
