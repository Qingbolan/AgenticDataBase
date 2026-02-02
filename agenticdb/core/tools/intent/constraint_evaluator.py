"""
Constraint evaluation tool for Intent validation.

Pure computation - no LLM calls. Evaluates safety constraints
against Intent and execution context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...models import (
    Intent,
    OperationType,
    SafetyConstraint,
)


@dataclass
class ConstraintViolation:
    """
    A constraint violation result.

    Attributes:
        constraint: The violated constraint
        message: Human-readable violation message
        severity: Violation severity (warning, error, critical)
        requires_confirmation: Whether user confirmation can override
        context: Additional context about the violation
    """

    constraint: SafetyConstraint
    message: str
    severity: str = "warning"
    requires_confirmation: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Result of constraint evaluation.

    Attributes:
        passed: Whether all constraints passed
        violations: List of constraint violations
        warnings: Non-blocking warnings
        requires_confirmation: Whether confirmation is needed
        confirmation_reason: Combined reason for confirmation
        can_proceed: Whether operation can proceed (possibly with confirmation)
    """

    passed: bool = True
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_reason: Optional[str] = None
    can_proceed: bool = True

    @classmethod
    def success(cls) -> "EvaluationResult":
        """Create a successful evaluation result."""
        return cls(passed=True, can_proceed=True)

    @classmethod
    def needs_confirmation(
        cls,
        reason: str,
        violations: Optional[List[ConstraintViolation]] = None,
    ) -> "EvaluationResult":
        """Create a result requiring confirmation."""
        return cls(
            passed=True,
            violations=violations or [],
            requires_confirmation=True,
            confirmation_reason=reason,
            can_proceed=True,
        )

    @classmethod
    def rejected(
        cls,
        violations: List[ConstraintViolation],
    ) -> "EvaluationResult":
        """Create a rejected result."""
        messages = [v.message for v in violations if v.severity == "critical"]
        return cls(
            passed=False,
            violations=violations,
            can_proceed=False,
            confirmation_reason="; ".join(messages) if messages else "Constraint violation",
        )


class ConstraintEvaluator:
    """
    Evaluate safety constraints against Intent and execution context.

    This is a pure computation tool (no LLM) that evaluates rules
    to determine if an operation is safe to execute.

    Default Constraints:
        - max_rows: Limit affected rows (default 1000)
        - no_delete_all: Require WHERE clause for DELETE
        - protected_tables: Require confirmation for sensitive tables
        - no_drop: Prohibit DROP operations

    Example:
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate(intent, {"affected_rows": 5000})
        # → EvaluationResult(passed=False, requires_confirmation=True, ...)
    """

    # Default constraints applied to all operations
    DEFAULT_CONSTRAINTS = [
        SafetyConstraint(
            constraint_type="max_rows",
            parameters={"limit": 1000},
            severity="warning",
            message="Operation affects more than 1000 rows",
        ),
        SafetyConstraint(
            constraint_type="no_drop",
            severity="critical",
            message="DROP operations are not allowed",
        ),
    ]

    # Additional constraints for DELETE operations
    DELETE_CONSTRAINTS = [
        SafetyConstraint(
            constraint_type="require_where",
            severity="error",
            message="DELETE requires a WHERE clause",
        ),
    ]

    # Tables that always require confirmation
    PROTECTED_TABLES = {"users", "accounts", "payments", "credentials", "secrets"}

    def __init__(
        self,
        custom_constraints: Optional[List[SafetyConstraint]] = None,
        max_rows_limit: int = 1000,
        protected_tables: Optional[set] = None,
    ):
        """
        Initialize the constraint evaluator.

        Args:
            custom_constraints: Additional constraints to apply
            max_rows_limit: Default max rows limit
            protected_tables: Set of table names requiring confirmation
        """
        self.custom_constraints = custom_constraints or []
        self.max_rows_limit = max_rows_limit
        self.protected_tables = protected_tables or self.PROTECTED_TABLES

    def evaluate(
        self,
        intent: Intent,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate all constraints against the Intent.

        Args:
            intent: The Intent to evaluate
            context: Execution context (affected_rows, sql, etc.)

        Returns:
            EvaluationResult with violations and recommendations
        """
        context = context or {}
        violations: List[ConstraintViolation] = []
        warnings: List[str] = []

        # Collect all applicable constraints
        constraints = self._get_applicable_constraints(intent)

        # Add intent's own constraints
        constraints.extend(intent.constraints)

        # Evaluate each constraint
        for constraint in constraints:
            passed, message = self._evaluate_constraint(constraint, intent, context)
            if not passed:
                violation = ConstraintViolation(
                    constraint=constraint,
                    message=message or constraint.message or f"Constraint {constraint.constraint_type} violated",
                    severity=constraint.severity,
                    requires_confirmation=constraint.severity in ("warning", "error"),
                    context=context,
                )
                violations.append(violation)
            elif message:
                # Constraint passed but has a warning
                warnings.append(message)

        # Determine overall result
        return self._compute_result(violations, warnings)

    def evaluate_row_count(
        self,
        affected_rows: int,
        operation: OperationType,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if the row count is acceptable.

        Args:
            affected_rows: Number of rows affected
            operation: The operation type

        Returns:
            Tuple of (is_acceptable, message_if_not)
        """
        # Queries can return any number of rows
        if operation == OperationType.QUERY:
            return True, None

        # Mutations have limits
        if affected_rows > self.max_rows_limit:
            return False, f"Operation affects {affected_rows} rows, exceeds limit of {self.max_rows_limit}"

        return True, None

    def evaluate_table_access(
        self,
        table_name: str,
        operation: OperationType,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if table access is allowed.

        Args:
            table_name: The table being accessed
            operation: The operation type

        Returns:
            Tuple of (needs_confirmation, reason)
        """
        table_lower = table_name.lower()

        # Protected tables require confirmation for mutations
        if table_lower in self.protected_tables:
            if operation in (OperationType.UPDATE, OperationType.DELETE):
                return True, f"Table '{table_name}' is protected and requires confirmation"

        return False, None

    def _get_applicable_constraints(self, intent: Intent) -> List[SafetyConstraint]:
        """Get all constraints applicable to this Intent."""
        constraints = list(self.DEFAULT_CONSTRAINTS)
        constraints.extend(self.custom_constraints)

        # Add operation-specific constraints
        if intent.operation == OperationType.DELETE:
            constraints.extend(self.DELETE_CONSTRAINTS)

        return constraints

    def _evaluate_constraint(
        self,
        constraint: SafetyConstraint,
        intent: Intent,
        context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate a single constraint.

        Returns:
            Tuple of (passed, message)
        """
        # Build full context
        full_context = {
            "operation": intent.operation,
            "target_table": intent.get_target_name(),
            "predicates": len(intent.predicates),
            "confirmed": context.get("confirmed", False),
            **context,
        }

        # Use constraint's own evaluate method
        return constraint.evaluate(full_context)

    def _compute_result(
        self,
        violations: List[ConstraintViolation],
        warnings: List[str],
    ) -> EvaluationResult:
        """Compute the final evaluation result."""
        if not violations:
            return EvaluationResult(
                passed=True,
                warnings=warnings,
                can_proceed=True,
            )

        # Check for critical violations (cannot proceed)
        critical = [v for v in violations if v.severity == "critical"]
        if critical:
            return EvaluationResult.rejected(violations)

        # Check for error violations (require confirmation)
        errors = [v for v in violations if v.severity == "error"]
        confirmable = [v for v in violations if v.requires_confirmation]

        if errors or confirmable:
            reasons = [v.message for v in (errors + confirmable)]
            return EvaluationResult.needs_confirmation(
                reason="; ".join(reasons),
                violations=violations,
            )

        # Only warnings - can proceed with warnings
        return EvaluationResult(
            passed=True,
            violations=violations,
            warnings=warnings + [v.message for v in violations],
            can_proceed=True,
        )

    def add_protected_table(self, table_name: str) -> None:
        """Add a table to the protected set."""
        self.protected_tables.add(table_name.lower())

    def remove_protected_table(self, table_name: str) -> None:
        """Remove a table from the protected set."""
        self.protected_tables.discard(table_name.lower())

    def set_max_rows_limit(self, limit: int) -> None:
        """Set the maximum rows limit."""
        self.max_rows_limit = limit
        # Update the default constraint
        for constraint in self.DEFAULT_CONSTRAINTS:
            if constraint.constraint_type == "max_rows":
                constraint.parameters["limit"] = limit
