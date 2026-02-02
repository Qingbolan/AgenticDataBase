# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Validation Agent for AgenticDB.

Validates Intent IR for safety and correctness.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from ...models import (
    Intent,
    IntentState,
    OperationType,
    SafetyConstraint,
)
from ...tools.intent.constraint_evaluator import ConstraintEvaluator, EvaluationResult
from ...tools.transaction.row_estimator import RowEstimator
from .types import ValidationResult, ValidationError


class ValidationAgent(BaseAgent[ValidationResult]):
    """
    Validate Intent IR for safety and correctness.

    This agent checks intents against safety constraints and
    determines if confirmation is required for dangerous operations.

    Validation Checks:
    1. Safety constraints (row limits, protected tables)
    2. Schema compatibility (tables/columns exist)
    3. Type compatibility (values match column types)
    4. Dangerous operation detection

    Example:
        agent = ValidationAgent()
        result = agent.run(ctx, intent)
        # → ValidationResult(valid=True, requires_confirmation=True, ...)
    """

    name = "validation"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        max_rows_limit: int = 1000,
        protected_tables: Optional[set] = None,
        custom_constraints: Optional[List[SafetyConstraint]] = None,
    ):
        """
        Initialize the Validation Agent.

        Args:
            model: LLM model to use
            prompts_dir: Directory containing prompt templates
            max_rows_limit: Maximum rows for operations without confirmation
            protected_tables: Tables requiring confirmation for mutations
            custom_constraints: Additional safety constraints
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "intent"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.constraint_evaluator = ConstraintEvaluator(
            max_rows_limit=max_rows_limit,
            protected_tables=protected_tables,
            custom_constraints=custom_constraints,
        )
        self.row_estimator = RowEstimator()

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("validation_system.md")
        except Exception:
            self.system_prompt = self._get_default_system_prompt()

    def run(
        self,
        ctx: AgentContext,
        intent: Intent,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate an Intent.

        Args:
            ctx: Agent context
            intent: Intent to validate
            context: Additional context (table stats, schema info)

        Returns:
            ValidationResult with validation outcome
        """
        context = context or {}
        errors: List[str] = []
        warnings: List[str] = []
        violated_constraints: List[SafetyConstraint] = []

        # Check if Intent is complete
        if intent.state == IntentState.PARTIAL:
            unbound = intent.get_unbound_slot_names()
            return ValidationResult(
                valid=False,
                intent=intent,
                errors=[f"Intent has unbound slots: {unbound}"],
            )

        if intent.state == IntentState.INVALID:
            return ValidationResult(
                valid=False,
                intent=intent,
                errors=["Intent is already marked as invalid"],
            )

        # Estimate affected rows
        row_estimate = self.row_estimator.estimate(intent)
        context["affected_rows"] = row_estimate.count

        # Evaluate constraints
        eval_result = self.constraint_evaluator.evaluate(intent, context)

        # Collect errors and warnings
        for violation in eval_result.violations:
            if violation.severity == "critical":
                errors.append(violation.message)
            elif violation.severity == "error":
                if violation.requires_confirmation:
                    warnings.append(violation.message)
                else:
                    errors.append(violation.message)
            else:
                warnings.append(violation.message)
            violated_constraints.append(violation.constraint)

        warnings.extend(eval_result.warnings)

        # Determine if confirmation is required
        requires_confirmation = eval_result.requires_confirmation
        confirmation_reason = eval_result.confirmation_reason

        # Additional checks for dangerous operations
        if intent.operation == OperationType.DELETE:
            if not intent.predicates:
                requires_confirmation = True
                confirmation_reason = "DELETE without WHERE clause will delete all rows"
                warnings.append(confirmation_reason)

            if not requires_confirmation and row_estimate.count > 100:
                requires_confirmation = True
                confirmation_reason = f"DELETE will affect approximately {row_estimate.count} rows"

        if intent.operation == OperationType.UPDATE:
            if not intent.predicates:
                requires_confirmation = True
                confirmation_reason = "UPDATE without WHERE clause will update all rows"
                warnings.append(confirmation_reason)

        # Check protected tables
        target = intent.get_target_name()
        if target:
            needs_confirm, reason = self.constraint_evaluator.evaluate_table_access(
                target, intent.operation
            )
            if needs_confirm and not requires_confirmation:
                requires_confirmation = True
                confirmation_reason = reason

        # Build result
        if errors and not eval_result.can_proceed:
            return ValidationResult.rejected(
                intent=intent,
                errors=errors,
                constraints=violated_constraints,
            )

        if requires_confirmation:
            return ValidationResult.needs_confirmation(
                intent=intent,
                reason=confirmation_reason or "Operation requires confirmation",
                affected_rows=row_estimate.count,
            )

        return ValidationResult(
            valid=True,
            intent=intent,
            warnings=warnings,
            affected_rows_estimate=row_estimate.count,
        )

    def validate_quick(self, intent: Intent) -> bool:
        """
        Quick validation check without full analysis.

        Args:
            intent: Intent to validate

        Returns:
            True if intent passes basic validation
        """
        # Must be complete
        if intent.state != IntentState.COMPLETE:
            return False

        # Must have target
        if not intent.get_target_name():
            return False

        return True

    def add_constraint(self, constraint: SafetyConstraint) -> None:
        """Add a custom safety constraint."""
        self.constraint_evaluator.custom_constraints.append(constraint)

    def add_protected_table(self, table_name: str) -> None:
        """Add a table to the protected set."""
        self.constraint_evaluator.add_protected_table(table_name)

    def set_max_rows_limit(self, limit: int) -> None:
        """Set the maximum rows limit."""
        self.constraint_evaluator.set_max_rows_limit(limit)

    def update_table_stats(self, table: str, row_count: int) -> None:
        """Update row count statistics for a table."""
        self.row_estimator.update_table_stats(table, row_count)

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a Validation Agent. Validate Intent objects for safety and correctness.

Check for:
1. Destructive operations without WHERE clause
2. Operations affecting many rows
3. Access to protected tables
4. Invalid column/table references
5. Type mismatches

Return JSON with:
- valid: true/false
- errors: list of error messages
- warnings: list of warning messages
- requires_confirmation: true/false
- confirmation_reason: reason if confirmation needed
- affected_rows_estimate: estimated rows affected
- risk_level: low/medium/high/critical"""
