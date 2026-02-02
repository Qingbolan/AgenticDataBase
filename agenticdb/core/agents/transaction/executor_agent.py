# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Transaction Executor Agent for AgenticDB.

Executes validated transactions against storage.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from ..base.base_agent import AgentContext, BaseAgent
from ...models import (
    Intent,
    OperationType,
    Transaction,
    TransactionState,
    TransactionResult,
)
from ...tools.intent.query_builder import QueryBuilder, SQLQuery
from .types import ExecutionResult


# Type for SQL executor function
SQLExecutor = Callable[[str, Dict[str, Any]], Any]


class ExecutorAgent(BaseAgent[ExecutionResult]):
    """
    Execute validated transactions.

    This agent takes validated Intents, builds SQL, and executes
    against the storage layer.

    Responsibilities:
    1. Build parameterized SQL from Intent
    2. Execute against storage
    3. Handle errors gracefully
    4. Record metrics

    Example:
        executor = ExecutorAgent(storage=sqlite_storage)
        result = executor.run(ctx, intent)
        # → ExecutionResult(success=True, data=[...], ...)
    """

    name = "executor"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        sql_executor: Optional[SQLExecutor] = None,
        default_limit: int = 100,
    ):
        """
        Initialize the Executor Agent.

        Args:
            model: LLM model to use
            prompts_dir: Directory containing prompt templates
            sql_executor: Function to execute SQL
            default_limit: Default LIMIT for queries
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "transaction"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.sql_executor = sql_executor
        self.query_builder = QueryBuilder(default_limit=default_limit)

        # Metrics
        self._execution_count = 0
        self._total_time_ms = 0.0

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("executor_system.md")
        except Exception:
            self.system_prompt = self._get_default_system_prompt()

    def run(
        self,
        ctx: AgentContext,
        intent: Intent,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute an Intent.

        Args:
            ctx: Agent context
            intent: Validated Intent to execute
            dry_run: If True, build SQL but don't execute

        Returns:
            ExecutionResult with execution outcome
        """
        # Build SQL
        build_result = self.query_builder.build(intent)

        if not build_result.success:
            return ExecutionResult.from_error(
                build_result.error or "Query build failed"
            )

        query = build_result.query

        if dry_run:
            return ExecutionResult(
                success=True,
                sql=query.sql,
                parameters=query.parameters,
            )

        # Execute
        return self.execute_sql(query.sql, query.parameters, intent.operation)

    def execute_sql(
        self,
        sql: str,
        parameters: Dict[str, Any],
        operation: Optional[OperationType] = None,
    ) -> ExecutionResult:
        """
        Execute raw SQL.

        Args:
            sql: SQL query
            parameters: Query parameters
            operation: Operation type (for result handling)

        Returns:
            ExecutionResult with execution outcome
        """
        if not self.sql_executor:
            return ExecutionResult.from_error("No SQL executor configured")

        start_time = time.time()

        try:
            result = self.sql_executor(sql, parameters)
            elapsed_ms = (time.time() - start_time) * 1000

            # Update metrics
            self._execution_count += 1
            self._total_time_ms += elapsed_ms

            # Handle result based on operation type
            if operation == OperationType.QUERY:
                data = result if isinstance(result, list) else [result] if result else []
                return ExecutionResult.from_query(
                    data=data,
                    execution_time_ms=elapsed_ms,
                    sql=sql,
                    parameters=parameters,
                )
            else:
                # For mutations, result is typically affected row count
                affected = result if isinstance(result, int) else 1
                return ExecutionResult.from_mutation(
                    affected_rows=affected,
                    execution_time_ms=elapsed_ms,
                    sql=sql,
                    parameters=parameters,
                )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(f"SQL execution failed: {e}")

            return ExecutionResult.from_error(
                error=str(e),
                sql=sql,
            )

    def execute_transaction(
        self,
        ctx: AgentContext,
        transaction: Transaction,
    ) -> ExecutionResult:
        """
        Execute a Transaction.

        Args:
            ctx: Agent context
            transaction: Transaction to execute

        Returns:
            ExecutionResult with execution outcome
        """
        if transaction.state != TransactionState.VALIDATED:
            return ExecutionResult.from_error(
                f"Transaction must be VALIDATED, got {transaction.state.value}"
            )

        if not transaction.intent:
            return ExecutionResult.from_error("Transaction has no Intent")

        return self.run(ctx, transaction.intent)

    def execute_from_pattern(
        self,
        template: str,
        parameters: Dict[str, Any],
        operation: Optional[OperationType] = None,
    ) -> ExecutionResult:
        """
        Execute from a pattern template.

        Args:
            template: SQL template with {placeholders}
            parameters: Values for placeholders
            operation: Operation type

        Returns:
            ExecutionResult with execution outcome
        """
        build_result = self.query_builder.build_from_pattern(template, parameters)

        if not build_result.success:
            return ExecutionResult.from_error(
                build_result.error or "Template build failed"
            )

        return self.execute_sql(
            build_result.query.sql,
            build_result.query.parameters,
            operation,
        )

    def set_sql_executor(self, executor: SQLExecutor) -> None:
        """Set the SQL executor function."""
        self.sql_executor = executor

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        avg_time = (
            self._total_time_ms / self._execution_count
            if self._execution_count > 0
            else 0.0
        )

        return {
            "execution_count": self._execution_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": avg_time,
        }

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self._execution_count = 0
        self._total_time_ms = 0.0

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a Transaction Executor. Execute validated Intent objects.

1. Build parameterized SQL from Intent
2. Execute against storage
3. Handle errors gracefully
4. Return structured results

Always use parameterized queries to prevent SQL injection."""
