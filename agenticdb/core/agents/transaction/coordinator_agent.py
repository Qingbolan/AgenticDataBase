# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Transaction Coordinator Agent for AgenticDB.

Orchestrates transaction state machine and manages lifecycle.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from ..intent import IntentParserAgent, BindingAgent, ValidationAgent
from ..intent.types import BindingContext
from ...models import (
    Intent,
    IntentState,
    Transaction,
    TransactionState,
    TransactionResult,
)
from ...tools.transaction.state_machine import StateMachine, TransitionError
from ...tools.intent.query_builder import QueryBuilder
from .types import CoordinationResult, ConfirmationRequest, ExecutionResult


class CoordinatorAgent(BaseAgent[CoordinationResult]):
    """
    Coordinate transaction state machine.

    This agent orchestrates the full transaction lifecycle:
    1. Parse natural language → Intent
    2. Resolve bindings if needed
    3. Validate Intent for safety
    4. Execute or request confirmation
    5. Return structured results

    Example:
        coordinator = CoordinatorAgent(storage=storage)
        result = coordinator.process("show orders from last week")
        # → CoordinationResult(state=COMPLETED, result=...)
    """

    name = "coordinator"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        parser: Optional[IntentParserAgent] = None,
        binder: Optional[BindingAgent] = None,
        validator: Optional[ValidationAgent] = None,
        available_tables: Optional[List[str]] = None,
    ):
        """
        Initialize the Coordinator Agent.

        Args:
            model: LLM model to use
            prompts_dir: Directory containing prompt templates
            parser: IntentParserAgent instance
            binder: BindingAgent instance
            validator: ValidationAgent instance
            available_tables: List of known tables
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "transaction"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.available_tables = available_tables or []

        # Initialize sub-agents
        self.parser = parser or IntentParserAgent(
            model=model,
            available_tables=self.available_tables,
        )
        self.binder = binder or BindingAgent(model=model)
        self.validator = validator or ValidationAgent(model=model)

        # Initialize tools
        self.state_machine = StateMachine(strict=False)
        self.query_builder = QueryBuilder()

        # Transaction registry
        self._transactions: Dict[str, Transaction] = {}

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("coordinator_system.md")
        except Exception:
            self.system_prompt = self._get_default_system_prompt()

    def run(
        self,
        ctx: AgentContext,
        query: str,
        bindings: Optional[Dict[str, Any]] = None,
        confirmed: bool = False,
    ) -> CoordinationResult:
        """
        Process a query through the transaction pipeline.

        Args:
            ctx: Agent context
            query: Natural language query
            bindings: Optional explicit bindings
            confirmed: Whether user has confirmed (for dangerous operations)

        Returns:
            CoordinationResult with transaction outcome
        """
        # Create transaction
        transaction = Transaction.create()
        self._transactions[transaction.id] = transaction

        # Transition to RECEIVED
        result = self.state_machine.transition(
            transaction, TransactionState.RECEIVED, "Transaction created"
        )
        transaction = result.transaction

        # Parse query into Intent
        parse_result = self.parser.run(ctx, query)
        if not parse_result.success:
            return self._reject(transaction, f"Parse failed: {parse_result.error}")

        intent = parse_result.intent
        transaction = transaction.with_intent(intent)

        # Transition to PARSED
        result = self.state_machine.transition(
            transaction, TransactionState.PARSED, "Intent parsed"
        )
        transaction = result.transaction

        # Check if binding is needed
        if intent.state == IntentState.PARTIAL:
            # Try to bind with provided bindings
            if bindings:
                bind_result = self.binder.run(ctx, intent, bindings=bindings)
                if bind_result.success and bind_result.intent.state == IntentState.COMPLETE:
                    intent = bind_result.intent
                    transaction = transaction.with_intent(intent)
                else:
                    # Still needs binding
                    result = self.state_machine.transition(
                        transaction, TransactionState.PENDING_BINDING, "Unbound slots remain"
                    )
                    return CoordinationResult.pending_binding(
                        result.transaction,
                        bind_result.remaining_slots,
                    )
            else:
                # Try auto-binding
                bind_context = BindingContext(
                    available_tables=self.available_tables,
                )
                bind_result = self.binder.run(ctx, intent, context=bind_context)

                if bind_result.intent.state != IntentState.COMPLETE:
                    result = self.state_machine.transition(
                        transaction, TransactionState.PENDING_BINDING, "Unbound slots"
                    )
                    transaction = result.transaction
                    return CoordinationResult.pending_binding(
                        transaction,
                        bind_result.remaining_slots,
                    )

                intent = bind_result.intent
                transaction = transaction.with_intent(intent)

        # Transition to BOUND
        result = self.state_machine.transition(
            transaction, TransactionState.BOUND, "All slots bound"
        )
        transaction = result.transaction

        # Validate
        validation = self.validator.run(ctx, intent)

        if not validation.valid:
            return self._reject(transaction, validation.errors[0] if validation.errors else "Validation failed")

        if validation.requires_confirmation and not confirmed:
            result = self.state_machine.transition(
                transaction, TransactionState.PENDING_CONFIRMATION, validation.confirmation_reason
            )
            transaction = result.transaction

            confirmation_request = ConfirmationRequest(
                transaction_id=transaction.id,
                reason=validation.confirmation_reason or "Operation requires confirmation",
                operation=intent.operation.value,
                affected_rows=validation.affected_rows_estimate,
                target=intent.get_target_name(),
            )

            return CoordinationResult.pending_confirmation(
                transaction,
                confirmation_request,
            )

        # Transition to VALIDATED
        result = self.state_machine.transition(
            transaction, TransactionState.VALIDATED, "Validation passed"
        )
        transaction = result.transaction

        # Build and execute query
        build_result = self.query_builder.build(intent)
        if not build_result.success:
            return self._fail(transaction, f"Query build failed: {build_result.error}")

        # Transition to EXECUTED (actual execution would happen here)
        result = self.state_machine.transition(
            transaction, TransactionState.EXECUTED, "Query executed"
        )
        transaction = result.transaction

        # Create execution result (in real implementation, this would be from actual DB execution)
        exec_result = ExecutionResult(
            success=True,
            sql=build_result.query.sql,
            parameters=build_result.query.parameters,
            data=[],  # Would be populated by actual execution
            affected_rows=0,
        )

        # Transition to COMPLETED
        result = self.state_machine.complete(
            transaction,
            exec_result.to_transaction_result(),
        )
        transaction = result.transaction
        self._transactions[transaction.id] = transaction

        return CoordinationResult.completed(transaction, exec_result)

    def bind(
        self,
        ctx: AgentContext,
        transaction_id: str,
        bindings: Dict[str, Any],
    ) -> CoordinationResult:
        """
        Provide bindings for a pending transaction.

        Args:
            ctx: Agent context
            transaction_id: ID of pending transaction
            bindings: Bindings to apply

        Returns:
            CoordinationResult with updated transaction
        """
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            return CoordinationResult.failed(
                Transaction.create(),
                f"Transaction not found: {transaction_id}",
            )

        if transaction.state != TransactionState.PENDING_BINDING:
            return CoordinationResult.failed(
                transaction,
                f"Transaction not in PENDING_BINDING state: {transaction.state.value}",
            )

        if not transaction.intent:
            return CoordinationResult.failed(transaction, "No intent on transaction")

        # Apply bindings
        bind_result = self.binder.run(ctx, transaction.intent, bindings=bindings)

        if not bind_result.success:
            return CoordinationResult.failed(transaction, bind_result.error or "Binding failed")

        # Update transaction with new intent
        transaction = transaction.with_intent(bind_result.intent)

        if bind_result.remaining_slots:
            # Still needs more bindings
            return CoordinationResult.pending_binding(
                transaction,
                bind_result.remaining_slots,
            )

        # Continue processing from BOUND state
        self._transactions[transaction.id] = transaction
        return self.run(
            ctx,
            transaction.intent.raw_input,
            confirmed=False,
        )

    def confirm(
        self,
        ctx: AgentContext,
        transaction_id: str,
        confirmed: bool = True,
    ) -> CoordinationResult:
        """
        Confirm or reject a pending confirmation.

        Args:
            ctx: Agent context
            transaction_id: ID of pending transaction
            confirmed: Whether to confirm

        Returns:
            CoordinationResult with updated transaction
        """
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            return CoordinationResult.failed(
                Transaction.create(),
                f"Transaction not found: {transaction_id}",
            )

        if transaction.state != TransactionState.PENDING_CONFIRMATION:
            return CoordinationResult.failed(
                transaction,
                f"Transaction not in PENDING_CONFIRMATION state: {transaction.state.value}",
            )

        if not confirmed:
            return self._reject(transaction, "User rejected operation")

        # Continue processing with confirmation
        transaction = transaction.with_confirmation(True)
        self._transactions[transaction.id] = transaction

        return self.run(
            ctx,
            transaction.intent.raw_input if transaction.intent else "",
            confirmed=True,
        )

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID."""
        return self._transactions.get(transaction_id)

    def update_available_tables(self, tables: List[str]) -> None:
        """Update the list of available tables."""
        self.available_tables = tables
        self.parser.update_available_tables(tables)

    def _reject(self, transaction: Transaction, reason: str) -> CoordinationResult:
        """Reject a transaction."""
        result = self.state_machine.reject(transaction, reason)
        self._transactions[result.transaction.id] = result.transaction
        return CoordinationResult.rejected(result.transaction, reason)

    def _fail(self, transaction: Transaction, error: str) -> CoordinationResult:
        """Mark transaction as failed."""
        result = self.state_machine.fail(transaction, error)
        self._transactions[result.transaction.id] = result.transaction
        return CoordinationResult.failed(result.transaction, error)

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a Transaction Coordinator. Manage transaction state transitions.

States: RECEIVED → PARSED → BOUND → VALIDATED → EXECUTED → COMPLETED
        Also: PENDING_BINDING, PENDING_CONFIRMATION, REJECTED, FAILED

Determine appropriate state transitions based on Intent completeness and validation results."""
