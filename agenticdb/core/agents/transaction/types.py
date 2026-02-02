"""
Type definitions for Transaction management agents.

These types define the input/output contracts for Transaction agents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...models import Transaction, TransactionState, TransactionResult


class TransactionError(Exception):
    """
    Error raised during transaction processing.

    This includes:
    - Invalid state transitions
    - Execution failures
    - Timeout errors
    """

    def __init__(
        self,
        message: str,
        transaction_id: Optional[str] = None,
        state: Optional[TransactionState] = None,
    ):
        super().__init__(message)
        self.transaction_id = transaction_id
        self.state = state


@dataclass
class ExecutionResult:
    """
    Result of executing a transaction.

    Attributes:
        success: Whether execution succeeded
        data: Query result data (for SELECT operations)
        affected_rows: Number of rows affected (for mutations)
        error: Error message if execution failed
        execution_time_ms: Execution time in milliseconds
        sql: The SQL that was executed
        parameters: Parameters used in the SQL
    """

    success: bool = True
    data: Optional[Any] = None
    affected_rows: int = 0
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    sql: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_query(
        cls,
        data: Any,
        execution_time_ms: float,
        sql: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Create a successful query result."""
        row_count = len(data) if isinstance(data, list) else 0
        return cls(
            success=True,
            data=data,
            affected_rows=row_count,
            execution_time_ms=execution_time_ms,
            sql=sql,
            parameters=parameters or {},
        )

    @classmethod
    def from_mutation(
        cls,
        affected_rows: int,
        execution_time_ms: float,
        sql: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Create a successful mutation result."""
        return cls(
            success=True,
            affected_rows=affected_rows,
            execution_time_ms=execution_time_ms,
            sql=sql,
            parameters=parameters or {},
        )

    @classmethod
    def from_error(cls, error: str, sql: Optional[str] = None) -> "ExecutionResult":
        """Create a failed execution result."""
        return cls(
            success=False,
            error=error,
            sql=sql,
        )

    def to_transaction_result(self) -> TransactionResult:
        """Convert to TransactionResult model."""
        return TransactionResult(
            success=self.success,
            data=self.data,
            affected_rows=self.affected_rows,
            error=self.error,
            execution_time_ms=self.execution_time_ms,
        )


@dataclass
class ConfirmationRequest:
    """
    Request for user confirmation before executing dangerous operation.

    Attributes:
        transaction_id: ID of the transaction requiring confirmation
        reason: Why confirmation is required
        operation: The operation type (DELETE, UPDATE, etc.)
        affected_rows: Estimated number of rows affected
        target: Target table or entity
        sql_preview: Preview of the SQL that will be executed
        warnings: List of warnings about this operation
        timeout_seconds: How long to wait for confirmation
    """

    transaction_id: str
    reason: str
    operation: str
    affected_rows: Optional[int] = None
    target: Optional[str] = None
    sql_preview: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timeout_seconds: int = 300  # 5 minutes default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "status": "pending_confirmation",
            "transaction_id": self.transaction_id,
            "reason": self.reason,
            "operation": self.operation,
            "affected_rows": self.affected_rows,
            "target": self.target,
            "sql_preview": self.sql_preview,
            "warnings": self.warnings,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class CoordinationResult:
    """
    Result of transaction coordination.

    This represents the outcome of processing a transaction through
    the state machine.

    Attributes:
        transaction: The transaction after coordination
        state: Final state of the transaction
        success: Whether coordination succeeded
        error: Error message if coordination failed
        needs_binding: Whether binding is required
        needs_confirmation: Whether confirmation is required
        pending_slots: Slots that need binding
        confirmation_request: Confirmation request if needed
        result: Execution result if completed
    """

    transaction: Transaction
    state: TransactionState
    success: bool = True
    error: Optional[str] = None

    # Pending states
    needs_binding: bool = False
    needs_confirmation: bool = False
    pending_slots: List[str] = field(default_factory=list)
    confirmation_request: Optional[ConfirmationRequest] = None

    # Execution result
    result: Optional[ExecutionResult] = None

    @classmethod
    def pending_binding(
        cls,
        transaction: Transaction,
        pending_slots: List[str],
    ) -> "CoordinationResult":
        """Create a result indicating binding is needed."""
        return cls(
            transaction=transaction,
            state=TransactionState.PENDING_BINDING,
            needs_binding=True,
            pending_slots=pending_slots,
        )

    @classmethod
    def pending_confirmation(
        cls,
        transaction: Transaction,
        confirmation_request: ConfirmationRequest,
    ) -> "CoordinationResult":
        """Create a result indicating confirmation is needed."""
        return cls(
            transaction=transaction,
            state=TransactionState.PENDING_CONFIRMATION,
            needs_confirmation=True,
            confirmation_request=confirmation_request,
        )

    @classmethod
    def completed(
        cls,
        transaction: Transaction,
        result: ExecutionResult,
    ) -> "CoordinationResult":
        """Create a completed result."""
        return cls(
            transaction=transaction,
            state=TransactionState.COMPLETED,
            result=result,
        )

    @classmethod
    def rejected(
        cls,
        transaction: Transaction,
        error: str,
    ) -> "CoordinationResult":
        """Create a rejected result."""
        return cls(
            transaction=transaction,
            state=TransactionState.REJECTED,
            success=False,
            error=error,
        )

    @classmethod
    def failed(
        cls,
        transaction: Transaction,
        error: str,
    ) -> "CoordinationResult":
        """Create a failed result."""
        return cls(
            transaction=transaction,
            state=TransactionState.FAILED,
            success=False,
            error=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        response: Dict[str, Any] = {
            "status": self.state.value,
            "transaction_id": self.transaction.id,
            "success": self.success,
        }

        if self.error:
            response["error"] = self.error

        if self.needs_binding:
            response["unbound_slots"] = self.pending_slots

        if self.needs_confirmation and self.confirmation_request:
            response.update(self.confirmation_request.to_dict())

        if self.result:
            response["data"] = self.result.data
            response["affected_rows"] = self.result.affected_rows
            response["execution_time_ms"] = self.result.execution_time_ms

        return response
