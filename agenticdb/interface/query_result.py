# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Query result types for AgenticDB.

Provides structured result objects for the Intent-Aware Transaction API.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.models import TransactionState


@dataclass
class QueryResult:
    """
    Result of a query operation.

    This is the primary return type for the Intent-Aware Transaction API.
    It encapsulates the status, data, and any pending information.

    Attributes:
        status: Current status (completed, pending_binding, pending_confirmation, rejected, failed)
        success: Whether the operation succeeded
        data: Query result data (for SELECT operations)
        affected_rows: Number of rows affected (for mutations)
        transaction_id: Transaction ID for pending operations
        unbound_slots: List of slots needing binding (if pending_binding)
        confirmation_reason: Reason for confirmation (if pending_confirmation)
        error: Error message if failed
        execution_time_ms: Execution time in milliseconds
    """

    status: str
    success: bool = True
    data: Optional[Any] = None
    affected_rows: int = 0
    transaction_id: Optional[str] = None
    unbound_slots: List[str] = field(default_factory=list)
    confirmation_reason: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    @classmethod
    def success(
        cls,
        data: Any = None,
        affected_rows: int = 0,
        execution_time_ms: float = 0.0,
    ) -> "QueryResult":
        """Create a successful result."""
        return cls(
            status="completed",
            success=True,
            data=data,
            affected_rows=affected_rows,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def pending_binding(
        cls,
        transaction_id: str,
        unbound_slots: List[str],
    ) -> "QueryResult":
        """Create a pending binding result."""
        return cls(
            status="pending_binding",
            success=True,
            transaction_id=transaction_id,
            unbound_slots=unbound_slots,
        )

    @classmethod
    def pending_confirmation(
        cls,
        transaction_id: str,
        reason: str,
        affected_rows: Optional[int] = None,
    ) -> "QueryResult":
        """Create a pending confirmation result."""
        return cls(
            status="pending_confirmation",
            success=True,
            transaction_id=transaction_id,
            confirmation_reason=reason,
            affected_rows=affected_rows or 0,
        )

    @classmethod
    def rejected(cls, reason: str, transaction_id: Optional[str] = None) -> "QueryResult":
        """Create a rejected result."""
        return cls(
            status="rejected",
            success=False,
            error=reason,
            transaction_id=transaction_id,
        )

    @classmethod
    def error(cls, message: str) -> "QueryResult":
        """Create an error result."""
        return cls(
            status="failed",
            success=False,
            error=message,
        )

    @classmethod
    def from_coordination_result(cls, coord_result) -> "QueryResult":
        """Create from a CoordinationResult."""
        if coord_result.needs_binding:
            return cls.pending_binding(
                coord_result.transaction.id,
                coord_result.pending_slots,
            )

        if coord_result.needs_confirmation:
            return cls.pending_confirmation(
                coord_result.transaction.id,
                coord_result.confirmation_request.reason if coord_result.confirmation_request else "Confirmation required",
                coord_result.confirmation_request.affected_rows if coord_result.confirmation_request else None,
            )

        if coord_result.state == TransactionState.REJECTED:
            return cls.rejected(
                coord_result.error or "Request rejected",
                coord_result.transaction.id,
            )

        if coord_result.state == TransactionState.FAILED:
            return cls.error(coord_result.error or "Operation failed")

        if coord_result.state == TransactionState.COMPLETED and coord_result.result:
            return cls.success(
                data=coord_result.result.data,
                affected_rows=coord_result.result.affected_rows,
                execution_time_ms=coord_result.result.execution_time_ms,
            )

        return cls(
            status=coord_result.state.value,
            success=coord_result.success,
            transaction_id=coord_result.transaction.id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "status": self.status,
            "success": self.success,
        }

        if self.data is not None:
            result["data"] = self.data

        if self.affected_rows > 0:
            result["affected_rows"] = self.affected_rows

        if self.transaction_id:
            result["transaction_id"] = self.transaction_id

        if self.unbound_slots:
            result["unbound_slots"] = self.unbound_slots

        if self.confirmation_reason:
            result["confirmation_reason"] = self.confirmation_reason

        if self.error:
            result["error"] = self.error

        if self.execution_time_ms > 0:
            result["execution_time_ms"] = self.execution_time_ms

        return result

    def __repr__(self) -> str:
        if self.success:
            if self.status == "completed":
                return f"QueryResult(status='completed', rows={self.affected_rows})"
            elif self.status == "pending_binding":
                return f"QueryResult(status='pending_binding', slots={self.unbound_slots})"
            elif self.status == "pending_confirmation":
                return f"QueryResult(status='pending_confirmation', reason='{self.confirmation_reason}')"
        return f"QueryResult(status='{self.status}', error='{self.error}')"
