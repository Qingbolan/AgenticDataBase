"""
Transaction management agents for AgenticDB.

This module provides agents for coordinating and executing transactions
through the state machine.

Agents:
    - CoordinatorAgent: Orchestrate transaction state machine
    - ExecutorAgent: Execute SQL operations
"""

from .types import (
    ExecutionResult,
    ConfirmationRequest,
    TransactionError,
    CoordinationResult,
)

__all__ = [
    "ExecutionResult",
    "ConfirmationRequest",
    "TransactionError",
    "CoordinationResult",
]
