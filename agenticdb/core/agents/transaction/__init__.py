# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
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
from .coordinator_agent import CoordinatorAgent
from .executor_agent import ExecutorAgent

__all__ = [
    # Types
    "ExecutionResult",
    "ConfirmationRequest",
    "TransactionError",
    "CoordinationResult",
    # Agents
    "CoordinatorAgent",
    "ExecutorAgent",
]
