"""
Transaction state machine tool.

Pure computation - no LLM calls. Manages transaction state transitions
with validation and audit logging.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ...models import (
    Transaction,
    TransactionState,
    TransactionResult,
)


class TransitionError(Exception):
    """
    Error raised when an invalid state transition is attempted.

    Attributes:
        from_state: Current state
        to_state: Attempted target state
        message: Error description
    """

    def __init__(
        self,
        from_state: TransactionState,
        to_state: TransactionState,
        message: str,
    ):
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state


@dataclass
class TransitionResult:
    """
    Result of a state transition attempt.

    Attributes:
        success: Whether the transition was valid
        transaction: The transaction after transition (or unchanged if failed)
        error: Error message if transition failed
    """

    success: bool
    transaction: Transaction
    error: Optional[str] = None


class StateMachine:
    """
    Manage transaction state transitions.

    This is a pure computation tool (no LLM) that validates and
    applies state transitions according to the state machine rules.

    State Machine:
        RECEIVED → PARSED → BOUND → VALIDATED → EXECUTED → COMPLETED
                     │         │         │
                     │ partial │ unsafe  │ invalid
                     ▼         ▼         ▼
               PENDING     PENDING    REJECTED
               BINDING  CONFIRMATION

    Example:
        machine = StateMachine()
        result = machine.transition(tx, TransactionState.PARSED)
        # → TransitionResult(success=True, transaction=...)
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[TransactionState, Set[TransactionState]] = {
        TransactionState.RECEIVED: {
            TransactionState.PARSED,
            TransactionState.FAILED,
        },
        TransactionState.PARSED: {
            TransactionState.PENDING_BINDING,
            TransactionState.BOUND,
            TransactionState.REJECTED,
            TransactionState.FAILED,
        },
        TransactionState.PENDING_BINDING: {
            TransactionState.BOUND,
            TransactionState.REJECTED,
            TransactionState.FAILED,
        },
        TransactionState.BOUND: {
            TransactionState.PENDING_CONFIRMATION,
            TransactionState.VALIDATED,
            TransactionState.REJECTED,
            TransactionState.FAILED,
        },
        TransactionState.PENDING_CONFIRMATION: {
            TransactionState.VALIDATED,
            TransactionState.REJECTED,
            TransactionState.FAILED,
        },
        TransactionState.VALIDATED: {
            TransactionState.EXECUTED,
            TransactionState.FAILED,
        },
        TransactionState.EXECUTED: {
            TransactionState.COMPLETED,
            TransactionState.FAILED,
        },
        # Terminal states have no outgoing transitions
        TransactionState.COMPLETED: set(),
        TransactionState.REJECTED: set(),
        TransactionState.FAILED: set(),
    }

    # States that can be retried
    RETRYABLE_STATES = {
        TransactionState.PENDING_BINDING,
        TransactionState.PENDING_CONFIRMATION,
    }

    # Terminal states
    TERMINAL_STATES = {
        TransactionState.COMPLETED,
        TransactionState.REJECTED,
        TransactionState.FAILED,
    }

    def __init__(self, strict: bool = True):
        """
        Initialize the state machine.

        Args:
            strict: If True, raise errors on invalid transitions.
                   If False, return failed TransitionResult.
        """
        self.strict = strict

    def can_transition(
        self,
        from_state: TransactionState,
        to_state: TransactionState,
    ) -> bool:
        """
        Check if a transition is valid.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if transition is valid
        """
        valid_targets = self.VALID_TRANSITIONS.get(from_state, set())
        return to_state in valid_targets

    def transition(
        self,
        transaction: Transaction,
        new_state: TransactionState,
        reason: Optional[str] = None,
    ) -> TransitionResult:
        """
        Attempt to transition the transaction to a new state.

        Args:
            transaction: The transaction to transition
            new_state: Target state
            reason: Optional reason for the transition

        Returns:
            TransitionResult with the outcome

        Raises:
            TransitionError: If strict mode and transition is invalid
        """
        current_state = transaction.state

        # Check if transition is valid
        if not self.can_transition(current_state, new_state):
            error = f"Invalid transition: {current_state.value} → {new_state.value}"
            if self.strict:
                raise TransitionError(current_state, new_state, error)
            return TransitionResult(
                success=False,
                transaction=transaction,
                error=error,
            )

        # Apply the transition
        new_transaction = transaction.transition_to(new_state, reason)

        return TransitionResult(
            success=True,
            transaction=new_transaction,
        )

    def get_valid_transitions(
        self,
        state: TransactionState,
    ) -> List[TransactionState]:
        """
        Get all valid transitions from a state.

        Args:
            state: Current state

        Returns:
            List of valid target states
        """
        return list(self.VALID_TRANSITIONS.get(state, set()))

    def is_terminal(self, state: TransactionState) -> bool:
        """Check if a state is terminal."""
        return state in self.TERMINAL_STATES

    def is_pending(self, state: TransactionState) -> bool:
        """Check if a state is a pending state."""
        return state in {
            TransactionState.PENDING_BINDING,
            TransactionState.PENDING_CONFIRMATION,
        }

    def is_retryable(self, state: TransactionState) -> bool:
        """Check if a state can be retried."""
        return state in self.RETRYABLE_STATES

    def process_to_bound(
        self,
        transaction: Transaction,
        has_unbound_slots: bool,
    ) -> TransitionResult:
        """
        Process a parsed transaction to BOUND or PENDING_BINDING.

        Args:
            transaction: Transaction after parsing
            has_unbound_slots: Whether the Intent has unbound slots

        Returns:
            TransitionResult with new state
        """
        if transaction.state != TransactionState.PARSED:
            return TransitionResult(
                success=False,
                transaction=transaction,
                error=f"Expected PARSED state, got {transaction.state.value}",
            )

        if has_unbound_slots:
            return self.transition(
                transaction,
                TransactionState.PENDING_BINDING,
                "Intent has unbound slots",
            )
        else:
            return self.transition(
                transaction,
                TransactionState.BOUND,
                "All slots bound",
            )

    def process_to_validated(
        self,
        transaction: Transaction,
        requires_confirmation: bool,
        confirmation_reason: Optional[str] = None,
    ) -> TransitionResult:
        """
        Process a bound transaction to VALIDATED or PENDING_CONFIRMATION.

        Args:
            transaction: Transaction after binding
            requires_confirmation: Whether confirmation is needed
            confirmation_reason: Reason for requiring confirmation

        Returns:
            TransitionResult with new state
        """
        if transaction.state != TransactionState.BOUND:
            return TransitionResult(
                success=False,
                transaction=transaction,
                error=f"Expected BOUND state, got {transaction.state.value}",
            )

        if requires_confirmation:
            return self.transition(
                transaction,
                TransactionState.PENDING_CONFIRMATION,
                confirmation_reason or "Requires user confirmation",
            )
        else:
            return self.transition(
                transaction,
                TransactionState.VALIDATED,
                "Validation passed",
            )

    def complete(
        self,
        transaction: Transaction,
        result: TransactionResult,
    ) -> TransitionResult:
        """
        Mark a transaction as completed with result.

        Args:
            transaction: Transaction to complete
            result: Execution result

        Returns:
            TransitionResult with completed transaction
        """
        if transaction.state != TransactionState.EXECUTED:
            return TransitionResult(
                success=False,
                transaction=transaction,
                error=f"Expected EXECUTED state, got {transaction.state.value}",
            )

        # Apply result to transaction
        tx_with_result = transaction.with_result(result)

        if result.success:
            return self.transition(
                tx_with_result,
                TransactionState.COMPLETED,
                "Execution successful",
            )
        else:
            return self.transition(
                tx_with_result,
                TransactionState.FAILED,
                result.error or "Execution failed",
            )

    def reject(
        self,
        transaction: Transaction,
        reason: str,
    ) -> TransitionResult:
        """
        Reject a transaction.

        Args:
            transaction: Transaction to reject
            reason: Rejection reason

        Returns:
            TransitionResult with rejected transaction
        """
        return self.transition(
            transaction,
            TransactionState.REJECTED,
            reason,
        )

    def fail(
        self,
        transaction: Transaction,
        error: str,
    ) -> TransitionResult:
        """
        Mark a transaction as failed.

        Args:
            transaction: Transaction that failed
            error: Error message

        Returns:
            TransitionResult with failed transaction
        """
        return self.transition(
            transaction,
            TransactionState.FAILED,
            error,
        )

    def get_path_to_completion(
        self,
        state: TransactionState,
    ) -> List[TransactionState]:
        """
        Get the happy path from a state to COMPLETED.

        Args:
            state: Starting state

        Returns:
            List of states on the path to completion
        """
        happy_path = [
            TransactionState.RECEIVED,
            TransactionState.PARSED,
            TransactionState.BOUND,
            TransactionState.VALIDATED,
            TransactionState.EXECUTED,
            TransactionState.COMPLETED,
        ]

        try:
            start_idx = happy_path.index(state)
            return happy_path[start_idx:]
        except ValueError:
            # State not on happy path (e.g., PENDING_BINDING)
            return []
