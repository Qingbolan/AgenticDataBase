"""
Tests for Transaction models.

Tests the Transaction, TransactionState, StateTransition, and TransactionResult
models used in the Intent-Aware Transaction Pipeline.
"""
import pytest
from datetime import datetime, timezone

from agenticdb.core.models import (
    Intent,
    IntentState,
    OperationType,
    ParameterSlot,
    SlotType,
    Transaction,
    TransactionState,
    StateTransition,
    TransactionResult,
)


class TestTransactionState:
    """Tests for TransactionState enum."""
    def test_all_states_defined(self):
        """Test all expected states are defined."""
        expected_states = [
            "RECEIVED",
            "PARSED",
            "PENDING_BINDING",
            "BOUND",
            "PENDING_CONFIRMATION",
            "VALIDATED",
            "EXECUTED",
            "COMPLETED",
            "REJECTED",
            "FAILED",
        ]
        for state in expected_states:
            assert hasattr(TransactionState, state)

    def test_state_values(self):
        """Test state values are lowercase strings."""
        assert TransactionState.RECEIVED.value == "received"
        assert TransactionState.PENDING_BINDING.value == "pending_binding"
        assert TransactionState.COMPLETED.value == "completed"


class TestStateTransition:
    """Tests for StateTransition model."""
    def test_create_transition(self):
        """Test creating a state transition."""
        transition = StateTransition(
            from_state=TransactionState.RECEIVED,
            to_state=TransactionState.PARSED,
            reason="Intent parsed successfully",
        )

        assert transition.from_state == TransactionState.RECEIVED
        assert transition.to_state == TransactionState.PARSED
        assert transition.reason == "Intent parsed successfully"
        assert transition.timestamp is not None

    def test_transition_with_metadata(self):
        """Test transition with metadata."""
        transition = StateTransition(
            from_state=TransactionState.BOUND,
            to_state=TransactionState.PENDING_CONFIRMATION,
            reason="Dangerous operation",
            metadata={"affected_rows": 5000},
        )

        assert transition.metadata["affected_rows"] == 5000


class TestTransactionResult:
    """Tests for TransactionResult model."""
    def test_success_result(self):
        """Test creating a success result."""
        result = TransactionResult(
            success=True,
            data=[{"id": 1}, {"id": 2}],
            affected_rows=2,
            execution_time_ms=45.5,
        )

        assert result.success
        assert len(result.data) == 2
        assert result.affected_rows == 2
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = TransactionResult(
            success=False,
            error="Table not found",
        )

        assert not result.success
        assert result.error == "Table not found"
        assert result.data is None


class TestTransaction:
    """Tests for Transaction model."""
    def test_create_transaction(self):
        """Test creating a transaction."""
        tx = Transaction.create()

        assert tx.id is not None
        assert tx.state == TransactionState.RECEIVED
        assert tx.intent is None
        assert len(tx.state_history) == 0

    def test_create_with_intent(self):
        """Test creating a transaction with Intent."""
        intent = Intent(
            operation=OperationType.QUERY,
            target="orders",
            raw_input="show orders",
        )
        tx = Transaction.create(intent=intent)

        assert tx.intent == intent

    def test_transition_to(self):
        """Test state transition."""
        tx = Transaction.create()
        new_tx = tx.transition_to(TransactionState.PARSED, "Intent parsed")

        # Original unchanged
        assert tx.state == TransactionState.RECEIVED
        assert len(tx.state_history) == 0

        # New transaction has transition
        assert new_tx.state == TransactionState.PARSED
        assert len(new_tx.state_history) == 1
        assert new_tx.state_history[0].from_state == TransactionState.RECEIVED
        assert new_tx.state_history[0].to_state == TransactionState.PARSED
        assert new_tx.state_history[0].reason == "Intent parsed"

    def test_with_intent(self):
        """Test associating Intent with Transaction."""
        tx = Transaction.create()

        target_slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        intent = Intent(
            operation=OperationType.QUERY,
            target=target_slot,
            raw_input="show records",
        )

        new_tx = tx.with_intent(intent)

        assert new_tx.intent == intent
        assert "target" in new_tx.pending_slots

    def test_with_binding(self):
        """Test applying binding to Transaction."""
        target_slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        intent = Intent(
            operation=OperationType.QUERY,
            target=target_slot,
            raw_input="show records",
        )
        tx = Transaction.create(intent=intent)
        tx = tx.with_intent(intent)

        new_tx = tx.with_binding("target", "orders")

        assert new_tx.intent.get_target_name() == "orders"
        assert len(new_tx.binding_history) == 1
        assert "target" not in new_tx.pending_slots

    def test_with_binding_no_intent_raises(self):
        """Test binding without Intent raises error."""
        tx = Transaction.create()

        with pytest.raises(ValueError, match="no Intent"):
            tx.with_binding("target", "orders")

    def test_with_confirmation(self):
        """Test setting confirmation."""
        tx = Transaction.create()
        new_tx = tx.with_confirmation(True, "User confirmed")

        assert new_tx.confirmed
        assert new_tx.confirmation_reason == "User confirmed"

    def test_with_result(self):
        """Test setting result."""
        tx = Transaction.create()
        result = TransactionResult(success=True, data=[{"id": 1}], affected_rows=1)

        new_tx = tx.with_result(result)

        assert new_tx.result == result

    def test_needs_binding(self):
        """Test needs_binding check."""
        tx = Transaction.create()

        tx_received = tx
        assert not tx_received.needs_binding()

        tx_pending = tx.transition_to(TransactionState.PENDING_BINDING)
        assert tx_pending.needs_binding()

    def test_needs_confirmation(self):
        """Test needs_confirmation check."""
        tx = Transaction.create()

        tx_received = tx
        assert not tx_received.needs_confirmation()

        tx_pending = tx.transition_to(TransactionState.PENDING_CONFIRMATION)
        assert tx_pending.needs_confirmation()

    def test_is_terminal(self):
        """Test terminal state detection."""
        tx = Transaction.create()

        # Non-terminal states
        for state in [
            TransactionState.RECEIVED,
            TransactionState.PARSED,
            TransactionState.PENDING_BINDING,
            TransactionState.BOUND,
            TransactionState.PENDING_CONFIRMATION,
            TransactionState.VALIDATED,
            TransactionState.EXECUTED,
        ]:
            tx_state = tx.transition_to(state)
            assert not tx_state.is_terminal()

        # Terminal states
        for state in [
            TransactionState.COMPLETED,
            TransactionState.REJECTED,
            TransactionState.FAILED,
        ]:
            tx_state = tx.transition_to(state)
            assert tx_state.is_terminal()

    def test_is_successful(self):
        """Test success detection."""
        tx = Transaction.create()

        # Not completed yet
        assert not tx.is_successful()

        # Completed but no result
        tx_completed = tx.transition_to(TransactionState.COMPLETED)
        assert not tx_completed.is_successful()

        # Completed with success result
        result = TransactionResult(success=True, data=[])
        tx_with_result = tx_completed.with_result(result)
        assert tx_with_result.is_successful()

        # Completed with failed result
        failed_result = TransactionResult(success=False, error="Error")
        tx_failed = tx_completed.with_result(failed_result)
        assert not tx_failed.is_successful()

    def test_state_history_accumulates(self):
        """Test that state history accumulates across transitions."""
        tx = Transaction.create()

        tx = tx.transition_to(TransactionState.PARSED, "Parsed")
        tx = tx.transition_to(TransactionState.BOUND, "Bound")
        tx = tx.transition_to(TransactionState.VALIDATED, "Validated")
        tx = tx.transition_to(TransactionState.EXECUTED, "Executed")
        tx = tx.transition_to(TransactionState.COMPLETED, "Completed")

        assert len(tx.state_history) == 5

        states = [t.to_state for t in tx.state_history]
        assert states == [
            TransactionState.PARSED,
            TransactionState.BOUND,
            TransactionState.VALIDATED,
            TransactionState.EXECUTED,
            TransactionState.COMPLETED,
        ]


class TestTransactionIntegration:
    """Integration tests for Transaction with Intent."""
    def test_full_happy_path(self):
        """Test complete happy path through transaction."""
        # Create intent
        intent = Intent(
            operation=OperationType.QUERY,
            target="orders",
            raw_input="show orders",
            state=IntentState.COMPLETE,
        )

        # Create and process transaction
        tx = Transaction.create()
        tx = tx.with_intent(intent)
        tx = tx.transition_to(TransactionState.PARSED, "Parsed")
        tx = tx.transition_to(TransactionState.BOUND, "All bound")
        tx = tx.transition_to(TransactionState.VALIDATED, "Validated")
        tx = tx.transition_to(TransactionState.EXECUTED, "Executed")

        result = TransactionResult(
            success=True,
            data=[{"id": 1, "total": 100}],
            affected_rows=1,
            execution_time_ms=25.0,
        )
        tx = tx.with_result(result)
        tx = tx.transition_to(TransactionState.COMPLETED, "Done")

        assert tx.state == TransactionState.COMPLETED
        assert tx.is_successful()
        assert tx.result.affected_rows == 1

    def test_pending_binding_flow(self):
        """Test flow through pending binding state."""
        # Create partial intent
        target_slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        intent = Intent(
            operation=OperationType.QUERY,
            target=target_slot,
            raw_input="show records",
        )

        # Create transaction
        tx = Transaction.create()
        tx = tx.with_intent(intent)
        tx = tx.transition_to(TransactionState.PARSED, "Parsed")
        tx = tx.transition_to(TransactionState.PENDING_BINDING, "Needs binding")

        assert tx.needs_binding()
        assert "target" in tx.pending_slots

        # Provide binding
        tx = tx.with_binding("target", "orders")
        tx = tx.transition_to(TransactionState.BOUND, "Binding resolved")

        assert not tx.needs_binding()
        assert tx.intent.state == IntentState.COMPLETE

    def test_pending_confirmation_flow(self):
        """Test flow through pending confirmation state."""
        # Create delete intent
        intent = Intent(
            operation=OperationType.DELETE,
            target="orders",
            raw_input="delete old orders",
            state=IntentState.COMPLETE,
        )

        # Create transaction
        tx = Transaction.create()
        tx = tx.with_intent(intent)
        tx = tx.transition_to(TransactionState.PARSED, "Parsed")
        tx = tx.transition_to(TransactionState.BOUND, "Bound")

        # Requires confirmation
        tx = Transaction(
            id=tx.id,
            state=tx.state,
            intent=tx.intent,
            pending_slots=tx.pending_slots,
            binding_history=tx.binding_history,
            requires_confirmation=True,
            confirmation_reason="DELETE affects 5000 rows",
            affected_rows_estimate=5000,
            confirmed=False,
            result=tx.result,
            error=tx.error,
            created_at=tx.created_at,
            state_history=tx.state_history,
            metadata=tx.metadata,
        )
        tx = tx.transition_to(TransactionState.PENDING_CONFIRMATION, "Dangerous operation")

        assert tx.needs_confirmation()
        assert tx.requires_confirmation
        assert tx.affected_rows_estimate == 5000

        # User confirms
        tx = tx.with_confirmation(True)
        tx = tx.transition_to(TransactionState.VALIDATED, "User confirmed")

        assert tx.confirmed
        assert not tx.needs_confirmation()

    def test_rejected_flow(self):
        """Test rejection flow."""
        intent = Intent(
            operation=OperationType.DELETE,
            target="users",
            raw_input="delete all users",
            state=IntentState.COMPLETE,
        )

        tx = Transaction.create()
        tx = tx.with_intent(intent)
        tx = tx.transition_to(TransactionState.PARSED, "Parsed")
        tx = tx.transition_to(TransactionState.REJECTED, "Protected table without confirmation")

        assert tx.state == TransactionState.REJECTED
        assert tx.is_terminal()
        assert not tx.is_successful()
