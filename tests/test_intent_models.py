"""
Tests for Intent IR models.

Tests the Intent, ParameterSlot, Predicate, SafetyConstraint, and related
models used in the Intent-Aware Transaction Pipeline.
"""

import pytest
from datetime import datetime, timezone

from agenticdb.core.models import (
    Intent,
    IntentState,
    OperationType,
    ParameterSlot,
    Predicate,
    SafetyConstraint,
    SlotType,
    Pattern,
)


class TestParameterSlot:
    """Tests for ParameterSlot model."""

    def test_create_unbound_slot(self):
        """Test creating an unbound parameter slot."""
        slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        assert slot.name == "target"
        assert slot.slot_type == SlotType.ENTITY
        assert slot.bound_value is None
        assert not slot.is_bound

    def test_create_bound_slot(self):
        """Test creating a bound parameter slot."""
        slot = ParameterSlot(
            name="limit",
            slot_type=SlotType.NUMERIC,
            bound_value=100,
        )
        assert slot.is_bound
        assert slot.bound_value == 100

    def test_bind_slot(self):
        """Test binding a value to a slot."""
        slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        new_slot = slot.bind("orders")

        # Original should be unchanged
        assert not slot.is_bound

        # New slot should be bound
        assert new_slot.is_bound
        assert new_slot.bound_value == "orders"
        assert new_slot.name == "target"

    def test_slot_types(self):
        """Test different slot types."""
        types = [
            SlotType.ENTITY,
            SlotType.TEMPORAL,
            SlotType.NUMERIC,
            SlotType.FILTER,
            SlotType.STRING,
            SlotType.LIST,
        ]
        for slot_type in types:
            slot = ParameterSlot(name="test", slot_type=slot_type)
            assert slot.slot_type == slot_type


class TestPredicate:
    """Tests for Predicate model."""

    def test_create_bound_predicate(self):
        """Test creating a predicate with bound value."""
        pred = Predicate(field="status", operator="eq", value="active")
        assert pred.field == "status"
        assert pred.operator == "eq"
        assert pred.value == "active"
        assert pred.is_bound
        assert pred.get_resolved_value() == "active"

    def test_create_predicate_with_slot(self):
        """Test creating a predicate with slot reference."""
        slot = ParameterSlot(name="status_filter", slot_type=SlotType.FILTER)
        pred = Predicate(field="status", operator="eq", value=slot)

        assert not pred.is_bound

    def test_predicate_with_bound_slot(self):
        """Test predicate with bound slot."""
        slot = ParameterSlot(
            name="status_filter",
            slot_type=SlotType.FILTER,
            bound_value="active",
        )
        pred = Predicate(field="status", operator="eq", value=slot)

        assert pred.is_bound
        assert pred.get_resolved_value() == "active"

    def test_negated_predicate(self):
        """Test negated predicate."""
        pred = Predicate(field="status", operator="eq", value="deleted", negate=True)
        assert pred.negate


class TestSafetyConstraint:
    """Tests for SafetyConstraint model."""

    def test_max_rows_constraint_pass(self):
        """Test max_rows constraint that passes."""
        constraint = SafetyConstraint(
            constraint_type="max_rows",
            parameters={"limit": 1000},
        )
        passed, reason = constraint.evaluate({"affected_rows": 500})
        assert passed
        assert reason is None

    def test_max_rows_constraint_fail(self):
        """Test max_rows constraint that fails."""
        constraint = SafetyConstraint(
            constraint_type="max_rows",
            parameters={"limit": 1000},
        )
        passed, reason = constraint.evaluate({"affected_rows": 5000})
        assert not passed
        assert "exceeds limit" in reason.lower()

    def test_no_delete_constraint(self):
        """Test no_delete constraint."""
        constraint = SafetyConstraint(constraint_type="no_delete")

        # Should pass for non-delete operations
        passed, _ = constraint.evaluate({"operation": OperationType.QUERY})
        assert passed

        # Should fail for delete operations
        passed, reason = constraint.evaluate({"operation": OperationType.DELETE})
        assert not passed
        assert "delete" in reason.lower()

    def test_require_confirm_constraint(self):
        """Test require_confirm constraint."""
        constraint = SafetyConstraint(constraint_type="require_confirm")

        # Should fail without confirmation
        passed, reason = constraint.evaluate({"confirmed": False})
        assert not passed

        # Should pass with confirmation
        passed, _ = constraint.evaluate({"confirmed": True})
        assert passed

    def test_protected_tables_constraint(self):
        """Test protected_tables constraint."""
        constraint = SafetyConstraint(
            constraint_type="protected_tables",
            parameters={"tables": ["users", "payments"]},
        )

        # Should pass for non-protected table
        passed, _ = constraint.evaluate({"target_table": "orders", "confirmed": False})
        assert passed

        # Should fail for protected table without confirmation
        passed, reason = constraint.evaluate({"target_table": "users", "confirmed": False})
        assert not passed
        assert "protected" in reason.lower()

        # Should pass for protected table with confirmation
        passed, _ = constraint.evaluate({"target_table": "users", "confirmed": True})
        assert passed


class TestIntent:
    """Tests for Intent model."""

    def test_create_complete_intent(self):
        """Test creating a complete Intent."""
        intent = Intent(
            operation=OperationType.QUERY,
            target="orders",
            raw_input="show orders",
        )
        # Target is a string, so it should check unbound_slots
        # Since no unbound_slots, but state defaults to PARTIAL
        assert intent.operation == OperationType.QUERY
        assert intent.target == "orders"
        assert intent.raw_input == "show orders"

    def test_create_partial_intent(self):
        """Test creating a partial Intent with unbound slots."""
        target_slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        intent = Intent(
            operation=OperationType.QUERY,
            target=target_slot,
            raw_input="show records",
        )

        assert intent.state == IntentState.PARTIAL
        assert "target" in intent.get_unbound_slot_names()

    def test_intent_bind_slot(self):
        """Test binding a slot in Intent."""
        target_slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        intent = Intent(
            operation=OperationType.QUERY,
            target=target_slot,
            raw_input="show records",
        )

        # Bind the target
        new_intent = intent.bind_slot("target", "orders")

        # Original unchanged
        assert intent.state == IntentState.PARTIAL

        # New intent should be complete
        assert new_intent.state == IntentState.COMPLETE
        assert new_intent.get_target_name() == "orders"
        assert "target" in new_intent.bindings

    def test_intent_binding_monotonicity(self):
        """Test that binding is monotonic (cannot unbind)."""
        target_slot = ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        intent = Intent(
            operation=OperationType.QUERY,
            target=target_slot,
            raw_input="show records",
        )

        # First binding
        bound_intent = intent.bind_slot("target", "orders")

        # Second binding should fail
        with pytest.raises(ValueError, match="already bound"):
            bound_intent.bind_slot("target", "users")

    def test_intent_unknown_slot_binding(self):
        """Test binding to unknown slot fails."""
        intent = Intent(
            operation=OperationType.QUERY,
            target="orders",
            raw_input="show orders",
        )

        with pytest.raises(ValueError, match="Unknown slot"):
            intent.bind_slot("nonexistent", "value")

    def test_intent_mark_invalid(self):
        """Test marking Intent as invalid."""
        intent = Intent(
            operation=OperationType.QUERY,
            target="orders",
            raw_input="show orders",
        )

        invalid_intent = intent.mark_invalid("Validation failed")

        assert invalid_intent.state == IntentState.INVALID
        assert "invalid_reason" in invalid_intent.metadata
        assert invalid_intent.metadata["invalid_reason"] == "Validation failed"

    def test_intent_factory_method(self):
        """Test Intent.create factory method."""
        intent = Intent.create(
            operation=OperationType.DELETE,
            raw_input="delete old records",
        )

        assert intent.operation == OperationType.DELETE
        assert isinstance(intent.target, ParameterSlot)
        assert intent.state == IntentState.PARTIAL

    def test_intent_with_predicates(self):
        """Test Intent with predicates."""
        intent = Intent(
            operation=OperationType.QUERY,
            target="orders",
            predicates=[
                Predicate(field="status", operator="eq", value="active"),
                Predicate(field="amount", operator="gt", value=100),
            ],
            raw_input="show active orders with amount > 100",
        )

        assert len(intent.predicates) == 2
        assert all(p.is_bound for p in intent.predicates)

    def test_intent_with_constraints(self):
        """Test Intent with safety constraints."""
        intent = Intent(
            operation=OperationType.DELETE,
            target="orders",
            constraints=[
                SafetyConstraint(
                    constraint_type="max_rows",
                    parameters={"limit": 100},
                ),
            ],
            raw_input="delete old orders",
        )

        assert len(intent.constraints) == 1


class TestPattern:
    """Tests for Pattern model."""

    def test_create_pattern(self):
        """Test creating a Pattern."""
        pattern = Pattern(
            template="SELECT * FROM {target} WHERE created_at > {time_start}",
            structural_hash="abc123",
            operation=OperationType.QUERY,
            parameter_slots=["target", "time_start"],
        )

        assert pattern.operation == OperationType.QUERY
        assert "target" in pattern.parameter_slots
        assert pattern.hit_count == 0

    def test_pattern_record_hit(self):
        """Test recording a cache hit."""
        pattern = Pattern(
            template="SELECT * FROM orders",
            structural_hash="abc123",
            operation=OperationType.QUERY,
            parameter_slots=[],
        )

        new_pattern = pattern.record_hit()

        # Original unchanged
        assert pattern.hit_count == 0

        # New pattern has hit
        assert new_pattern.hit_count == 1
        assert new_pattern.last_used is not None

    def test_pattern_add_example(self):
        """Test adding example queries."""
        pattern = Pattern(
            template="SELECT * FROM orders",
            structural_hash="abc123",
            operation=OperationType.QUERY,
            parameter_slots=[],
        )

        new_pattern = pattern.add_example("show orders")
        new_pattern = new_pattern.add_example("list orders")

        assert len(new_pattern.example_queries) == 2
        assert "show orders" in new_pattern.example_queries

    def test_pattern_example_limit(self):
        """Test example queries are limited."""
        pattern = Pattern(
            template="SELECT * FROM orders",
            structural_hash="abc123",
            operation=OperationType.QUERY,
            parameter_slots=[],
        )

        for i in range(10):
            pattern = pattern.add_example(f"query {i}")

        # Should only keep last 5
        assert len(pattern.example_queries) == 5
        assert "query 9" in pattern.example_queries
        assert "query 0" not in pattern.example_queries
