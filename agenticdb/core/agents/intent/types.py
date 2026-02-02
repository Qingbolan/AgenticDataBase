"""
Type definitions for Intent processing agents.

These types define the input/output contracts for Intent agents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...models import Intent, IntentState, OperationType, ParameterSlot, SafetyConstraint


class BindingError(Exception):
    """
    Error raised when binding fails.

    This includes:
    - Attempting to bind an already-bound slot (monotonicity violation)
    - Invalid value type for slot
    - Unknown slot name
    """

    def __init__(self, message: str, slot_name: Optional[str] = None):
        super().__init__(message)
        self.slot_name = slot_name


class ValidationError(Exception):
    """
    Error raised when validation fails.

    This includes:
    - Safety constraint violations
    - Invalid operation on target
    - Permission denied
    """

    def __init__(
        self,
        message: str,
        constraint: Optional[SafetyConstraint] = None,
        requires_confirmation: bool = False,
    ):
        super().__init__(message)
        self.constraint = constraint
        self.requires_confirmation = requires_confirmation


@dataclass
class IntentParseResult:
    """
    Result of parsing natural language into Intent IR.

    Attributes:
        intent: The parsed Intent (may be PARTIAL or COMPLETE)
        success: Whether parsing succeeded
        error: Error message if parsing failed
        confidence: Confidence score of the parse [0, 1]
        alternatives: Alternative parses if ambiguous
        raw_response: Raw LLM response for debugging
    """

    intent: Optional[Intent] = None
    success: bool = True
    error: Optional[str] = None
    confidence: float = 1.0
    alternatives: List[Intent] = field(default_factory=list)
    raw_response: Optional[str] = None

    @classmethod
    def from_intent(cls, intent: Intent, confidence: float = 1.0) -> "IntentParseResult":
        """Create a successful result from an Intent."""
        return cls(intent=intent, success=True, confidence=confidence)

    @classmethod
    def from_error(cls, error: str) -> "IntentParseResult":
        """Create a failed result from an error message."""
        return cls(success=False, error=error, confidence=0.0)

    @property
    def needs_binding(self) -> bool:
        """Check if the intent needs binding resolution."""
        return self.intent is not None and self.intent.state == IntentState.PARTIAL

    @property
    def unbound_slots(self) -> List[str]:
        """Get names of unbound slots."""
        if self.intent is None:
            return []
        return self.intent.get_unbound_slot_names()


@dataclass
class BindingResult:
    """
    Result of binding resolution.

    Attributes:
        intent: The Intent after binding (may still be PARTIAL)
        success: Whether binding succeeded
        error: Error message if binding failed
        bound_slots: Names of slots that were bound
        remaining_slots: Names of slots still unbound
        binding_source: Source of the binding (user, context, default)
    """

    intent: Optional[Intent] = None
    success: bool = True
    error: Optional[str] = None
    bound_slots: List[str] = field(default_factory=list)
    remaining_slots: List[str] = field(default_factory=list)
    binding_source: str = "user"

    @classmethod
    def from_intent(
        cls,
        intent: Intent,
        bound_slots: List[str],
        source: str = "user",
    ) -> "BindingResult":
        """Create a successful result from a bound Intent."""
        return cls(
            intent=intent,
            success=True,
            bound_slots=bound_slots,
            remaining_slots=intent.get_unbound_slot_names(),
            binding_source=source,
        )

    @classmethod
    def from_error(cls, error: str, slot_name: Optional[str] = None) -> "BindingResult":
        """Create a failed result from an error."""
        return cls(
            success=False,
            error=error,
            bound_slots=[],
            remaining_slots=[slot_name] if slot_name else [],
        )

    @property
    def is_complete(self) -> bool:
        """Check if binding is complete."""
        return self.intent is not None and self.intent.state == IntentState.COMPLETE


@dataclass
class ValidationResult:
    """
    Result of safety validation.

    Attributes:
        valid: Whether the Intent passed validation
        intent: The validated Intent (may be marked INVALID)
        errors: List of validation errors
        warnings: List of validation warnings
        requires_confirmation: Whether user confirmation is needed
        confirmation_reason: Reason for requiring confirmation
        affected_rows_estimate: Estimated rows affected
        violated_constraints: Constraints that were violated
    """

    valid: bool = True
    intent: Optional[Intent] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_reason: Optional[str] = None
    affected_rows_estimate: Optional[int] = None
    violated_constraints: List[SafetyConstraint] = field(default_factory=list)

    @classmethod
    def passed(cls, intent: Intent) -> "ValidationResult":
        """Create a passed validation result."""
        return cls(valid=True, intent=intent)

    @classmethod
    def needs_confirmation(
        cls,
        intent: Intent,
        reason: str,
        affected_rows: Optional[int] = None,
        constraint: Optional[SafetyConstraint] = None,
    ) -> "ValidationResult":
        """Create a result that requires confirmation."""
        return cls(
            valid=True,
            intent=intent,
            requires_confirmation=True,
            confirmation_reason=reason,
            affected_rows_estimate=affected_rows,
            violated_constraints=[constraint] if constraint else [],
        )

    @classmethod
    def rejected(
        cls,
        intent: Intent,
        errors: List[str],
        constraints: Optional[List[SafetyConstraint]] = None,
    ) -> "ValidationResult":
        """Create a rejected validation result."""
        return cls(
            valid=False,
            intent=intent.mark_invalid(errors[0] if errors else "Validation failed"),
            errors=errors,
            violated_constraints=constraints or [],
        )


@dataclass
class SlotSuggestion:
    """
    Suggestion for binding an unbound slot.

    Used by BindingAgent to suggest possible values.
    """

    slot_name: str
    suggested_value: Any
    confidence: float = 1.0
    source: str = "context"  # context, default, schema
    reason: Optional[str] = None


@dataclass
class BindingContext:
    """
    Context for binding resolution.

    Provides information to help resolve unbound slots.
    """

    # Known schema information
    available_tables: List[str] = field(default_factory=list)
    available_columns: Dict[str, List[str]] = field(default_factory=dict)

    # Session context
    recent_targets: List[str] = field(default_factory=list)
    conversation_entities: Dict[str, Any] = field(default_factory=dict)

    # Default values
    default_limit: int = 100
    default_time_range: str = "7d"

    # User preferences
    preferred_tables: List[str] = field(default_factory=list)
