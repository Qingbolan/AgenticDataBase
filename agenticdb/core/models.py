"""
Core entity models for AgenticDB.

This module defines the fundamental data primitives that make agent behavior
first-class citizens in the data model:

- Entity: Base class for all storable objects
- Event: Immutable facts that happened (cannot be changed)
- Claim: Structured assertions with provenance (source, confidence, timestamp)
- Action: Agent behaviors with explicit dependency tracking

Design Philosophy:
    Traditional databases store "what data is".
    AgenticDB stores "how state became this way".
"""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import ulid
from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Enumeration of entity types in AgenticDB."""

    EVENT = "event"
    CLAIM = "claim"
    ACTION = "action"


class EntityStatus(str, Enum):
    """Status of an entity in the system."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"  # Replaced by a newer version
    INVALIDATED = "invalidated"  # Dependencies changed, no longer valid
    RETRACTED = "retracted"  # Explicitly withdrawn


def generate_id() -> str:
    """Generate a unique, sortable ID using ULID."""
    return str(ulid.new())


def compute_content_hash(data: dict[str, Any]) -> str:
    """
    Compute a content-addressable hash for data.

    This enables deduplication and integrity verification.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class Entity(BaseModel, ABC):
    """
    Base class for all storable entities in AgenticDB.

    Every entity has:
    - A unique ID (ULID for sortability)
    - A content hash (for integrity and deduplication)
    - Timestamps for creation and modification
    - Version tracking for optimistic concurrency

    Entities are immutable once created. Updates create new versions.
    """

    id: str = Field(default_factory=generate_id, description="Unique entity identifier")
    entity_type: EntityType = Field(..., description="Type discriminator")
    status: EntityStatus = Field(default=EntityStatus.ACTIVE, description="Entity status")
    content_hash: Optional[str] = Field(default=None, description="Content-addressable hash")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    version: int = Field(default=1, description="Version number for optimistic concurrency")

    branch_id: Optional[str] = Field(default=None, description="Branch this entity belongs to")
    parent_id: Optional[str] = Field(default=None, description="Previous version of this entity")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")

    model_config = {"frozen": False, "extra": "forbid"}

    def model_post_init(self, __context: Any) -> None:
        """Compute content hash after initialization."""
        if self.content_hash is None:
            self.content_hash = self._compute_hash()

    @abstractmethod
    def _get_hashable_content(self) -> dict[str, Any]:
        """Return the content that should be hashed for this entity."""
        pass

    def _compute_hash(self) -> str:
        """Compute the content hash for this entity."""
        return compute_content_hash(self._get_hashable_content())

    def is_active(self) -> bool:
        """Check if this entity is currently active."""
        return self.status == EntityStatus.ACTIVE

    def supersede(self, new_entity: Entity) -> None:
        """Mark this entity as superseded by a new version."""
        self.status = EntityStatus.SUPERSEDED
        new_entity.parent_id = self.id
        new_entity.version = self.version + 1


class Event(Entity):
    """
    An immutable fact that happened.

    Events are the ground truth of the system. They represent things that
    occurred and cannot be changed or disputed - only new events can be
    recorded to represent new facts.

    Examples:
        - UserRegistered(user_id="u123", email="alice@example.com")
        - PaymentReceived(order_id="o456", amount=99.99)
        - ModelTrainingCompleted(model_id="m789", accuracy=0.95)

    Events form the foundation of the causal chain. Actions and Claims
    often depend on Events as their source of truth.
    """

    entity_type: EntityType = Field(default=EntityType.EVENT, frozen=True)

    event_type: str = Field(..., description="Type/name of the event")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")

    # Source tracking
    source_agent: Optional[str] = Field(default=None, description="Agent that emitted this event")
    source_system: Optional[str] = Field(default=None, description="System that emitted this event")

    # Correlation
    correlation_id: Optional[str] = Field(
        default=None,
        description="ID to correlate related events across the system"
    )
    causation_id: Optional[str] = Field(
        default=None,
        description="ID of the event/action that caused this event"
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Ensure event type is not empty."""
        if not v or not v.strip():
            raise ValueError("event_type cannot be empty")
        return v.strip()

    def _get_hashable_content(self) -> dict[str, Any]:
        """Events are hashed by their type and data."""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "source_agent": self.source_agent,
            "source_system": self.source_system,
        }


class Claim(Entity):
    """
    A structured assertion with provenance.

    Claims represent beliefs or computed values that have a source,
    confidence level, and validity period. Unlike Events (which are
    immutable facts), Claims can be superseded by newer claims or
    invalidated when their dependencies change.

    Examples:
        - Claim(subject="user.u123.risk_score", value=0.7, source="risk_model_v2")
        - Claim(subject="order.o456.fraud_probability", value=0.1, source="fraud_detector")
        - Claim(subject="product.p789.recommended", value=True, source="recommendation_engine")

    Claims are essential for:
    - Tracking where computed values come from
    - Managing confidence and uncertainty
    - Invalidating downstream when sources update
    """

    entity_type: EntityType = Field(default=EntityType.CLAIM, frozen=True)

    # The assertion itself
    subject: str = Field(..., description="What this claim is about (e.g., 'user.u123.risk_score')")
    predicate: str = Field(default="is", description="The relationship (default: 'is')")
    value: Any = Field(..., description="The claimed value")

    # Provenance
    source: str = Field(..., description="Where this claim comes from")
    source_version: Optional[str] = Field(default=None, description="Version of the source")

    # Confidence and validity
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score [0, 1]"
    )
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this claim becomes valid"
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="When this claim expires (None = no expiration)"
    )

    # Dependencies - what this claim was derived from
    derived_from: list[str] = Field(
        default_factory=list,
        description="IDs of entities this claim was derived from"
    )

    # Conflict resolution
    priority: int = Field(
        default=0,
        description="Priority for conflict resolution (higher wins)"
    )

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v: str) -> str:
        """Ensure subject is not empty."""
        if not v or not v.strip():
            raise ValueError("subject cannot be empty")
        return v.strip()

    def _get_hashable_content(self) -> dict[str, Any]:
        """Claims are hashed by their subject, value, and source."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "source": self.source,
            "source_version": self.source_version,
        }

    def is_valid_at(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if this claim is valid at the given timestamp."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return self.is_active()

    def conflicts_with(self, other: Claim) -> bool:
        """Check if this claim conflicts with another claim."""
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.value != other.value
            and self.is_active()
            and other.is_active()
        )


class ActionStatus(str, Enum):
    """Status of an action execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Action(Entity):
    """
    An agent behavior with explicit dependencies.

    Actions represent things agents do - decisions, computations, side effects.
    Unlike Events (passive observations) or Claims (assertions), Actions are
    active behaviors that transform state.

    The key innovation is explicit dependency tracking:
    - depends_on: What this action requires to execute
    - produces: What entities this action creates
    - invalidates: What this action makes stale

    Examples:
        - Action(type="ApproveOrder", depends_on=[risk_claim_id, inventory_event_id])
        - Action(type="SendNotification", depends_on=[user_event_id, preference_claim_id])
        - Action(type="RetrainModel", depends_on=[dataset_event_ids])

    Actions enable:
    - Answering "why did this happen?" (trace depends_on chain)
    - Answering "what breaks if X changes?" (trace produces/invalidates)
    - Replaying decisions for debugging and auditing
    """

    entity_type: EntityType = Field(default=EntityType.ACTION, frozen=True)

    # Action definition
    action_type: str = Field(..., description="Type/name of the action")
    action_status: ActionStatus = Field(
        default=ActionStatus.PENDING,
        description="Current execution status"
    )

    # Agent attribution
    agent_id: str = Field(..., description="ID of the agent performing this action")
    agent_type: Optional[str] = Field(default=None, description="Type of agent")

    # Inputs and outputs
    inputs: dict[str, Any] = Field(default_factory=dict, description="Action inputs")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Action outputs")

    # Explicit dependency tracking - THE CORE INNOVATION
    depends_on: list[str] = Field(
        default_factory=list,
        description="Entity IDs this action depends on"
    )
    produces: list[str] = Field(
        default_factory=list,
        description="Entity IDs this action produces"
    )
    invalidates: list[str] = Field(
        default_factory=list,
        description="Entity IDs this action invalidates"
    )

    # Execution tracking
    started_at: Optional[datetime] = Field(default=None, description="When execution started")
    completed_at: Optional[datetime] = Field(default=None, description="When execution completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # Reasoning trace - for LLM/agent debugging
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for this action (for audit/debug)"
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in reasoning (for cost tracking)"
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        """Ensure action type is not empty."""
        if not v or not v.strip():
            raise ValueError("action_type cannot be empty")
        return v.strip()

    def _get_hashable_content(self) -> dict[str, Any]:
        """Actions are hashed by their type, agent, and inputs."""
        return {
            "action_type": self.action_type,
            "agent_id": self.agent_id,
            "inputs": self.inputs,
            "depends_on": sorted(self.depends_on),
        }

    def start(self) -> None:
        """Mark the action as started."""
        self.action_status = ActionStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self, outputs: Optional[dict[str, Any]] = None) -> None:
        """Mark the action as completed."""
        self.action_status = ActionStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if outputs:
            self.outputs = outputs

    def fail(self, error: str) -> None:
        """Mark the action as failed."""
        self.action_status = ActionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error = error

    def add_dependency(self, entity_id: str) -> None:
        """Add a dependency to this action."""
        if entity_id not in self.depends_on:
            self.depends_on.append(entity_id)

    def add_product(self, entity_id: str) -> None:
        """Record an entity produced by this action."""
        if entity_id not in self.produces:
            self.produces.append(entity_id)


# =============================================================================
# Intent IR Models
# =============================================================================


class OperationType(str, Enum):
    """Operation types in Intent IR."""

    QUERY = "query"
    STORE = "store"
    UPDATE = "update"
    DELETE = "delete"


class IntentState(str, Enum):
    """
    Intent completeness state.

    COMPLETE: All parameter slots are bound, ready for execution
    PARTIAL: Has unbound slots, requires binding resolution
    INVALID: Failed validation, cannot be executed
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    INVALID = "invalid"


class SlotType(str, Enum):
    """Types of parameter slots in Intent."""

    ENTITY = "entity"       # Target entity reference (e.g., "orders", "users")
    TEMPORAL = "temporal"   # Time-based constraint (e.g., "last month")
    NUMERIC = "numeric"     # Numeric value (e.g., limit, threshold)
    FILTER = "filter"       # Filter predicate (e.g., "active", "pending")
    STRING = "string"       # String value
    LIST = "list"           # List of values


class ParameterSlot(BaseModel):
    """
    A parameter slot in the Intent.

    Represents an unbound or bound parameter that can be resolved
    during the PENDING_BINDING phase. Slots enforce binding monotonicity:
    once bound, a slot cannot be unbound.

    Examples:
        - ParameterSlot(name="target", slot_type=SlotType.ENTITY)
        - ParameterSlot(name="time_range", slot_type=SlotType.TEMPORAL, bound_value="7d")
    """

    name: str = Field(..., description="Slot identifier")
    slot_type: SlotType = Field(..., description="Type of the slot")
    bound_value: Optional[Any] = Field(default=None, description="Bound value if resolved")
    description: Optional[str] = Field(default=None, description="Human-readable description")
    required: bool = Field(default=True, description="Whether this slot must be bound")

    model_config = {"frozen": False}

    @property
    def is_bound(self) -> bool:
        """Check if this slot has been bound to a value."""
        return self.bound_value is not None

    def bind(self, value: Any) -> "ParameterSlot":
        """
        Create a new slot with the bound value (immutable pattern).

        Args:
            value: The value to bind to this slot

        Returns:
            New ParameterSlot with the bound value
        """
        return ParameterSlot(
            name=self.name,
            slot_type=self.slot_type,
            bound_value=value,
            description=self.description,
            required=self.required,
        )


class Predicate(BaseModel):
    """
    A predicate in the Intent.

    Represents a condition that filters or constrains the operation.
    Predicates can reference bound values or parameter slots.

    Examples:
        - Predicate(field="created_at", operator="gt", value="2024-01-01")
        - Predicate(field="status", operator="eq", value=slot_ref)
    """

    field: str = Field(..., description="Field to filter on")
    operator: str = Field(..., description="Comparison operator: eq, gt, lt, gte, lte, contains, between, in")
    value: Union[Any, ParameterSlot] = Field(..., description="Value or slot reference")
    negate: bool = Field(default=False, description="Negate the predicate (NOT)")

    model_config = {"frozen": False}

    @property
    def is_bound(self) -> bool:
        """Check if the predicate value is bound."""
        if isinstance(self.value, ParameterSlot):
            return self.value.is_bound
        return True

    def get_resolved_value(self) -> Any:
        """Get the resolved value of this predicate."""
        if isinstance(self.value, ParameterSlot):
            return self.value.bound_value
        return self.value


class SafetyConstraint(BaseModel):
    """
    Safety constraint for validation.

    Defines rules that trigger PENDING_CONFIRMATION or REJECTED states.
    Used to prevent dangerous operations without explicit user confirmation.

    Constraint Types:
        - max_rows: Limit affected rows (parameters: {"limit": 1000})
        - no_delete: Prohibit DELETE operations
        - require_confirm: Always require confirmation
        - no_drop: Prohibit DROP TABLE operations
        - protected_tables: List of tables that require confirmation

    Examples:
        - SafetyConstraint(constraint_type="max_rows", parameters={"limit": 1000})
        - SafetyConstraint(constraint_type="protected_tables", parameters={"tables": ["users", "payments"]})
    """

    constraint_type: str = Field(..., description="Type of safety constraint")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Constraint parameters")
    severity: str = Field(default="warning", description="Severity: warning, error, critical")
    message: Optional[str] = Field(default=None, description="Human-readable constraint message")

    model_config = {"frozen": False}

    def evaluate(self, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Evaluate if the constraint is satisfied.

        Args:
            context: Execution context with operation details

        Returns:
            Tuple of (is_satisfied, reason_if_not)
        """
        if self.constraint_type == "max_rows":
            limit = self.parameters.get("limit", 1000)
            affected = context.get("affected_rows", 0)
            if affected > limit:
                return False, f"Operation affects {affected} rows, exceeds limit of {limit}"
            return True, None

        elif self.constraint_type == "no_delete":
            operation = context.get("operation")
            if operation == OperationType.DELETE:
                return False, "DELETE operations are not allowed"
            return True, None

        elif self.constraint_type == "require_confirm":
            confirmed = context.get("confirmed", False)
            if not confirmed:
                return False, "This operation requires explicit confirmation"
            return True, None

        elif self.constraint_type == "no_drop":
            sql = context.get("sql", "").upper()
            if "DROP TABLE" in sql or "DROP DATABASE" in sql:
                return False, "DROP operations are not allowed"
            return True, None

        elif self.constraint_type == "protected_tables":
            tables = self.parameters.get("tables", [])
            target = context.get("target_table", "")
            if target in tables:
                confirmed = context.get("confirmed", False)
                if not confirmed:
                    return False, f"Table '{target}' is protected and requires confirmation"
            return True, None

        # Unknown constraint type - pass by default
        return True, None


class Intent(BaseModel):
    """
    Intent as Intermediate Representation.

    This is a formal IR within the transaction pipeline, not a semantic
    interpretation. It captures the operation, target, predicates, and
    binding state of a user request.

    Intent represents:
        operation   : QUERY | STORE | UPDATE | DELETE
        target      : EntityReference | Unbound
        predicates  : List<Predicate | UnboundPredicate>
        bindings    : Map<ParameterSlot, BoundValue | Pending>
        constraints : List<SafetyConstraint>
        state       : COMPLETE | PARTIAL | INVALID

    State Transitions:
        PARTIAL → COMPLETE (all slots bound)
        PARTIAL → INVALID (validation failed)
        COMPLETE → INVALID (validation failed)

    Formal Properties:
        - Binding Monotonicity: Once bound, slots cannot be unbound
        - Safety-Preserving Refinement: Binding cannot introduce violations
        - Deterministic Resolution: Same bindings produce same result
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique intent identifier")
    operation: OperationType = Field(..., description="Operation type")
    target: Union[str, ParameterSlot] = Field(..., description="Target entity or unbound slot")
    predicates: List[Predicate] = Field(default_factory=list, description="Filter predicates")
    bindings: Dict[str, Any] = Field(default_factory=dict, description="Resolved bindings")
    unbound_slots: List[ParameterSlot] = Field(default_factory=list, description="Pending slots")
    constraints: List[SafetyConstraint] = Field(default_factory=list, description="Safety constraints")
    state: IntentState = Field(default=IntentState.PARTIAL, description="Completeness state")

    # Original input
    raw_input: str = Field(default="", description="Original natural language input")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Parse confidence")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default=None)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")

    model_config = {"frozen": False}

    @classmethod
    def create(
        cls,
        operation: OperationType,
        raw_input: str,
        target: Optional[Union[str, ParameterSlot]] = None,
    ) -> "Intent":
        """
        Factory method to create a new Intent.

        Args:
            operation: The operation type
            raw_input: Original natural language input
            target: Target entity or unbound slot

        Returns:
            New Intent instance
        """
        if target is None:
            target = ParameterSlot(name="target", slot_type=SlotType.ENTITY)

        return cls(
            operation=operation,
            target=target,
            raw_input=raw_input,
        )

    def is_complete(self) -> bool:
        """Check if all slots are bound and intent is ready for execution."""
        return self.state == IntentState.COMPLETE

    def get_unbound_slot_names(self) -> List[str]:
        """Get names of all unbound slots."""
        names = []

        # Check target
        if isinstance(self.target, ParameterSlot) and not self.target.is_bound:
            names.append(self.target.name)

        # Check predicates
        for pred in self.predicates:
            if isinstance(pred.value, ParameterSlot) and not pred.value.is_bound:
                names.append(pred.value.name)

        # Check explicit unbound slots
        for slot in self.unbound_slots:
            if not slot.is_bound and slot.name not in names:
                names.append(slot.name)

        return names

    def bind_slot(self, slot_name: str, value: Any) -> "Intent":
        """
        Bind a value to a slot, returning a new Intent.

        Enforces binding monotonicity: once bound, cannot be unbound.

        Args:
            slot_name: Name of the slot to bind
            value: Value to bind

        Returns:
            New Intent with the binding applied

        Raises:
            ValueError: If slot is already bound or doesn't exist
        """
        # Check target slot
        new_target = self.target
        if isinstance(self.target, ParameterSlot) and self.target.name == slot_name:
            if self.target.is_bound:
                raise ValueError(f"Slot '{slot_name}' is already bound (monotonicity violation)")
            new_target = self.target.bind(value)

        # Check and update unbound slots
        new_unbound_slots = []
        found = False
        for slot in self.unbound_slots:
            if slot.name == slot_name:
                if slot.is_bound:
                    raise ValueError(f"Slot '{slot_name}' is already bound (monotonicity violation)")
                new_unbound_slots.append(slot.bind(value))
                found = True
            else:
                new_unbound_slots.append(slot)

        # Check predicates for slot references
        new_predicates = []
        for pred in self.predicates:
            if isinstance(pred.value, ParameterSlot) and pred.value.name == slot_name:
                if pred.value.is_bound:
                    raise ValueError(f"Slot '{slot_name}' is already bound (monotonicity violation)")
                new_predicates.append(
                    Predicate(
                        field=pred.field,
                        operator=pred.operator,
                        value=pred.value.bind(value),
                        negate=pred.negate,
                    )
                )
                found = True
            else:
                new_predicates.append(pred)

        if not found and not (isinstance(self.target, ParameterSlot) and self.target.name == slot_name):
            raise ValueError(f"Unknown slot: {slot_name}")

        # Update bindings
        new_bindings = {**self.bindings, slot_name: value}

        # Determine new state
        new_intent = Intent(
            id=self.id,
            operation=self.operation,
            target=new_target,
            predicates=new_predicates,
            bindings=new_bindings,
            unbound_slots=new_unbound_slots,
            constraints=self.constraints,
            state=self.state,
            raw_input=self.raw_input,
            confidence=self.confidence,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata=self.metadata,
        )

        # Check if all slots are now bound
        if not new_intent.get_unbound_slot_names():
            new_intent.state = IntentState.COMPLETE

        return new_intent

    def mark_invalid(self, reason: str) -> "Intent":
        """
        Mark the intent as invalid.

        Args:
            reason: Reason for invalidation

        Returns:
            New Intent with INVALID state
        """
        return Intent(
            id=self.id,
            operation=self.operation,
            target=self.target,
            predicates=self.predicates,
            bindings=self.bindings,
            unbound_slots=self.unbound_slots,
            constraints=self.constraints,
            state=IntentState.INVALID,
            raw_input=self.raw_input,
            confidence=self.confidence,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata={**self.metadata, "invalid_reason": reason},
        )

    def get_target_name(self) -> Optional[str]:
        """Get the resolved target name if bound."""
        if isinstance(self.target, str):
            return self.target
        elif isinstance(self.target, ParameterSlot) and self.target.is_bound:
            return str(self.target.bound_value)
        return None


# =============================================================================
# Transaction State Models
# =============================================================================


class TransactionState(str, Enum):
    """
    Transaction lifecycle states.

    State Machine:
        RECEIVED → PARSED → BOUND → VALIDATED → EXECUTED → COMPLETED
                     │         │         │
                     │ partial │ unsafe  │ invalid
                     ▼         ▼         ▼
               PENDING     PENDING    REJECTED
               BINDING  CONFIRMATION
    """

    RECEIVED = "received"
    PARSED = "parsed"
    PENDING_BINDING = "pending_binding"
    BOUND = "bound"
    PENDING_CONFIRMATION = "pending_confirmation"
    VALIDATED = "validated"
    EXECUTED = "executed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class StateTransition(BaseModel):
    """
    Record of a state transition for audit.

    Every state change is recorded to enable:
    - Audit trail for debugging
    - Replay capability
    - Understanding transaction history
    """

    from_state: TransactionState = Field(..., description="State before transition")
    to_state: TransactionState = Field(..., description="State after transition")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: Optional[str] = Field(default=None, description="Reason for transition")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional transition data")

    model_config = {"frozen": True}


class TransactionResult(BaseModel):
    """
    Result of a completed transaction.

    Contains execution outcome including data, affected rows, and timing.
    """

    success: bool = Field(..., description="Whether execution succeeded")
    data: Optional[Any] = Field(default=None, description="Query result data")
    affected_rows: int = Field(default=0, description="Number of rows affected")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")

    model_config = {"frozen": False}


class Transaction(BaseModel):
    """
    Transaction with explicit state tracking.

    Transactions can exist in pending states awaiting binding or confirmation.
    All state transitions are recorded for audit and replay.

    Key States:
        - PENDING_BINDING: Waiting for slot resolution
        - PENDING_CONFIRMATION: Waiting for user to confirm dangerous operation
        - REJECTED: Validation failed, cannot proceed

    Formal Properties:
        - Binding Monotonicity: Transaction bindings are append-only
        - Safety-Preserving Refinement: Confirmation cannot bypass validation
        - Deterministic Resolution: Same inputs produce same state transitions
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique transaction identifier")
    state: TransactionState = Field(default=TransactionState.RECEIVED, description="Current state")
    intent: Optional[Intent] = Field(default=None, description="Associated Intent IR")

    # Binding state
    pending_slots: List[str] = Field(default_factory=list, description="Slots awaiting binding")
    binding_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of bindings")

    # Confirmation state
    requires_confirmation: bool = Field(default=False, description="Whether confirmation is needed")
    confirmation_reason: Optional[str] = Field(default=None, description="Why confirmation is required")
    affected_rows_estimate: Optional[int] = Field(default=None, description="Estimated affected rows")
    confirmed: bool = Field(default=False, description="Whether user has confirmed")

    # Result
    result: Optional[TransactionResult] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error if failed/rejected")

    # Audit trail
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_history: List[StateTransition] = Field(default_factory=list, description="State transition history")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")

    model_config = {"frozen": False}

    @classmethod
    def create(cls, intent: Optional[Intent] = None) -> "Transaction":
        """
        Factory method to create a new Transaction.

        Args:
            intent: Optional Intent to associate with this transaction

        Returns:
            New Transaction instance in RECEIVED state
        """
        return cls(
            state=TransactionState.RECEIVED,
            intent=intent,
        )

    def transition_to(
        self,
        new_state: TransactionState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Transaction":
        """
        Transition to a new state, recording the change.

        Returns a new Transaction instance (immutable pattern).

        Args:
            new_state: Target state
            reason: Optional reason for transition
            metadata: Optional metadata for transition

        Returns:
            New Transaction in the new state
        """
        transition = StateTransition(
            from_state=self.state,
            to_state=new_state,
            reason=reason,
            metadata=metadata or {},
        )

        new_history = [*self.state_history, transition]

        return Transaction(
            id=self.id,
            state=new_state,
            intent=self.intent,
            pending_slots=self.pending_slots,
            binding_history=self.binding_history,
            requires_confirmation=self.requires_confirmation,
            confirmation_reason=self.confirmation_reason,
            affected_rows_estimate=self.affected_rows_estimate,
            confirmed=self.confirmed,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            state_history=new_history,
            metadata=self.metadata,
        )

    def with_intent(self, intent: Intent) -> "Transaction":
        """
        Associate an Intent with this transaction.

        Args:
            intent: The Intent to associate

        Returns:
            New Transaction with the Intent
        """
        pending = intent.get_unbound_slot_names()

        return Transaction(
            id=self.id,
            state=self.state,
            intent=intent,
            pending_slots=pending,
            binding_history=self.binding_history,
            requires_confirmation=self.requires_confirmation,
            confirmation_reason=self.confirmation_reason,
            affected_rows_estimate=self.affected_rows_estimate,
            confirmed=self.confirmed,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            state_history=self.state_history,
            metadata=self.metadata,
        )

    def with_binding(self, slot_name: str, value: Any) -> "Transaction":
        """
        Apply a binding to the transaction's Intent.

        Args:
            slot_name: Name of slot to bind
            value: Value to bind

        Returns:
            New Transaction with updated Intent

        Raises:
            ValueError: If no Intent or binding fails
        """
        if not self.intent:
            raise ValueError("Cannot bind: no Intent associated")

        new_intent = self.intent.bind_slot(slot_name, value)

        binding_record = {
            "slot": slot_name,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        new_pending = [s for s in self.pending_slots if s != slot_name]

        return Transaction(
            id=self.id,
            state=self.state,
            intent=new_intent,
            pending_slots=new_pending,
            binding_history=[*self.binding_history, binding_record],
            requires_confirmation=self.requires_confirmation,
            confirmation_reason=self.confirmation_reason,
            affected_rows_estimate=self.affected_rows_estimate,
            confirmed=self.confirmed,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            state_history=self.state_history,
            metadata=self.metadata,
        )

    def with_confirmation(self, confirmed: bool, reason: Optional[str] = None) -> "Transaction":
        """
        Set confirmation status.

        Args:
            confirmed: Whether user confirmed
            reason: Optional confirmation reason

        Returns:
            New Transaction with confirmation set
        """
        return Transaction(
            id=self.id,
            state=self.state,
            intent=self.intent,
            pending_slots=self.pending_slots,
            binding_history=self.binding_history,
            requires_confirmation=self.requires_confirmation,
            confirmation_reason=reason or self.confirmation_reason,
            affected_rows_estimate=self.affected_rows_estimate,
            confirmed=confirmed,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            state_history=self.state_history,
            metadata=self.metadata,
        )

    def with_result(self, result: TransactionResult) -> "Transaction":
        """
        Set the execution result.

        Args:
            result: The execution result

        Returns:
            New Transaction with result set
        """
        return Transaction(
            id=self.id,
            state=self.state,
            intent=self.intent,
            pending_slots=self.pending_slots,
            binding_history=self.binding_history,
            requires_confirmation=self.requires_confirmation,
            confirmation_reason=self.confirmation_reason,
            affected_rows_estimate=self.affected_rows_estimate,
            confirmed=self.confirmed,
            result=result,
            error=result.error,
            created_at=self.created_at,
            state_history=self.state_history,
            metadata=self.metadata,
        )

    def needs_binding(self) -> bool:
        """Check if transaction is waiting for bindings."""
        return self.state == TransactionState.PENDING_BINDING

    def needs_confirmation(self) -> bool:
        """Check if transaction is waiting for confirmation."""
        return self.state == TransactionState.PENDING_CONFIRMATION

    def is_terminal(self) -> bool:
        """Check if transaction is in a terminal state."""
        return self.state in (
            TransactionState.COMPLETED,
            TransactionState.REJECTED,
            TransactionState.FAILED,
        )

    def is_successful(self) -> bool:
        """Check if transaction completed successfully."""
        return self.state == TransactionState.COMPLETED and self.result is not None and self.result.success


# =============================================================================
# Pattern Models (for Workload Memoization)
# =============================================================================


class Pattern(BaseModel):
    """
    Pattern template for workload memoization.

    Patterns capture the structural form of queries to enable direct SQL
    execution for similar future queries without re-parsing through LLM.

    Pattern matching:
        - Extract parameters from query
        - Compute structural hash
        - Match against cached patterns
        - Execute with bound parameters

    Examples:
        "show orders from last week" → Pattern(template="SELECT * FROM {target} WHERE created_at > {time}")
        "show orders from last month" → Same pattern, different time binding
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique pattern identifier")
    template: str = Field(..., description="SQL template with placeholders")
    structural_hash: str = Field(..., description="Hash for structural matching")

    # Pattern structure
    operation: OperationType = Field(..., description="Operation type")
    target_table: Optional[str] = Field(default=None, description="Target table if fixed")
    parameter_slots: List[str] = Field(default_factory=list, description="Named parameter slots")

    # Statistics
    hit_count: int = Field(default=0, description="Number of cache hits")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    example_queries: List[str] = Field(default_factory=list, description="Example queries that match")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Pattern confidence")

    model_config = {"frozen": False}

    def record_hit(self) -> "Pattern":
        """Record a cache hit for this pattern."""
        return Pattern(
            id=self.id,
            template=self.template,
            structural_hash=self.structural_hash,
            operation=self.operation,
            target_table=self.target_table,
            parameter_slots=self.parameter_slots,
            hit_count=self.hit_count + 1,
            last_used=datetime.now(timezone.utc),
            created_at=self.created_at,
            example_queries=self.example_queries,
            confidence=self.confidence,
        )

    def add_example(self, query: str, max_examples: int = 5) -> "Pattern":
        """Add an example query to this pattern."""
        if query in self.example_queries:
            return self

        examples = [*self.example_queries, query]
        if len(examples) > max_examples:
            examples = examples[-max_examples:]

        return Pattern(
            id=self.id,
            template=self.template,
            structural_hash=self.structural_hash,
            operation=self.operation,
            target_table=self.target_table,
            parameter_slots=self.parameter_slots,
            hit_count=self.hit_count,
            last_used=self.last_used,
            created_at=self.created_at,
            example_queries=examples,
            confidence=self.confidence,
        )
