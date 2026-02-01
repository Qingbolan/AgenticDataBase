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
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

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
