"""
Reactive subscriptions for AgenticDB.

This module provides subscription capabilities for reacting to
state changes in real-time - enabling event-driven architectures
without external message queues.

Design Philosophy:
    Traditional approach: Poll for changes or use external pub/sub
    AgenticDB approach: Subscribe to dependency changes natively
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agenticdb.core.models import Entity, EntityType
from agenticdb.core.dependency import DependencyGraph, EdgeType


class SubscriptionType(str, Enum):
    """Types of subscriptions."""

    ENTITY = "entity"  # Subscribe to a specific entity
    TYPE = "type"  # Subscribe to all entities of a type
    SUBJECT = "subject"  # Subscribe to claims with a subject pattern
    DEPENDENCY = "dependency"  # Subscribe to changes affecting dependencies


class SubscriptionEvent(BaseModel):
    """An event delivered to a subscription."""

    subscription_id: str = Field(..., description="Subscription that matched")
    event_type: str = Field(..., description="Type of change: create, update, delete, invalidate")
    entity_id: str = Field(..., description="Entity that changed")
    entity_type: EntityType = Field(..., description="Type of entity")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this event occurred"
    )

    # Optional details
    old_value: Optional[Any] = Field(default=None, description="Previous value (for updates)")
    new_value: Optional[Any] = Field(default=None, description="New value")
    caused_by: Optional[str] = Field(default=None, description="Entity that caused this change")

    model_config = {"extra": "forbid"}


# Type alias for subscription callbacks
SubscriptionCallback = Callable[[SubscriptionEvent], None]


class Subscription(BaseModel):
    """
    A subscription to state changes.

    Subscriptions allow clients to react to changes without polling.
    They can filter by entity, type, subject pattern, or dependencies.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="Subscription ID")
    subscription_type: SubscriptionType = Field(..., description="Type of subscription")

    # Filters (depending on subscription_type)
    entity_id: Optional[str] = Field(default=None, description="For ENTITY subscriptions")
    entity_type_filter: Optional[EntityType] = Field(default=None, description="For TYPE subscriptions")
    subject_pattern: Optional[str] = Field(default=None, description="For SUBJECT subscriptions")
    watch_entities: list[str] = Field(
        default_factory=list,
        description="For DEPENDENCY subscriptions - entities to watch"
    )

    # Subscription metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When subscription was created"
    )
    active: bool = Field(default=True, description="Whether subscription is active")

    # Delivery tracking
    events_delivered: int = Field(default=0, description="Number of events delivered")
    last_event_at: Optional[datetime] = Field(default=None, description="Last event delivery time")

    model_config = {"extra": "forbid"}

    def matches(self, entity: Entity, event_type: str) -> bool:
        """
        Check if an entity change matches this subscription.

        Args:
            entity: The changed entity
            event_type: Type of change

        Returns:
            True if this subscription should receive the event
        """
        if not self.active:
            return False

        if self.subscription_type == SubscriptionType.ENTITY:
            return entity.id == self.entity_id

        elif self.subscription_type == SubscriptionType.TYPE:
            return entity.entity_type == self.entity_type_filter

        elif self.subscription_type == SubscriptionType.SUBJECT:
            if self.subject_pattern and hasattr(entity, "subject"):
                subject = getattr(entity, "subject", "")
                # Simple prefix matching (could be extended to glob/regex)
                return subject.startswith(self.subject_pattern.rstrip("*"))
            return False

        elif self.subscription_type == SubscriptionType.DEPENDENCY:
            return entity.id in self.watch_entities

        return False


class SubscriptionManager:
    """
    Manages subscriptions and event delivery.

    Usage:
        ```python
        manager = SubscriptionManager(graph)

        # Subscribe to an entity
        sub = manager.subscribe_entity(
            entity_id="user_123",
            callback=lambda e: print(f"User changed: {e}")
        )

        # Subscribe to all claims about a subject
        sub = manager.subscribe_subject(
            pattern="user.*.risk_score",
            callback=handle_risk_change
        )

        # When an entity changes, notify subscribers
        manager.notify(entity, "update")
        ```
    """

    def __init__(self, graph: DependencyGraph):
        """
        Initialize the subscription manager.

        Args:
            graph: Dependency graph for tracking relationships
        """
        self._graph = graph
        self._lock = RLock()

        # Subscriptions
        self._subscriptions: dict[str, Subscription] = {}
        self._callbacks: dict[str, SubscriptionCallback] = {}

        # Indexes for fast matching
        self._entity_index: dict[str, set[str]] = {}  # entity_id -> subscription_ids
        self._type_index: dict[EntityType, set[str]] = {}  # entity_type -> subscription_ids
        self._subject_subscriptions: set[str] = set()  # subscriptions with subject patterns
        self._dependency_subscriptions: set[str] = set()  # subscriptions watching dependencies

    def subscribe_entity(
        self,
        entity_id: str,
        callback: SubscriptionCallback,
    ) -> Subscription:
        """
        Subscribe to changes on a specific entity.

        Args:
            entity_id: Entity to watch
            callback: Function to call on changes

        Returns:
            Created subscription
        """
        sub = Subscription(
            subscription_type=SubscriptionType.ENTITY,
            entity_id=entity_id,
        )
        return self._register(sub, callback)

    def subscribe_type(
        self,
        entity_type: EntityType,
        callback: SubscriptionCallback,
    ) -> Subscription:
        """
        Subscribe to all entities of a type.

        Args:
            entity_type: Type to watch
            callback: Function to call on changes

        Returns:
            Created subscription
        """
        sub = Subscription(
            subscription_type=SubscriptionType.TYPE,
            entity_type_filter=entity_type,
        )
        return self._register(sub, callback)

    def subscribe_subject(
        self,
        pattern: str,
        callback: SubscriptionCallback,
    ) -> Subscription:
        """
        Subscribe to claims matching a subject pattern.

        Args:
            pattern: Subject pattern (supports * wildcard at end)
            callback: Function to call on changes

        Returns:
            Created subscription
        """
        sub = Subscription(
            subscription_type=SubscriptionType.SUBJECT,
            subject_pattern=pattern,
        )
        return self._register(sub, callback)

    def subscribe_dependencies(
        self,
        entity_ids: list[str],
        callback: SubscriptionCallback,
    ) -> Subscription:
        """
        Subscribe to changes affecting a set of entities.

        Also watches transitive dependencies through the graph.

        Args:
            entity_ids: Entities to watch
            callback: Function to call on changes

        Returns:
            Created subscription
        """
        # Expand to include transitive dependencies
        all_watched = set(entity_ids)
        for entity_id in entity_ids:
            impact = self._graph.impact(entity_id)
            all_watched.update(impact.entities)

        sub = Subscription(
            subscription_type=SubscriptionType.DEPENDENCY,
            watch_entities=list(all_watched),
        )
        return self._register(sub, callback)

    def _register(
        self,
        sub: Subscription,
        callback: SubscriptionCallback,
    ) -> Subscription:
        """Register a subscription and update indexes."""
        with self._lock:
            self._subscriptions[sub.id] = sub
            self._callbacks[sub.id] = callback

            # Update indexes
            if sub.subscription_type == SubscriptionType.ENTITY and sub.entity_id:
                if sub.entity_id not in self._entity_index:
                    self._entity_index[sub.entity_id] = set()
                self._entity_index[sub.entity_id].add(sub.id)

            elif sub.subscription_type == SubscriptionType.TYPE and sub.entity_type_filter:
                if sub.entity_type_filter not in self._type_index:
                    self._type_index[sub.entity_type_filter] = set()
                self._type_index[sub.entity_type_filter].add(sub.id)

            elif sub.subscription_type == SubscriptionType.SUBJECT:
                self._subject_subscriptions.add(sub.id)

            elif sub.subscription_type == SubscriptionType.DEPENDENCY:
                self._dependency_subscriptions.add(sub.id)
                for entity_id in sub.watch_entities:
                    if entity_id not in self._entity_index:
                        self._entity_index[entity_id] = set()
                    self._entity_index[entity_id].add(sub.id)

            return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription.

        Args:
            subscription_id: Subscription to remove

        Returns:
            True if subscription existed and was removed
        """
        with self._lock:
            sub = self._subscriptions.pop(subscription_id, None)
            if sub is None:
                return False

            self._callbacks.pop(subscription_id, None)

            # Clean up indexes
            if sub.entity_id and sub.entity_id in self._entity_index:
                self._entity_index[sub.entity_id].discard(subscription_id)

            if sub.entity_type_filter and sub.entity_type_filter in self._type_index:
                self._type_index[sub.entity_type_filter].discard(subscription_id)

            self._subject_subscriptions.discard(subscription_id)
            self._dependency_subscriptions.discard(subscription_id)

            return True

    def notify(
        self,
        entity: Entity,
        event_type: str,
        old_value: Optional[Any] = None,
        caused_by: Optional[str] = None,
    ) -> int:
        """
        Notify subscribers of an entity change.

        Args:
            entity: The changed entity
            event_type: Type of change (create, update, delete, invalidate)
            old_value: Previous value (for updates)
            caused_by: Entity that caused this change

        Returns:
            Number of subscribers notified
        """
        with self._lock:
            # Find matching subscriptions
            matching_ids: set[str] = set()

            # Check entity-specific subscriptions
            matching_ids.update(self._entity_index.get(entity.id, set()))

            # Check type subscriptions
            matching_ids.update(self._type_index.get(entity.entity_type, set()))

            # Check subject subscriptions
            for sub_id in self._subject_subscriptions:
                sub = self._subscriptions.get(sub_id)
                if sub and sub.matches(entity, event_type):
                    matching_ids.add(sub_id)

            # Make a copy of subscriptions to notify (avoid holding lock during callbacks)
            to_notify = [
                (self._subscriptions[sid], self._callbacks[sid])
                for sid in matching_ids
                if sid in self._subscriptions and sid in self._callbacks
            ]

        # Deliver events (outside lock)
        count = 0
        for sub, callback in to_notify:
            if sub.matches(entity, event_type):
                event = SubscriptionEvent(
                    subscription_id=sub.id,
                    event_type=event_type,
                    entity_id=entity.id,
                    entity_type=entity.entity_type,
                    old_value=old_value,
                    new_value=entity.model_dump() if hasattr(entity, "model_dump") else None,
                    caused_by=caused_by,
                )

                try:
                    callback(event)
                    sub.events_delivered += 1
                    sub.last_event_at = datetime.now(timezone.utc)
                    count += 1
                except Exception:
                    # Log but don't fail on callback errors
                    pass

        return count

    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get a subscription by ID."""
        with self._lock:
            return self._subscriptions.get(subscription_id)

    def list_subscriptions(self) -> list[Subscription]:
        """List all active subscriptions."""
        with self._lock:
            return [s for s in self._subscriptions.values() if s.active]

    def subscription_count(self) -> int:
        """Get the number of active subscriptions."""
        with self._lock:
            return sum(1 for s in self._subscriptions.values() if s.active)
