"""
Storage engine for AgenticDB.

This module provides the storage abstraction layer, enabling pluggable
backends while maintaining consistent semantics:

- Event-sourced: All changes are appended, never modified
- Versioned: Every mutation creates a new version
- Content-addressable: Entities can be retrieved by content hash

Design Philosophy:
    Storage is separated from semantics. The storage engine knows
    how to persist and retrieve entities, but doesn't understand
    the semantic relationships between them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Generic, Iterator, Optional, TypeVar

from agenticdb.core.models import Entity, EntityStatus, EntityType, Event, Claim, Action
from agenticdb.core.version import Branch, Version, Snapshot


T = TypeVar("T", bound=Entity)


class StorageEngine(ABC):
    """
    Abstract base class for storage backends.

    Storage engines must implement basic CRUD operations while
    respecting the event-sourced, versioned nature of AgenticDB.

    Implementations:
        - InMemoryStorage: Development and testing
        - SQLiteStorage: Single-node persistence (planned)
        - PostgresStorage: Production deployment (planned)
    """

    @abstractmethod
    def store(self, entity: Entity, branch_id: str) -> Version:
        """
        Store an entity and create a new version.

        Args:
            entity: Entity to store
            branch_id: Branch to store in

        Returns:
            Version created by this operation
        """
        pass

    @abstractmethod
    def get(self, entity_id: str, branch_id: str) -> Optional[Entity]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: Entity identifier
            branch_id: Branch to search in

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    def get_at_version(
        self,
        entity_id: str,
        branch_id: str,
        version: int
    ) -> Optional[Entity]:
        """
        Retrieve an entity as it was at a specific version.

        Args:
            entity_id: Entity identifier
            branch_id: Branch to search in
            version: Version number

        Returns:
            Entity as it was at that version, None if not found
        """
        pass

    @abstractmethod
    def query(
        self,
        branch_id: str,
        entity_type: Optional[EntityType] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Entity]:
        """
        Query entities with optional filters.

        Args:
            branch_id: Branch to search in
            entity_type: Filter by entity type
            filters: Additional filters
            limit: Maximum results

        Yields:
            Matching entities
        """
        pass

    @abstractmethod
    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """Get a branch by ID."""
        pass

    @abstractmethod
    def create_branch(self, branch: Branch) -> Branch:
        """Create a new branch."""
        pass

    @abstractmethod
    def update_branch(self, branch: Branch) -> Branch:
        """Update a branch."""
        pass

    @abstractmethod
    def get_versions(
        self,
        branch_id: str,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
    ) -> list[Version]:
        """Get versions for a branch within a range."""
        pass

    @abstractmethod
    def create_snapshot(self, branch_id: str, version: int) -> Snapshot:
        """Create a snapshot at a specific version."""
        pass


class InMemoryStorage(StorageEngine):
    """
    In-memory storage implementation for development and testing.

    This implementation stores all data in memory using dictionaries.
    Data is lost when the process exits.

    Thread Safety:
        This implementation is thread-safe using a reentrant lock.
        However, iterators returned by query() are not thread-safe.
    """

    def __init__(self):
        """Initialize empty storage."""
        self._lock = RLock()

        # Branch storage
        self._branches: dict[str, Branch] = {}

        # Entity storage: branch_id -> entity_id -> list of (version, entity)
        self._entities: dict[str, dict[str, list[tuple[int, Entity]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Version storage: branch_id -> list of versions
        self._versions: dict[str, list[Version]] = defaultdict(list)

        # Index by content hash: hash -> entity_id
        self._content_index: dict[str, str] = {}

        # Index by type: branch_id -> entity_type -> set of entity_ids
        self._type_index: dict[str, dict[EntityType, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def store(self, entity: Entity, branch_id: str) -> Version:
        """
        Store an entity and create a new version.

        The entity is appended to the history for its ID, creating
        an immutable record of all changes.
        """
        with self._lock:
            # Get or create branch
            branch = self._branches.get(branch_id)
            if branch is None:
                raise ValueError(f"Branch not found: {branch_id}")

            # Increment version
            new_version_num = branch.increment_version()

            # Set entity metadata
            entity.branch_id = branch_id
            entity.version = new_version_num

            # Store entity
            self._entities[branch_id][entity.id].append((new_version_num, entity))

            # Update indexes
            if entity.content_hash:
                self._content_index[entity.content_hash] = entity.id
            self._type_index[branch_id][entity.entity_type].add(entity.id)

            # Create version record
            version = Version(
                version_number=new_version_num,
                branch_id=branch_id,
                entity_id=entity.id,
                operation="create" if len(self._entities[branch_id][entity.id]) == 1 else "update",
            )
            self._versions[branch_id].append(version)

            return version

    def get(self, entity_id: str, branch_id: str) -> Optional[Entity]:
        """Get the latest version of an entity."""
        with self._lock:
            history = self._entities.get(branch_id, {}).get(entity_id, [])
            if not history:
                return None
            # Return the latest version
            return history[-1][1]

    def get_at_version(
        self,
        entity_id: str,
        branch_id: str,
        version: int
    ) -> Optional[Entity]:
        """Get an entity as it was at a specific version."""
        with self._lock:
            history = self._entities.get(branch_id, {}).get(entity_id, [])

            # Find the version that was active at the requested version
            result = None
            for v, entity in history:
                if v <= version:
                    result = entity
                else:
                    break

            return result

    def query(
        self,
        branch_id: str,
        entity_type: Optional[EntityType] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Entity]:
        """Query entities with optional filters."""
        with self._lock:
            # Get candidate entity IDs
            if entity_type is not None:
                entity_ids = self._type_index.get(branch_id, {}).get(entity_type, set())
            else:
                entity_ids = set()
                for type_ids in self._type_index.get(branch_id, {}).values():
                    entity_ids.update(type_ids)

            # Make a copy to avoid modification during iteration
            entity_ids = list(entity_ids)

        # Yield matching entities (outside lock to avoid holding it too long)
        count = 0
        for entity_id in entity_ids:
            if limit is not None and count >= limit:
                break

            entity = self.get(entity_id, branch_id)
            if entity is None:
                continue

            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if hasattr(entity, key):
                        if getattr(entity, key) != value:
                            match = False
                            break
                    elif key in entity.metadata:
                        if entity.metadata[key] != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                if not match:
                    continue

            yield entity
            count += 1

    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """Get a branch by ID."""
        with self._lock:
            return self._branches.get(branch_id)

    def create_branch(self, branch: Branch) -> Branch:
        """Create a new branch."""
        with self._lock:
            if branch.id in self._branches:
                raise ValueError(f"Branch already exists: {branch.id}")
            self._branches[branch.id] = branch
            return branch

    def update_branch(self, branch: Branch) -> Branch:
        """Update a branch."""
        with self._lock:
            if branch.id not in self._branches:
                raise ValueError(f"Branch not found: {branch.id}")
            self._branches[branch.id] = branch
            return branch

    def get_versions(
        self,
        branch_id: str,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
    ) -> list[Version]:
        """Get versions for a branch within a range."""
        with self._lock:
            versions = self._versions.get(branch_id, [])

            if from_version is not None:
                versions = [v for v in versions if v.version_number >= from_version]
            if to_version is not None:
                versions = [v for v in versions if v.version_number <= to_version]

            return list(versions)

    def create_snapshot(self, branch_id: str, version: int) -> Snapshot:
        """Create a snapshot at a specific version."""
        with self._lock:
            branch = self._branches.get(branch_id)
            if branch is None:
                raise ValueError(f"Branch not found: {branch_id}")

            events: list[str] = []
            claims: list[str] = []
            actions: list[str] = []

            # Collect all entities that existed at this version
            for entity_id, history in self._entities.get(branch_id, {}).items():
                for v, entity in history:
                    if v <= version and entity.is_active():
                        if entity.entity_type == EntityType.EVENT:
                            events.append(entity_id)
                        elif entity.entity_type == EntityType.CLAIM:
                            claims.append(entity_id)
                        elif entity.entity_type == EntityType.ACTION:
                            actions.append(entity_id)
                        break  # Only count once per entity

            # Get timestamp from the version
            version_records = [v for v in self._versions.get(branch_id, []) if v.version_number == version]
            timestamp = version_records[0].created_at if version_records else datetime.now(timezone.utc)

            return Snapshot(
                branch_id=branch_id,
                version=version,
                timestamp=timestamp,
                events=events,
                claims=claims,
                actions=actions,
                entity_count=len(events) + len(claims) + len(actions),
            )

    def get_by_content_hash(self, content_hash: str, branch_id: str) -> Optional[Entity]:
        """Get an entity by its content hash."""
        with self._lock:
            entity_id = self._content_index.get(content_hash)
            if entity_id is None:
                return None
            return self.get(entity_id, branch_id)

    def get_all_entities(self, branch_id: str) -> list[Entity]:
        """Get all current entities in a branch."""
        with self._lock:
            entities = []
            for entity_id in self._entities.get(branch_id, {}):
                entity = self.get(entity_id, branch_id)
                if entity is not None:
                    entities.append(entity)
            return entities

    def clear(self) -> None:
        """Clear all data (for testing)."""
        with self._lock:
            self._branches.clear()
            self._entities.clear()
            self._versions.clear()
            self._content_index.clear()
            self._type_index.clear()
