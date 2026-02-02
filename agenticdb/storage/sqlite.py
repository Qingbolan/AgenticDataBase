"""
SQLite storage backend for AgenticDB.

This module provides a persistent storage implementation using SQLite,
suitable for single-node deployments and development.

Design Philosophy:
    SQLite provides ACID guarantees, making it ideal for:
    - Development and testing
    - Single-user applications
    - Embedded deployments
    - Small to medium datasets

Schema Design:
    - entities: Core entity data with JSON serialization
    - versions: Version history for time-travel queries
    - branches: Branch metadata
    - content_index: Content-addressable lookup

Thread Safety:
    SQLite in WAL mode supports concurrent reads with a single writer.
    This implementation uses connection pooling and proper locking
    for thread-safe operation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from agenticdb.core.models import (
    Entity,
    EntityType,
    EntityStatus,
    Event,
    Claim,
    Action,
)
from agenticdb.core.version import Branch, BranchStatus, Version, Snapshot
from agenticdb.storage.engine import StorageEngine


class SQLiteStorage(StorageEngine):
    """
    SQLite-based storage backend for AgenticDB.

    Provides persistent storage with:
    - ACID guarantees
    - Event-sourced history
    - Content-addressable entities
    - Efficient queries via indexes

    Usage:
        ```python
        storage = SQLiteStorage("./data.db")

        branch = Branch(name="main")
        storage.create_branch(branch)

        event = Event(event_type="Test", data={})
        storage.store(event, branch.id)

        retrieved = storage.get(event.id, branch.id)
        ```

    Thread Safety:
        This implementation is thread-safe. Multiple threads can read
        concurrently, and writes are serialized via locking.
    """

    def __init__(self, db_path: str | Path, timeout: float = 30.0):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
            timeout: Connection timeout in seconds
        """
        self._db_path = Path(db_path)
        self._timeout = timeout
        self._lock = threading.RLock()

        # Thread-local connections
        self._local = threading.local()

        # Initialize database schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                timeout=self._timeout,
                check_same_thread=False,
            )
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Branches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS branches (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    parent_branch_id TEXT,
                    fork_version INTEGER DEFAULT 0,
                    head_version INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """)

            # Entities table (stores all entity types)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    entity_type TEXT NOT NULL,
                    content_hash TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (id, branch_id, version),
                    FOREIGN KEY (branch_id) REFERENCES branches(id)
                )
            """)

            # Versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_number INTEGER NOT NULL,
                    branch_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (branch_id) REFERENCES branches(id)
                )
            """)

            # Content hash index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON entities(content_hash, branch_id)
            """)

            # Entity type index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_type
                ON entities(entity_type, branch_id)
            """)

            # Version index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_version
                ON entities(branch_id, version)
            """)

            conn.commit()

    def store(self, entity: Entity, branch_id: str) -> Version:
        """
        Store an entity and create a new version.

        Args:
            entity: Entity to store
            branch_id: Branch to store in

        Returns:
            Version created by this operation

        Raises:
            ValueError: If branch doesn't exist
        """
        with self._lock:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Verify branch exists
                cursor.execute(
                    "SELECT id, head_version FROM branches WHERE id = ?",
                    (branch_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"Branch not found: {branch_id}")

                # Increment version
                new_version = row['head_version'] + 1
                cursor.execute(
                    "UPDATE branches SET head_version = ? WHERE id = ?",
                    (new_version, branch_id)
                )

                # Set entity metadata
                entity.branch_id = branch_id
                entity.version = new_version

                # Serialize entity
                entity_data = self._serialize_entity(entity)

                # Check if this is create or update
                cursor.execute(
                    "SELECT COUNT(*) FROM entities WHERE id = ? AND branch_id = ?",
                    (entity.id, branch_id)
                )
                is_update = cursor.fetchone()[0] > 0
                operation = "update" if is_update else "create"

                # Store entity
                cursor.execute("""
                    INSERT INTO entities (
                        id, branch_id, version, entity_type, content_hash,
                        status, data, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    branch_id,
                    new_version,
                    entity.entity_type.value,
                    entity.content_hash,
                    entity.status.value,
                    entity_data,
                    datetime.now(timezone.utc).isoformat(),
                ))

                # Create version record
                cursor.execute("""
                    INSERT INTO versions (
                        version_number, branch_id, entity_id, operation, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    new_version,
                    branch_id,
                    entity.id,
                    operation,
                    datetime.now(timezone.utc).isoformat(),
                ))

                return Version(
                    version_number=new_version,
                    branch_id=branch_id,
                    entity_id=entity.id,
                    operation=operation,
                )

    def get(self, entity_id: str, branch_id: str) -> Optional[Entity]:
        """
        Get the latest version of an entity.

        Args:
            entity_id: Entity identifier
            branch_id: Branch to search in

        Returns:
            Entity if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT data, entity_type FROM entities
            WHERE id = ? AND branch_id = ?
            ORDER BY version DESC
            LIMIT 1
        """, (entity_id, branch_id))

        row = cursor.fetchone()
        if row is None:
            return None

        return self._deserialize_entity(row['data'], row['entity_type'])

    def get_at_version(
        self,
        entity_id: str,
        branch_id: str,
        version: int
    ) -> Optional[Entity]:
        """
        Get an entity as it was at a specific version.

        Args:
            entity_id: Entity identifier
            branch_id: Branch to search in
            version: Version number

        Returns:
            Entity as it was at that version, None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT data, entity_type FROM entities
            WHERE id = ? AND branch_id = ? AND version <= ?
            ORDER BY version DESC
            LIMIT 1
        """, (entity_id, branch_id, version))

        row = cursor.fetchone()
        if row is None:
            return None

        return self._deserialize_entity(row['data'], row['entity_type'])

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
            filters: Additional filters (matched against entity attributes)
            limit: Maximum results

        Yields:
            Matching entities
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query to get latest version of each entity
        sql = """
            SELECT e.data, e.entity_type
            FROM entities e
            INNER JOIN (
                SELECT id, MAX(version) as max_version
                FROM entities
                WHERE branch_id = ?
                GROUP BY id
            ) latest ON e.id = latest.id AND e.version = latest.max_version
            WHERE e.branch_id = ?
        """
        params: list = [branch_id, branch_id]

        if entity_type is not None:
            sql += " AND e.entity_type = ?"
            params.append(entity_type.value)

        if limit is not None:
            sql += f" LIMIT {limit}"

        cursor.execute(sql, params)

        count = 0
        for row in cursor:
            entity = self._deserialize_entity(row['data'], row['entity_type'])
            if entity is None:
                continue

            # Apply attribute filters
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
            if limit is not None and count >= limit:
                break

    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """Get a branch by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM branches WHERE id = ?",
            (branch_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._deserialize_branch(row)

    def create_branch(self, branch: Branch) -> Branch:
        """Create a new branch."""
        with self._lock:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Check if branch exists
                cursor.execute(
                    "SELECT id FROM branches WHERE id = ?",
                    (branch.id,)
                )
                if cursor.fetchone() is not None:
                    raise ValueError(f"Branch already exists: {branch.id}")

                cursor.execute("""
                    INSERT INTO branches (
                        id, name, description, status, parent_branch_id,
                        fork_version, head_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    branch.id,
                    branch.name,
                    branch.description,
                    branch.status.value,
                    branch.parent_branch_id,
                    branch.fork_version,
                    branch.head_version,
                    datetime.now(timezone.utc).isoformat(),
                ))

                return branch

    def update_branch(self, branch: Branch) -> Branch:
        """Update a branch."""
        with self._lock:
            with self._transaction() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT id FROM branches WHERE id = ?",
                    (branch.id,)
                )
                if cursor.fetchone() is None:
                    raise ValueError(f"Branch not found: {branch.id}")

                cursor.execute("""
                    UPDATE branches SET
                        name = ?,
                        description = ?,
                        status = ?,
                        head_version = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    branch.name,
                    branch.description,
                    branch.status.value,
                    branch.head_version,
                    datetime.now(timezone.utc).isoformat(),
                    branch.id,
                ))

                return branch

    def get_versions(
        self,
        branch_id: str,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
    ) -> list[Version]:
        """Get versions for a branch within a range."""
        conn = self._get_connection()
        cursor = conn.cursor()

        sql = "SELECT * FROM versions WHERE branch_id = ?"
        params: list = [branch_id]

        if from_version is not None:
            sql += " AND version_number >= ?"
            params.append(from_version)

        if to_version is not None:
            sql += " AND version_number <= ?"
            params.append(to_version)

        sql += " ORDER BY version_number"

        cursor.execute(sql, params)

        versions = []
        for row in cursor:
            versions.append(Version(
                version_number=row['version_number'],
                branch_id=row['branch_id'],
                entity_id=row['entity_id'],
                operation=row['operation'],
                created_at=datetime.fromisoformat(row['created_at']),
            ))
        return versions

    def create_snapshot(self, branch_id: str, version: int) -> Snapshot:
        """Create a snapshot at a specific version."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Verify branch exists
        cursor.execute(
            "SELECT id FROM branches WHERE id = ?",
            (branch_id,)
        )
        if cursor.fetchone() is None:
            raise ValueError(f"Branch not found: {branch_id}")

        # Get all entities that existed at this version
        cursor.execute("""
            SELECT DISTINCT e.id, e.entity_type
            FROM entities e
            INNER JOIN (
                SELECT id, MAX(version) as max_version
                FROM entities
                WHERE branch_id = ? AND version <= ?
                GROUP BY id
            ) latest ON e.id = latest.id AND e.version = latest.max_version
            WHERE e.branch_id = ? AND e.status = 'active'
        """, (branch_id, version, branch_id))

        events: list[str] = []
        claims: list[str] = []
        actions: list[str] = []

        for row in cursor:
            entity_type = row['entity_type']
            entity_id = row['id']

            if entity_type == EntityType.EVENT.value:
                events.append(entity_id)
            elif entity_type == EntityType.CLAIM.value:
                claims.append(entity_id)
            elif entity_type == EntityType.ACTION.value:
                actions.append(entity_id)

        # Get timestamp from version
        cursor.execute("""
            SELECT created_at FROM versions
            WHERE branch_id = ? AND version_number = ?
        """, (branch_id, version))
        row = cursor.fetchone()
        timestamp = (
            datetime.fromisoformat(row['created_at'])
            if row else datetime.now(timezone.utc)
        )

        return Snapshot(
            branch_id=branch_id,
            version=version,
            timestamp=timestamp,
            events=events,
            claims=claims,
            actions=actions,
            entity_count=len(events) + len(claims) + len(actions),
        )

    def get_by_content_hash(
        self,
        content_hash: str,
        branch_id: str
    ) -> Optional[Entity]:
        """Get an entity by its content hash."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT data, entity_type FROM entities
            WHERE content_hash = ? AND branch_id = ?
            ORDER BY version DESC
            LIMIT 1
        """, (content_hash, branch_id))

        row = cursor.fetchone()
        if row is None:
            return None

        return self._deserialize_entity(row['data'], row['entity_type'])

    def get_all_entities(self, branch_id: str) -> list[Entity]:
        """Get all current entities in a branch."""
        return list(self.query(branch_id))

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM versions")
                cursor.execute("DELETE FROM entities")
                cursor.execute("DELETE FROM branches")

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # =========================================================================
    # Serialization Helpers
    # =========================================================================

    def _serialize_entity(self, entity: Entity) -> str:
        """Serialize an entity to JSON string."""
        return entity.model_dump_json()

    def _deserialize_entity(
        self,
        data: str,
        entity_type: str
    ) -> Optional[Entity]:
        """Deserialize an entity from JSON string."""
        try:
            if entity_type == EntityType.EVENT.value:
                return Event.model_validate_json(data)
            elif entity_type == EntityType.CLAIM.value:
                return Claim.model_validate_json(data)
            elif entity_type == EntityType.ACTION.value:
                return Action.model_validate_json(data)
            else:
                return None
        except Exception:
            return None

    def _deserialize_branch(self, row: sqlite3.Row) -> Branch:
        """Deserialize a branch from database row."""
        return Branch(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            status=BranchStatus(row['status']),
            parent_branch_id=row['parent_branch_id'],
            fork_version=row['fork_version'],
            head_version=row['head_version'],
            created_at=datetime.fromisoformat(row['created_at']),
        )
