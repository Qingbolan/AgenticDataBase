"""
Tests for SQLite storage backend.

This module tests the SQLiteStorage implementation that provides persistent
storage for AgenticDB using SQLite.

Test Coverage:
    - Basic CRUD operations (create, read, update, delete)
    - Entity versioning
    - Branch management
    - Content-addressable storage
    - Snapshot creation
    - Query operations
    - Concurrent access
    - Database persistence
"""

import pytest
import os
import tempfile
import time
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from agenticdb.core.models import (
    Entity,
    Event,
    Claim,
    Action,
    EntityType,
    EntityStatus,
)
from agenticdb.core.version import Branch, BranchStatus


class TestSQLiteStorageBasic:
    """Basic CRUD operation tests."""

    def test_store_and_retrieve_event(self, sqlite_storage):
        """Should store and retrieve an event."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        event = Event(
            event_type="UserRegistered",
            data={"user_id": "u123", "email": "test@example.com"}
        )

        version = sqlite_storage.store(event, branch.id)
        assert version is not None
        assert version.version_number == 1

        retrieved = sqlite_storage.get(event.id, branch.id)
        assert retrieved is not None
        assert retrieved.id == event.id
        assert retrieved.event_type == "UserRegistered"
        assert retrieved.data["user_id"] == "u123"

    def test_store_and_retrieve_claim(self, sqlite_storage):
        """Should store and retrieve a claim."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        claim = Claim(
            subject="user.risk_score",
            predicate="equals",
            value=0.5,
            source="risk_model_v1"
        )

        version = sqlite_storage.store(claim, branch.id)
        assert version is not None

        retrieved = sqlite_storage.get(claim.id, branch.id)
        assert retrieved is not None
        assert retrieved.subject == "user.risk_score"
        assert retrieved.value == 0.5

    def test_store_and_retrieve_action(self, sqlite_storage):
        """Should store and retrieve an action."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        action = Action(
            action_type="ApproveUser",
            agent_id="test-agent",
            inputs={"user_id": "u123"},
            depends_on=[]
        )

        version = sqlite_storage.store(action, branch.id)
        assert version is not None

        retrieved = sqlite_storage.get(action.id, branch.id)
        assert retrieved is not None
        assert retrieved.action_type == "ApproveUser"
        assert retrieved.agent_id == "test-agent"

    def test_get_nonexistent_entity(self, sqlite_storage):
        """Should return None for nonexistent entity."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        result = sqlite_storage.get("nonexistent", branch.id)
        assert result is None

    def test_store_without_branch_fails(self, sqlite_storage):
        """Should fail when storing to nonexistent branch."""
        event = Event(event_type="Test", data={})

        with pytest.raises(ValueError):
            sqlite_storage.store(event, "nonexistent-branch")


class TestSQLiteVersioning:
    """Tests for entity versioning."""

    def test_multiple_versions_of_entity(self, sqlite_storage):
        """Should track multiple versions of the same entity."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        # Store initial version
        event = Event(event_type="Test", data={"v": 1})
        v1 = sqlite_storage.store(event, branch.id)
        assert v1.version_number == 1

        # Store update (same entity, new version)
        event.data = {"v": 2}
        v2 = sqlite_storage.store(event, branch.id)
        assert v2.version_number == 2

        # Latest version should be v2
        current = sqlite_storage.get(event.id, branch.id)
        assert current.data == {"v": 2}

    def test_get_at_version(self, sqlite_storage):
        """Should retrieve entity at specific version."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        # Create multiple entities with different versions
        e1 = Event(event_type="E1", data={"n": 1})
        sqlite_storage.store(e1, branch.id)  # v1

        e2 = Event(event_type="E2", data={"n": 2})
        sqlite_storage.store(e2, branch.id)  # v2

        e3 = Event(event_type="E3", data={"n": 3})
        sqlite_storage.store(e3, branch.id)  # v3

        # Get at version 2 - should see e1 and e2 but not e3
        at_v2 = sqlite_storage.get_at_version(e2.id, branch.id, 2)
        assert at_v2 is not None
        assert at_v2.event_type == "E2"

        at_v2_e3 = sqlite_storage.get_at_version(e3.id, branch.id, 2)
        assert at_v2_e3 is None  # e3 didn't exist at v2

    def test_get_versions_range(self, sqlite_storage):
        """Should get versions within a range."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        # Create 5 entities
        for i in range(5):
            event = Event(event_type=f"E{i}", data={"i": i})
            sqlite_storage.store(event, branch.id)

        # Get versions 2-4
        versions = sqlite_storage.get_versions(branch.id, from_version=2, to_version=4)
        assert len(versions) == 3
        assert all(2 <= v.version_number <= 4 for v in versions)


class TestSQLiteBranchManagement:
    """Tests for branch operations."""

    def test_create_branch(self, sqlite_storage):
        """Should create a new branch."""
        branch = Branch(name="feature-branch", description="Test branch")
        created = sqlite_storage.create_branch(branch)

        assert created.id == branch.id
        assert created.name == "feature-branch"

    def test_get_branch(self, sqlite_storage):
        """Should retrieve a branch by ID."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        retrieved = sqlite_storage.get_branch(branch.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_update_branch(self, sqlite_storage):
        """Should update branch state."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        branch.description = "Updated description"
        branch.increment_version()
        updated = sqlite_storage.update_branch(branch)

        assert updated.description == "Updated description"
        assert updated.head_version == 1

    def test_duplicate_branch_fails(self, sqlite_storage):
        """Should fail when creating duplicate branch."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        with pytest.raises(ValueError):
            sqlite_storage.create_branch(branch)


class TestSQLiteQueryOperations:
    """Tests for query operations."""

    def test_query_by_entity_type(self, sqlite_storage):
        """Should query entities by type."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        # Store different entity types
        e1 = Event(event_type="E1", data={})
        e2 = Event(event_type="E2", data={})
        c1 = Claim(subject="s1", predicate="p", value=1, source="src")

        sqlite_storage.store(e1, branch.id)
        sqlite_storage.store(e2, branch.id)
        sqlite_storage.store(c1, branch.id)

        # Query events only
        events = list(sqlite_storage.query(branch.id, entity_type=EntityType.EVENT))
        assert len(events) == 2
        assert all(e.entity_type == EntityType.EVENT for e in events)

        # Query claims only
        claims = list(sqlite_storage.query(branch.id, entity_type=EntityType.CLAIM))
        assert len(claims) == 1

    def test_query_with_filters(self, sqlite_storage):
        """Should query with attribute filters."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        e1 = Event(event_type="OrderPlaced", data={"amount": 100})
        e2 = Event(event_type="OrderPlaced", data={"amount": 200})
        e3 = Event(event_type="OrderCancelled", data={"amount": 100})

        sqlite_storage.store(e1, branch.id)
        sqlite_storage.store(e2, branch.id)
        sqlite_storage.store(e3, branch.id)

        # Query by event_type
        placed = list(sqlite_storage.query(
            branch.id,
            entity_type=EntityType.EVENT,
            filters={"event_type": "OrderPlaced"}
        ))
        assert len(placed) == 2

    def test_query_with_limit(self, sqlite_storage):
        """Should limit query results."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        # Store 10 events
        for i in range(10):
            event = Event(event_type=f"E{i}", data={})
            sqlite_storage.store(event, branch.id)

        # Query with limit
        limited = list(sqlite_storage.query(branch.id, limit=5))
        assert len(limited) == 5


class TestSQLiteContentAddressing:
    """Tests for content-addressable storage."""

    def test_content_hash_stored(self, sqlite_storage):
        """Should store and index content hash."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        event = Event(event_type="Test", data={"key": "value"})
        sqlite_storage.store(event, branch.id)

        # Should be able to retrieve by content hash
        retrieved = sqlite_storage.get_by_content_hash(event.content_hash, branch.id)
        assert retrieved is not None
        assert retrieved.id == event.id

    def test_same_content_same_hash(self, sqlite_storage):
        """Same content should produce same hash."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        e1 = Event(event_type="Test", data={"key": "value"})
        e2 = Event(event_type="Test", data={"key": "value"})

        assert e1.content_hash == e2.content_hash


class TestSQLiteSnapshots:
    """Tests for snapshot creation."""

    def test_create_snapshot(self, sqlite_storage):
        """Should create snapshot at specific version."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        # Add some entities
        e1 = Event(event_type="E1", data={})
        e2 = Event(event_type="E2", data={})
        c1 = Claim(subject="s1", predicate="p", value=1, source="src")

        sqlite_storage.store(e1, branch.id)  # v1
        sqlite_storage.store(e2, branch.id)  # v2
        sqlite_storage.store(c1, branch.id)  # v3

        # Snapshot at v2 - should only include e1 and e2
        snapshot = sqlite_storage.create_snapshot(branch.id, 2)

        assert snapshot.version == 2
        assert snapshot.entity_count == 2
        assert len(snapshot.events) == 2
        assert len(snapshot.claims) == 0  # c1 was added at v3


class TestSQLiteConcurrency:
    """Tests for concurrent access."""

    def test_concurrent_writes(self, sqlite_storage):
        """Should handle concurrent writes safely."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        errors = []
        stored_ids = []

        def write_event(i):
            try:
                event = Event(event_type=f"E{i}", data={"i": i})
                sqlite_storage.store(event, branch.id)
                stored_ids.append(event.id)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_event, i) for i in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(stored_ids) == 20

    def test_concurrent_reads(self, sqlite_storage):
        """Should handle concurrent reads safely."""
        branch = Branch(name="test")
        sqlite_storage.create_branch(branch)

        event = Event(event_type="Test", data={"value": 42})
        sqlite_storage.store(event, branch.id)

        results = []
        errors = []

        def read_event():
            try:
                for _ in range(50):
                    result = sqlite_storage.get(event.id, branch.id)
                    if result:
                        results.append(result.data["value"])
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=read_event) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(v == 42 for v in results)


class TestSQLitePersistence:
    """Tests for data persistence across connections."""

    def test_data_persists_after_close(self):
        """Data should persist after closing and reopening."""
        from agenticdb.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create storage, add data, close
            storage1 = SQLiteStorage(db_path)
            branch = Branch(name="test")
            storage1.create_branch(branch)

            event = Event(event_type="Persistent", data={"key": "value"})
            storage1.store(event, branch.id)
            event_id = event.id
            storage1.close()

            # Open new connection, verify data exists
            storage2 = SQLiteStorage(db_path)
            retrieved = storage2.get(event_id, branch.id)
            assert retrieved is not None
            assert retrieved.event_type == "Persistent"
            storage2.close()
        finally:
            os.unlink(db_path)

    def test_clear_data(self):
        """Clear should remove all data."""
        from agenticdb.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)
            branch = Branch(name="test")
            storage.create_branch(branch)

            event = Event(event_type="Test", data={})
            storage.store(event, branch.id)

            storage.clear()

            # Branch should be gone
            assert storage.get_branch(branch.id) is None
            storage.close()
        finally:
            os.unlink(db_path)


class TestSQLiteIntegration:
    """Integration tests with AgenticDB client."""

    def test_use_with_agenticdb(self):
        """Should work as storage backend for AgenticDB."""
        from agenticdb.storage.sqlite import SQLiteStorage
        from agenticdb.interface.client import AgenticDB

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)
            db = AgenticDB(storage=storage)

            branch = db.create_branch("test-branch")

            # Record events
            event = branch.record(Event(
                event_type="UserSignup",
                data={"user_id": "u123"}
            ))

            # Record claims
            claim = branch.record(Claim(
                subject="user.verified",
                predicate="equals",
                value=True,
                source="verification_service",
                derived_from=[event.id]
            ))

            # Execute action
            action = branch.execute(Action(
                action_type="SendWelcomeEmail",
                agent_id="email-agent",
                depends_on=[claim.id]
            ))

            # Verify data
            retrieved_event = branch.get(event.id)
            assert retrieved_event is not None

            # Test why() query
            chain = branch.why(action.id)
            assert len(chain.steps) > 0

            storage.close()
        finally:
            os.unlink(db_path)


# Fixtures
@pytest.fixture
def sqlite_storage():
    """Create a temporary SQLite storage for testing."""
    from agenticdb.storage.sqlite import SQLiteStorage

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage
    storage.close()
    os.unlink(db_path)
