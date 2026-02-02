# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Pattern Store for AgenticDB.

Provides SQLite-based persistence for query patterns used in workload memoization.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import Pattern, OperationType


class PatternStore:
    """
    SQLite-based storage for query patterns.

    Provides persistence for the pattern cache, enabling patterns to
    survive across sessions and be shared across instances.

    Example:
        store = PatternStore("patterns.db")
        store.save(pattern)
        patterns = store.get_all()
        match = store.find_by_hash("abc123")
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the pattern store.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

        # Create patterns table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                template TEXT NOT NULL,
                structural_hash TEXT NOT NULL,
                operation TEXT NOT NULL,
                target_table TEXT,
                parameter_slots TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_used TEXT,
                created_at TEXT NOT NULL,
                example_queries TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                metadata TEXT
            )
        """)

        # Create index on structural hash for fast lookup
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_hash
            ON patterns(structural_hash)
        """)

        # Create index on operation for filtering
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_operation
            ON patterns(operation)
        """)

        self._conn.commit()

    def save(self, pattern: Pattern) -> None:
        """
        Save or update a pattern.

        Args:
            pattern: Pattern to save
        """
        self._conn.execute("""
            INSERT OR REPLACE INTO patterns
            (id, template, structural_hash, operation, target_table,
             parameter_slots, hit_count, last_used, created_at,
             example_queries, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.id,
            pattern.template,
            pattern.structural_hash,
            pattern.operation.value,
            pattern.target_table,
            json.dumps(pattern.parameter_slots),
            pattern.hit_count,
            pattern.last_used.isoformat() if pattern.last_used else None,
            pattern.created_at.isoformat(),
            json.dumps(pattern.example_queries),
            pattern.confidence,
            None,  # metadata
        ))
        self._conn.commit()

    def save_many(self, patterns: List[Pattern]) -> None:
        """
        Save multiple patterns in a batch.

        Args:
            patterns: List of patterns to save
        """
        for pattern in patterns:
            self.save(pattern)

    def get(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get a pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern or None if not found
        """
        cursor = self._conn.execute(
            "SELECT * FROM patterns WHERE id = ?",
            (pattern_id,)
        )
        row = cursor.fetchone()
        return self._row_to_pattern(row) if row else None

    def get_all(self) -> List[Pattern]:
        """
        Get all patterns.

        Returns:
            List of all patterns
        """
        cursor = self._conn.execute("SELECT * FROM patterns")
        return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def find_by_hash(self, structural_hash: str) -> Optional[Pattern]:
        """
        Find a pattern by structural hash.

        Args:
            structural_hash: Structural hash to search for

        Returns:
            Pattern or None if not found
        """
        cursor = self._conn.execute(
            "SELECT * FROM patterns WHERE structural_hash = ?",
            (structural_hash,)
        )
        row = cursor.fetchone()
        return self._row_to_pattern(row) if row else None

    def find_by_operation(self, operation: OperationType) -> List[Pattern]:
        """
        Find patterns by operation type.

        Args:
            operation: Operation type to filter by

        Returns:
            List of matching patterns
        """
        cursor = self._conn.execute(
            "SELECT * FROM patterns WHERE operation = ?",
            (operation.value,)
        )
        return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def find_by_table(self, table_name: str) -> List[Pattern]:
        """
        Find patterns by target table.

        Args:
            table_name: Table name to search for

        Returns:
            List of matching patterns
        """
        cursor = self._conn.execute(
            "SELECT * FROM patterns WHERE target_table = ?",
            (table_name,)
        )
        return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def record_hit(self, pattern_id: str) -> None:
        """
        Record a cache hit for a pattern.

        Args:
            pattern_id: ID of pattern that was hit
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("""
            UPDATE patterns
            SET hit_count = hit_count + 1, last_used = ?
            WHERE id = ?
        """, (now, pattern_id))
        self._conn.commit()

    def delete(self, pattern_id: str) -> bool:
        """
        Delete a pattern.

        Args:
            pattern_id: ID of pattern to delete

        Returns:
            True if pattern was deleted
        """
        cursor = self._conn.execute(
            "DELETE FROM patterns WHERE id = ?",
            (pattern_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """
        Clear all patterns.

        Returns:
            Number of patterns deleted
        """
        cursor = self._conn.execute("DELETE FROM patterns")
        self._conn.commit()
        return cursor.rowcount

    def get_top_patterns(self, limit: int = 10) -> List[Pattern]:
        """
        Get the most frequently used patterns.

        Args:
            limit: Maximum number of patterns to return

        Returns:
            List of top patterns sorted by hit count
        """
        cursor = self._conn.execute(
            "SELECT * FROM patterns ORDER BY hit_count DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pattern store.

        Returns:
            Dictionary with statistics
        """
        cursor = self._conn.execute("""
            SELECT
                COUNT(*) as total_patterns,
                SUM(hit_count) as total_hits,
                AVG(hit_count) as avg_hits,
                AVG(confidence) as avg_confidence
            FROM patterns
        """)
        row = cursor.fetchone()

        # Get patterns by operation
        cursor = self._conn.execute("""
            SELECT operation, COUNT(*) as count
            FROM patterns
            GROUP BY operation
        """)
        by_operation = {r["operation"]: r["count"] for r in cursor.fetchall()}

        return {
            "total_patterns": row["total_patterns"] or 0,
            "total_hits": row["total_hits"] or 0,
            "avg_hits": row["avg_hits"] or 0.0,
            "avg_confidence": row["avg_confidence"] or 0.0,
            "by_operation": by_operation,
        }

    def prune_unused(self, min_hits: int = 0, older_than_days: int = 30) -> int:
        """
        Remove patterns that haven't been used.

        Args:
            min_hits: Minimum hit count to keep
            older_than_days: Remove if last used more than this many days ago

        Returns:
            Number of patterns removed
        """
        cutoff = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=older_than_days)

        cursor = self._conn.execute("""
            DELETE FROM patterns
            WHERE hit_count <= ?
            AND (last_used IS NULL OR last_used < ?)
        """, (min_hits, cutoff.isoformat()))
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_pattern(self, row: sqlite3.Row) -> Pattern:
        """Convert a database row to a Pattern object."""
        return Pattern(
            id=row["id"],
            template=row["template"],
            structural_hash=row["structural_hash"],
            operation=OperationType(row["operation"]),
            target_table=row["target_table"],
            parameter_slots=json.loads(row["parameter_slots"]),
            hit_count=row["hit_count"],
            last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            example_queries=json.loads(row["example_queries"]),
            confidence=row["confidence"],
        )

    def __enter__(self) -> "PatternStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
