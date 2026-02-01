"""
Version control for AgenticDB.

This module provides Git-like branching and versioning for agent state,
enabling:

- Branch: Isolated workspaces for parallel agent workflows
- Version: Immutable snapshots of state at a point in time
- Snapshot: Reconstructable view of state at any version

Design Philosophy:
    Agent workflows often need to explore alternatives, rollback on failure,
    or compare different decision paths. Version control is not an add-on
    but a fundamental requirement for reproducible agent systems.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from agenticdb.core.models import Entity, generate_id


class BranchStatus(str, Enum):
    """Status of a branch."""

    ACTIVE = "active"
    MERGED = "merged"
    ABANDONED = "abandoned"


class Branch(BaseModel):
    """
    An isolated workspace for agent state.

    Branches allow agents to:
    - Work in isolation without affecting other workflows
    - Explore alternative decision paths
    - Merge successful outcomes back to main
    - Abandon failed experiments cleanly

    Similar to Git branches but designed for state evolution, not files.
    """

    id: str = Field(default_factory=generate_id, description="Unique branch identifier")
    name: str = Field(..., description="Human-readable branch name")
    status: BranchStatus = Field(default=BranchStatus.ACTIVE, description="Branch status")

    # Lineage
    parent_branch_id: Optional[str] = Field(
        default=None,
        description="Branch this was forked from (None for root)"
    )
    fork_version: Optional[int] = Field(
        default=None,
        description="Version number at fork point"
    )

    # State tracking
    head_version: int = Field(default=0, description="Current version number")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Branch creation time"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )

    # Metadata
    description: Optional[str] = Field(default=None, description="Branch description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")

    model_config = {"extra": "forbid"}

    def increment_version(self) -> int:
        """
        Increment the head version and return the new version number.

        Each mutation creates a new version, ensuring immutability of past states.
        """
        self.head_version += 1
        self.updated_at = datetime.now(timezone.utc)
        return self.head_version

    def fork(self, name: str, description: Optional[str] = None) -> Branch:
        """
        Create a new branch forked from the current state.

        Args:
            name: Name for the new branch
            description: Optional description

        Returns:
            New branch instance forked from current head
        """
        return Branch(
            name=name,
            parent_branch_id=self.id,
            fork_version=self.head_version,
            description=description,
        )

    def mark_merged(self) -> None:
        """Mark this branch as merged."""
        self.status = BranchStatus.MERGED
        self.updated_at = datetime.now(timezone.utc)

    def mark_abandoned(self) -> None:
        """Mark this branch as abandoned."""
        self.status = BranchStatus.ABANDONED
        self.updated_at = datetime.now(timezone.utc)

    def is_active(self) -> bool:
        """Check if this branch is active."""
        return self.status == BranchStatus.ACTIVE


class Version(BaseModel):
    """
    An immutable snapshot marker at a point in time.

    Versions are created automatically when state changes. They provide:
    - A reference point for time travel queries
    - Audit trail for all mutations
    - Foundation for reproducibility

    Versions are lightweight - they don't copy all data, just mark
    the point in the event stream.
    """

    version_number: int = Field(..., description="Sequential version number")
    branch_id: str = Field(..., description="Branch this version belongs to")

    # Causation
    entity_id: str = Field(..., description="Entity that triggered this version")
    operation: str = Field(..., description="Operation type: create, update, delete")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this version was created"
    )

    # Optional commit-like metadata
    message: Optional[str] = Field(default=None, description="Version message/description")
    author: Optional[str] = Field(default=None, description="Who/what created this version")

    # State hash for integrity verification
    state_hash: Optional[str] = Field(
        default=None,
        description="Hash of the entire branch state at this version"
    )

    model_config = {"frozen": True, "extra": "forbid"}


class Snapshot(BaseModel):
    """
    A reconstructable view of state at a specific version.

    Snapshots allow querying "what was the state at version X?" without
    having to replay all events from the beginning.

    This is essential for:
    - Debugging: "What did the agent see when it made this decision?"
    - Auditing: "What was the system state when this action occurred?"
    - Reproducibility: "Replay this workflow from this exact state"
    """

    branch_id: str = Field(..., description="Branch ID")
    version: int = Field(..., description="Version number")
    timestamp: datetime = Field(..., description="Timestamp of this snapshot")

    # Entity collections at this point in time
    events: list[str] = Field(default_factory=list, description="Event IDs at this version")
    claims: list[str] = Field(default_factory=list, description="Active claim IDs at this version")
    actions: list[str] = Field(default_factory=list, description="Action IDs at this version")

    # State summary
    entity_count: int = Field(default=0, description="Total entities at this version")
    state_hash: Optional[str] = Field(default=None, description="Hash for integrity check")

    model_config = {"frozen": True, "extra": "forbid"}


class VersionDiff(BaseModel):
    """
    Difference between two versions.

    Useful for understanding what changed between states, enabling
    efficient sync and debugging.
    """

    from_version: int = Field(..., description="Starting version")
    to_version: int = Field(..., description="Ending version")
    branch_id: str = Field(..., description="Branch ID")

    # Changes
    added_entities: list[str] = Field(default_factory=list, description="New entity IDs")
    modified_entities: list[str] = Field(default_factory=list, description="Changed entity IDs")
    removed_entities: list[str] = Field(default_factory=list, description="Deleted entity IDs")

    # Summary
    total_changes: int = Field(default=0, description="Total number of changes")

    model_config = {"frozen": True, "extra": "forbid"}

    def model_post_init(self, __context: Any) -> None:
        """Calculate total changes."""
        object.__setattr__(
            self,
            "total_changes",
            len(self.added_entities) + len(self.modified_entities) + len(self.removed_entities)
        )
