"""
Schema Evolution for AgenticDB.

This module handles schema proposals and evolution:
- Detect unknown entity types from ingestion
- Propose schema changes (new types, fields, relations)
- Apply changes with version tracking

Design Philosophy:
    Schemas are not fixed at design time. They evolve with the data.
    When agents produce new concepts, the system proposes schema
    changes that can be reviewed and applied.

    This is like database migrations, but for semantic types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from agenticdb.core.models import generate_id


class FieldType(str, Enum):
    """Supported field types in schema."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ENUM = "enum"
    JSON = "json"
    REFERENCE = "reference"  # Reference to another entity


class SchemaField(BaseModel):
    """A field in an entity schema."""

    name: str = Field(..., description="Field name")
    field_type: FieldType = Field(..., description="Field type")
    required: bool = Field(default=False, description="Whether field is required")
    default: Optional[Any] = Field(default=None, description="Default value")

    # For enum types
    enum_values: Optional[list[str]] = Field(default=None, description="Allowed enum values")

    # For reference types
    reference_type: Optional[str] = Field(default=None, description="Referenced entity type")

    model_config = {"extra": "forbid"}


class EntitySchema(BaseModel):
    """Schema for an entity type."""

    name: str = Field(..., description="Entity type name")
    description: Optional[str] = Field(default=None, description="Human description")
    fields: list[SchemaField] = Field(default_factory=list, description="Entity fields")

    # Semantic hints
    category: str = Field(default="claim", description="event, claim, or action")
    derived_from: list[str] = Field(
        default_factory=list,
        description="What this type is typically derived from"
    )

    model_config = {"extra": "forbid"}


class SchemaChangeType(str, Enum):
    """Types of schema changes."""

    ADD_TYPE = "add_type"
    REMOVE_TYPE = "remove_type"
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    MODIFY_FIELD = "modify_field"
    ADD_RELATION = "add_relation"


class SchemaChange(BaseModel):
    """A single schema change."""

    change_type: SchemaChangeType = Field(..., description="Type of change")
    entity_type: str = Field(..., description="Affected entity type")

    # For field changes
    field: Optional[SchemaField] = Field(default=None, description="Field being changed")
    old_field: Optional[SchemaField] = Field(default=None, description="Previous field state")

    # For type changes
    schema: Optional[EntitySchema] = Field(default=None, description="New type schema")

    # Metadata
    reason: Optional[str] = Field(default=None, description="Why this change is proposed")

    model_config = {"extra": "forbid"}


class SchemaProposal(BaseModel):
    """
    A proposed set of schema changes.

    Proposals are generated when:
    - New entity types are detected during ingestion
    - Fields are used that don't exist in current schema
    - Relations are implied that aren't defined

    Proposals can be:
    - Reviewed by humans
    - Auto-applied based on policy
    - Rejected if inappropriate
    """

    id: str = Field(default_factory=generate_id, description="Proposal ID")
    changes: list[SchemaChange] = Field(default_factory=list, description="Proposed changes")

    # Context
    triggered_by: str = Field(default="ingestion", description="What triggered this proposal")
    source_text: Optional[str] = Field(default=None, description="Text that triggered proposal")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When proposal was created"
    )
    confidence: float = Field(default=0.8, description="Confidence in proposal")

    model_config = {"extra": "forbid"}

    @property
    def is_empty(self) -> bool:
        """Check if proposal has no changes."""
        return len(self.changes) == 0

    def summary(self) -> str:
        """Get a human-readable summary of changes."""
        if self.is_empty:
            return "No changes proposed"

        lines = ["Schema Proposal:"]
        for change in self.changes:
            if change.change_type == SchemaChangeType.ADD_TYPE:
                lines.append(f"  + Add type: {change.entity_type}")
            elif change.change_type == SchemaChangeType.ADD_FIELD:
                lines.append(f"  + Add field: {change.entity_type}.{change.field.name}")
            elif change.change_type == SchemaChangeType.REMOVE_TYPE:
                lines.append(f"  - Remove type: {change.entity_type}")
            elif change.change_type == SchemaChangeType.REMOVE_FIELD:
                lines.append(f"  - Remove field: {change.entity_type}.{change.field.name}")
        return "\n".join(lines)


class SchemaCommit(BaseModel):
    """
    A committed schema version.

    Like a git commit, but for schemas.
    """

    id: str = Field(default_factory=generate_id, description="Commit ID")
    version: int = Field(..., description="Schema version number")
    parent_id: Optional[str] = Field(default=None, description="Parent commit ID")

    # Changes in this commit
    proposal_id: str = Field(..., description="Proposal that was applied")
    changes: list[SchemaChange] = Field(default_factory=list, description="Applied changes")

    # Metadata
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When committed"
    )
    committed_by: Optional[str] = Field(default=None, description="Who committed")
    message: Optional[str] = Field(default=None, description="Commit message")

    model_config = {"frozen": True, "extra": "forbid"}


class SchemaProposer:
    """
    Proposes schema changes based on extracted data.

    The proposer analyzes:
    - Unknown entity types
    - New fields in known types
    - Implied relations

    And generates SchemaProposals that can be reviewed and applied.
    """

    def __init__(
        self,
        existing_types: Optional[list[str]] = None,
        strict: bool = False,
    ):
        """
        Initialize the proposer.

        Args:
            existing_types: Known entity types
            strict: If True, don't propose new types
        """
        self._existing_types = set(existing_types or [
            "UserRegistered", "PaymentReceived", "OrderPlaced",  # Common events
            "risk_score", "trust_level", "fraud_score",  # Common claims
            "ApproveUser", "RejectUser", "FlagForReview",  # Common actions
        ])
        self._strict = strict

    def propose(
        self,
        unknown_types: list[str],
        context: Optional[list[dict[str, Any]]] = None,
    ) -> SchemaProposal:
        """
        Generate a schema proposal for unknown types.

        Args:
            unknown_types: Types not in current schema
            context: Extraction context for inferring fields

        Returns:
            SchemaProposal with suggested changes
        """
        changes = []

        for type_name in unknown_types:
            if type_name in self._existing_types:
                continue

            if self._strict:
                continue

            # Infer schema from context
            schema = self._infer_schema(type_name, context)
            changes.append(SchemaChange(
                change_type=SchemaChangeType.ADD_TYPE,
                entity_type=type_name,
                schema=schema,
                reason=f"New type detected during ingestion: {type_name}",
            ))

        return SchemaProposal(
            changes=changes,
            triggered_by="ingestion",
            confidence=0.7 if changes else 1.0,
        )

    def _infer_schema(
        self,
        type_name: str,
        context: Optional[list[dict[str, Any]]] = None,
    ) -> EntitySchema:
        """Infer schema for a new type."""
        # Determine category from naming convention
        category = "claim"
        if type_name.endswith("ed") or type_name.endswith("Event"):
            category = "event"
        elif type_name.startswith("Approve") or type_name.startswith("Reject"):
            category = "action"

        # Infer fields from context if available
        fields = []
        if context:
            for extraction in context:
                for key, value in extraction.get("data", {}).items():
                    field_type = self._infer_field_type(value)
                    fields.append(SchemaField(
                        name=key,
                        field_type=field_type,
                        required=False,
                    ))

        return EntitySchema(
            name=type_name,
            description=f"Auto-generated schema for {type_name}",
            fields=fields,
            category=category,
        )

    def _infer_field_type(self, value: Any) -> FieldType:
        """Infer field type from a value."""
        if isinstance(value, bool):
            return FieldType.BOOLEAN
        if isinstance(value, int):
            return FieldType.INTEGER
        if isinstance(value, float):
            return FieldType.FLOAT
        if isinstance(value, str):
            # Check if it looks like a datetime
            if "T" in value and ":" in value:
                return FieldType.DATETIME
            return FieldType.STRING
        if isinstance(value, dict):
            return FieldType.JSON
        return FieldType.STRING


class SchemaRegistry:
    """
    Registry for entity schemas with version history.

    Provides:
    - Current schema lookup
    - Schema evolution (apply proposals)
    - Version history and diff
    """

    def __init__(self):
        """Initialize empty registry."""
        self._schemas: dict[str, EntitySchema] = {}
        self._commits: list[SchemaCommit] = []
        self._version: int = 0

    @property
    def version(self) -> int:
        """Current schema version."""
        return self._version

    def get(self, type_name: str) -> Optional[EntitySchema]:
        """Get schema for a type."""
        return self._schemas.get(type_name)

    def list_types(self) -> list[str]:
        """List all known types."""
        return list(self._schemas.keys())

    def has_type(self, type_name: str) -> bool:
        """Check if a type exists."""
        return type_name in self._schemas

    def diff(self, proposal: SchemaProposal) -> str:
        """
        Get a diff-style view of what would change.

        Args:
            proposal: Proposal to diff

        Returns:
            Human-readable diff
        """
        lines = [f"Schema Diff (v{self._version} → v{self._version + 1}):", ""]

        for change in proposal.changes:
            if change.change_type == SchemaChangeType.ADD_TYPE:
                lines.append(f"+ type {change.entity_type}")
                if change.schema:
                    for field in change.schema.fields:
                        lines.append(f"    + field {field.name}: {field.field_type.value}")
            elif change.change_type == SchemaChangeType.REMOVE_TYPE:
                lines.append(f"- type {change.entity_type}")
            elif change.change_type == SchemaChangeType.ADD_FIELD:
                lines.append(f"  + field {change.entity_type}.{change.field.name}")
            elif change.change_type == SchemaChangeType.REMOVE_FIELD:
                lines.append(f"  - field {change.entity_type}.{change.field.name}")

        return "\n".join(lines)

    def apply(
        self,
        proposal: SchemaProposal,
        message: Optional[str] = None,
        auto_migrate: bool = False,
    ) -> SchemaCommit:
        """
        Apply a schema proposal.

        Args:
            proposal: Proposal to apply
            message: Commit message
            auto_migrate: Auto-migrate existing data

        Returns:
            SchemaCommit representing this change
        """
        # Apply changes
        for change in proposal.changes:
            if change.change_type == SchemaChangeType.ADD_TYPE and change.schema:
                self._schemas[change.entity_type] = change.schema
            elif change.change_type == SchemaChangeType.REMOVE_TYPE:
                self._schemas.pop(change.entity_type, None)
            # TODO: Handle field-level changes

        # Create commit
        self._version += 1
        parent_id = self._commits[-1].id if self._commits else None

        commit = SchemaCommit(
            version=self._version,
            parent_id=parent_id,
            proposal_id=proposal.id,
            changes=proposal.changes,
            message=message or f"Applied proposal {proposal.id}",
        )
        self._commits.append(commit)

        return commit

    def history(self) -> list[SchemaCommit]:
        """Get schema version history."""
        return list(self._commits)

    def at_version(self, version: int) -> dict[str, EntitySchema]:
        """
        Get schema as it was at a specific version.

        NOTE: This is a simplified implementation.
        Full implementation would replay changes.
        """
        if version == self._version:
            return dict(self._schemas)

        # TODO: Implement schema reconstruction from history
        raise NotImplementedError("Schema time travel not yet implemented")
