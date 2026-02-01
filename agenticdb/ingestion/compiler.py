"""
Trace Compiler for AgenticDB.

Compiles text input (agent traces, logs, descriptions) into
structured semantic objects with inferred dependencies.

This is the core of the "text-to-state" thesis:
- Input: Unstructured or semi-structured text
- Output: Event/Claim/Action objects with dependency graph

Design Philosophy:
    The compiler is deterministic given the same extractor output.
    Non-determinism (if any) is isolated to the extractor layer,
    making the core system reproducible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, TYPE_CHECKING

from pydantic import BaseModel, Field

from agenticdb.core.models import Event, Claim, Action, Entity
from agenticdb.ingestion.extractor import Extractor, ExtractorResult, RuleBasedExtractor
from agenticdb.ingestion.schema_proposer import SchemaProposer, SchemaProposal

if TYPE_CHECKING:
    from agenticdb.interface.client import BranchHandle


class IngestionMode(str, Enum):
    """Mode of ingestion determining extraction strategy."""

    AGENT_TRACE = "agent_trace"  # Structured agent execution trace
    LOG = "log"  # System logs
    DESCRIPTION = "description"  # Natural language description
    STRUCTURED = "structured"  # Already structured, just validate


class CompilationResult(BaseModel):
    """
    Result of compiling text into semantic objects.

    Contains:
    - Extracted entities (events, claims, actions)
    - Inferred dependency edges
    - Schema proposals (if new types detected)
    - Compilation metadata
    """

    # Extracted entities
    events: list[Event] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)

    # Schema evolution
    schema_proposal: Optional[SchemaProposal] = Field(default=None)

    # Metadata
    source_text: str = Field(default="")
    mode: IngestionMode = Field(default=IngestionMode.DESCRIPTION)
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extractor_used: str = Field(default="unknown")

    # Diagnostics
    warnings: list[str] = Field(default_factory=list)
    extraction_confidence: float = Field(default=1.0)

    model_config = {"extra": "forbid"}

    @property
    def entity_count(self) -> int:
        """Total number of extracted entities."""
        return len(self.events) + len(self.claims) + len(self.actions)

    @property
    def has_schema_changes(self) -> bool:
        """Whether schema evolution is proposed."""
        return self.schema_proposal is not None and not self.schema_proposal.is_empty


class TraceCompiler:
    """
    Compiles text into structured semantic objects.

    The compiler orchestrates:
    1. Extraction: Text → candidate entities (via pluggable Extractor)
    2. Validation: Check against current schema
    3. Dependency inference: Build edges between entities
    4. Schema proposal: Suggest new types if needed

    Usage:
        ```python
        compiler = TraceCompiler(extractor=LLMExtractor(model="gpt-4"))

        result = compiler.compile('''
            User u123 registered with alice@example.com.
            Risk model computed score = 0.3.
            User approved for onboarding.
        ''', mode=IngestionMode.AGENT_TRACE)

        print(result.events)  # [Event(UserRegistered)]
        print(result.claims)  # [Claim(risk_score=0.3)]
        print(result.actions)  # [Action(ApproveUser)]
        ```
    """

    def __init__(
        self,
        extractor: Optional[Extractor] = None,
        schema_proposer: Optional[SchemaProposer] = None,
        strict_schema: bool = False,
    ):
        """
        Initialize the compiler.

        Args:
            extractor: Entity extractor (default: RuleBasedExtractor)
            schema_proposer: Schema evolution proposer
            strict_schema: If True, reject unknown types instead of proposing schema
        """
        self._extractor = extractor or RuleBasedExtractor()
        self._schema_proposer = schema_proposer or SchemaProposer()
        self._strict_schema = strict_schema

    def compile(
        self,
        text: str,
        mode: IngestionMode = IngestionMode.DESCRIPTION,
        context: Optional[dict[str, Any]] = None,
    ) -> CompilationResult:
        """
        Compile text into semantic objects.

        Args:
            text: Input text (trace, log, description)
            mode: Ingestion mode affecting extraction strategy
            context: Optional context (existing entities, schema hints)

        Returns:
            CompilationResult with extracted entities and metadata
        """
        # Step 1: Extract candidate entities
        extraction = self._extractor.extract(text, mode=mode.value, context=context)

        # Step 2: Convert to semantic objects
        events = self._build_events(extraction)
        claims = self._build_claims(extraction)
        actions = self._build_actions(extraction, events, claims)

        # Step 3: Check schema and propose changes if needed
        schema_proposal = None
        if extraction.unknown_types and not self._strict_schema:
            schema_proposal = self._schema_proposer.propose(
                unknown_types=extraction.unknown_types,
                context=extraction.raw_extractions,
            )

        # Step 4: Build result
        warnings = []
        if self._strict_schema and extraction.unknown_types:
            warnings.append(f"Unknown types rejected: {extraction.unknown_types}")

        return CompilationResult(
            events=events,
            claims=claims,
            actions=actions,
            schema_proposal=schema_proposal,
            source_text=text,
            mode=mode,
            extractor_used=self._extractor.name,
            warnings=warnings,
            extraction_confidence=extraction.confidence,
        )

    def _build_events(self, extraction: ExtractorResult) -> list[Event]:
        """Convert extracted event data to Event objects."""
        events = []
        for event_data in extraction.events:
            event = Event(
                event_type=event_data.get("type", "Unknown"),
                data=event_data.get("data", {}),
                source_agent=event_data.get("source_agent"),
                source_system=event_data.get("source_system"),
                metadata={"extracted": True, "raw": event_data},
            )
            events.append(event)
        return events

    def _build_claims(self, extraction: ExtractorResult) -> list[Claim]:
        """Convert extracted claim data to Claim objects."""
        claims = []
        for claim_data in extraction.claims:
            claim = Claim(
                subject=claim_data.get("subject", "unknown"),
                value=claim_data.get("value"),
                source=claim_data.get("source", "extraction"),
                confidence=claim_data.get("confidence", 0.8),
                metadata={"extracted": True, "raw": claim_data},
            )
            claims.append(claim)
        return claims

    def _build_actions(
        self,
        extraction: ExtractorResult,
        events: list[Event],
        claims: list[Claim],
    ) -> list[Action]:
        """Convert extracted action data to Action objects with dependencies."""
        actions = []
        for action_data in extraction.actions:
            # Infer dependencies from extraction
            depends_on = []

            # Add dependencies based on extracted references
            for ref in action_data.get("references", []):
                # Match to events
                for event in events:
                    if self._matches_reference(event, ref):
                        depends_on.append(event.id)
                # Match to claims
                for claim in claims:
                    if self._matches_reference(claim, ref):
                        depends_on.append(claim.id)

            action = Action(
                action_type=action_data.get("type", "Unknown"),
                agent_id=action_data.get("agent_id", "extracted-agent"),
                inputs=action_data.get("inputs", {}),
                depends_on=depends_on,
                reasoning=action_data.get("reasoning"),
                metadata={"extracted": True, "raw": action_data},
            )
            actions.append(action)
        return actions

    def _matches_reference(self, entity: Entity, reference: dict[str, Any]) -> bool:
        """Check if an entity matches a reference from extraction."""
        # Simple matching by type and key fields
        if hasattr(entity, "event_type"):
            return entity.event_type == reference.get("type")
        if hasattr(entity, "subject"):
            return entity.subject == reference.get("subject")
        return False


class IngestResult(BaseModel):
    """
    Result of ingesting text into a branch.

    Wraps CompilationResult with branch-specific information
    (recorded entity IDs, version numbers).
    """

    compilation: CompilationResult = Field(...)

    # Recorded entity IDs (after storage)
    event_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    action_ids: list[str] = Field(default_factory=list)

    # Version info
    start_version: int = Field(default=0)
    end_version: int = Field(default=0)

    model_config = {"extra": "forbid"}

    @property
    def events(self) -> list[Event]:
        """Get recorded events."""
        return self.compilation.events

    @property
    def claims(self) -> list[Claim]:
        """Get recorded claims."""
        return self.compilation.claims

    @property
    def actions(self) -> list[Action]:
        """Get recorded actions."""
        return self.compilation.actions

    @property
    def schema_proposal(self) -> Optional[SchemaProposal]:
        """Get schema proposal if any."""
        return self.compilation.schema_proposal
