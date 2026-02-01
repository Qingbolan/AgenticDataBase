# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Type definitions for Ingestion Agents.

These types represent the intermediate results from extraction
before they are converted to core Entity models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EdgeType(str, Enum):
    """Types of dependency edges between entities."""
    DEPENDS_ON = "DEPENDS_ON"
    PRODUCES = "PRODUCES"
    DERIVED_FROM = "DERIVED_FROM"
    INVALIDATES = "INVALIDATES"


@dataclass
class ExtractedEvent:
    """An event extracted from text, before conversion to Event model."""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source_agent: Optional[str] = None
    source_system: Optional[str] = None
    correlation_id: Optional[str] = None
    # Reference ID for dependency linking
    ref_id: Optional[str] = None


@dataclass
class ExtractedClaim:
    """A claim extracted from text, before conversion to Claim model."""
    subject: str
    value: Any
    source: str
    predicate: str = "is"
    source_version: Optional[str] = None
    confidence: float = 1.0
    derived_from_refs: List[str] = field(default_factory=list)
    # Reference ID for dependency linking
    ref_id: Optional[str] = None


@dataclass
class ExtractedAction:
    """An action extracted from text, before conversion to Action model."""
    action_type: str
    agent_id: str
    agent_type: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    depends_on_refs: List[str] = field(default_factory=list)
    produces_refs: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    # Reference ID for dependency linking
    ref_id: Optional[str] = None


@dataclass
class InferredEdge:
    """A dependency edge inferred between entities."""
    from_ref: str
    to_ref: str
    edge_type: EdgeType
    reasoning: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result from a single extractor agent."""
    events: List[ExtractedEvent] = field(default_factory=list)
    claims: List[ExtractedClaim] = field(default_factory=list)
    actions: List[ExtractedAction] = field(default_factory=list)
    edges: List[InferredEdge] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class IngestionResult:
    """Complete result from the ingestion pipeline."""
    events: List[ExtractedEvent] = field(default_factory=list)
    claims: List[ExtractedClaim] = field(default_factory=list)
    actions: List[ExtractedAction] = field(default_factory=list)
    edges: List[InferredEdge] = field(default_factory=list)

    # Mapping from ref_id to actual entity ID after storage
    ref_to_id: Dict[str, str] = field(default_factory=dict)

    # Overall metrics
    event_confidence: float = 0.0
    claim_confidence: float = 0.0
    action_confidence: float = 0.0
    edge_confidence: float = 0.0

    @property
    def total_entities(self) -> int:
        """Total number of extracted entities."""
        return len(self.events) + len(self.claims) + len(self.actions)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all extractions."""
        confidences = [
            self.event_confidence,
            self.claim_confidence,
            self.action_confidence,
            self.edge_confidence,
        ]
        non_zero = [c for c in confidences if c > 0]
        return sum(non_zero) / len(non_zero) if non_zero else 0.0
