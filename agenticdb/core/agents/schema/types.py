# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Type definitions for Schema Agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChangeType(str, Enum):
    """Types of schema changes."""
    ADD_EVENT_TYPE = "ADD_EVENT_TYPE"
    ADD_CLAIM_PATTERN = "ADD_CLAIM_PATTERN"
    ADD_ACTION_TYPE = "ADD_ACTION_TYPE"
    MODIFY_TYPE = "MODIFY_TYPE"


@dataclass
class FieldDefinition:
    """Definition of a field in a schema."""
    name: str
    type: str  # string, number, boolean, object, array
    required: bool = False
    description: Optional[str] = None


@dataclass
class DetectedEventType:
    """A newly detected event type."""
    type_name: str
    fields: List[FieldDefinition] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class DetectedClaimSubject:
    """A newly detected claim subject pattern."""
    subject_pattern: str
    value_type: str
    common_sources: List[str] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class DetectedActionType:
    """A newly detected action type."""
    type_name: str
    input_fields: List[FieldDefinition] = field(default_factory=list)
    output_fields: List[FieldDefinition] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class SchemaDetectionResult:
    """Result from schema detection."""
    new_event_types: List[DetectedEventType] = field(default_factory=list)
    new_claim_subjects: List[DetectedClaimSubject] = field(default_factory=list)
    new_action_types: List[DetectedActionType] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class SchemaChange:
    """A single schema change."""
    change_type: ChangeType
    target: str
    definition: Dict[str, Any] = field(default_factory=dict)
    breaking: bool = False
    migration: Optional[str] = None


@dataclass
class ImpactAnalysis:
    """Impact analysis for a schema proposal."""
    affected_entities: int = 0
    backwards_compatible: bool = True
    requires_migration: bool = False


@dataclass
class SchemaProposal:
    """A schema evolution proposal."""
    id: str
    title: str
    description: str
    changes: List[SchemaChange] = field(default_factory=list)
    impact_analysis: ImpactAnalysis = field(default_factory=ImpactAnalysis)


@dataclass
class SchemaProposalResult:
    """Result from schema proposal generation."""
    proposal: Optional[SchemaProposal] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
