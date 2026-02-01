# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
AgenticDB Agents - AI-powered agents for semantic data processing.

Agent Hierarchy:
- IngestionCoordinator: Orchestrates text → entities pipeline
  - EventExtractorAgent: Extracts immutable events
  - ClaimExtractorAgent: Extracts sourced assertions
  - ActionExtractorAgent: Extracts agent behaviors
  - DependencyInferenceAgent: Infers relationships
- Schema Agents: Manage schema evolution
  - SchemaDetectorAgent: Detects new types/fields
  - SchemaProposalAgent: Generates change proposals
- Query Agents: Answer causal queries
  - CausalReasoningAgent: Explains why(x)
  - ImpactAnalysisAgent: Explains impact(x)
- Memory Agents: Manage entity state
  - EventMemoryAgent: Event table memory
  - ClaimMemoryAgent: Claim table memory
  - ActionMemoryAgent: Action table memory

Import paths:
- agenticdb.core.agents.base: BaseAgent, AgentContext
- agenticdb.core.agents.ingestion: IngestionCoordinator, *ExtractorAgent
- agenticdb.core.agents.schema: SchemaDetectorAgent, SchemaProposalAgent
- agenticdb.core.agents.query: CausalReasoningAgent, ImpactAnalysisAgent
- agenticdb.core.agents.memory: EventMemoryAgent, ClaimMemoryAgent, ActionMemoryAgent
"""

# Base classes
from .base import BaseAgent, AgentContext

# Ingestion agents
from .ingestion import (
    IngestionCoordinator,
    EventExtractorAgent,
    ClaimExtractorAgent,
    ActionExtractorAgent,
    DependencyInferenceAgent,
    # Types
    EdgeType,
    ExtractedEvent,
    ExtractedClaim,
    ExtractedAction,
    InferredEdge,
    ExtractionResult,
    IngestionResult,
)

# Schema agents
from .schema import (
    SchemaDetectorAgent,
    SchemaProposalAgent,
    # Types
    ChangeType,
    FieldDefinition,
    DetectedEventType,
    DetectedClaimSubject,
    DetectedActionType,
    SchemaDetectionResult,
    SchemaChange,
    ImpactAnalysis,
    SchemaProposal,
    SchemaProposalResult,
)

# Query agents
from .query import (
    CausalReasoningAgent,
    ImpactAnalysisAgent,
    # Types
    CausalExplanation,
    KeyFactor,
    ImpactExplanation,
    AffectedCount,
    CriticalImpact,
    Severity,
)

# Memory agents
from .memory import (
    EventMemoryAgent,
    ClaimMemoryAgent,
    ActionMemoryAgent,
    # Types
    RecalledEvent,
    RecalledClaim,
    RecalledAction,
    ClaimConflict,
    ActionPattern,
    EventRecallResult,
    ClaimRecallResult,
    ActionRecallResult,
    MemorySummary,
    MemoryStats,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentContext",
    # Ingestion
    "IngestionCoordinator",
    "EventExtractorAgent",
    "ClaimExtractorAgent",
    "ActionExtractorAgent",
    "DependencyInferenceAgent",
    "EdgeType",
    "ExtractedEvent",
    "ExtractedClaim",
    "ExtractedAction",
    "InferredEdge",
    "ExtractionResult",
    "IngestionResult",
    # Schema
    "SchemaDetectorAgent",
    "SchemaProposalAgent",
    "ChangeType",
    "FieldDefinition",
    "DetectedEventType",
    "DetectedClaimSubject",
    "DetectedActionType",
    "SchemaDetectionResult",
    "SchemaChange",
    "ImpactAnalysis",
    "SchemaProposal",
    "SchemaProposalResult",
    # Query
    "CausalReasoningAgent",
    "ImpactAnalysisAgent",
    "CausalExplanation",
    "KeyFactor",
    "ImpactExplanation",
    "AffectedCount",
    "CriticalImpact",
    "Severity",
    # Memory
    "EventMemoryAgent",
    "ClaimMemoryAgent",
    "ActionMemoryAgent",
    "RecalledEvent",
    "RecalledClaim",
    "RecalledAction",
    "ClaimConflict",
    "ActionPattern",
    "EventRecallResult",
    "ClaimRecallResult",
    "ActionRecallResult",
    "MemorySummary",
    "MemoryStats",
]
