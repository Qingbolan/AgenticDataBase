"""
AgenticDB - Agent-native data runtime.

The database that turns text into evolvable state, with
dependency-aware history, replay, and invalidation.

Two interfaces:
- Semantic Ingestion: Text → structured state (product interface)
- Structured API: Event/Claim/Action objects (engineering interface)

Agent Architecture:
- IngestionCoordinator: Text → Entities pipeline
- Schema Agents: Schema evolution management
- Query Agents: Causal reasoning (why/impact)
- Memory Agents: Entity state management
"""
from agenticdb.core.models import Action, Claim, Entity, Event
from agenticdb.core.version import Branch, Snapshot, Version
from agenticdb.core.dependency import DependencyEdge, DependencyGraph
from agenticdb.storage.engine import StorageEngine, InMemoryStorage
from agenticdb.query.engine import QueryEngine
from agenticdb.query.operators import WhyQuery, ImpactQuery
from agenticdb.runtime.cache import DependencyAwareCache
from agenticdb.interface.client import AgenticDB

# Semantic Ingestion Layer
from agenticdb.ingestion.compiler import TraceCompiler, CompilationResult, IngestResult
from agenticdb.ingestion.extractor import Extractor, RuleBasedExtractor, LLMExtractor
from agenticdb.ingestion.schema_proposer import (
    SchemaProposal,
    SchemaCommit,
    SchemaRegistry,
    EntitySchema,
)

# Agent Layer
from agenticdb.core.agents import (
    # Base
    BaseAgent,
    AgentContext,
    # Ingestion
    IngestionCoordinator,
    EventExtractorAgent,
    ClaimExtractorAgent,
    ActionExtractorAgent,
    DependencyInferenceAgent,
    IngestionResult,
    # Schema
    SchemaDetectorAgent,
    SchemaProposalAgent,
    SchemaDetectionResult,
    SchemaProposalResult,
    # Query
    CausalReasoningAgent,
    ImpactAnalysisAgent,
    CausalExplanation,
    ImpactExplanation,
    # Memory
    EventMemoryAgent,
    ClaimMemoryAgent,
    ActionMemoryAgent,
)

__version__ = "0.1.0"

__all__ = [
    # Core models
    "Entity",
    "Event",
    "Claim",
    "Action",
    # Versioning
    "Branch",
    "Version",
    "Snapshot",
    # Dependency
    "DependencyGraph",
    "DependencyEdge",
    # Storage
    "StorageEngine",
    "InMemoryStorage",
    # Query
    "QueryEngine",
    "WhyQuery",
    "ImpactQuery",
    # Runtime
    "DependencyAwareCache",
    # Client
    "AgenticDB",
    # Semantic Ingestion
    "TraceCompiler",
    "CompilationResult",
    "IngestResult",
    "Extractor",
    "RuleBasedExtractor",
    "LLMExtractor",
    # Schema Evolution
    "SchemaProposal",
    "SchemaCommit",
    "SchemaRegistry",
    "EntitySchema",
    # Agents - Base
    "BaseAgent",
    "AgentContext",
    # Agents - Ingestion
    "IngestionCoordinator",
    "EventExtractorAgent",
    "ClaimExtractorAgent",
    "ActionExtractorAgent",
    "DependencyInferenceAgent",
    "IngestionResult",
    # Agents - Schema
    "SchemaDetectorAgent",
    "SchemaProposalAgent",
    "SchemaDetectionResult",
    "SchemaProposalResult",
    # Agents - Query
    "CausalReasoningAgent",
    "ImpactAnalysisAgent",
    "CausalExplanation",
    "ImpactExplanation",
    # Agents - Memory
    "EventMemoryAgent",
    "ClaimMemoryAgent",
    "ActionMemoryAgent",
]
