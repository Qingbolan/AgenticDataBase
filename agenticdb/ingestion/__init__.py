"""
Semantic Ingestion Layer for AgenticDB.

This module compiles text (agent traces, logs, descriptions) into
structured semantic objects (Event, Claim, Action) with automatic
dependency inference and schema evolution.

This is the "product interface" — what agents and systems talk to.
The Structured API (Event/Claim/Action) is the "engineering interface"
that this layer compiles down to.
"""

from agenticdb.ingestion.compiler import TraceCompiler, CompilationResult
from agenticdb.ingestion.extractor import (
    Extractor,
    ExtractorResult,
    LLMExtractor,
    RuleBasedExtractor,
)
from agenticdb.ingestion.schema_proposer import SchemaProposer, SchemaProposal

__all__ = [
    "TraceCompiler",
    "CompilationResult",
    "Extractor",
    "ExtractorResult",
    "LLMExtractor",
    "RuleBasedExtractor",
    "SchemaProposer",
    "SchemaProposal",
]
