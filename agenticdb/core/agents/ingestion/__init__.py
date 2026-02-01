# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Ingestion Agents - Transform text into semantic entities.

This module provides agents for extracting Events, Claims, and Actions
from natural language text, as well as inferring dependencies between them.
"""
from .action_extractor_agent import ActionExtractorAgent
from .claim_extractor_agent import ClaimExtractorAgent
from .coordinator import IngestionCoordinator
from .dependency_inference_agent import DependencyInferenceAgent
from .event_extractor_agent import EventExtractorAgent
from .types import (
    EdgeType,
    ExtractedAction,
    ExtractedClaim,
    ExtractedEvent,
    ExtractionResult,
    InferredEdge,
    IngestionResult,
)

__all__ = [
    # Coordinator
    "IngestionCoordinator",
    # Agents
    "EventExtractorAgent",
    "ClaimExtractorAgent",
    "ActionExtractorAgent",
    "DependencyInferenceAgent",
    # Types
    "EdgeType",
    "ExtractedEvent",
    "ExtractedClaim",
    "ExtractedAction",
    "InferredEdge",
    "ExtractionResult",
    "IngestionResult",
]
