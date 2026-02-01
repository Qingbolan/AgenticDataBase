# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Schema Agents - Manage schema evolution.

This module provides agents for detecting new schema patterns and
generating evolution proposals.
"""
from .schema_detector_agent import SchemaDetectorAgent
from .schema_proposal_agent import SchemaProposalAgent
from .types import (
    ChangeType,
    DetectedActionType,
    DetectedClaimSubject,
    DetectedEventType,
    FieldDefinition,
    ImpactAnalysis,
    SchemaChange,
    SchemaDetectionResult,
    SchemaProposal,
    SchemaProposalResult,
)

__all__ = [
    # Agents
    "SchemaDetectorAgent",
    "SchemaProposalAgent",
    # Types
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
]
