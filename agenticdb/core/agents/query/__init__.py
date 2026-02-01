# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Query Agents - Answer causal queries with natural language.

This module provides agents for explaining why things happened
and what would be affected by changes.
"""
from .causal_reasoning_agent import CausalReasoningAgent
from .impact_analysis_agent import ImpactAnalysisAgent
from .types import (
    AffectedCount,
    CausalExplanation,
    CriticalImpact,
    ImpactExplanation,
    KeyFactor,
    Severity,
)

__all__ = [
    # Agents
    "CausalReasoningAgent",
    "ImpactAnalysisAgent",
    # Types
    "CausalExplanation",
    "KeyFactor",
    "ImpactExplanation",
    "AffectedCount",
    "CriticalImpact",
    "Severity",
]
