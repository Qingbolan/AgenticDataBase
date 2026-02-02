# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Pattern learning agents for AgenticDB.

This module provides agents for extracting and matching query patterns
to enable workload memoization and reduce LLM calls.

Agents:
    - PatternExtractorAgent: Extract patterns from successful queries
    - PatternMatcherAgent: Match queries against cached patterns
"""

from .types import (
    PatternMatch,
    PatternScore,
    PatternExtractionResult,
    PatternMatchResult,
    PatternCacheStats,
)
from .extractor_agent import PatternExtractorAgent
from .matcher_agent import PatternMatcherAgent

__all__ = [
    # Types
    "PatternMatch",
    "PatternScore",
    "PatternExtractionResult",
    "PatternMatchResult",
    "PatternCacheStats",
    # Agents
    "PatternExtractorAgent",
    "PatternMatcherAgent",
]
