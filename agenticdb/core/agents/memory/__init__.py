# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Memory Agents - Manage entity state and retrieval.

This module provides agents for each entity type that combine:
- RAG-like semantic retrieval (recall)
- State summarization
- Hot data caching
- Change tracking
"""
from .action_memory_agent import ActionMemoryAgent
from .claim_memory_agent import ClaimMemoryAgent
from .event_memory_agent import EventMemoryAgent
from .types import (
    ActionPattern,
    ActionRecallResult,
    ClaimConflict,
    ClaimRecallResult,
    EventRecallResult,
    MemoryStats,
    MemorySummary,
    RecalledAction,
    RecalledClaim,
    RecalledEvent,
)

__all__ = [
    # Agents
    "EventMemoryAgent",
    "ClaimMemoryAgent",
    "ActionMemoryAgent",
    # Types
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
