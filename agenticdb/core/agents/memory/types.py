# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Type definitions for Memory Agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RecalledEvent:
    """An event recalled from memory."""
    event_type: str
    data: Dict[str, Any]
    relevance_score: float
    summary: str
    created_at: Optional[datetime] = None
    entity_id: Optional[str] = None


@dataclass
class RecalledClaim:
    """A claim recalled from memory."""
    subject: str
    value: Any
    source: str
    confidence: float
    relevance_score: float
    entity_id: Optional[str] = None
    valid_until: Optional[datetime] = None


@dataclass
class RecalledAction:
    """An action recalled from memory."""
    action_type: str
    agent_id: str
    status: str
    relevance_score: float
    summary: str
    entity_id: Optional[str] = None


@dataclass
class ClaimConflict:
    """A conflict between two claims."""
    claim1_id: str
    claim2_id: str
    nature: str


@dataclass
class ActionPattern:
    """A pattern identified in agent actions."""
    pattern: str
    frequency: str


@dataclass
class EventRecallResult:
    """Result from event memory recall."""
    events: List[RecalledEvent] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class ClaimRecallResult:
    """Result from claim memory recall."""
    claims: List[RecalledClaim] = field(default_factory=list)
    conflicts: List[ClaimConflict] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class ActionRecallResult:
    """Result from action memory recall."""
    actions: List[RecalledAction] = field(default_factory=list)
    patterns: List[ActionPattern] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class MemorySummary:
    """Summary of memory contents."""
    summary: str
    key_points: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    total_entities: int = 0


@dataclass
class MemoryStats:
    """Statistics about memory contents."""
    total_events: int = 0
    total_claims: int = 0
    total_actions: int = 0
    active_claims: int = 0
    event_types: List[str] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)
    claim_subjects: List[str] = field(default_factory=list)
