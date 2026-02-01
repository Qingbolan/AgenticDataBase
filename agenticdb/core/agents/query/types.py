# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Type definitions for Query Agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Severity levels for impact analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class KeyFactor:
    """A key factor in a causal chain."""
    factor: str
    role: str


@dataclass
class CausalExplanation:
    """Natural language explanation of a causal chain."""
    summary: str
    explanation: str
    key_factors: List[KeyFactor] = field(default_factory=list)
    causal_depth: int = 0
    confidence: float = 0.0


@dataclass
class CriticalImpact:
    """A critical impact from a change."""
    entity_type: str  # "claim", "action", "event"
    entity_ref: str
    impact: str
    severity: Severity = Severity.MEDIUM


@dataclass
class AffectedCount:
    """Count of affected entities by type."""
    events: int = 0
    claims: int = 0
    actions: int = 0

    @property
    def total(self) -> int:
        return self.events + self.claims + self.actions


@dataclass
class ImpactExplanation:
    """Natural language explanation of change impact."""
    summary: str
    affected_count: AffectedCount = field(default_factory=AffectedCount)
    critical_impacts: List[CriticalImpact] = field(default_factory=list)
    cascade_effects: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    confidence: float = 0.0
