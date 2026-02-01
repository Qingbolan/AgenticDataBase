# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Claim Memory Agent - Manages claim entity memory.

This agent provides RAG-like semantic retrieval and summarization
for claim entities stored in AgenticDB.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agenticdb.core.models import Claim
from ..base.base_agent import AgentContext, BaseAgent
from .types import (
    ClaimConflict,
    ClaimRecallResult,
    MemoryStats,
    MemorySummary,
    RecalledClaim,
)


class ClaimMemoryAgent(BaseAgent[ClaimRecallResult]):
    """
    Agent for managing claim memory.

    Provides semantic retrieval (recall), conflict detection,
    and summarization for Claim entities.
    """

    name = "claim_memory"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Claim Memory Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "memory"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.1)

        # In-memory cache
        self._claim_cache: Dict[str, Claim] = {}
        self._claim_by_subject: Dict[str, List[str]] = {}

    def run(
        self,
        ctx: AgentContext,
        query: str,
        claims: List[Claim],
    ) -> ClaimRecallResult:
        """
        Recall claims relevant to a query.

        Args:
            ctx: Agent context
            query: Natural language query
            claims: Claims to search through

        Returns:
            ClaimRecallResult with relevant claims
        """
        return self.recall(query, claims)

    def recall(
        self,
        query: str,
        claims: List[Claim],
        limit: int = 10,
        active_only: bool = True,
    ) -> ClaimRecallResult:
        """
        Semantic retrieval of claims matching a query.

        Args:
            query: Natural language query
            claims: Claims to search
            limit: Maximum claims to return
            active_only: Only return currently valid claims

        Returns:
            ClaimRecallResult with relevant claims
        """
        # Filter to active claims if requested
        if active_only:
            now = datetime.now(timezone.utc)
            claims = [c for c in claims if c.is_valid_at(now)]

        if not claims:
            return ClaimRecallResult(
                claims=[],
                reasoning="No claims to search",
            )

        # Convert claims to JSON for LLM
        claims_json = json.dumps([
            {
                "id": c.id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "source": c.source,
                "confidence": c.confidence,
            }
            for c in claims[:100]
        ], indent=2, default=str)

        system_prompt = self._load_prompt("claim_memory_system.md")

        user_prompt = f"""Find claims relevant to this query:

Query: {query}

Available Claims:
{claims_json}

Return a JSON object with "relevant_claims" (list with subject, value, source, confidence, relevance_score), "conflicts", and "reasoning".
"""

        response = self.query_json(system_prompt, user_prompt)

        return self._parse_recall_result(response, claims)

    def summarize(
        self,
        claims: List[Claim],
        focus: Optional[str] = None,
    ) -> MemorySummary:
        """
        Summarize a list of claims.

        Args:
            claims: Claims to summarize
            focus: Optional focus area

        Returns:
            MemorySummary
        """
        if not claims:
            return MemorySummary(
                summary="No claims to summarize",
                total_entities=0,
            )

        claims_json = json.dumps([
            {
                "subject": c.subject,
                "value": c.value,
                "source": c.source,
                "confidence": c.confidence,
            }
            for c in claims[:50]
        ], indent=2, default=str)

        system_prompt = self._load_prompt("claim_memory_system.md")

        focus_text = f"\nFocus on: {focus}" if focus else ""
        user_prompt = f"""Summarize these claims:{focus_text}

{claims_json}

Return a JSON object with "summary", "by_subject", "source_reliability", and "uncertainties".
"""

        response = self.query_json(system_prompt, user_prompt)

        # Extract key points from by_subject
        by_subject = response.get("by_subject", {})
        key_points = [f"{k}: {v}" for k, v in by_subject.items()]

        return MemorySummary(
            summary=response.get("summary", ""),
            key_points=key_points,
            patterns=response.get("uncertainties", []),
            total_entities=len(claims),
        )

    def get_latest(
        self,
        subject: str,
        claims: List[Claim],
    ) -> Optional[Claim]:
        """
        Get the most recent claim for a subject.

        Args:
            subject: Claim subject to find
            claims: Claims to search

        Returns:
            Most recent claim for the subject, or None
        """
        matching = [c for c in claims if c.subject == subject and c.is_active()]
        if not matching:
            return None

        return max(matching, key=lambda c: c.created_at or datetime.min)

    def find_conflicts(self, claims: List[Claim]) -> List[ClaimConflict]:
        """
        Find conflicting claims.

        Args:
            claims: Claims to check

        Returns:
            List of conflicts found
        """
        conflicts = []
        active_claims = [c for c in claims if c.is_active()]

        for i, claim1 in enumerate(active_claims):
            for claim2 in active_claims[i + 1:]:
                if claim1.conflicts_with(claim2):
                    conflicts.append(ClaimConflict(
                        claim1_id=claim1.id,
                        claim2_id=claim2.id,
                        nature=f"Conflicting values for {claim1.subject}: {claim1.value} vs {claim2.value}",
                    ))

        return conflicts

    def track_changes(
        self,
        since: datetime,
        claims: List[Claim],
    ) -> List[Claim]:
        """
        Get claims created since a given time.

        Args:
            since: Timestamp to filter from
            claims: Claims to filter

        Returns:
            Claims created after the timestamp
        """
        return [
            c for c in claims
            if c.created_at and c.created_at > since
        ]

    def get_stats(self, claims: List[Claim]) -> MemoryStats:
        """
        Get statistics about claims.

        Args:
            claims: Claims to analyze

        Returns:
            MemoryStats
        """
        subjects = list(set(c.subject for c in claims))
        active = len([c for c in claims if c.is_active()])

        return MemoryStats(
            total_claims=len(claims),
            active_claims=active,
            claim_subjects=subjects,
        )

    def _parse_recall_result(
        self,
        response: dict,
        claims: List[Claim],
    ) -> ClaimRecallResult:
        """Parse recall result from LLM response."""
        recalled = []
        claims_by_subject = {c.subject: c for c in claims}

        for raw in self._extract_list(response, "relevant_claims"):
            if not isinstance(raw, dict):
                continue

            subject = raw.get("subject", "")
            if subject in claims_by_subject:
                claim = claims_by_subject[subject]
                recalled.append(RecalledClaim(
                    subject=subject,
                    value=claim.value,
                    source=claim.source,
                    confidence=claim.confidence,
                    relevance_score=raw.get("relevance_score", 0.0),
                    entity_id=claim.id,
                    valid_until=claim.valid_until,
                ))

        # Parse conflicts
        conflicts = []
        for raw in self._extract_list(response, "conflicts"):
            if isinstance(raw, dict):
                conflicts.append(ClaimConflict(
                    claim1_id=raw.get("claim1", ""),
                    claim2_id=raw.get("claim2", ""),
                    nature=raw.get("nature", ""),
                ))

        return ClaimRecallResult(
            claims=recalled,
            conflicts=conflicts,
            reasoning=response.get("reasoning"),
        )

    # Cache management
    def cache_claim(self, claim: Claim) -> None:
        """Add a claim to the hot cache."""
        self._claim_cache[claim.id] = claim
        if claim.subject not in self._claim_by_subject:
            self._claim_by_subject[claim.subject] = []
        if claim.id not in self._claim_by_subject[claim.subject]:
            self._claim_by_subject[claim.subject].append(claim.id)

    def get_cached(self, claim_id: str) -> Optional[Claim]:
        """Get a claim from cache."""
        return self._claim_cache.get(claim_id)

    def clear_cache(self) -> None:
        """Clear the hot cache."""
        self._claim_cache.clear()
        self._claim_by_subject.clear()
