# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Impact Analysis Agent - Explains what would be affected by changes.

This agent takes impact results from the dependency graph and
generates natural language explanations of downstream effects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import (
    AffectedCount,
    CriticalImpact,
    ImpactExplanation,
    Severity,
)


class ImpactAnalysisAgent(BaseAgent[ImpactExplanation]):
    """
    Agent for generating natural language impact analysis.

    Given the dependency graph's impact result (from impact() query),
    this agent produces human-readable explanations of what would
    be affected by a change.
    """

    name = "impact_analysis"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Impact Analysis Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "query"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.2)

    def run(
        self,
        ctx: AgentContext,
        source_entity: Dict[str, Any],
        affected_entities: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> ImpactExplanation:
        """
        Generate an impact analysis explanation.

        Args:
            ctx: Agent context
            source_entity: The entity that would change
            affected_entities: Entities that would be affected
            edges: Dependency edges to affected entities

        Returns:
            ImpactExplanation with natural language analysis
        """
        # Build context for prompt
        impact_json = json.dumps({
            "source": source_entity,
            "affected": affected_entities,
            "edges": edges,
        }, indent=2, default=str)

        # Load system prompt
        system_prompt = self._load_prompt("impact_analysis_system.md")

        # Build user prompt
        user_prompt = f"""Analyze the impact if this entity changes:

## Source Entity (what would change)
{json.dumps(source_entity, indent=2, default=str)}

## Affected Entities
{json.dumps(affected_entities, indent=2, default=str)}

## Dependency Edges
{json.dumps(edges, indent=2, default=str)}

Return your analysis as a JSON object with "summary", "affected_count", "critical_impacts", "cascade_effects", "recommended_actions", and "confidence" fields.
"""

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        return self._parse_result(response)

    def _parse_result(self, response: dict) -> ImpactExplanation:
        """Parse impact analysis from LLM response."""
        # Parse affected count
        count_data = response.get("affected_count", {})
        affected_count = AffectedCount(
            events=count_data.get("events", 0),
            claims=count_data.get("claims", 0),
            actions=count_data.get("actions", 0),
        )

        # Parse critical impacts
        critical_impacts = []
        for raw in self._extract_list(response, "critical_impacts"):
            if not isinstance(raw, dict):
                continue
            try:
                severity = Severity(raw.get("severity", "medium"))
            except ValueError:
                severity = Severity.MEDIUM

            critical_impacts.append(CriticalImpact(
                entity_type=raw.get("entity_type", ""),
                entity_ref=raw.get("entity_ref", ""),
                impact=raw.get("impact", ""),
                severity=severity,
            ))

        return ImpactExplanation(
            summary=response.get("summary", "No summary available"),
            affected_count=affected_count,
            critical_impacts=critical_impacts,
            cascade_effects=self._extract_list(response, "cascade_effects"),
            recommended_actions=self._extract_list(response, "recommended_actions"),
            confidence=response.get("confidence", 0.0),
        )

    def analyze_simple(
        self,
        ctx: AgentContext,
        source_description: str,
        affected_descriptions: List[str],
    ) -> ImpactExplanation:
        """
        Generate a simple analysis from text descriptions.

        Args:
            ctx: Agent context
            source_description: Description of the changing entity
            affected_descriptions: Descriptions of affected entities

        Returns:
            ImpactExplanation
        """
        source_entity = {"description": source_description}
        affected_entities = [{"description": d} for d in affected_descriptions]
        edges = []

        return self.run(ctx, source_entity, affected_entities, edges)
