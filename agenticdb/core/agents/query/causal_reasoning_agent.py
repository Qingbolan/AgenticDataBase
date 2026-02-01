# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Causal Reasoning Agent - Explains why things happened.

This agent takes a causal chain from the dependency graph and
generates natural language explanations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import CausalExplanation, KeyFactor


class CausalReasoningAgent(BaseAgent[CausalExplanation]):
    """
    Agent for generating natural language explanations of causal chains.

    Given the dependency graph's causal chain (from why() query),
    this agent produces human-readable explanations.
    """

    name = "causal_reasoning"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Causal Reasoning Agent.

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
        target_entity: Dict[str, Any],
        causal_chain: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> CausalExplanation:
        """
        Generate a causal explanation.

        Args:
            ctx: Agent context
            target_entity: The entity we're explaining
            causal_chain: List of entities in the causal chain
            edges: Dependency edges between entities

        Returns:
            CausalExplanation with natural language explanation
        """
        # Build context for prompt
        chain_json = json.dumps({
            "target": target_entity,
            "chain": causal_chain,
            "edges": edges,
        }, indent=2, default=str)

        # Load system prompt
        system_prompt = self._load_prompt("causal_reasoning_system.md")

        # Build user prompt
        user_prompt = f"""Explain why this entity came to be:

## Target Entity
{json.dumps(target_entity, indent=2, default=str)}

## Causal Chain (from target back to root causes)
{json.dumps(causal_chain, indent=2, default=str)}

## Dependency Edges
{json.dumps(edges, indent=2, default=str)}

Return your explanation as a JSON object with "summary", "explanation", "key_factors", "causal_depth", and "confidence" fields.
"""

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        return self._parse_result(response, len(causal_chain))

    def _parse_result(
        self,
        response: dict,
        chain_length: int,
    ) -> CausalExplanation:
        """Parse explanation from LLM response."""
        key_factors = []
        for raw in self._extract_list(response, "key_factors"):
            if isinstance(raw, dict):
                key_factors.append(KeyFactor(
                    factor=raw.get("factor", ""),
                    role=raw.get("role", ""),
                ))

        return CausalExplanation(
            summary=response.get("summary", "No summary available"),
            explanation=response.get("explanation", "No explanation available"),
            key_factors=key_factors,
            causal_depth=response.get("causal_depth", chain_length),
            confidence=response.get("confidence", 0.0),
        )

    def explain_simple(
        self,
        ctx: AgentContext,
        target_description: str,
        causes: List[str],
    ) -> CausalExplanation:
        """
        Generate a simple explanation from text descriptions.

        Args:
            ctx: Agent context
            target_description: Description of what we're explaining
            causes: List of cause descriptions

        Returns:
            CausalExplanation
        """
        # Convert to simplified format
        target_entity = {"description": target_description}
        causal_chain = [{"description": c} for c in causes]
        edges = []

        return self.run(ctx, target_entity, causal_chain, edges)
