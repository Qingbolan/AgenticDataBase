# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Dependency Inference Agent - Infers causal relationships between entities.

This agent analyzes extracted entities and determines how they depend on,
produce, derive from, or invalidate each other.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import (
    EdgeType,
    ExtractedAction,
    ExtractedClaim,
    ExtractedEvent,
    ExtractionResult,
    InferredEdge,
)


class DependencyInferenceAgent(BaseAgent[ExtractionResult]):
    """
    Agent for inferring dependency relationships between entities.

    This agent takes extracted events, claims, and actions and determines
    the causal and dependency edges between them.
    """

    name = "dependency_inference"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Dependency Inference Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "ingestion"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.0)

    def run(
        self,
        ctx: AgentContext,
        events: List[ExtractedEvent],
        claims: List[ExtractedClaim],
        actions: List[ExtractedAction],
    ) -> ExtractionResult:
        """
        Infer dependencies between extracted entities.

        Args:
            ctx: Agent context
            events: Extracted events
            claims: Extracted claims
            actions: Extracted actions

        Returns:
            ExtractionResult containing inferred edges
        """
        # Convert entities to JSON for prompt
        events_json = json.dumps(
            [self._event_to_dict(e) for e in events],
            indent=2,
            default=str,
        )
        claims_json = json.dumps(
            [self._claim_to_dict(c) for c in claims],
            indent=2,
            default=str,
        )
        actions_json = json.dumps(
            [self._action_to_dict(a) for a in actions],
            indent=2,
            default=str,
        )

        # Load prompts
        system_prompt = self._load_prompt("dependency_inference_system.md")
        user_template = self._load_prompt("dependency_inference_user.md")
        user_prompt = (
            user_template
            .replace("{events_json}", events_json)
            .replace("{claims_json}", claims_json)
            .replace("{actions_json}", actions_json)
        )

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        edges = self._parse_edges(response)
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "")

        return ExtractionResult(
            edges=edges,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _event_to_dict(self, event: ExtractedEvent) -> dict:
        """Convert ExtractedEvent to dict for JSON serialization."""
        return {
            "ref_id": event.ref_id,
            "event_type": event.event_type,
            "data": event.data,
            "source_agent": event.source_agent,
            "source_system": event.source_system,
        }

    def _claim_to_dict(self, claim: ExtractedClaim) -> dict:
        """Convert ExtractedClaim to dict for JSON serialization."""
        return {
            "ref_id": claim.ref_id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "source": claim.source,
            "confidence": claim.confidence,
        }

    def _action_to_dict(self, action: ExtractedAction) -> dict:
        """Convert ExtractedAction to dict for JSON serialization."""
        return {
            "ref_id": action.ref_id,
            "action_type": action.action_type,
            "agent_id": action.agent_id,
            "inputs": action.inputs,
            "outputs": action.outputs,
            "depends_on_refs": action.depends_on_refs,
            "produces_refs": action.produces_refs,
        }

    def _parse_edges(self, response: dict) -> List[InferredEdge]:
        """Parse inferred edges from LLM response."""
        edges = []
        raw_edges = self._extract_list(response, "edges")

        for raw in raw_edges:
            if not isinstance(raw, dict):
                continue

            from_ref = raw.get("from_ref", "")
            to_ref = raw.get("to_ref", "")
            edge_type_str = raw.get("edge_type", "")

            if not from_ref or not to_ref or not edge_type_str:
                continue

            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                continue

            edge = InferredEdge(
                from_ref=from_ref,
                to_ref=to_ref,
                edge_type=edge_type,
                reasoning=raw.get("reasoning"),
            )
            edges.append(edge)

        return edges
