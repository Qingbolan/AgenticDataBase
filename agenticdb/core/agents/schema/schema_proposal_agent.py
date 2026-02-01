# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Schema Proposal Agent - Generates schema evolution proposals.

This agent takes detected schema changes and generates well-structured
proposals that can be reviewed and applied.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import (
    ChangeType,
    ImpactAnalysis,
    SchemaChange,
    SchemaDetectionResult,
    SchemaProposal,
    SchemaProposalResult,
)


class SchemaProposalAgent(BaseAgent[SchemaProposalResult]):
    """
    Agent for generating schema evolution proposals.

    Takes detection results and generates formal proposals
    for schema changes.
    """

    name = "schema_proposal"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Schema Proposal Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "schema"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.0)

    def run(
        self,
        ctx: AgentContext,
        detection_result: SchemaDetectionResult,
    ) -> SchemaProposalResult:
        """
        Generate a schema proposal from detection results.

        Args:
            ctx: Agent context
            detection_result: Schema detection result

        Returns:
            SchemaProposalResult with the generated proposal
        """
        # Convert detection result to JSON
        detection_json = json.dumps({
            "new_event_types": [
                {
                    "type_name": e.type_name,
                    "fields": [
                        {"name": f.name, "type": f.type, "required": f.required}
                        for f in e.fields
                    ],
                    "description": e.description,
                }
                for e in detection_result.new_event_types
            ],
            "new_claim_subjects": [
                {
                    "subject_pattern": c.subject_pattern,
                    "value_type": c.value_type,
                    "common_sources": c.common_sources,
                    "description": c.description,
                }
                for c in detection_result.new_claim_subjects
            ],
            "new_action_types": [
                {
                    "type_name": a.type_name,
                    "input_fields": [
                        {"name": f.name, "type": f.type}
                        for f in a.input_fields
                    ],
                    "output_fields": [
                        {"name": f.name, "type": f.type}
                        for f in a.output_fields
                    ],
                    "description": a.description,
                }
                for a in detection_result.new_action_types
            ],
        }, indent=2)

        # Check if there's anything to propose
        if not (
            detection_result.new_event_types
            or detection_result.new_claim_subjects
            or detection_result.new_action_types
        ):
            return SchemaProposalResult(
                proposal=None,
                confidence=1.0,
                reasoning="No schema changes detected",
            )

        # Load system prompt
        system_prompt = self._load_prompt("schema_proposal_system.md")

        # Build user prompt
        user_prompt = f"""Generate a schema proposal for the following detected changes:

{detection_json}

Return your proposal as a JSON object with "proposal", "confidence", and "reasoning" fields.
"""

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        return self._parse_result(response)

    def _parse_result(self, response: dict) -> SchemaProposalResult:
        """Parse proposal result from LLM response."""
        proposal_data = response.get("proposal", {})

        if not proposal_data:
            return SchemaProposalResult(
                proposal=None,
                confidence=response.get("confidence", 0.0),
                reasoning=response.get("reasoning"),
            )

        # Parse changes
        changes = []
        for raw in proposal_data.get("changes", []):
            if not isinstance(raw, dict):
                continue
            try:
                change_type = ChangeType(raw.get("change_type", ""))
            except ValueError:
                continue

            changes.append(SchemaChange(
                change_type=change_type,
                target=raw.get("target", ""),
                definition=raw.get("definition", {}),
                breaking=raw.get("breaking", False),
                migration=raw.get("migration"),
            ))

        # Parse impact analysis
        impact_data = proposal_data.get("impact_analysis", {})
        impact = ImpactAnalysis(
            affected_entities=impact_data.get("affected_entities", 0),
            backwards_compatible=impact_data.get("backwards_compatible", True),
            requires_migration=impact_data.get("requires_migration", False),
        )

        proposal = SchemaProposal(
            id=proposal_data.get("id", str(uuid.uuid4())[:8]),
            title=proposal_data.get("title", "Schema Update"),
            description=proposal_data.get("description", ""),
            changes=changes,
            impact_analysis=impact,
        )

        return SchemaProposalResult(
            proposal=proposal,
            confidence=response.get("confidence", 0.0),
            reasoning=response.get("reasoning"),
        )
