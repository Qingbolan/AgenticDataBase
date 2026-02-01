# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Schema Detector Agent - Detects new types and fields from extracted entities.

This agent analyzes extracted entities to identify patterns that suggest
new schema elements should be added to the registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from ..ingestion.types import (
    ExtractedAction,
    ExtractedClaim,
    ExtractedEvent,
)
from .types import (
    DetectedActionType,
    DetectedClaimSubject,
    DetectedEventType,
    FieldDefinition,
    SchemaDetectionResult,
)


class SchemaDetectorAgent(BaseAgent[SchemaDetectionResult]):
    """
    Agent for detecting schema patterns in extracted entities.

    Analyzes extracted events, claims, and actions to identify new types
    or fields that should be added to the schema registry.
    """

    name = "schema_detector"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Schema Detector Agent.

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
        events: List[ExtractedEvent],
        claims: List[ExtractedClaim],
        actions: List[ExtractedAction],
        known_event_types: Optional[List[str]] = None,
        known_claim_patterns: Optional[List[str]] = None,
        known_action_types: Optional[List[str]] = None,
    ) -> SchemaDetectionResult:
        """
        Detect new schema patterns from extracted entities.

        Args:
            ctx: Agent context
            events: Extracted events
            claims: Extracted claims
            actions: Extracted actions
            known_event_types: Already known event types
            known_claim_patterns: Already known claim subject patterns
            known_action_types: Already known action types

        Returns:
            SchemaDetectionResult with detected new types
        """
        # Build context for the prompt
        entities_json = json.dumps({
            "events": [
                {"event_type": e.event_type, "data": e.data}
                for e in events
            ],
            "claims": [
                {"subject": c.subject, "value": c.value, "source": c.source}
                for c in claims
            ],
            "actions": [
                {"action_type": a.action_type, "inputs": a.inputs, "outputs": a.outputs}
                for a in actions
            ],
        }, indent=2, default=str)

        known_types_json = json.dumps({
            "event_types": known_event_types or [],
            "claim_patterns": known_claim_patterns or [],
            "action_types": known_action_types or [],
        }, indent=2)

        # Load system prompt
        system_prompt = self._load_prompt("schema_detection_system.md")

        # Build user prompt
        user_prompt = f"""Analyze these extracted entities and detect new schema patterns.

## Current Known Types
{known_types_json}

## Extracted Entities
{entities_json}

Return your analysis as a JSON object with "new_event_types", "new_claim_subjects", "new_action_types", "confidence", and "reasoning" fields.
"""

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        return self._parse_result(response)

    def _parse_result(self, response: dict) -> SchemaDetectionResult:
        """Parse detection result from LLM response."""
        # Parse event types
        event_types = []
        for raw in self._extract_list(response, "new_event_types"):
            if not isinstance(raw, dict):
                continue
            fields = [
                FieldDefinition(
                    name=f.get("name", ""),
                    type=f.get("type", "string"),
                    required=f.get("required", False),
                )
                for f in raw.get("fields", [])
                if isinstance(f, dict) and f.get("name")
            ]
            event_types.append(DetectedEventType(
                type_name=raw.get("type_name", ""),
                fields=fields,
                description=raw.get("description"),
            ))

        # Parse claim subjects
        claim_subjects = []
        for raw in self._extract_list(response, "new_claim_subjects"):
            if not isinstance(raw, dict):
                continue
            claim_subjects.append(DetectedClaimSubject(
                subject_pattern=raw.get("subject_pattern", ""),
                value_type=raw.get("value_type", "string"),
                common_sources=raw.get("common_sources", []),
                description=raw.get("description"),
            ))

        # Parse action types
        action_types = []
        for raw in self._extract_list(response, "new_action_types"):
            if not isinstance(raw, dict):
                continue
            input_fields = [
                FieldDefinition(
                    name=f.get("name", ""),
                    type=f.get("type", "string"),
                    required=f.get("required", False),
                )
                for f in raw.get("input_fields", [])
                if isinstance(f, dict) and f.get("name")
            ]
            output_fields = [
                FieldDefinition(
                    name=f.get("name", ""),
                    type=f.get("type", "string"),
                    required=f.get("required", False),
                )
                for f in raw.get("output_fields", [])
                if isinstance(f, dict) and f.get("name")
            ]
            action_types.append(DetectedActionType(
                type_name=raw.get("type_name", ""),
                input_fields=input_fields,
                output_fields=output_fields,
                description=raw.get("description"),
            ))

        return SchemaDetectionResult(
            new_event_types=event_types,
            new_claim_subjects=claim_subjects,
            new_action_types=action_types,
            confidence=response.get("confidence", 0.0),
            reasoning=response.get("reasoning"),
        )
