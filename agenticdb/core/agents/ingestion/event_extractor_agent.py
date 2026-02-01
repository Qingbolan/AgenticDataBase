# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Event Extractor Agent - Extracts immutable events from text.

Events are facts that happened and cannot be changed.
Examples: UserRegistered, PaymentReceived, ModelTrainingCompleted
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import ExtractedEvent, ExtractionResult


class EventExtractorAgent(BaseAgent[ExtractionResult]):
    """
    Agent for extracting Events from text.

    Events are immutable facts - things that happened and cannot be disputed.
    This agent identifies events and their associated data from natural language.
    """

    name = "event_extractor"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Event Extractor Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        # Default prompts directory
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "ingestion"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.0)

    def run(self, ctx: AgentContext, text: str) -> ExtractionResult:
        """
        Extract events from the given text.

        Args:
            ctx: Agent context
            text: Text to extract events from

        Returns:
            ExtractionResult containing extracted events
        """
        # Load prompts
        system_prompt = self._load_prompt("event_extraction_system.md")
        user_template = self._load_prompt("event_extraction_user.md")
        user_prompt = user_template.replace("{text}", text)

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        events = self._parse_events(response)
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "")

        return ExtractionResult(
            events=events,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _parse_events(self, response: dict) -> List[ExtractedEvent]:
        """Parse extracted events from LLM response."""
        events = []
        raw_events = self._extract_list(response, "events")

        for i, raw in enumerate(raw_events):
            if not isinstance(raw, dict):
                continue

            event_type = raw.get("event_type", "")
            if not event_type:
                continue

            event = ExtractedEvent(
                event_type=event_type,
                data=raw.get("data", {}),
                source_agent=raw.get("source_agent"),
                source_system=raw.get("source_system"),
                correlation_id=raw.get("correlation_id"),
                ref_id=f"event_{i}",
            )
            events.append(event)

        return events
