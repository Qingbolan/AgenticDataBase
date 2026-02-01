# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Action Extractor Agent - Extracts agent behaviors from text.

Actions represent things agents do - decisions, computations, side effects.
They have explicit dependency tracking.
Examples: ApproveOrder, SendNotification, RetrainModel
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import ExtractedAction, ExtractionResult


class ActionExtractorAgent(BaseAgent[ExtractionResult]):
    """
    Agent for extracting Actions from text.

    Actions are agent behaviors with explicit dependencies. They transform state
    and can produce new events, claims, or trigger other actions.
    """

    name = "action_extractor"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Action Extractor Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "ingestion"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.0)

    def run(self, ctx: AgentContext, text: str) -> ExtractionResult:
        """
        Extract actions from the given text.

        Args:
            ctx: Agent context
            text: Text to extract actions from

        Returns:
            ExtractionResult containing extracted actions
        """
        # Load prompts
        system_prompt = self._load_prompt("action_extraction_system.md")
        user_template = self._load_prompt("action_extraction_user.md")
        user_prompt = user_template.replace("{text}", text)

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        actions = self._parse_actions(response)
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "")

        return ExtractionResult(
            actions=actions,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _parse_actions(self, response: dict) -> List[ExtractedAction]:
        """Parse extracted actions from LLM response."""
        actions = []
        raw_actions = self._extract_list(response, "actions")

        for i, raw in enumerate(raw_actions):
            if not isinstance(raw, dict):
                continue

            action_type = raw.get("action_type", "")
            agent_id = raw.get("agent_id", "")
            if not action_type or not agent_id:
                continue

            action = ExtractedAction(
                action_type=action_type,
                agent_id=agent_id,
                agent_type=raw.get("agent_type"),
                inputs=raw.get("inputs", {}),
                outputs=raw.get("outputs", {}),
                depends_on_refs=raw.get("depends_on_refs", []),
                produces_refs=raw.get("produces_refs", []),
                reasoning=raw.get("reasoning"),
                ref_id=f"action_{i}",
            )
            actions.append(action)

        return actions
