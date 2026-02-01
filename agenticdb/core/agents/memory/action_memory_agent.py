# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Action Memory Agent - Manages action entity memory.

This agent provides RAG-like semantic retrieval and summarization
for action entities stored in AgenticDB.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agenticdb.core.models import Action, ActionStatus
from ..base.base_agent import AgentContext, BaseAgent
from .types import (
    ActionPattern,
    ActionRecallResult,
    MemoryStats,
    MemorySummary,
    RecalledAction,
)


class ActionMemoryAgent(BaseAgent[ActionRecallResult]):
    """
    Agent for managing action memory.

    Provides semantic retrieval (recall), pattern recognition,
    and summarization for Action entities.
    """

    name = "action_memory"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Action Memory Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "memory"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.1)

        # In-memory cache
        self._action_cache: Dict[str, Action] = {}
        self._action_by_type: Dict[str, List[str]] = {}
        self._action_by_agent: Dict[str, List[str]] = {}

    def run(
        self,
        ctx: AgentContext,
        query: str,
        actions: List[Action],
    ) -> ActionRecallResult:
        """
        Recall actions relevant to a query.

        Args:
            ctx: Agent context
            query: Natural language query
            actions: Actions to search through

        Returns:
            ActionRecallResult with relevant actions
        """
        return self.recall(query, actions)

    def recall(
        self,
        query: str,
        actions: List[Action],
        limit: int = 10,
    ) -> ActionRecallResult:
        """
        Semantic retrieval of actions matching a query.

        Args:
            query: Natural language query
            actions: Actions to search
            limit: Maximum actions to return

        Returns:
            ActionRecallResult with relevant actions
        """
        if not actions:
            return ActionRecallResult(
                actions=[],
                reasoning="No actions to search",
            )

        # Convert actions to JSON for LLM
        actions_json = json.dumps([
            {
                "id": a.id,
                "action_type": a.action_type,
                "agent_id": a.agent_id,
                "status": a.action_status.value,
                "inputs": a.inputs,
                "outputs": a.outputs,
            }
            for a in actions[:100]
        ], indent=2, default=str)

        system_prompt = self._load_prompt("action_memory_system.md")

        user_prompt = f"""Find actions relevant to this query:

Query: {query}

Available Actions:
{actions_json}

Return a JSON object with "relevant_actions" (list with action_type, agent_id, status, relevance_score, summary), "patterns", and "reasoning".
"""

        response = self.query_json(system_prompt, user_prompt)

        return self._parse_recall_result(response, actions)

    def summarize(
        self,
        actions: List[Action],
        focus: Optional[str] = None,
    ) -> MemorySummary:
        """
        Summarize a list of actions.

        Args:
            actions: Actions to summarize
            focus: Optional focus area

        Returns:
            MemorySummary
        """
        if not actions:
            return MemorySummary(
                summary="No actions to summarize",
                total_entities=0,
            )

        actions_json = json.dumps([
            {
                "action_type": a.action_type,
                "agent_id": a.agent_id,
                "status": a.action_status.value,
                "inputs": a.inputs,
            }
            for a in actions[:50]
        ], indent=2, default=str)

        system_prompt = self._load_prompt("action_memory_system.md")

        focus_text = f"\nFocus on: {focus}" if focus else ""
        user_prompt = f"""Summarize these actions:{focus_text}

{actions_json}

Return a JSON object with "summary", "by_agent", "by_type", "success_rate", and "common_failures".
"""

        response = self.query_json(system_prompt, user_prompt)

        # Extract key points from by_agent
        by_agent = response.get("by_agent", {})
        key_points = [f"{k}: {v}" for k, v in by_agent.items()]

        return MemorySummary(
            summary=response.get("summary", ""),
            key_points=key_points,
            patterns=response.get("common_failures", []),
            total_entities=len(actions),
        )

    def get_latest(
        self,
        action_type: str,
        actions: List[Action],
    ) -> Optional[Action]:
        """
        Get the most recent action of a given type.

        Args:
            action_type: Type of action to find
            actions: Actions to search

        Returns:
            Most recent action of the type, or None
        """
        matching = [a for a in actions if a.action_type == action_type]
        if not matching:
            return None

        return max(matching, key=lambda a: a.created_at or datetime.min)

    def get_by_agent(
        self,
        agent_id: str,
        actions: List[Action],
    ) -> List[Action]:
        """
        Get all actions by a specific agent.

        Args:
            agent_id: Agent ID to filter by
            actions: Actions to filter

        Returns:
            Actions performed by the agent
        """
        return [a for a in actions if a.agent_id == agent_id]

    def get_failures(self, actions: List[Action]) -> List[Action]:
        """
        Get all failed actions.

        Args:
            actions: Actions to filter

        Returns:
            Failed actions
        """
        return [a for a in actions if a.action_status == ActionStatus.FAILED]

    def track_changes(
        self,
        since: datetime,
        actions: List[Action],
    ) -> List[Action]:
        """
        Get actions created since a given time.

        Args:
            since: Timestamp to filter from
            actions: Actions to filter

        Returns:
            Actions created after the timestamp
        """
        return [
            a for a in actions
            if a.created_at and a.created_at > since
        ]

    def get_stats(self, actions: List[Action]) -> MemoryStats:
        """
        Get statistics about actions.

        Args:
            actions: Actions to analyze

        Returns:
            MemoryStats
        """
        action_types = list(set(a.action_type for a in actions))

        return MemoryStats(
            total_actions=len(actions),
            action_types=action_types,
        )

    def _parse_recall_result(
        self,
        response: dict,
        actions: List[Action],
    ) -> ActionRecallResult:
        """Parse recall result from LLM response."""
        recalled = []
        actions_by_type = {a.action_type: a for a in actions}

        for raw in self._extract_list(response, "relevant_actions"):
            if not isinstance(raw, dict):
                continue

            action_type = raw.get("action_type", "")
            if action_type in actions_by_type:
                action = actions_by_type[action_type]
                recalled.append(RecalledAction(
                    action_type=action_type,
                    agent_id=action.agent_id,
                    status=action.action_status.value,
                    relevance_score=raw.get("relevance_score", 0.0),
                    summary=raw.get("summary", ""),
                    entity_id=action.id,
                ))

        # Parse patterns
        patterns = []
        for raw in self._extract_list(response, "patterns"):
            if isinstance(raw, dict):
                patterns.append(ActionPattern(
                    pattern=raw.get("pattern", ""),
                    frequency=raw.get("frequency", ""),
                ))

        return ActionRecallResult(
            actions=recalled,
            patterns=patterns,
            reasoning=response.get("reasoning"),
        )

    # Cache management
    def cache_action(self, action: Action) -> None:
        """Add an action to the hot cache."""
        self._action_cache[action.id] = action

        if action.action_type not in self._action_by_type:
            self._action_by_type[action.action_type] = []
        if action.id not in self._action_by_type[action.action_type]:
            self._action_by_type[action.action_type].append(action.id)

        if action.agent_id not in self._action_by_agent:
            self._action_by_agent[action.agent_id] = []
        if action.id not in self._action_by_agent[action.agent_id]:
            self._action_by_agent[action.agent_id].append(action.id)

    def get_cached(self, action_id: str) -> Optional[Action]:
        """Get an action from cache."""
        return self._action_cache.get(action_id)

    def clear_cache(self) -> None:
        """Clear the hot cache."""
        self._action_cache.clear()
        self._action_by_type.clear()
        self._action_by_agent.clear()
