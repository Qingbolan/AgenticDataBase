# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Event Memory Agent - Manages event entity memory.

This agent provides RAG-like semantic retrieval and summarization
for event entities stored in AgenticDB.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agenticdb.core.models import Event
from ..base.base_agent import AgentContext, BaseAgent
from .types import (
    EventRecallResult,
    MemoryStats,
    MemorySummary,
    RecalledEvent,
)


class EventMemoryAgent(BaseAgent[EventRecallResult]):
    """
    Agent for managing event memory.

    Provides semantic retrieval (recall), summarization, and
    state tracking for Event entities.
    """

    name = "event_memory"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Event Memory Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "memory"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.1)

        # In-memory cache for hot data
        self._event_cache: Dict[str, Event] = {}
        self._event_by_type: Dict[str, List[str]] = {}

    def run(
        self,
        ctx: AgentContext,
        query: str,
        events: List[Event],
    ) -> EventRecallResult:
        """
        Recall events relevant to a query.

        Args:
            ctx: Agent context
            query: Natural language query
            events: Events to search through

        Returns:
            EventRecallResult with relevant events
        """
        return self.recall(query, events)

    def recall(
        self,
        query: str,
        events: List[Event],
        limit: int = 10,
    ) -> EventRecallResult:
        """
        Semantic retrieval of events matching a query.

        Args:
            query: Natural language query
            events: Events to search
            limit: Maximum events to return

        Returns:
            EventRecallResult with relevant events
        """
        if not events:
            return EventRecallResult(
                events=[],
                reasoning="No events to search",
            )

        # Convert events to JSON for LLM
        events_json = json.dumps([
            {
                "id": e.id,
                "event_type": e.event_type,
                "data": e.data,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events[:100]  # Limit context size
        ], indent=2, default=str)

        # Load system prompt
        system_prompt = self._load_prompt("event_memory_system.md")

        # Build user prompt
        user_prompt = f"""Find events relevant to this query:

Query: {query}

Available Events:
{events_json}

Return a JSON object with "relevant_events" (list with event_type, relevance_score, summary) and "reasoning".
"""

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse and return
        return self._parse_recall_result(response, events)

    def summarize(
        self,
        events: List[Event],
        focus: Optional[str] = None,
    ) -> MemorySummary:
        """
        Summarize a list of events.

        Args:
            events: Events to summarize
            focus: Optional focus area for summary

        Returns:
            MemorySummary
        """
        if not events:
            return MemorySummary(
                summary="No events to summarize",
                total_entities=0,
            )

        # Convert events to JSON
        events_json = json.dumps([
            {
                "event_type": e.event_type,
                "data": e.data,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events[:50]
        ], indent=2, default=str)

        system_prompt = self._load_prompt("event_memory_system.md")

        focus_text = f"\nFocus on: {focus}" if focus else ""
        user_prompt = f"""Summarize these events:{focus_text}

{events_json}

Return a JSON object with "summary", "key_events", "timeline", and "patterns".
"""

        response = self.query_json(system_prompt, user_prompt)

        return MemorySummary(
            summary=response.get("summary", ""),
            key_points=response.get("key_events", []),
            patterns=response.get("patterns", []),
            total_entities=len(events),
        )

    def get_latest(
        self,
        event_type: str,
        events: List[Event],
    ) -> Optional[Event]:
        """
        Get the most recent event of a given type.

        Args:
            event_type: Type of event to find
            events: Events to search

        Returns:
            Most recent event of the type, or None
        """
        matching = [e for e in events if e.event_type == event_type]
        if not matching:
            return None

        return max(matching, key=lambda e: e.created_at or datetime.min)

    def track_changes(
        self,
        since: datetime,
        events: List[Event],
    ) -> List[Event]:
        """
        Get events created since a given time.

        Args:
            since: Timestamp to filter from
            events: Events to filter

        Returns:
            Events created after the timestamp
        """
        return [
            e for e in events
            if e.created_at and e.created_at > since
        ]

    def get_stats(self, events: List[Event]) -> MemoryStats:
        """
        Get statistics about events.

        Args:
            events: Events to analyze

        Returns:
            MemoryStats
        """
        event_types = list(set(e.event_type for e in events))

        return MemoryStats(
            total_events=len(events),
            event_types=event_types,
        )

    def _parse_recall_result(
        self,
        response: dict,
        events: List[Event],
    ) -> EventRecallResult:
        """Parse recall result from LLM response."""
        recalled = []
        events_by_type = {e.event_type: e for e in events}

        for raw in self._extract_list(response, "relevant_events"):
            if not isinstance(raw, dict):
                continue

            event_type = raw.get("event_type", "")
            if event_type in events_by_type:
                event = events_by_type[event_type]
                recalled.append(RecalledEvent(
                    event_type=event_type,
                    data=event.data,
                    relevance_score=raw.get("relevance_score", 0.0),
                    summary=raw.get("summary", ""),
                    created_at=event.created_at,
                    entity_id=event.id,
                ))

        return EventRecallResult(
            events=recalled,
            reasoning=response.get("reasoning"),
        )

    # Cache management methods
    def cache_event(self, event: Event) -> None:
        """Add an event to the hot cache."""
        self._event_cache[event.id] = event
        if event.event_type not in self._event_by_type:
            self._event_by_type[event.event_type] = []
        if event.id not in self._event_by_type[event.event_type]:
            self._event_by_type[event.event_type].append(event.id)

    def get_cached(self, event_id: str) -> Optional[Event]:
        """Get an event from cache."""
        return self._event_cache.get(event_id)

    def clear_cache(self) -> None:
        """Clear the hot cache."""
        self._event_cache.clear()
        self._event_by_type.clear()
