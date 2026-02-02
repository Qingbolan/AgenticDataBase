# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Binding Agent for AgenticDB.

Resolves unbound parameter slots in Intent IR.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from ...models import (
    Intent,
    IntentState,
    ParameterSlot,
    SlotType,
)
from .types import BindingResult, BindingError, BindingContext, SlotSuggestion


class BindingAgent(BaseAgent[BindingResult]):
    """
    Resolve unbound parameter slots in Intent.

    This agent uses LLM and context to determine appropriate values
    for unbound slots, enabling Intent to transition from PARTIAL
    to COMPLETE state.

    Binding Principles:
    1. Monotonicity: Once bound, slots cannot be unbound
    2. Safety: Bindings should not introduce risks
    3. Context-awareness: Use conversation history and schema

    Example:
        agent = BindingAgent()
        result = agent.run(ctx, intent, context={"recent_table": "orders"})
        # → BindingResult(intent=Intent(state=COMPLETE), ...)
    """

    name = "binding"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        default_time_range: str = "7d",
        default_limit: int = 100,
    ):
        """
        Initialize the Binding Agent.

        Args:
            model: LLM model to use
            prompts_dir: Directory containing prompt templates
            default_time_range: Default time range for temporal slots
            default_limit: Default limit for numeric slots
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "intent"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.default_time_range = default_time_range
        self.default_limit = default_limit

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("binding_system.md")
        except Exception:
            self.system_prompt = self._get_default_system_prompt()

    def run(
        self,
        ctx: AgentContext,
        intent: Intent,
        bindings: Optional[Dict[str, Any]] = None,
        context: Optional[BindingContext] = None,
    ) -> BindingResult:
        """
        Resolve unbound slots in an Intent.

        Args:
            ctx: Agent context
            intent: Intent with unbound slots
            bindings: Explicit bindings to apply
            context: Binding context with available information

        Returns:
            BindingResult with updated Intent
        """
        # If no unbound slots, return immediately
        unbound = intent.get_unbound_slot_names()
        if not unbound:
            return BindingResult.from_intent(intent, bound_slots=[], source="none")

        # Apply explicit bindings first
        bound_slots = []
        current_intent = intent

        if bindings:
            for slot_name, value in bindings.items():
                if slot_name in unbound:
                    try:
                        current_intent = current_intent.bind_slot(slot_name, value)
                        bound_slots.append(slot_name)
                    except ValueError as e:
                        return BindingResult.from_error(str(e), slot_name)

        # Check if any slots remain
        remaining = current_intent.get_unbound_slot_names()
        if not remaining:
            return BindingResult.from_intent(
                current_intent,
                bound_slots=bound_slots,
                source="explicit",
            )

        # Try to resolve remaining slots using context and LLM
        context = context or BindingContext()
        suggestions = self._get_binding_suggestions(current_intent, remaining, context)

        for suggestion in suggestions:
            if suggestion.confidence >= 0.8:
                try:
                    current_intent = current_intent.bind_slot(
                        suggestion.slot_name,
                        suggestion.suggested_value,
                    )
                    bound_slots.append(suggestion.slot_name)
                except ValueError:
                    continue

        return BindingResult.from_intent(
            current_intent,
            bound_slots=bound_slots,
            source="context" if suggestions else "explicit",
        )

    def bind_explicit(
        self,
        intent: Intent,
        slot_name: str,
        value: Any,
    ) -> BindingResult:
        """
        Explicitly bind a single slot.

        Args:
            intent: Intent to bind
            slot_name: Name of slot to bind
            value: Value to bind

        Returns:
            BindingResult with updated Intent
        """
        try:
            new_intent = intent.bind_slot(slot_name, value)
            return BindingResult.from_intent(
                new_intent,
                bound_slots=[slot_name],
                source="user",
            )
        except ValueError as e:
            return BindingResult.from_error(str(e), slot_name)

    def suggest_bindings(
        self,
        intent: Intent,
        context: Optional[BindingContext] = None,
    ) -> List[SlotSuggestion]:
        """
        Suggest bindings for unbound slots without applying them.

        Args:
            intent: Intent with unbound slots
            context: Binding context

        Returns:
            List of binding suggestions
        """
        unbound = intent.get_unbound_slot_names()
        if not unbound:
            return []

        context = context or BindingContext()
        return self._get_binding_suggestions(intent, unbound, context)

    def _get_binding_suggestions(
        self,
        intent: Intent,
        unbound_slots: List[str],
        context: BindingContext,
    ) -> List[SlotSuggestion]:
        """Get binding suggestions using context and heuristics."""
        suggestions = []

        for slot_name in unbound_slots:
            suggestion = self._suggest_for_slot(intent, slot_name, context)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _suggest_for_slot(
        self,
        intent: Intent,
        slot_name: str,
        context: BindingContext,
    ) -> Optional[SlotSuggestion]:
        """Suggest a binding for a specific slot."""
        # Find the slot
        slot = None
        for s in intent.unbound_slots:
            if s.name == slot_name:
                slot = s
                break

        if isinstance(intent.target, ParameterSlot) and intent.target.name == slot_name:
            slot = intent.target

        if not slot:
            return None

        # Generate suggestion based on slot type
        if slot.slot_type == SlotType.ENTITY:
            return self._suggest_entity(slot_name, context)
        elif slot.slot_type == SlotType.TEMPORAL:
            return self._suggest_temporal(slot_name, context)
        elif slot.slot_type == SlotType.NUMERIC:
            return self._suggest_numeric(slot_name, context)
        elif slot.slot_type == SlotType.FILTER:
            return self._suggest_filter(slot_name, context)

        return None

    def _suggest_entity(
        self,
        slot_name: str,
        context: BindingContext,
    ) -> Optional[SlotSuggestion]:
        """Suggest an entity binding."""
        # Use recent targets if available
        if context.recent_targets:
            return SlotSuggestion(
                slot_name=slot_name,
                suggested_value=context.recent_targets[0],
                confidence=0.7,
                source="context",
                reason="Most recently used table",
            )

        # Use available tables
        if context.available_tables:
            return SlotSuggestion(
                slot_name=slot_name,
                suggested_value=context.available_tables[0],
                confidence=0.5,
                source="schema",
                reason="First available table",
            )

        return None

    def _suggest_temporal(
        self,
        slot_name: str,
        context: BindingContext,
    ) -> Optional[SlotSuggestion]:
        """Suggest a temporal binding."""
        # Parse default time range
        range_str = context.default_time_range or self.default_time_range
        date_value = self._parse_time_range(range_str)

        return SlotSuggestion(
            slot_name=slot_name,
            suggested_value=date_value,
            confidence=0.8,
            source="default",
            reason=f"Default time range: {range_str}",
        )

    def _suggest_numeric(
        self,
        slot_name: str,
        context: BindingContext,
    ) -> Optional[SlotSuggestion]:
        """Suggest a numeric binding."""
        limit = context.default_limit or self.default_limit

        return SlotSuggestion(
            slot_name=slot_name,
            suggested_value=limit,
            confidence=0.9,
            source="default",
            reason=f"Default limit: {limit}",
        )

    def _suggest_filter(
        self,
        slot_name: str,
        context: BindingContext,
    ) -> Optional[SlotSuggestion]:
        """Suggest a filter binding."""
        # Default to no filter (include all)
        return SlotSuggestion(
            slot_name=slot_name,
            suggested_value=None,
            confidence=0.5,
            source="default",
            reason="No filter (include all)",
        )

    def _parse_time_range(self, range_str: str) -> datetime:
        """Parse time range string to datetime."""
        now = datetime.now(timezone.utc)

        # Parse patterns like "7d", "1w", "1m"
        import re

        match = re.match(r"(\d+)([dwmyh])", range_str.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)

            if unit == "d":
                return now - timedelta(days=amount)
            elif unit == "w":
                return now - timedelta(weeks=amount)
            elif unit == "m":
                return now - timedelta(days=amount * 30)
            elif unit == "y":
                return now - timedelta(days=amount * 365)
            elif unit == "h":
                return now - timedelta(hours=amount)

        # Default to 7 days ago
        return now - timedelta(days=7)

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a Binding Resolution Agent. Given an Intent with unbound slots and context, suggest appropriate bindings.

Return JSON with:
- bindings: list of {slot_name, value, confidence, source, reasoning}
- remaining_unbound: slots that couldn't be bound
- requires_user_input: true if user must provide value
- questions: list of questions for user if needed"""
