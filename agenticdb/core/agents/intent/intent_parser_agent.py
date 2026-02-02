# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Intent Parser Agent for AgenticDB.

Parses natural language queries into Intent IR using LLM.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from ...models import (
    Intent,
    IntentState,
    OperationType,
    ParameterSlot,
    Predicate,
    SlotType,
)
from ...tools.intent.slot_detector import SlotDetector
from .types import IntentParseResult


class IntentParserAgent(BaseAgent[IntentParseResult]):
    """
    Parse natural language queries into Intent IR.

    This agent uses LLM to understand the user's intent and convert
    it into a structured Intent object that can be processed by
    the transaction pipeline.

    The agent:
    1. Identifies the operation type (QUERY, STORE, UPDATE, DELETE)
    2. Extracts the target entity/table
    3. Parses predicates and filters
    4. Identifies unbound parameter slots

    Example:
        agent = IntentParserAgent()
        result = agent.run(ctx, query="show orders from last week")
        # → IntentParseResult(intent=Intent(operation=QUERY, target="orders", ...))
    """

    name = "intent_parser"

    def __init__(
        self,
        model: Optional[str] = None,
        available_tables: Optional[List[str]] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Intent Parser Agent.

        Args:
            model: LLM model to use
            available_tables: List of known table names for context
            prompts_dir: Directory containing prompt templates
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "intent"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.available_tables = available_tables or []
        self.slot_detector = SlotDetector(known_entities=set(self.available_tables))

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("parser_system.md")
            self.user_template = self._load_prompt("parser_user.md")
        except Exception:
            # Fallback to embedded prompts
            self.system_prompt = self._get_default_system_prompt()
            self.user_template = self._get_default_user_template()

    def run(
        self,
        ctx: AgentContext,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentParseResult:
        """
        Parse a natural language query into Intent IR.

        Args:
            ctx: Agent context
            query: Natural language query
            context: Optional additional context

        Returns:
            IntentParseResult with parsed Intent
        """
        context = context or {}

        # First, use slot detector for initial analysis
        slot_analysis = self.slot_detector.analyze(query)

        # Build user prompt
        user_prompt = self.user_template.format(
            available_tables=", ".join(self.available_tables) if self.available_tables else "Unknown",
            query=query,
            context=str(context) if context else "None",
        )

        try:
            # Query LLM
            response = self.query_json(self.system_prompt, user_prompt)

            # Parse response into Intent
            intent = self._parse_response(response, query)

            return IntentParseResult.from_intent(
                intent=intent,
                confidence=response.get("confidence", 0.8),
            )

        except Exception as e:
            self.logger.error(f"Intent parsing failed: {e}")

            # Fallback to slot detector analysis
            return self._fallback_parse(query, slot_analysis)

    def parse_simple(self, query: str) -> IntentParseResult:
        """
        Parse a query without full LLM (uses slot detector only).

        Useful for fast, simple queries or when LLM is unavailable.

        Args:
            query: Natural language query

        Returns:
            IntentParseResult with basic Intent
        """
        analysis = self.slot_detector.analyze(query)
        return self._fallback_parse(query, analysis)

    def update_available_tables(self, tables: List[str]) -> None:
        """Update the list of available tables."""
        self.available_tables = tables
        self.slot_detector = SlotDetector(known_entities=set(tables))

    def _parse_response(self, response: Dict[str, Any], raw_input: str) -> Intent:
        """Parse LLM response into Intent object."""
        # Extract operation
        op_str = response.get("operation", "query").lower()
        operation = self._parse_operation(op_str)

        # Extract target
        target_str = response.get("target")
        target_resolved = response.get("target_resolved", target_str is not None)

        if target_resolved and target_str:
            target = target_str
        else:
            target = ParameterSlot(
                name="target",
                slot_type=SlotType.ENTITY,
                description="The target table/entity for this operation",
            )

        # Extract predicates
        predicates = []
        for pred_data in response.get("predicates", []):
            pred = Predicate(
                field=pred_data.get("field", ""),
                operator=pred_data.get("operator", "eq"),
                value=pred_data.get("value"),
                negate=pred_data.get("negate", False),
            )
            predicates.append(pred)

        # Extract bindings
        bindings = response.get("bindings", {})

        # Extract unbound slots
        unbound_slots = []
        for slot_data in response.get("unbound_slots", []):
            slot = ParameterSlot(
                name=slot_data.get("name", "unknown"),
                slot_type=self._parse_slot_type(slot_data.get("slot_type", "string")),
                description=slot_data.get("description"),
            )
            unbound_slots.append(slot)

        # Determine state
        state = IntentState.COMPLETE
        if unbound_slots or not target_resolved:
            state = IntentState.PARTIAL

        return Intent(
            operation=operation,
            target=target,
            predicates=predicates,
            bindings=bindings,
            unbound_slots=unbound_slots,
            state=state,
            raw_input=raw_input,
            confidence=response.get("confidence", 0.8),
        )

    def _fallback_parse(
        self,
        query: str,
        analysis: Dict[str, Any],
    ) -> IntentParseResult:
        """Fallback parsing using slot detector analysis."""
        operation = analysis.get("operation") or OperationType.QUERY

        # Determine target
        if analysis.get("target_resolved"):
            target = self._extract_target_from_query(query)
        else:
            target = ParameterSlot(
                name="target",
                slot_type=SlotType.ENTITY,
            )

        # Convert detected slots to ParameterSlots
        unbound_slots = []
        for slot in analysis.get("slots", []):
            unbound_slots.append(slot.to_parameter_slot())

        # Determine state
        state = IntentState.PARTIAL if analysis.get("needs_binding") else IntentState.COMPLETE

        intent = Intent(
            operation=operation,
            target=target,
            unbound_slots=unbound_slots,
            state=state,
            raw_input=query,
            confidence=0.6,  # Lower confidence for fallback
        )

        return IntentParseResult.from_intent(intent, confidence=0.6)

    def _parse_operation(self, op_str: str) -> OperationType:
        """Parse operation string to OperationType."""
        op_map = {
            "query": OperationType.QUERY,
            "select": OperationType.QUERY,
            "store": OperationType.STORE,
            "insert": OperationType.STORE,
            "update": OperationType.UPDATE,
            "delete": OperationType.DELETE,
            "remove": OperationType.DELETE,
        }
        return op_map.get(op_str.lower(), OperationType.QUERY)

    def _parse_slot_type(self, type_str: str) -> SlotType:
        """Parse slot type string to SlotType."""
        type_map = {
            "entity": SlotType.ENTITY,
            "temporal": SlotType.TEMPORAL,
            "numeric": SlotType.NUMERIC,
            "filter": SlotType.FILTER,
            "string": SlotType.STRING,
            "list": SlotType.LIST,
        }
        return type_map.get(type_str.lower(), SlotType.STRING)

    def _extract_target_from_query(self, query: str) -> str:
        """Extract target table from query using simple patterns."""
        import re

        # Check for "from X" pattern
        from_match = re.search(r"\bfrom\s+(\w+)", query.lower())
        if from_match:
            return from_match.group(1)

        # Check for known tables
        for table in self.available_tables:
            if table.lower() in query.lower():
                return table

        # Default to first available table or "unknown"
        return self.available_tables[0] if self.available_tables else "unknown"

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt if file not found."""
        return """You are an Intent Parser that converts natural language queries into structured Intent objects.

Return JSON with:
- operation: query|store|update|delete
- target: table name or null if ambiguous
- target_resolved: true/false
- predicates: list of {field, operator, value}
- bindings: dict of resolved values
- unbound_slots: list of {name, slot_type, description}
- confidence: 0-1
- reasoning: explanation"""

    def _get_default_user_template(self) -> str:
        """Return default user template if file not found."""
        return """Parse this query: {query}

Available tables: {available_tables}
Context: {context}"""
