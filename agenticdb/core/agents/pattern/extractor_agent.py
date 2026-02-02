# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Pattern Extractor Agent for AgenticDB.

Extracts reusable query patterns from successful queries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from ...models import Intent, Pattern, OperationType
from ...tools.pattern.pattern_hasher import PatternHasher
from .types import PatternExtractionResult


class PatternExtractorAgent(BaseAgent[PatternExtractionResult]):
    """
    Extract reusable patterns from queries.

    This agent analyzes successful queries and extracts patterns
    that can be reused for similar future queries, reducing LLM calls.

    Example:
        extractor = PatternExtractorAgent()
        result = extractor.run(ctx, intent, sql="SELECT * FROM orders WHERE ...")
        # → PatternExtractionResult(pattern=Pattern(...), ...)
    """

    name = "pattern_extractor"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Pattern Extractor Agent.

        Args:
            model: LLM model to use
            prompts_dir: Directory containing prompt templates
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "pattern"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.hasher = PatternHasher()

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("extractor_system.md")
        except Exception:
            self.system_prompt = self._get_default_system_prompt()

    def run(
        self,
        ctx: AgentContext,
        intent: Intent,
        sql: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> PatternExtractionResult:
        """
        Extract a pattern from an Intent and its generated SQL.

        Args:
            ctx: Agent context
            intent: The source Intent
            sql: Generated SQL
            parameters: SQL parameters

        Returns:
            PatternExtractionResult with extracted pattern
        """
        parameters = parameters or {}

        try:
            # Create template from SQL
            template = self.hasher.extract_template(sql, parameters)

            # Create pattern
            pattern = self.hasher.create_pattern(intent, template)

            return PatternExtractionResult.from_pattern(
                pattern=pattern,
                raw_query=intent.raw_input,
                parameters=parameters,
            )

        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return PatternExtractionResult.from_error(
                str(e),
                intent.raw_input,
            )

    def extract_simple(
        self,
        query: str,
        sql: str,
        operation: OperationType,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> PatternExtractionResult:
        """
        Extract a pattern without full Intent analysis.

        Args:
            query: Original natural language query
            sql: Generated SQL
            operation: Operation type
            parameters: SQL parameters

        Returns:
            PatternExtractionResult with extracted pattern
        """
        parameters = parameters or {}

        try:
            # Create template
            template = self.hasher.extract_template(sql, parameters)

            # Compute hash from SQL
            structural_hash = self.hasher.hash_sql(sql)

            # Extract parameter slots
            import re
            parameter_slots = re.findall(r"\{(\w+)\}", template)

            # Create pattern
            pattern = Pattern(
                template=template,
                structural_hash=structural_hash,
                operation=operation,
                parameter_slots=parameter_slots,
                example_queries=[query],
            )

            return PatternExtractionResult.from_pattern(
                pattern=pattern,
                raw_query=query,
                parameters=parameters,
            )

        except Exception as e:
            return PatternExtractionResult.from_error(str(e), query)

    def should_extract(self, intent: Intent) -> bool:
        """
        Determine if a pattern should be extracted from this Intent.

        Args:
            intent: The Intent to check

        Returns:
            True if pattern extraction is worthwhile
        """
        # Don't extract from low-confidence intents
        if intent.confidence < 0.7:
            return False

        # Don't extract from very simple queries
        if not intent.predicates and not intent.bindings:
            return False

        # Extract from queries with reasonable complexity
        return True

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a Pattern Extractor. Extract reusable patterns from queries.

Create SQL templates with {placeholders} for variable parts:
- {target}: Table name
- {time_start}, {time_end}: Time boundaries
- {limit}: Result limit
- {filter_value}: Filter values

Return JSON with template, operation, parameter_slots, and confidence."""
