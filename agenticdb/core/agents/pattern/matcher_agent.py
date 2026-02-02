# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Pattern Matcher Agent for AgenticDB.

Matches queries against cached patterns for fast execution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base.base_agent import AgentContext, BaseAgent
from ...models import Intent, Pattern
from ...tools.pattern.pattern_hasher import PatternHasher
from ...tools.pattern.similarity_scorer import SimilarityScorer, SimilarityScore
from .types import PatternMatch, PatternMatchResult, PatternScore


class PatternMatcherAgent(BaseAgent[PatternMatchResult]):
    """
    Match queries against cached patterns.

    This agent checks if incoming queries match cached patterns,
    enabling direct SQL execution without LLM parsing.

    Example:
        matcher = PatternMatcherAgent(patterns=cached_patterns)
        result = matcher.run(ctx, query="show orders from last week")
        if result.found:
            # Execute directly from pattern
            sql = result.match.pattern.template
    """

    name = "pattern_matcher"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        patterns: Optional[List[Pattern]] = None,
        match_threshold: float = 0.8,
    ):
        """
        Initialize the Pattern Matcher Agent.

        Args:
            model: LLM model to use
            prompts_dir: Directory containing prompt templates
            patterns: List of cached patterns
            match_threshold: Minimum score for a match
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "pattern"

        super().__init__(model=model, prompts_dir=prompts_dir)

        self.patterns = patterns or []
        self.match_threshold = match_threshold
        self.hasher = PatternHasher()
        self.scorer = SimilarityScorer(hasher=self.hasher)

        # Load prompts
        try:
            self.system_prompt = self._load_prompt("matcher_system.md")
        except Exception:
            self.system_prompt = self._get_default_system_prompt()

    def run(
        self,
        ctx: AgentContext,
        query: str,
        intent: Optional[Intent] = None,
    ) -> PatternMatchResult:
        """
        Match a query against cached patterns.

        Args:
            ctx: Agent context
            query: Natural language query
            intent: Optional pre-parsed Intent

        Returns:
            PatternMatchResult with match information
        """
        if not self.patterns:
            return PatternMatchResult.miss(query)

        # If we have an Intent, use structured matching
        if intent:
            return self._match_intent(intent, query)

        # Otherwise, use query-based matching
        return self._match_query(query)

    def match_fast(self, query: str) -> Optional[PatternMatch]:
        """
        Fast pattern matching without full analysis.

        Args:
            query: Natural language query

        Returns:
            Best match or None
        """
        result = self._match_query(query)
        return result.match if result.found else None

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the cache."""
        # Check for duplicates
        for existing in self.patterns:
            if existing.structural_hash == pattern.structural_hash:
                # Update existing pattern
                existing.example_queries.extend(pattern.example_queries)
                return

        self.patterns.append(pattern)

    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern from the cache."""
        for i, pattern in enumerate(self.patterns):
            if pattern.id == pattern_id:
                self.patterns.pop(i)
                return True
        return False

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        for pattern in self.patterns:
            if pattern.id == pattern_id:
                return pattern
        return None

    def clear_patterns(self) -> None:
        """Clear all cached patterns."""
        self.patterns.clear()

    def _match_intent(self, intent: Intent, query: str) -> PatternMatchResult:
        """Match using structured Intent."""
        best_match = self.scorer.find_best_match(
            intent,
            self.patterns,
            self.match_threshold,
        )

        if best_match:
            pattern, score = best_match

            # Extract parameters
            params = self._extract_parameters(intent, pattern)

            # Generate SQL
            sql = self._generate_sql(pattern, params)

            match = PatternMatch(
                pattern=pattern,
                score=PatternScore(
                    structural_similarity=score.structural,
                    semantic_similarity=score.semantic,
                    parameter_compatibility=score.parameter,
                    overall_score=score.overall,
                ),
                extracted_parameters=params,
                sql=sql,
                confidence=score.overall,
            )

            return PatternMatchResult.hit(match, query)

        return PatternMatchResult.miss(query)

    def _match_query(self, query: str) -> PatternMatchResult:
        """Match using raw query text."""
        best_pattern = None
        best_score = None

        for pattern in self.patterns:
            score = self.scorer.score_query(query, pattern)

            if score.overall >= self.match_threshold:
                if best_score is None or score.overall > best_score.overall:
                    best_pattern = pattern
                    best_score = score

        if best_pattern and best_score:
            # Extract parameters from query
            params = self._extract_parameters_from_query(query, best_pattern)

            # Generate SQL
            sql = self._generate_sql(best_pattern, params)

            match = PatternMatch(
                pattern=best_pattern,
                score=PatternScore(
                    structural_similarity=best_score.structural,
                    semantic_similarity=best_score.semantic,
                    parameter_compatibility=best_score.parameter,
                    overall_score=best_score.overall,
                ),
                extracted_parameters=params,
                sql=sql,
                confidence=best_score.overall,
            )

            return PatternMatchResult.hit(match, query)

        return PatternMatchResult.miss(query)

    def _extract_parameters(
        self,
        intent: Intent,
        pattern: Pattern,
    ) -> Dict[str, Any]:
        """Extract parameters from Intent for pattern."""
        params = {}

        for slot_name in pattern.parameter_slots:
            if slot_name == "target":
                params["target"] = intent.get_target_name()
            elif slot_name in intent.bindings:
                params[slot_name] = intent.bindings[slot_name]
            elif slot_name == "limit":
                params["limit"] = intent.bindings.get("limit", 100)

        return params

    def _extract_parameters_from_query(
        self,
        query: str,
        pattern: Pattern,
    ) -> Dict[str, Any]:
        """Extract parameters from raw query text."""
        import re
        from datetime import datetime, timedelta, timezone

        params = {}
        query_lower = query.lower()

        for slot_name in pattern.parameter_slots:
            if slot_name == "target":
                # Try to extract table name
                if pattern.target_table:
                    params["target"] = pattern.target_table
                else:
                    # Look for "from X" pattern
                    match = re.search(r"\bfrom\s+(\w+)", query_lower)
                    if match:
                        params["target"] = match.group(1)

            elif slot_name in ("time_start", "time_relative"):
                # Parse temporal references
                now = datetime.now(timezone.utc)

                if "last week" in query_lower:
                    params[slot_name] = now - timedelta(weeks=1)
                elif "last month" in query_lower:
                    params[slot_name] = now - timedelta(days=30)
                elif "yesterday" in query_lower:
                    params[slot_name] = now - timedelta(days=1)
                elif "today" in query_lower:
                    params[slot_name] = now.replace(hour=0, minute=0, second=0)
                else:
                    params[slot_name] = now - timedelta(days=7)  # Default

            elif slot_name == "limit":
                # Look for limit numbers
                match = re.search(r"\b(?:top|limit|first)\s+(\d+)", query_lower)
                if match:
                    params["limit"] = int(match.group(1))
                else:
                    params["limit"] = 100  # Default

        return params

    def _generate_sql(
        self,
        pattern: Pattern,
        params: Dict[str, Any],
    ) -> str:
        """Generate SQL from pattern and parameters."""
        sql = pattern.template

        for name, value in params.items():
            placeholder = f"{{{name}}}"
            if placeholder in sql:
                # Format value appropriately
                if isinstance(value, str):
                    formatted = f"'{value}'"
                elif isinstance(value, datetime):
                    formatted = f"'{value.isoformat()}'"
                else:
                    formatted = str(value)

                sql = sql.replace(placeholder, formatted)

        return sql

    def _get_default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a Pattern Matcher. Match queries against cached patterns.

Score matches on:
- Structural similarity (same operation, predicates)
- Semantic similarity (similar keywords)
- Parameter compatibility

Extract parameters for matching patterns and generate SQL."""


# Import datetime for type hints
from datetime import datetime
