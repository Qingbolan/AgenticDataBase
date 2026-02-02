"""
Similarity scoring tool for pattern matching.

Pure computation - no LLM calls. Computes similarity scores
between queries and cached patterns.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ...models import Intent, Pattern, OperationType
from ..pattern.pattern_hasher import PatternHasher, PatternStructure


@dataclass
class SimilarityScore:
    """
    Detailed similarity score between a query and pattern.

    Attributes:
        overall: Combined similarity score [0, 1]
        structural: Structural similarity [0, 1]
        semantic: Semantic/keyword similarity [0, 1]
        parameter: Parameter compatibility [0, 1]
        breakdown: Detailed score breakdown
    """

    overall: float
    structural: float = 0.0
    semantic: float = 0.0
    parameter: float = 1.0
    breakdown: Dict[str, float] = None

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {
                "structural": self.structural,
                "semantic": self.semantic,
                "parameter": self.parameter,
            }

    @property
    def is_match(self) -> bool:
        """Check if this is a usable match."""
        return self.overall >= 0.8

    @property
    def is_exact(self) -> bool:
        """Check if this is an exact match."""
        return self.overall >= 0.95


class SimilarityScorer:
    """
    Score similarity between queries and patterns.

    This is a pure computation tool (no LLM) that computes
    multi-dimensional similarity scores for pattern matching.

    Scoring Dimensions:
        - Structural: Same operation, predicate structure
        - Semantic: Similar keywords and intent
        - Parameter: Compatible parameter types

    Example:
        scorer = SimilarityScorer()
        score = scorer.score(intent, pattern)
        # → SimilarityScore(overall=0.92, ...)
    """

    # Keywords by operation type for semantic matching
    OPERATION_KEYWORDS = {
        OperationType.QUERY: {
            "show", "display", "get", "find", "list", "retrieve",
            "select", "fetch", "search", "lookup", "view", "read",
        },
        OperationType.STORE: {
            "store", "save", "insert", "add", "create", "record",
            "put", "write", "log", "new",
        },
        OperationType.UPDATE: {
            "update", "modify", "change", "edit", "set", "alter",
            "patch", "adjust", "revise",
        },
        OperationType.DELETE: {
            "delete", "remove", "drop", "clear", "purge", "erase",
            "destroy", "wipe", "trash",
        },
    }

    # Temporal keywords for semantic matching
    TEMPORAL_KEYWORDS = {
        "today", "yesterday", "tomorrow", "week", "month", "year",
        "last", "next", "recent", "past", "current", "since", "before",
        "after", "between", "during",
    }

    def __init__(
        self,
        structural_weight: float = 0.5,
        semantic_weight: float = 0.3,
        parameter_weight: float = 0.2,
        hasher: Optional[PatternHasher] = None,
    ):
        """
        Initialize the similarity scorer.

        Args:
            structural_weight: Weight for structural similarity
            semantic_weight: Weight for semantic similarity
            parameter_weight: Weight for parameter compatibility
            hasher: Optional PatternHasher instance
        """
        self.structural_weight = structural_weight
        self.semantic_weight = semantic_weight
        self.parameter_weight = parameter_weight
        self.hasher = hasher or PatternHasher()

        # Normalize weights
        total = structural_weight + semantic_weight + parameter_weight
        self.structural_weight /= total
        self.semantic_weight /= total
        self.parameter_weight /= total

    def score(
        self,
        intent: Intent,
        pattern: Pattern,
    ) -> SimilarityScore:
        """
        Score similarity between an Intent and a Pattern.

        Args:
            intent: The query Intent
            pattern: The cached Pattern

        Returns:
            SimilarityScore with detailed breakdown
        """
        # Structural similarity
        structural = self._score_structural(intent, pattern)

        # Semantic similarity
        semantic = self._score_semantic(intent, pattern)

        # Parameter compatibility
        parameter = self._score_parameters(intent, pattern)

        # Compute weighted overall score
        overall = (
            self.structural_weight * structural +
            self.semantic_weight * semantic +
            self.parameter_weight * parameter
        )

        return SimilarityScore(
            overall=overall,
            structural=structural,
            semantic=semantic,
            parameter=parameter,
        )

    def score_query(
        self,
        query: str,
        pattern: Pattern,
    ) -> SimilarityScore:
        """
        Score similarity between a raw query and a Pattern.

        Args:
            query: Natural language query
            pattern: The cached Pattern

        Returns:
            SimilarityScore with detailed breakdown
        """
        # Extract keywords from query
        query_keywords = self._extract_keywords(query.lower())

        # Score semantic similarity against example queries
        max_semantic = 0.0
        for example in pattern.example_queries:
            example_keywords = self._extract_keywords(example.lower())
            semantic = self._jaccard_similarity(query_keywords, example_keywords)
            max_semantic = max(max_semantic, semantic)

        # Check operation match
        query_op = self._detect_operation(query)
        op_match = 1.0 if query_op == pattern.operation else 0.0

        # Structural score based on operation match and keyword overlap
        structural = op_match

        # Parameter score based on detected slots
        parameter = 1.0  # Assume compatible until we parse

        overall = (
            self.structural_weight * structural +
            self.semantic_weight * max_semantic +
            self.parameter_weight * parameter
        )

        return SimilarityScore(
            overall=overall,
            structural=structural,
            semantic=max_semantic,
            parameter=parameter,
        )

    def find_best_match(
        self,
        intent: Intent,
        patterns: List[Pattern],
        threshold: float = 0.8,
    ) -> Optional[Tuple[Pattern, SimilarityScore]]:
        """
        Find the best matching pattern.

        Args:
            intent: The query Intent
            patterns: List of candidate patterns
            threshold: Minimum score threshold

        Returns:
            Tuple of (best pattern, score) or None
        """
        best_pattern = None
        best_score = None

        for pattern in patterns:
            score = self.score(intent, pattern)
            if score.overall >= threshold:
                if best_score is None or score.overall > best_score.overall:
                    best_pattern = pattern
                    best_score = score

        if best_pattern and best_score:
            return best_pattern, best_score
        return None

    def find_all_matches(
        self,
        intent: Intent,
        patterns: List[Pattern],
        threshold: float = 0.8,
    ) -> List[Tuple[Pattern, SimilarityScore]]:
        """
        Find all patterns above threshold, sorted by score.

        Args:
            intent: The query Intent
            patterns: List of candidate patterns
            threshold: Minimum score threshold

        Returns:
            List of (pattern, score) tuples sorted by score descending
        """
        matches = []

        for pattern in patterns:
            score = self.score(intent, pattern)
            if score.overall >= threshold:
                matches.append((pattern, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1].overall, reverse=True)
        return matches

    def _score_structural(
        self,
        intent: Intent,
        pattern: Pattern,
    ) -> float:
        """Score structural similarity."""
        score = 0.0
        max_score = 0.0

        # Operation type match (highest weight)
        max_score += 0.4
        if intent.operation == pattern.operation:
            score += 0.4

        # Table match (if both have tables)
        max_score += 0.3
        intent_table = intent.get_target_name()
        if intent_table and pattern.target_table:
            if intent_table.lower() == pattern.target_table.lower():
                score += 0.3

        # Hash match (exact structural match)
        max_score += 0.3
        intent_hash = self.hasher.hash_intent(intent)
        if intent_hash == pattern.structural_hash:
            score += 0.3

        return score / max_score if max_score > 0 else 0.0

    def _score_semantic(
        self,
        intent: Intent,
        pattern: Pattern,
    ) -> float:
        """Score semantic similarity."""
        if not intent.raw_input:
            return 0.5  # Default if no raw input

        intent_keywords = self._extract_keywords(intent.raw_input.lower())

        # Compare against example queries
        max_similarity = 0.0
        for example in pattern.example_queries:
            example_keywords = self._extract_keywords(example.lower())
            similarity = self._jaccard_similarity(intent_keywords, example_keywords)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _score_parameters(
        self,
        intent: Intent,
        pattern: Pattern,
    ) -> float:
        """Score parameter compatibility."""
        # Check if Intent has values for all pattern slots
        pattern_slots = set(pattern.parameter_slots)

        if not pattern_slots:
            return 1.0  # No parameters = fully compatible

        # Get available bindings from Intent
        available = set(intent.bindings.keys())

        # Check target
        if "target" in pattern_slots:
            if intent.get_target_name():
                available.add("target")

        # Compute overlap
        matched = pattern_slots & available
        return len(matched) / len(pattern_slots) if pattern_slots else 1.0

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Remove punctuation and split
        words = re.findall(r"\b[a-z]+\b", text.lower())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "can", "just", "now", "me", "my", "i", "you", "your", "we",
            "our", "it", "its", "this", "that", "these", "those",
        }

        keywords = {w for w in words if w not in stop_words and len(w) > 2}
        return keywords

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _detect_operation(self, query: str) -> Optional[OperationType]:
        """Detect operation type from query text."""
        query_lower = query.lower()

        for op_type, keywords in self.OPERATION_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{keyword}\b", query_lower):
                    return op_type

        return None
