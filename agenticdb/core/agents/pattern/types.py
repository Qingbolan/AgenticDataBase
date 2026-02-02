# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Type definitions for Pattern learning agents.

These types define the input/output contracts for Pattern agents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...models import Pattern, OperationType


@dataclass
class PatternScore:
    """
    Score representing how well a pattern matches a query.

    Attributes:
        structural_similarity: How similar the query structure is [0, 1]
        semantic_similarity: How similar the meaning is [0, 1]
        parameter_compatibility: How compatible parameters are [0, 1]
        overall_score: Weighted combination of all scores [0, 1]
    """

    structural_similarity: float = 0.0
    semantic_similarity: float = 0.0
    parameter_compatibility: float = 1.0
    overall_score: float = 0.0

    def __post_init__(self):
        """Compute overall score if not provided."""
        if self.overall_score == 0.0:
            # Weighted average: structure most important, then semantics
            self.overall_score = (
                self.structural_similarity * 0.5 +
                self.semantic_similarity * 0.3 +
                self.parameter_compatibility * 0.2
            )

    @property
    def is_match(self) -> bool:
        """Check if this represents a valid match (score > threshold)."""
        return self.overall_score >= 0.8

    @property
    def is_strong_match(self) -> bool:
        """Check if this is a strong match (score > 0.95)."""
        return self.overall_score >= 0.95


@dataclass
class PatternMatch:
    """
    A pattern match result.

    Attributes:
        pattern: The matched pattern
        score: Match quality score
        extracted_parameters: Parameters extracted from the query
        sql: Generated SQL from pattern with parameters
        confidence: Overall confidence in this match
    """

    pattern: Pattern
    score: PatternScore
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    sql: Optional[str] = None
    confidence: float = 1.0

    @property
    def is_usable(self) -> bool:
        """Check if this match can be used directly."""
        return self.score.is_match and self.confidence >= 0.8

    @property
    def is_exact(self) -> bool:
        """Check if this is an exact match."""
        return self.score.is_strong_match and self.confidence >= 0.95


@dataclass
class PatternExtractionResult:
    """
    Result of pattern extraction from a query.

    Attributes:
        pattern: The extracted pattern
        success: Whether extraction succeeded
        error: Error message if extraction failed
        raw_query: The original query
        parameters: Parameters identified in the query
        template: SQL template with placeholders
    """

    pattern: Optional[Pattern] = None
    success: bool = True
    error: Optional[str] = None
    raw_query: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    template: Optional[str] = None

    @classmethod
    def from_pattern(
        cls,
        pattern: Pattern,
        raw_query: str,
        parameters: Dict[str, Any],
    ) -> "PatternExtractionResult":
        """Create a successful extraction result."""
        return cls(
            pattern=pattern,
            success=True,
            raw_query=raw_query,
            parameters=parameters,
            template=pattern.template,
        )

    @classmethod
    def from_error(cls, error: str, raw_query: str) -> "PatternExtractionResult":
        """Create a failed extraction result."""
        return cls(
            success=False,
            error=error,
            raw_query=raw_query,
        )


@dataclass
class PatternMatchResult:
    """
    Result of pattern matching against the cache.

    Attributes:
        found: Whether a matching pattern was found
        match: The best match if found
        all_matches: All matches above threshold, sorted by score
        cache_hit: Whether this was a cache hit
        query: The original query
    """

    found: bool = False
    match: Optional[PatternMatch] = None
    all_matches: List[PatternMatch] = field(default_factory=list)
    cache_hit: bool = False
    query: str = ""

    @classmethod
    def hit(
        cls,
        match: PatternMatch,
        query: str,
        all_matches: Optional[List[PatternMatch]] = None,
    ) -> "PatternMatchResult":
        """Create a cache hit result."""
        return cls(
            found=True,
            match=match,
            all_matches=all_matches or [match],
            cache_hit=True,
            query=query,
        )

    @classmethod
    def miss(cls, query: str) -> "PatternMatchResult":
        """Create a cache miss result."""
        return cls(
            found=False,
            cache_hit=False,
            query=query,
        )

    @property
    def can_execute_directly(self) -> bool:
        """Check if we can execute directly from cache."""
        return self.found and self.match is not None and self.match.is_usable


@dataclass
class PatternCacheStats:
    """
    Statistics about pattern cache performance.

    Attributes:
        total_patterns: Total patterns in cache
        total_hits: Total cache hits
        total_misses: Total cache misses
        hit_rate: Cache hit rate [0, 1]
        avg_match_score: Average match score
        top_patterns: Most frequently hit patterns
    """

    total_patterns: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    avg_match_score: float = 0.0
    top_patterns: List[str] = field(default_factory=list)  # Pattern IDs

    def record_hit(self, score: float) -> None:
        """Record a cache hit."""
        self.total_hits += 1
        total = self.total_hits + self.total_misses
        self.hit_rate = self.total_hits / total if total > 0 else 0.0
        # Update running average
        self.avg_match_score = (
            (self.avg_match_score * (self.total_hits - 1) + score) / self.total_hits
        )

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.total_misses += 1
        total = self.total_hits + self.total_misses
        self.hit_rate = self.total_hits / total if total > 0 else 0.0
