# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Evaluation Tools - Pure computation for brand visibility analysis.

NOT a service - does not make external API calls.
All classes here are pure computation.

Usage:
    from core.tools.evaluation import EvaluationCalculator, Sentiment, BrandConfig

    calculator = EvaluationCalculator()
    metrics = calculator.evaluate_single(answer, brand)
"""

from .types import (
    # Enums
    Sentiment,
    VisibilityTier,
    # Data classes
    BrandConfig,
    PersonaContext,
    ProductMention,
    SingleResultMetrics,
    AggregatedMetrics,
    EvaluationResult,
    RunAnalysisResult,
    # Constants
    SENTIMENT_LEXICON,
)

from .calculator import EvaluationCalculator

__all__ = [
    # Enums
    "Sentiment",
    "VisibilityTier",
    # Data classes
    "BrandConfig",
    "PersonaContext",
    "ProductMention",
    "SingleResultMetrics",
    "AggregatedMetrics",
    "EvaluationResult",
    "RunAnalysisResult",
    # Constants
    "SENTIMENT_LEXICON",
    # Calculator
    "EvaluationCalculator",
]
