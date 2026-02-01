# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Core tools for GEO-SCOPE.

Pure computation classes without service-layer dependencies.

Includes:
- Utility tools: PromptLoader, PromptBuilder, JsonStore, mention_extractor
- Calculation tools: MetricsCalculator

Architecture rule:
- Tools should NOT depend on services or agents
- Tools should NOT make external API calls
- Tools should be pure functions or stateless classes
"""

from .prompt_loader import PromptLoader, PromptLoaderError
from .prompt_builder import PromptBuilder, PromptBuilderError
from .mention_extractor import extract_mentions, first_mention_rank
from .json_store import JsonStore

# Metrics calculation tool (pure computation, no service dependencies)
from .metrics import MetricsCalculator

# Evaluation calculation tools
from .evaluation import (
    EvaluationCalculator,
    Sentiment,
    VisibilityTier,
    BrandConfig,
    SingleResultMetrics,
    AggregatedMetrics,
    SENTIMENT_LEXICON,
)

__all__ = [
    # Prompt tools
    "PromptLoader",
    "PromptLoaderError",
    "PromptBuilder",
    "PromptBuilderError",
    # Utility tools
    "extract_mentions",
    "first_mention_rank",
    "JsonStore",
    # Calculation tools
    "MetricsCalculator",
    # Evaluation tools
    "EvaluationCalculator",
    "Sentiment",
    "VisibilityTier",
    "BrandConfig",
    "SingleResultMetrics",
    "AggregatedMetrics",
    "SENTIMENT_LEXICON",
]
