# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Evaluation Types - Enums, constants, and data classes for brand visibility analysis.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse


# =============================================================================
# Enums
# =============================================================================

class Sentiment(Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class VisibilityTier(Enum):
    """Visibility score tiers."""
    EXCELLENT = "excellent"  # 70-100
    GOOD = "good"            # 50-69
    MODERATE = "moderate"    # 30-49
    LOW = "low"              # 0-29


# =============================================================================
# Constants
# =============================================================================

# Sentiment keywords with weights
SENTIMENT_LEXICON = {
    "positive": {
        # English
        "excellent": 1.0, "great": 0.8, "best": 0.9, "recommend": 0.85,
        "outstanding": 0.9, "amazing": 0.85, "perfect": 0.95, "superior": 0.8,
        "leading": 0.7, "innovative": 0.7, "reliable": 0.75, "trusted": 0.8,
        # Chinese
        "优秀": 1.0, "推荐": 0.85, "领先": 0.8, "优质": 0.85, "首选": 0.9,
        "卓越": 0.9, "可靠": 0.75, "高效": 0.7, "专业": 0.75, "信赖": 0.8,
        "满意": 0.7, "好评": 0.75, "出色": 0.85, "顶尖": 0.9, "一流": 0.85,
    },
    "negative": {
        # English
        "bad": 0.8, "poor": 0.85, "avoid": 0.9, "problem": 0.7, "issue": 0.6,
        "expensive": 0.5, "terrible": 0.95, "worst": 1.0, "unreliable": 0.8,
        "disappointing": 0.75, "mediocre": 0.6, "overpriced": 0.6,
        # Chinese
        "问题": 0.6, "缺点": 0.7, "不足": 0.65, "昂贵": 0.5, "复杂": 0.4,
        "困难": 0.5, "差": 0.85, "不好": 0.8, "不推荐": 0.9, "谨慎": 0.6,
        "糟糕": 0.9, "失望": 0.75, "劣质": 0.9, "坑": 0.7, "骗": 0.95,
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BrandConfig:
    """Brand configuration for evaluation."""
    name: str
    website: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        """Get all brand name variants."""
        return [self.name] + self.aliases

    @property
    def domain(self) -> str:
        """Extract domain from website URL."""
        if not self.website:
            return ""
        parsed = urlparse(self.website)
        return parsed.netloc.lower() or self.website.lower()


# Benchmark scenario 中文描述映射
SCENARIO_DESCRIPTIONS = {
    "legal_validity": "评估电子签名的法律效力和合规性",
    "api_integration": "评估 API 集成能力和技术对接便捷性",
    "security_compliance": "评估安全合规能力和数据保护措施",
    "competitor_compare": "与竞品进行功能和服务对比",
    "pricing_value": "评估价格和性价比",
    "industry_solution": "评估行业解决方案和场景适配性",
}


@dataclass
class PersonaContext:
    """
    Persona context for evaluation.

    Provides persona-specific information to enhance analysis
    by considering the user's goals and pain points.
    """
    persona_name: str
    persona_role: str = ""
    persona_description: str = ""  # Persona 的详细描述/背景
    goals: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    intent: str = ""  # Journey stage: AWARE, INTEREST, CONSIDER, etc.
    is_custom_question: bool = False  # 是否为用户自定义问题（无 persona）
    benchmark_scenario: str = ""  # Benchmark 场景背景

    def has_context(self) -> bool:
        """Check if meaningful persona context is available."""
        return bool(self.goals or self.pain_points or self.persona_description)

    def get_keywords(self) -> List[str]:
        """Extract keywords from goals and pain points for matching."""
        keywords = []
        for text in self.goals + self.pain_points:
            # Extract first few meaningful words as keywords
            words = [w for w in text.lower().split()[:5] if len(w) > 2]
            keywords.extend(words)
        return list(set(keywords))


@dataclass
class ProductMention:
    """Product mention detection result."""
    product_name: str = ""
    language: str = ""                    # 检测到的语言版本
    is_brand_associated: bool = False     # 是否与品牌在同一上下文
    context_snippet: str = ""             # 上下文片段
    sentiment: str = "neutral"            # 产品相关情感


@dataclass
class SingleResultMetrics:
    """Metrics for a single simulation result."""
    # Visibility
    visibility_score: int = 0
    visibility_tier: VisibilityTier = VisibilityTier.LOW

    # Brand mention
    brand_mentioned: bool = False
    mention_count: int = 0
    first_mention_position: float = 1.0  # Relative position (0-1)

    # Ranking
    ranking: int = -1  # -1 means not mentioned
    is_first_mention: bool = False
    is_top3: bool = False

    # Sentiment
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_confidence: float = 0.0
    sentiment_reason: str = ""

    # Citations
    brand_cited: bool = False
    brand_citation_count: int = 0
    total_citation_count: int = 0

    # Semantic similarity (if embedding available)
    semantic_relevance: float = 0.0

    # Product association (品牌+产品关联检测)
    products_mentioned: List[ProductMention] = field(default_factory=list)
    has_brand_product_linkage: bool = False


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple results."""
    count: int = 0
    avg_visibility: float = 0.0
    avg_ranking: float = 0.0
    mention_rate: float = 0.0
    first_mention_rate: float = 0.0
    top3_rate: float = 0.0
    positive_rate: float = 0.0
    neutral_rate: float = 0.0
    negative_rate: float = 0.0
    citation_rate: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result for a run."""
    # Overall metrics
    overall: AggregatedMetrics = field(default_factory=AggregatedMetrics)

    # Breakdown by dimensions
    by_provider: Dict[str, AggregatedMetrics] = field(default_factory=dict)
    by_intent: Dict[str, AggregatedMetrics] = field(default_factory=dict)
    by_persona: Dict[str, AggregatedMetrics] = field(default_factory=dict)

    # Citation analysis
    citation_sources: List[Dict[str, Any]] = field(default_factory=list)
    brand_citation_domains: List[str] = field(default_factory=list)

    # Competitor analysis
    competitor_metrics: Dict[str, AggregatedMetrics] = field(default_factory=dict)
    brand_vs_competitors_rank: int = 0

    # Raw metrics for each result
    individual_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Summary
    summary_text: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RunAnalysisResult:
    """Analysis result format expected by RunExecutor."""
    visibility_scores: Dict[str, Any] = field(default_factory=dict)
    brand_mention_counts: Dict[str, Any] = field(default_factory=dict)
    ranking_stats: Dict[str, Any] = field(default_factory=dict)
    sentiment_breakdown: Dict[str, Any] = field(default_factory=dict)
    citation_stats: Dict[str, Any] = field(default_factory=dict)
    competitor_analysis: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
