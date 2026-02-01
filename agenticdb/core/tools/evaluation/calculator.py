# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Evaluation Calculator - Pure computation for brand visibility analysis.

NOT a service - does not make external API calls.
Does not use parallel processing (that stays in EvaluationService).

This class contains the core calculation logic for:
- Visibility scoring
- Ranking detection
- Sentiment analysis
- Citation analysis
- Product mention detection
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .types import (
    Sentiment,
    VisibilityTier,
    BrandConfig,
    ProductMention,
    SingleResultMetrics,
    AggregatedMetrics,
    SENTIMENT_LEXICON,
)


class EvaluationCalculator:
    """
    Calculator for brand visibility evaluation.

    Pure computation class - no external API calls, no parallel processing.
    Use this directly for single result evaluation, or through EvaluationService
    for parallel batch processing.

    Usage:
        calculator = EvaluationCalculator()
        metrics = calculator.evaluate_single({"final_answer": "..."}, brand)
    """

    # =========================================================================
    # Main Evaluation Methods
    # =========================================================================

    def evaluate_single(
        self,
        result: Dict[str, Any],
        brand: BrandConfig,
    ) -> SingleResultMetrics:
        """
        Evaluate a single simulation result.

        Args:
            result: Simulation result dict with 'final_answer' or 'simulated_response'
            brand: Brand configuration

        Returns:
            SingleResultMetrics with all computed metrics
        """
        answer = result.get("final_answer", "") or result.get("simulated_response", "")
        citations = result.get("citations", [])

        metrics = SingleResultMetrics()

        if not answer:
            return metrics

        # 1. Visibility and mention analysis
        vis_score, mention_count, first_pos = self.calculate_visibility(answer, brand)
        metrics.visibility_score = vis_score
        metrics.visibility_tier = self.get_visibility_tier(vis_score)
        metrics.brand_mentioned = mention_count > 0
        metrics.mention_count = mention_count
        metrics.first_mention_position = first_pos

        # 2. Ranking analysis
        ranking = self.calculate_ranking(answer, brand)
        metrics.ranking = ranking
        metrics.is_first_mention = ranking == 1
        metrics.is_top3 = 0 < ranking <= 3

        # 3. Sentiment analysis
        sentiment, score, confidence, reason = self.analyze_sentiment(answer, brand)
        metrics.sentiment = sentiment
        metrics.sentiment_score = score
        metrics.sentiment_confidence = confidence
        metrics.sentiment_reason = reason

        # 4. Citation analysis
        brand_cited, brand_count, total_count = self.analyze_citations(citations, brand)
        metrics.brand_cited = brand_cited
        metrics.brand_citation_count = brand_count
        metrics.total_citation_count = total_count

        return metrics

    def aggregate_metrics(
        self,
        metrics_list: List[SingleResultMetrics],
    ) -> AggregatedMetrics:
        """
        Aggregate a list of SingleResultMetrics.

        Args:
            metrics_list: List of individual metrics

        Returns:
            AggregatedMetrics with averages and rates
        """
        if not metrics_list:
            return AggregatedMetrics()

        n = len(metrics_list)

        # Calculate average ranking (only for mentioned results)
        rankings = [m.ranking for m in metrics_list if m.ranking > 0]
        avg_ranking = sum(rankings) / len(rankings) if rankings else 0.0

        return AggregatedMetrics(
            count=n,
            avg_visibility=sum(m.visibility_score for m in metrics_list) / n,
            avg_ranking=avg_ranking,
            mention_rate=sum(1 for m in metrics_list if m.brand_mentioned) / n,
            first_mention_rate=sum(1 for m in metrics_list if m.is_first_mention) / n,
            top3_rate=sum(1 for m in metrics_list if m.is_top3) / n,
            positive_rate=sum(1 for m in metrics_list if m.sentiment == Sentiment.POSITIVE) / n,
            neutral_rate=sum(1 for m in metrics_list if m.sentiment == Sentiment.NEUTRAL) / n,
            negative_rate=sum(1 for m in metrics_list if m.sentiment == Sentiment.NEGATIVE) / n,
            citation_rate=sum(1 for m in metrics_list if m.brand_cited) / n,
        )

    # =========================================================================
    # Visibility Calculation
    # =========================================================================

    def calculate_visibility(
        self,
        answer: str,
        brand: BrandConfig,
    ) -> Tuple[int, int, float]:
        """
        Calculate visibility score for brand in answer.

        Args:
            answer: AI response text
            brand: Brand configuration

        Returns:
            Tuple of (visibility_score, mention_count, first_mention_position)
        """
        if not answer:
            return 0, 0, 1.0

        text_lower = answer.lower()

        # Count mentions across all brand name variants
        total_mentions = 0
        first_position = len(answer)

        for name in brand.all_names:
            name_lower = name.lower()
            count = text_lower.count(name_lower)
            total_mentions += count

            if count > 0:
                pos = text_lower.find(name_lower)
                first_position = min(first_position, pos)

        if total_mentions == 0:
            return 0, 0, 1.0

        # Calculate relative position (0 = start, 1 = end)
        relative_pos = first_position / len(answer) if len(answer) > 0 else 1.0

        # Base score for mention
        score = 50

        # Early mention bonus (up to +30)
        if relative_pos < 0.1:
            score += 30
        elif relative_pos < 0.2:
            score += 25
        elif relative_pos < 0.3:
            score += 20
        elif relative_pos < 0.5:
            score += 10

        # Multiple mention bonus (up to +20)
        if total_mentions > 1:
            score += min(20, (total_mentions - 1) * 5)

        return min(100, score), total_mentions, relative_pos

    def get_visibility_tier(self, score: int) -> VisibilityTier:
        """Get visibility tier from score."""
        if score >= 70:
            return VisibilityTier.EXCELLENT
        elif score >= 50:
            return VisibilityTier.GOOD
        elif score >= 30:
            return VisibilityTier.MODERATE
        else:
            return VisibilityTier.LOW

    # =========================================================================
    # Ranking Calculation
    # =========================================================================

    def calculate_ranking(self, answer: str, brand: BrandConfig) -> int:
        """
        Calculate ranking (position) of brand mention in answer.

        Args:
            answer: AI response text
            brand: Brand configuration

        Returns:
            1-based ranking, or -1 if not mentioned
        """
        if not answer:
            return -1

        # Split into sentences/segments
        segments = re.split(r'[.!?。！？\n]+', answer)

        for i, segment in enumerate(segments):
            segment_lower = segment.lower()
            for name in brand.all_names:
                if name.lower() in segment_lower:
                    return i + 1

        return -1

    # =========================================================================
    # Sentiment Analysis
    # =========================================================================

    def analyze_sentiment(
        self,
        answer: str,
        brand: BrandConfig,
    ) -> Tuple[Sentiment, float, float, str]:
        """
        Analyze sentiment toward brand in answer.

        Args:
            answer: AI response text
            brand: Brand configuration

        Returns:
            Tuple of (sentiment, score, confidence, reason)
        """
        if not answer:
            return Sentiment.NEUTRAL, 0.0, 0.0, "Empty answer"

        # Check if brand is mentioned
        text_lower = answer.lower()
        brand_mentioned = any(name.lower() in text_lower for name in brand.all_names)

        if not brand_mentioned:
            return Sentiment.NEUTRAL, 0.0, 0.0, "Brand not mentioned"

        # Extract context around brand mentions
        context = self._extract_brand_context(answer, brand)

        # Calculate sentiment using lexicon
        return self._lexicon_sentiment(context)

    def _extract_brand_context(
        self,
        answer: str,
        brand: BrandConfig,
        window: int = 200,
    ) -> str:
        """Extract text context around brand mentions."""
        contexts = []
        text_lower = answer.lower()

        for name in brand.all_names:
            name_lower = name.lower()
            start = 0
            while True:
                pos = text_lower.find(name_lower, start)
                if pos == -1:
                    break

                # Extract window around mention
                ctx_start = max(0, pos - window)
                ctx_end = min(len(answer), pos + len(name) + window)
                contexts.append(answer[ctx_start:ctx_end])
                start = pos + 1

        return " ".join(contexts)

    def _lexicon_sentiment(
        self,
        text: str,
    ) -> Tuple[Sentiment, float, float, str]:
        """Calculate sentiment using weighted lexicon."""
        text_lower = text.lower()

        positive_score = 0.0
        negative_score = 0.0
        positive_words = []
        negative_words = []

        for word, weight in SENTIMENT_LEXICON["positive"].items():
            count = text_lower.count(word)
            if count > 0:
                positive_score += weight * count
                positive_words.append(word)

        for word, weight in SENTIMENT_LEXICON["negative"].items():
            count = text_lower.count(word)
            if count > 0:
                negative_score += weight * count
                negative_words.append(word)

        # Calculate net score (-1 to 1)
        total = positive_score + negative_score
        if total == 0:
            return Sentiment.NEUTRAL, 0.0, 0.5, "No sentiment indicators found"

        net_score = (positive_score - negative_score) / (positive_score + negative_score)
        confidence = min(0.9, 0.5 + total * 0.1)

        # Determine sentiment
        if net_score > 0.2:
            sentiment = Sentiment.POSITIVE
            reason = f"Positive indicators: {', '.join(positive_words[:3])}"
        elif net_score < -0.2:
            sentiment = Sentiment.NEGATIVE
            reason = f"Negative indicators: {', '.join(negative_words[:3])}"
        else:
            sentiment = Sentiment.NEUTRAL
            reason = "Mixed or neutral sentiment"

        return sentiment, net_score, confidence, reason

    # =========================================================================
    # Citation Analysis
    # =========================================================================

    def analyze_citations(
        self,
        citations: List[Dict],
        brand: BrandConfig,
    ) -> Tuple[bool, int, int]:
        """
        Analyze citations for brand website presence.

        Args:
            citations: List of citation dicts with 'url' key
            brand: Brand configuration

        Returns:
            Tuple of (brand_cited, brand_citation_count, total_citation_count)
        """
        total_count = len(citations)

        if not citations or not brand.domain:
            return False, 0, total_count

        brand_count = 0
        brand_domain = brand.domain

        for citation in citations:
            url = citation.get("url", "") or citation.get("source_url", "")
            if url:
                domain = urlparse(url).netloc.lower()
                if brand_domain in domain or domain in brand_domain:
                    brand_count += 1

        return brand_count > 0, brand_count, total_count

    # =========================================================================
    # Product Mention Detection
    # =========================================================================

    def detect_product_mentions(
        self,
        answer: str,
        brand_name: str,
        products: List[Dict[str, Any]],
        context_window: int = 200,
    ) -> Tuple[List[ProductMention], bool]:
        """
        检测产品在响应中的提及，并判断是否与品牌关联。

        Args:
            answer: AI 回答文本
            brand_name: 品牌名称
            products: 产品列表，每个产品包含 versions (地区版本列表)
            context_window: 上下文窗口大小（字符数）

        Returns:
            Tuple of (产品提及列表, 是否有品牌+产品关联)
        """
        if not answer or not products:
            return [], False

        text_lower = answer.lower()
        brand_lower = brand_name.lower()

        # 找到所有品牌提及的位置
        brand_positions = []
        start = 0
        while True:
            pos = text_lower.find(brand_lower, start)
            if pos == -1:
                break
            brand_positions.append(pos)
            start = pos + 1

        mentions = []
        has_linkage = False

        for product in products:
            # 获取所有产品名称变体（包括多语言）
            product_names = self._get_product_names(product)

            for name_info in product_names:
                name = name_info["name"]
                language = name_info.get("language", "default")

                if not name:
                    continue

                name_lower = name.lower()

                # 检测产品名称是否在回答中
                pos = text_lower.find(name_lower)
                if pos == -1:
                    continue

                # 检查是否与品牌关联（在同一上下文窗口内）
                is_associated = False
                context_snippet = ""

                for brand_pos in brand_positions:
                    distance = abs(pos - brand_pos)
                    if distance <= context_window:
                        is_associated = True
                        # 提取上下文片段
                        ctx_start = max(0, min(pos, brand_pos) - 50)
                        ctx_end = min(len(answer), max(pos + len(name), brand_pos + len(brand_name)) + 50)
                        context_snippet = answer[ctx_start:ctx_end]
                        break

                # 如果没有品牌位置，检查同一句话
                if not brand_positions and not is_associated:
                    sentences = re.split(r'[.!?。！？\n]+', answer)
                    for sentence in sentences:
                        if name_lower in sentence.lower() and brand_lower in sentence.lower():
                            is_associated = True
                            context_snippet = sentence.strip()
                            break

                if is_associated:
                    has_linkage = True

                mentions.append(ProductMention(
                    product_name=name,
                    language=language,
                    is_brand_associated=is_associated,
                    context_snippet=context_snippet[:200] if context_snippet else "",
                    sentiment="neutral",
                ))

        return mentions, has_linkage

    def _get_product_names(self, product: Dict[str, Any]) -> List[Dict[str, str]]:
        """获取产品的所有名称变体（从地区版本中提取）"""
        names = []

        # 从地区版本中提取名称
        for ver in product.get("versions", []):
            if ver.get("name"):
                names.append({"name": ver["name"], "language": ver.get("language", "")})

        return names

    # =========================================================================
    # Summary Generation
    # =========================================================================

    def generate_simple_summary(
        self,
        brand_name: str,
        avg_visibility: float,
        mention_rate: float,
        ranking_stats: Dict,
        sentiment: Dict,
    ) -> str:
        """Generate a simple summary text."""
        total = sentiment.get("positive", 0) + sentiment.get("neutral", 0) + sentiment.get("negative", 0)
        if total == 0:
            return f"品牌 {brand_name} 暂无评估数据。"

        pos_rate = sentiment.get("positive", 0) / total * 100
        neg_rate = sentiment.get("negative", 0) / total * 100

        # 确保数值不为 None（.get() 不会处理值为 None 的情况）
        avg_visibility_safe = avg_visibility if avg_visibility is not None else 0
        mention_rate_safe = mention_rate if mention_rate is not None else 0
        avg_rank = ranking_stats.get('average_rank')
        avg_rank_safe = avg_rank if avg_rank is not None else 0
        first_mention_count = ranking_stats.get('first_mention_count') or 0
        top3_count = ranking_stats.get('top3_count') or 0

        summary = f"""品牌 "{brand_name}" AI搜索可见性分析：

可见性评分：{avg_visibility_safe:.1f}/100
品牌提及率：{mention_rate_safe*100:.0f}%
平均排名：第{avg_rank_safe:.1f}位
首位提及次数：{first_mention_count}次
前三位提及次数：{top3_count}次

情感分析：
- 正面评价：{pos_rate:.0f}%
- 负面评价：{neg_rate:.0f}%"""

        return summary

    def generate_recommendations(
        self,
        overall: AggregatedMetrics,
        by_provider: Optional[Dict[str, AggregatedMetrics]] = None,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Visibility recommendations
        if overall.avg_visibility < 50:
            recommendations.append("建议优化品牌关键词策略，增加品牌在AI回答中的曝光率")

        if overall.first_mention_rate < 0.2:
            recommendations.append("建议创建更多高质量内容，争取在AI回答中获得更靠前的位置")

        # Sentiment recommendations
        if overall.negative_rate > 0.2:
            recommendations.append("建议关注负面评价内容，采取措施改善品牌形象")

        # Citation recommendations
        if overall.citation_rate < 0.1:
            recommendations.append("建议优化品牌官网SEO，增加被AI引用的机会")

        # Provider-specific recommendations
        if by_provider:
            for provider, metrics in by_provider.items():
                if metrics.avg_visibility < overall.avg_visibility * 0.7:
                    recommendations.append(f"建议针对{provider}平台优化内容策略")

        return recommendations
