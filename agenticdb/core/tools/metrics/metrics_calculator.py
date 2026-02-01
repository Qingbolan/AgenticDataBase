# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Metrics Calculator - Pure computation for visibility metrics.

NOT an agent - does not use LLM.
Extracted from the original MetricsAgent for proper separation of concerns.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..mention_extractor import extract_mentions


class MetricsCalculator:
    """
    Calculator for visibility and comparison metrics.

    This is a pure computation class that does not inherit from BaseAgent
    since it doesn't use any LLM capabilities.
    """

    def compute(
        self,
        results: List,
        brand_name: Optional[str] = None,
        competitors: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute metrics from simulation results.

        Args:
            results: List of simulation results (objects with brand_mentioned, ranking, etc.)
            brand_name: Brand name for analysis
            competitors: List of competitor names

        Returns:
            Dictionary with visibility and comparison metrics:
            - visibility_rate: Percentage of results where brand is mentioned
            - top_position_rate: Percentage where brand ranks in top 3
            - avg_visibility_score: Average visibility score
            - vs_competitors: Comparative mention frequency vs competitors
        """
        # SEE-only MVP metrics + competitor comparison.
        if not results:
            return {
                "visibility_rate": 0.0,
                "top_position_rate": 0.0,
                "avg_visibility_score": 0.0,
                "vs_competitors": {},
            }

        mentioned = [r for r in results if getattr(r, "brand_mentioned", False)]
        visibility_rate = len(mentioned) / max(1, len(results))

        top = [
            r for r in mentioned
            if (getattr(r, "ranking", -1) > 0 and getattr(r, "ranking", -1) <= 3)
        ]
        top_position_rate = len(top) / max(1, len(results))

        avg_score = sum(
            getattr(r, "visibility_score", 0) for r in results
        ) / max(1, len(results))

        vs: Dict[str, float] = {}
        if brand_name and competitors:
            # Compare mention frequency in the same set of responses.
            brand_hits = 0
            comp_hits = {c: 0 for c in competitors if (c or "").strip()}

            for r in results:
                text = getattr(r, "simulated_response", "") or ""
                m = extract_mentions(text, [brand_name] + list(comp_hits.keys()))
                if m.get(brand_name):
                    brand_hits += 1
                for c in comp_hits.keys():
                    if m.get(c):
                        comp_hits[c] += 1

            total = max(1, len(results))
            brand_rate = brand_hits / total
            for c, cnt in comp_hits.items():
                vs[c] = brand_rate - (cnt / total)

        return {
            "visibility_rate": visibility_rate,
            "top_position_rate": top_position_rate,
            "avg_visibility_score": avg_score,
            "vs_competitors": vs,
        }
