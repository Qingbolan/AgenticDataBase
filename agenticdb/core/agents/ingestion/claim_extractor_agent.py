# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Claim Extractor Agent - Extracts structured assertions from text.

Claims are assertions with provenance - they have a source, confidence,
and can be superseded or invalidated.
Examples: risk_score = 0.7 from model_v2, user.tier = "premium" from rules_engine
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..base.base_agent import AgentContext, BaseAgent
from .types import ExtractedClaim, ExtractionResult


class ClaimExtractorAgent(BaseAgent[ExtractionResult]):
    """
    Agent for extracting Claims from text.

    Claims are structured assertions with provenance. Unlike events (immutable facts),
    claims can be superseded by newer claims or invalidated when dependencies change.
    """

    name = "claim_extractor"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Claim Extractor Agent.

        Args:
            model: LLM model name
            prompts_dir: Path to prompts directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "ingestion"

        super().__init__(model=model, prompts_dir=prompts_dir, temperature=0.0)

    def run(self, ctx: AgentContext, text: str) -> ExtractionResult:
        """
        Extract claims from the given text.

        Args:
            ctx: Agent context
            text: Text to extract claims from

        Returns:
            ExtractionResult containing extracted claims
        """
        # Load prompts
        system_prompt = self._load_prompt("claim_extraction_system.md")
        user_template = self._load_prompt("claim_extraction_user.md")
        user_prompt = user_template.replace("{text}", text)

        # Query LLM
        response = self.query_json(system_prompt, user_prompt)

        # Parse response
        claims = self._parse_claims(response)
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "")

        return ExtractionResult(
            claims=claims,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _parse_claims(self, response: dict) -> List[ExtractedClaim]:
        """Parse extracted claims from LLM response."""
        claims = []
        raw_claims = self._extract_list(response, "claims")

        for i, raw in enumerate(raw_claims):
            if not isinstance(raw, dict):
                continue

            subject = raw.get("subject", "")
            source = raw.get("source", "")
            if not subject or not source:
                continue

            claim = ExtractedClaim(
                subject=subject,
                value=raw.get("value"),
                source=source,
                predicate=raw.get("predicate", "is"),
                source_version=raw.get("source_version"),
                confidence=raw.get("confidence", 1.0),
                derived_from_refs=raw.get("derived_from", []),
                ref_id=f"claim_{i}",
            )
            claims.append(claim)

        return claims
