"""
Entity Extractors for AgenticDB.

Extractors convert unstructured text into candidate entities.
They are pluggable — you can use:
- Rule-based extraction (deterministic, fast)
- LLM-based extraction (flexible, handles ambiguity)
- Hybrid approaches

Design Philosophy:
    Extractors are the ONLY place where non-determinism (LLM) is allowed.
    The rest of the pipeline is deterministic, making the system
    reproducible given the same extraction output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import re

from pydantic import BaseModel, Field


class ExtractorResult(BaseModel):
    """
    Result of entity extraction.

    Contains candidate entities in a normalized format,
    ready for the compiler to convert to semantic objects.
    """

    # Extracted candidates
    events: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)

    # Schema hints
    unknown_types: list[str] = Field(default_factory=list)

    # Raw extraction data (for debugging/auditing)
    raw_extractions: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extractor_name: str = Field(default="unknown")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"extra": "forbid"}

    @property
    def is_empty(self) -> bool:
        """Check if no entities were extracted."""
        return not self.events and not self.claims and not self.actions


class Extractor(ABC):
    """
    Abstract base class for entity extractors.

    Extractors take text and produce candidate entities.
    They should be stateless and idempotent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor name for logging/debugging."""
        pass

    @abstractmethod
    def extract(
        self,
        text: str,
        mode: str = "description",
        context: Optional[dict[str, Any]] = None,
    ) -> ExtractorResult:
        """
        Extract entities from text.

        Args:
            text: Input text
            mode: Extraction mode (agent_trace, log, description, structured)
            context: Optional context (existing schema, hints)

        Returns:
            ExtractorResult with candidate entities
        """
        pass


class RuleBasedExtractor(Extractor):
    """
    Rule-based entity extractor.

    Uses regex patterns and heuristics to extract entities.
    Deterministic, fast, but limited to known patterns.

    Patterns recognized:
    - Events: "X registered", "X happened", "X occurred"
    - Claims: "X = Y", "X is Y", "score of X is Y"
    - Actions: "approved", "rejected", "processed"
    """

    @property
    def name(self) -> str:
        return "rule-based"

    def __init__(self):
        """Initialize with default patterns."""
        # Event patterns
        self._event_patterns = [
            r"(?P<entity>\w+)\s+(?P<user_id>\w+)\s+registered\s+with\s+(?:email\s+)?(?P<email>\S+)",
            r"(?P<event_type>\w+)\s+event\s+occurred",
            r"(?P<entity>\w+)\s+was\s+created",
            r"(?P<entity>\w+)\s+(?P<id>\w+)\s+signed\s+up",
        ]

        # Claim patterns
        self._claim_patterns = [
            r"(?P<subject>[\w.]+)\s*=\s*(?P<value>[\d.]+)",
            r"(?P<model>\w+)\s+computed\s+(?P<subject>\w+)\s*=\s*(?P<value>[\d.]+)",
            r"(?P<subject>\w+)\s+(?:score|value)\s+(?:is|was|of)\s+(?P<value>[\d.]+)",
            r"confidence\s+(?:of\s+)?(?P<value>[\d.]+)",
        ]

        # Action patterns
        self._action_patterns = [
            r"(?P<entity>\w+)\s+was\s+(?P<action>approved|rejected|flagged)",
            r"(?P<action>approved|rejected)\s+(?:for\s+)?(?P<reason>[\w\s]+)",
            r"user\s+was\s+(?P<action>\w+)",
        ]

    def extract(
        self,
        text: str,
        mode: str = "description",
        context: Optional[dict[str, Any]] = None,
    ) -> ExtractorResult:
        """Extract entities using rule-based patterns."""
        events = self._extract_events(text)
        claims = self._extract_claims(text)
        actions = self._extract_actions(text, events, claims)

        return ExtractorResult(
            events=events,
            claims=claims,
            actions=actions,
            confidence=0.7 if events or claims or actions else 0.3,
            extractor_name=self.name,
            raw_extractions=[{"text": text, "mode": mode}],
        )

    def _extract_events(self, text: str) -> list[dict[str, Any]]:
        """Extract event candidates."""
        events = []
        text_lower = text.lower()

        # Check for registration events
        if "registered" in text_lower:
            # Try to extract details
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
            user_match = re.search(r"[Uu]ser\s+(\w+)", text)

            event = {
                "type": "UserRegistered",
                "data": {},
                "source_system": "extraction",
            }
            if email_match:
                event["data"]["email"] = email_match.group()
            if user_match:
                event["data"]["user_id"] = user_match.group(1)
            events.append(event)

        # Check for other event types
        for event_type in ["created", "updated", "deleted", "completed", "started"]:
            if event_type in text_lower:
                events.append({
                    "type": event_type.capitalize(),
                    "data": {},
                    "source_system": "extraction",
                })

        return events

    def _extract_claims(self, text: str) -> list[dict[str, Any]]:
        """Extract claim candidates."""
        claims = []

        # Look for score/value patterns
        score_pattern = r"(\w+)[_\s]?score\s*(?:=|is|was|of)\s*([\d.]+)"
        for match in re.finditer(score_pattern, text, re.IGNORECASE):
            claims.append({
                "subject": f"{match.group(1).lower()}_score",
                "value": float(match.group(2)),
                "source": "extraction",
                "confidence": 0.8,
            })

        # Look for explicit assignments
        assign_pattern = r"(\w+)\s*=\s*([\d.]+)"
        for match in re.finditer(assign_pattern, text):
            subject = match.group(1).lower()
            # Skip if already captured as score
            if not any(c["subject"] == f"{subject}_score" for c in claims):
                claims.append({
                    "subject": subject,
                    "value": float(match.group(2)),
                    "source": "extraction",
                    "confidence": 0.7,
                })

        # Look for confidence mentions
        conf_pattern = r"confidence\s*(?:of\s*)?([\d.]+)"
        for match in re.finditer(conf_pattern, text, re.IGNORECASE):
            # Attach confidence to last claim if exists
            if claims:
                claims[-1]["confidence"] = float(match.group(1))

        # Look for model sources
        model_pattern = r"(\w+[_\s]?model[_\s]?\w*)"
        for match in re.finditer(model_pattern, text, re.IGNORECASE):
            # Attach source to last claim if exists
            if claims:
                claims[-1]["source"] = match.group(1).replace(" ", "_").lower()

        return claims

    def _extract_actions(
        self,
        text: str,
        events: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract action candidates."""
        actions = []
        text_lower = text.lower()

        # Approval/rejection actions
        for action_type in ["approved", "rejected", "flagged", "processed"]:
            if action_type in text_lower:
                # Build references to other entities
                references = []
                for event in events:
                    references.append({"type": event["type"]})
                for claim in claims:
                    references.append({"subject": claim["subject"]})

                # Extract reasoning if present
                reasoning = None
                reason_patterns = [
                    r"because\s+(.+?)(?:\.|$)",
                    r"since\s+(.+?)(?:\.|$)",
                    r"due to\s+(.+?)(?:\.|$)",
                ]
                for pattern in reason_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        reasoning = match.group(1).strip()
                        break

                actions.append({
                    "type": f"{action_type.capitalize()}User",
                    "agent_id": "extracted-agent",
                    "inputs": {},
                    "references": references,
                    "reasoning": reasoning,
                })

        return actions


class LLMExtractor(Extractor):
    """
    LLM-based entity extractor.

    Uses a language model to extract entities from text.
    More flexible than rule-based, handles ambiguity.

    NOTE: This is a stub. Actual implementation would integrate
    with OpenAI, Anthropic, or other LLM providers.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the LLM extractor.

        Args:
            model: LLM model name
            api_key: API key (from env if not provided)
            temperature: Sampling temperature (0 for determinism)
        """
        self._model = model
        self._api_key = api_key
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"llm:{self._model}"

    def extract(
        self,
        text: str,
        mode: str = "description",
        context: Optional[dict[str, Any]] = None,
    ) -> ExtractorResult:
        """
        Extract entities using LLM.

        NOTE: This is a stub implementation.
        Real implementation would call the LLM API.
        """
        # For now, fall back to rule-based
        # TODO: Implement actual LLM extraction
        fallback = RuleBasedExtractor()
        result = fallback.extract(text, mode, context)

        # Mark as LLM-extracted (even though it's stubbed)
        result.extractor_name = self.name
        result.confidence = 0.9  # LLM typically more confident

        return result

    def _build_prompt(self, text: str, mode: str, context: Optional[dict]) -> str:
        """Build the extraction prompt for the LLM."""
        schema_hint = ""
        if context and "schema" in context:
            schema_hint = f"\nKnown types: {context['schema']}"

        return f"""Extract semantic entities from the following text.

Mode: {mode}
{schema_hint}

Text:
{text}

Extract:
1. EVENTS: Facts that happened (immutable)
2. CLAIMS: Assertions with source and confidence
3. ACTIONS: Agent behaviors with dependencies

Output as JSON with keys: events, claims, actions
Each should have: type, data/value, source, references (to other entities)
"""


class HybridExtractor(Extractor):
    """
    Hybrid extractor combining rule-based and LLM approaches.

    Strategy:
    1. Try rule-based first (fast, deterministic)
    2. If low confidence, enhance with LLM
    3. Use LLM for unknown patterns
    """

    def __init__(
        self,
        rule_extractor: Optional[RuleBasedExtractor] = None,
        llm_extractor: Optional[LLMExtractor] = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize hybrid extractor.

        Args:
            rule_extractor: Rule-based extractor
            llm_extractor: LLM extractor
            confidence_threshold: Below this, use LLM enhancement
        """
        self._rule_extractor = rule_extractor or RuleBasedExtractor()
        self._llm_extractor = llm_extractor
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "hybrid"

    def extract(
        self,
        text: str,
        mode: str = "description",
        context: Optional[dict[str, Any]] = None,
    ) -> ExtractorResult:
        """Extract using hybrid approach."""
        # Start with rule-based
        result = self._rule_extractor.extract(text, mode, context)

        # Enhance with LLM if confidence is low
        if result.confidence < self._confidence_threshold and self._llm_extractor:
            llm_result = self._llm_extractor.extract(text, mode, context)
            result = self._merge_results(result, llm_result)

        result.extractor_name = self.name
        return result

    def _merge_results(
        self,
        rule_result: ExtractorResult,
        llm_result: ExtractorResult,
    ) -> ExtractorResult:
        """Merge results from both extractors."""
        # Simple merge: prefer LLM for new entities, keep rule-based for confirmed
        merged_events = rule_result.events + [
            e for e in llm_result.events
            if e not in rule_result.events
        ]
        merged_claims = rule_result.claims + [
            c for c in llm_result.claims
            if c not in rule_result.claims
        ]
        merged_actions = rule_result.actions + [
            a for a in llm_result.actions
            if a not in rule_result.actions
        ]

        return ExtractorResult(
            events=merged_events,
            claims=merged_claims,
            actions=merged_actions,
            unknown_types=list(set(rule_result.unknown_types + llm_result.unknown_types)),
            confidence=(rule_result.confidence + llm_result.confidence) / 2,
            extractor_name="hybrid",
        )
