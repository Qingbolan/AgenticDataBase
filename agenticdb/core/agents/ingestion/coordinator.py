# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Ingestion Coordinator - Orchestrates the text → entities pipeline.

This coordinator runs the extraction agents in the optimal order and
combines their results into a unified ingestion result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..base.base_agent import AgentContext
from .action_extractor_agent import ActionExtractorAgent
from .claim_extractor_agent import ClaimExtractorAgent
from .dependency_inference_agent import DependencyInferenceAgent
from .event_extractor_agent import EventExtractorAgent
from .types import IngestionResult


class IngestionCoordinator:
    """
    Coordinator for the ingestion pipeline.

    Orchestrates the following flow:
    1. EventExtractorAgent: Extract events from text
    2. ClaimExtractorAgent: Extract claims from text
    3. ActionExtractorAgent: Extract actions from text
    4. DependencyInferenceAgent: Infer relationships

    The coordinator can run extractors in parallel for performance,
    then runs dependency inference on the combined results.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Ingestion Coordinator.

        Args:
            model: LLM model name (shared across all agents)
            prompts_dir: Path to prompts directory
        """
        self.model = model
        self.prompts_dir = prompts_dir

        # Initialize agents
        self._event_extractor = EventExtractorAgent(
            model=model, prompts_dir=prompts_dir
        )
        self._claim_extractor = ClaimExtractorAgent(
            model=model, prompts_dir=prompts_dir
        )
        self._action_extractor = ActionExtractorAgent(
            model=model, prompts_dir=prompts_dir
        )
        self._dependency_inference = DependencyInferenceAgent(
            model=model, prompts_dir=prompts_dir
        )

    def ingest(
        self,
        text: str,
        ctx: Optional[AgentContext] = None,
        infer_dependencies: bool = True,
    ) -> IngestionResult:
        """
        Ingest text and extract all semantic entities.

        Args:
            text: Input text to process
            ctx: Agent context (optional)
            infer_dependencies: Whether to run dependency inference

        Returns:
            IngestionResult with all extracted entities and edges
        """
        if ctx is None:
            ctx = AgentContext()

        # Run extractors (could be parallelized in production)
        event_result = self._event_extractor.run(ctx, text)
        claim_result = self._claim_extractor.run(ctx, text)
        action_result = self._action_extractor.run(ctx, text)

        # Combine extracted entities
        all_events = event_result.events
        all_claims = claim_result.claims
        all_actions = action_result.actions

        # Infer dependencies if requested
        all_edges = []
        edge_confidence = 0.0

        if infer_dependencies and (all_events or all_claims or all_actions):
            edge_result = self._dependency_inference.run(
                ctx, all_events, all_claims, all_actions
            )
            all_edges = edge_result.edges
            edge_confidence = edge_result.confidence

        return IngestionResult(
            events=all_events,
            claims=all_claims,
            actions=all_actions,
            edges=all_edges,
            event_confidence=event_result.confidence,
            claim_confidence=claim_result.confidence,
            action_confidence=action_result.confidence,
            edge_confidence=edge_confidence,
        )

    def extract_events(
        self,
        text: str,
        ctx: Optional[AgentContext] = None,
    ) -> IngestionResult:
        """
        Extract only events from text.

        Args:
            text: Input text
            ctx: Agent context

        Returns:
            IngestionResult with extracted events
        """
        if ctx is None:
            ctx = AgentContext()

        result = self._event_extractor.run(ctx, text)
        return IngestionResult(
            events=result.events,
            event_confidence=result.confidence,
        )

    def extract_claims(
        self,
        text: str,
        ctx: Optional[AgentContext] = None,
    ) -> IngestionResult:
        """
        Extract only claims from text.

        Args:
            text: Input text
            ctx: Agent context

        Returns:
            IngestionResult with extracted claims
        """
        if ctx is None:
            ctx = AgentContext()

        result = self._claim_extractor.run(ctx, text)
        return IngestionResult(
            claims=result.claims,
            claim_confidence=result.confidence,
        )

    def extract_actions(
        self,
        text: str,
        ctx: Optional[AgentContext] = None,
    ) -> IngestionResult:
        """
        Extract only actions from text.

        Args:
            text: Input text
            ctx: Agent context

        Returns:
            IngestionResult with extracted actions
        """
        if ctx is None:
            ctx = AgentContext()

        result = self._action_extractor.run(ctx, text)
        return IngestionResult(
            actions=result.actions,
            action_confidence=result.confidence,
        )
