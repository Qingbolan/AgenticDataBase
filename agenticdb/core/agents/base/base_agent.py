# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Base Agent for AgenticDB.

Provides unified interface for all LLM-based agents via Oracle.

IMPORTANT: This is for LLM agents only.
For pure computation, use classes in core/tools instead.
For prompt assembly without LLM, use core/tools/prompt_builder.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

from ...tools.prompt_loader import PromptLoader
from ...utils.oracle import Oracle
from ...utils.logger import ModernLogger


# Type variable for agent output types
T = TypeVar("T")


@dataclass
class AgentContext:
    """Context for agent execution."""
    session_id: Optional[str] = None
    branch_id: Optional[str] = None
    language: str = "en"
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ModernLogger, Generic[T]):
    """
    Base class for LLM-based agents in AgenticDB.

    All agents that use LLM should inherit from this class.
    For pure computation or prompt-only operations, use core/tools instead.

    Usage:
        class MyAgent(BaseAgent[MyOutput]):
            name = "my_agent"

            def run(self, ctx: AgentContext, text: str) -> MyOutput:
                prompt = self._load_prompt("my_prompt.md")
                response = self.query(prompt, text)
                return self._parse_response(response)

    Architecture:
        - BaseAgent provides Oracle (LLM client) and optional prompt loading
        - All LLM calls go through Oracle (self._oracle)
        - Subclasses implement specific business logic via run()
    """

    name: str = "base"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize BaseAgent with LLM support.

        Args:
            model: LLM model name (defaults to DEFAULT_MODEL)
            prompts_dir: Optional path to prompts directory for loading templates
            temperature: Default temperature for LLM calls

        Raises:
            RuntimeError: If Oracle initialization fails (e.g., missing API key)
        """
        super().__init__()
        # Initialize Oracle (LLM client)
        self._model_name = model or self.DEFAULT_MODEL
        self._temperature = temperature
        self._oracle: Optional[Oracle] = None
        self.system_prompt = ""

        self._oracle = Oracle(
            model=self._model_name,
            agent_name=self.name,
        )

        # Optional prompt loading
        self.prompts_dir = prompts_dir
        self._prompt_loader = PromptLoader(prompts_dir) if prompts_dir else None

    # =========================================================================
    # LLM Query Methods
    # =========================================================================

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Query LLM with system and user prompts.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input/question
            temperature: LLM temperature (0.0-1.0), uses default if not specified

        Returns:
            LLM response text

        Raises:
            RuntimeError: If Oracle not initialized
        """
        if not self._oracle:
            raise RuntimeError("Oracle not initialized")
        temp = temperature if temperature is not None else self._temperature
        return self._oracle.query(system_prompt, user_prompt, temp)

    def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Query LLM and parse response as JSON.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input/question
            temperature: LLM temperature

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If Oracle not initialized
            ValueError: If response is not valid JSON
        """
        response = self.query(system_prompt, user_prompt, temperature)
        return self._parse_json(response)

    def query_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Query LLM with multiple user prompts in parallel.

        Args:
            system_prompt: System/instruction prompt (shared)
            user_prompts: List of user inputs
            temperature: LLM temperature

        Returns:
            List of LLM responses

        Raises:
            RuntimeError: If Oracle not initialized
        """
        if not self._oracle:
            raise RuntimeError("Oracle not initialized")
        temp = temperature if temperature is not None else self._temperature
        return self._oracle.query_all(system_prompt, user_prompts, temp)

    # =========================================================================
    # Legacy Compatibility Methods
    # =========================================================================

    def answer(self, question: str) -> str:
        """
        Query LLM using instance's system_prompt.

        Legacy method for backward compatibility.
        Prefer using query() with explicit system prompt.
        """
        return self.query(self.system_prompt, question)

    def answer_multiple(self, questions: List[str]) -> List[str]:
        """
        Query LLM with multiple questions using instance's system_prompt.

        Legacy method for backward compatibility.
        Prefer using query_batch() with explicit system prompt.
        """
        return self.query_batch(self.system_prompt, questions)

    # =========================================================================
    # Prompt Loading Methods
    # =========================================================================

    def _load_prompt(self, filename: str) -> str:
        """
        Load prompt template from file.

        Args:
            filename: Name of the prompt file

        Returns:
            Prompt content

        Raises:
            RuntimeError: If prompts_dir not configured
        """
        if not self._prompt_loader:
            raise RuntimeError("prompts_dir not configured")
        return self._prompt_loader.load(filename)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def model(self) -> Optional[Oracle]:
        """Get Oracle client."""
        return self._oracle

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    # =========================================================================
    # JSON Parsing Helpers
    # =========================================================================

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        Args:
            text: Raw LLM response

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {text[:500]}")

    def _extract_list(self, data: Dict[str, Any], key: str) -> List[Any]:
        """
        Safely extract a list from parsed JSON.

        Args:
            data: Parsed JSON dict
            key: Key to extract

        Returns:
            List value or empty list
        """
        value = data.get(key, [])
        return value if isinstance(value, list) else []

    # =========================================================================
    # Abstract Methods (for subclasses to override)
    # =========================================================================

    def run(self, ctx: AgentContext, **kwargs) -> T:
        """
        Run the agent with context.

        Override this method in subclasses.

        Args:
            ctx: Agent context with session info
            **kwargs: Additional arguments

        Returns:
            Agent result of type T
        """
        raise NotImplementedError("Subclasses must implement run()")
