# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Oracle - Multi-platform LLM client with unified interface.

Supports:
- OpenAI (GPT-4, GPT-4o, etc.)
- DeepInfra (Llama, Mixtral)
- Doubao (字节跳动豆包)
- Anthropic (Claude)
- Google Gemini
"""
from __future__ import annotations

import json
import os
import logging
from typing import List, Optional

from .parallel import ParallelProcessor
from .token_usage_logger import get_token_logger
from openai import OpenAI

# Disable verbose HTTP logging
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.ERROR)



class OracleError(Exception):
    """Oracle error."""
    pass


class InvalidResponseError(OracleError):
    """Raised when LLM returns invalid response format (e.g., search query instead of answer)."""
    pass


def _is_invalid_response(text: str) -> bool:
    """
    Check if response is invalid format (e.g., search query JSON instead of actual answer).

    Only returns True when the response appears to be a malformed LLM response
    that contains BOTH search_query AND response_length fields - this pattern
    indicates the LLM returned its internal query structure instead of answering.

    This is more conservative than checking for either field alone, which could
    cause false positives if users legitimately need JSON with these field names.
    """
    if not text or len(text) < 10:
        return False

    text = text.strip()
    if not text.startswith("{"):
        return False

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Require BOTH fields to be present - more conservative check
            # to avoid false positives with legitimate JSON responses
            if "search_query" in data and "response_length" in data:
                return True
    except (json.JSONDecodeError, ValueError):
        pass

    return False


class Oracle(ParallelProcessor):
    """
    Unified LLM client supporting multiple platforms.

    Usage:
        oracle = Oracle("gpt-4o")
        response = oracle.query("You are helpful.", "Hello!")
    """

    # OpenAI models
    MODEL_GPT4o_MINI = "gpt-4o-mini"
    MODEL_GPT4o = "gpt-4o"
    MODEL_GPT4_TURBO = "gpt-4-turbo"

    # DeepInfra models
    MODEL_LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_LLAMA_3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
    MODEL_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Doubao models (字节跳动豆包/火山引擎)
    MODEL_DOUBAO_SEED = "doubao-seed-1-6-251015"
    MODEL_DEEPSEEK_V3 = "deepseek-v3-2-251201"

    # Anthropic models
    MODEL_CLAUDE_3_OPUS = "claude-3-opus-20240229"
    MODEL_CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    MODEL_CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    MODEL_CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"

    # Gemini models
    MODEL_GEMINI_3_FLASH = "gemini-3-flash-preview"
    MODEL_GEMINI_25_FLASH = "gemini-2.5-flash"
    MODEL_GEMINI_25_PRO = "gemini-2.5-pro"

    # Platform base URLs
    DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
    DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_name: str = "unknown",
    ):
        """
        Initialize Oracle with specified model.

        Args:
            model: Model name/ID
            api_key: API key (uses env var if not specified)
            base_url: Base URL (uses default if not specified)
            agent_name: Name of the agent using this Oracle (for billing logs)
        """
        super().__init__()
        self.model = model
        self.model_name = model  # Alias for compatibility
        self.agent_name = agent_name
        self._api_key = api_key
        self._base_url = base_url
        self._token_logger = get_token_logger()
        self._init_openai_compatible()

    def _init_openai_compatible(self):
        """Initialize OpenAI-compatible client."""
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        base_url = self._base_url or os.getenv("BASE_URL", "https://api.poixe.com/v1")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def query(
        self,
        prompt_sys: str,
        prompt_user: str,
        temp: float = 0.0,
        top_p: float = 0.9,
        logprobs: bool = False,
        query_key: Optional[str] = None,
        method_name: str = "query",
    ) -> str:
        """
        Query the model with system and user prompts.

        Args:
            prompt_sys: System prompt
            prompt_user: User prompt
            temp: Temperature (0.0 - 1.0)
            top_p: Top-p sampling parameter
            logprobs: Whether to return log probabilities (OpenAI only)
            query_key: Optional key for the query
            method_name: Name of the calling method (for billing logs)

        Returns:
            Model response text
        """
        result = ""
        try:
            # All requests go through one api (OpenAI-compatible)
            result = self._query_openai(prompt_sys, prompt_user, temp, top_p, logprobs)

            # Log token usage
            self._token_logger.log_usage(
                agent_name=self.agent_name,
                method=method_name,
                model=self.model,
                input_text=prompt_user,
                output_text=result,
                system_prompt=prompt_sys,
            )

            return result
        except InvalidResponseError:
            # Let InvalidResponseError propagate for retry
            raise
        except Exception as e:
            return f"QUERY_FAILED: {str(e)}"

    def _query_openai(
        self,
        prompt_sys: str,
        prompt_user: str,
        temp: float,
        top_p: float,
        logprobs: bool,
    ) -> str:
        """Query OpenAI-compatible API."""
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_sys},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=temp,
                top_p=top_p,
                logprobs=logprobs,
            )
            result = completion.choices[0].message.content or ""
        except Exception:
            # Retry without logprobs
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_sys},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=temp,
                top_p=top_p,
            )
            result = completion.choices[0].message.content or ""

        # Validate response format
        if _is_invalid_response(result):
            raise InvalidResponseError(f"Invalid response format: {result[:200]}...")

        return result

    def query_all(
        self,
        prompt_sys: str,
        prompt_user_all: List[str],
        workers: Optional[int] = None,
        temp: float = 0.0,
        top_p: float = 0.9,
        query_key_list: Optional[List[str]] = None,
        batch_size: int = 10,
        max_retries: int = 2,
        timeout: int = 3000,
        **kwargs,
    ) -> List[str]:
        """
        Query all prompts in parallel.

        Args:
            prompt_sys: System prompt
            prompt_user_all: List of user prompts
            workers: Number of worker threads
            temp: Temperature
            top_p: Top-p sampling parameter
            query_key_list: Optional list of query keys
            batch_size: Batch size for processing
            max_retries: Maximum retries per query
            timeout: Timeout in seconds

        Returns:
            List of model responses
        """
        query_key_list = query_key_list or []

        query_items = []
        for i, prompt in enumerate(prompt_user_all):
            key = query_key_list[i] if i < len(query_key_list) else None
            query_items.append((prompt, key))

        def process_func(item, prompt_sys=prompt_sys, temp=temp, top_p=top_p):
            prompt, key = item
            try:
                return self.query(prompt_sys, prompt, temp, top_p, query_key=key, method_name="query_all")
            except Exception as e:
                return f"QUERY_FAILED: {str(e)}"

        workers = workers or min(32, (os.cpu_count() or 4) * 4)

        return self.parallel_process(
            items=query_items,
            process_func=process_func,
            workers=workers,
            batch_size=batch_size,
            max_retries=max_retries,
            timeout=timeout,
            task_description=f"Processing queries ({self.model})",
        )

    def __repr__(self) -> str:
        return f"Oracle(model={self.model})"
