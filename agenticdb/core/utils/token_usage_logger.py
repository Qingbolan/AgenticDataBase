# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Token Usage Logger for LLM billing tracking.

Records all LLM API calls with token counts to a log file for cost analysis.
Uses tiktoken for accurate token counting across different models.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from functools import lru_cache

import tiktoken


class TokenUsageLogger:
    """
    Logger for tracking LLM token usage and API calls.

    Records to a dedicated log file with the following fields:
    - timestamp: ISO format datetime
    - agent_name: Name of the calling agent
    - method: Query method (query, query_batch, etc.)
    - model: LLM model name
    - input_tokens: Number of input tokens
    - output_tokens: Number of output tokens
    - total_tokens: Sum of input and output tokens
    """

    _instance: Optional["TokenUsageLogger"] = None

    # Default log file path
    DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
    DEFAULT_LOG_FILE = "token_usage.log"

    # Model to tiktoken encoding mapping
    MODEL_ENCODING_MAP = {
        # OpenAI models
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        # Claude models (use cl100k_base as approximation)
        "claude": "cl100k_base",
        # Gemini models (use cl100k_base as approximation)
        "gemini": "cl100k_base",
        # Other models (use cl100k_base as default)
        "default": "cl100k_base",
    }

    def __new__(cls, log_dir: Optional[Path] = None):
        """Singleton pattern to ensure single log file."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the token usage logger.

        Args:
            log_dir: Directory for log files. Defaults to Backend/logs/
        """
        if self._initialized:
            return

        self._initialized = True
        self._log_dir = log_dir or self.DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._log_file = self._log_dir / self.DEFAULT_LOG_FILE

        # Setup dedicated file logger
        self._logger = logging.getLogger("token_usage")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        # File handler with JSON lines format
        file_handler = logging.FileHandler(self._log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self._logger.propagate = False

    @lru_cache(maxsize=10)
    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """
        Get tiktoken encoding for a model.

        Args:
            model: Model name

        Returns:
            tiktoken Encoding object
        """
        model_lower = model.lower()

        # Find matching encoding
        encoding_name = self.MODEL_ENCODING_MAP["default"]
        for key, enc in self.MODEL_ENCODING_MAP.items():
            if key in model_lower:
                encoding_name = enc
                break

        return tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model: Model name for encoding selection

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            encoding = self._get_encoding(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ~ 4 chars for English, ~2 for Chinese)
            return len(text) // 3

    def log_usage(
        self,
        agent_name: str,
        method: str,
        model: str,
        input_text: str,
        output_text: str,
        system_prompt: str = "",
    ) -> dict:
        """
        Log a single LLM API call with token counts.

        Args:
            agent_name: Name of the calling agent
            method: Query method name
            model: LLM model name
            input_text: User input/prompt text
            output_text: Model response text
            system_prompt: System prompt (counted as input)

        Returns:
            Dict with logged data including token counts
        """
        # Count tokens
        input_tokens = self.count_tokens(system_prompt + input_text, model)
        output_tokens = self.count_tokens(output_text, model)
        total_tokens = input_tokens + output_tokens

        # Build log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "method": method,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        # Write to log file as JSON line
        self._logger.info(json.dumps(log_entry, ensure_ascii=False))

        return log_entry

    def get_log_file_path(self) -> Path:
        """Get the path to the log file."""
        return self._log_file


# Global singleton instance
_token_logger: Optional[TokenUsageLogger] = None


def get_token_logger() -> TokenUsageLogger:
    """Get the global TokenUsageLogger instance."""
    global _token_logger
    if _token_logger is None:
        _token_logger = TokenUsageLogger()
    return _token_logger
