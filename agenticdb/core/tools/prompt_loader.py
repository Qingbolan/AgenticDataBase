# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Prompt Loader - Unified prompt loading with language fallback support.

All agents should use this class to load prompts, ensuring consistent:
- Encoding handling (UTF-8)
- Language fallback (zh → en → default)
- Error handling for missing files
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptLoaderError(Exception):
    """Error loading prompt file."""
    pass


class PromptLoader:
    """
    Loads prompt templates with language fallback support.

    Usage:
        loader = PromptLoader(Path("prompts"))
        prompt = loader.load("result_analysis.md")

        # With language fallback
        prompt = loader.load_with_fallback("result_analysis", language="zh")
    """

    # Supported encodings to try in order
    ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "latin-1"]

    def __init__(self, prompts_dir: Path):
        """
        Initialize PromptLoader.

        Args:
            prompts_dir: Path to the prompts directory
        """
        self.prompts_dir = prompts_dir
        self._cache: dict = {}

    def load(self, name: str, use_cache: bool = True) -> str:
        """
        Load a prompt file by name.

        Args:
            name: Filename of the prompt (e.g., "result_analysis.md")
            use_cache: Whether to use cached content

        Returns:
            Prompt content as string

        Raises:
            PromptLoaderError: If file cannot be loaded
        """
        if use_cache and name in self._cache:
            return self._cache[name]

        path = self.prompts_dir / name

        if not path.exists():
            raise PromptLoaderError(f"Prompt file not found: {path}")

        content = self._read_file(path)

        if use_cache:
            self._cache[name] = content

        return content

    def load_with_fallback(
        self,
        base_name: str,
        language: str = "zh",
        extension: str = ".md",
        use_cache: bool = True,
    ) -> str:
        """
        Load a prompt with language fallback.

        Attempts to load in order:
        1. {base_name}_{language}{extension} (e.g., result_analysis_zh.md)
        2. {base_name}{extension} (e.g., result_analysis.md)

        Args:
            base_name: Base name of the prompt file (without extension)
            language: Language code (zh, en, etc.)
            extension: File extension (default: .md)
            use_cache: Whether to use cached content

        Returns:
            Prompt content as string

        Raises:
            PromptLoaderError: If no matching file can be loaded
        """
        # Try language-specific file first
        lang_specific = f"{base_name}_{language}{extension}"
        try:
            return self.load(lang_specific, use_cache=use_cache)
        except PromptLoaderError:
            pass

        # Try default file
        default = f"{base_name}{extension}"
        try:
            return self.load(default, use_cache=use_cache)
        except PromptLoaderError:
            pass

        # If language was not default, try without language suffix
        if language != "zh":
            zh_specific = f"{base_name}_zh{extension}"
            try:
                return self.load(zh_specific, use_cache=use_cache)
            except PromptLoaderError:
                pass

        raise PromptLoaderError(
            f"No prompt file found for '{base_name}' in {self.prompts_dir}. "
            f"Tried: {lang_specific}, {default}"
        )

    def _read_file(self, path: Path) -> str:
        """
        Read file content with encoding fallback.

        Args:
            path: Path to the file

        Returns:
            File content as string

        Raises:
            PromptLoaderError: If file cannot be read with any encoding
        """
        for encoding in self.ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read {path} with {encoding}: {e}")
                continue

        raise PromptLoaderError(
            f"Failed to read {path} with any of the supported encodings: {self.ENCODINGS}"
        )

    def exists(self, name: str) -> bool:
        """
        Check if a prompt file exists.

        Args:
            name: Filename of the prompt

        Returns:
            True if file exists
        """
        return (self.prompts_dir / name).exists()

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()

    def list_prompts(self, pattern: str = "*.md") -> list:
        """
        List all prompt files matching a pattern.

        Args:
            pattern: Glob pattern (default: *.md)

        Returns:
            List of prompt filenames
        """
        return [p.name for p in self.prompts_dir.glob(pattern)]
