# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Prompt Builder - Assembles system prompts for AI providers.

NOT an agent - does not use LLM.
Pure tool for loading and assembling prompts from files.

Design principle: Single source of truth
- All prompts MUST come from prompts/ directory
- NO fallback dicts, NO hardcoded prompts
- If file missing → raise error (fail fast)
"""
from pathlib import Path
from typing import Dict, Optional
import logging

from .prompt_loader import PromptLoader, PromptLoaderError

logger = logging.getLogger(__name__)


class PromptBuilderError(Exception):
    """Prompt building error."""
    pass


class PromptBuilder:
    """
    Builder for AI provider system prompts.

    Loads prompts from markdown files and assembles them with
    brand context and language instructions.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build("chatgpt", brand_context="e签宝...", language="zh")
    """

    # Provider → prompt file mapping
    # All files must exist in prompts/polling/
    # Language-specific versions use suffix: polling_chatgpt_zh.md, polling_chatgpt_en.md, etc.
    PROVIDER_PROMPTS = {
        "chatgpt": "polling_chatgpt.md",
        "claude": "polling_claude.md",
        "deepseek": "polling_deepseek.md",
        "doubao": "polling_doubao.md",
    }

    # Supported languages: zh, en, ja, ko, fr, de, es
    SUPPORTED_LANGUAGES = ["zh", "en", "ja", "ko", "fr", "de", "es"]

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize PromptBuilder.

        Args:
            prompts_dir: Path to polling prompts directory.
                         Defaults to core/prompts/polling/
        """
        self.prompts_dir = prompts_dir or Path(__file__).parent.parent / "prompts" / "polling"
        self._loader = PromptLoader(self.prompts_dir)
        self._cache: Dict[str, str] = {}

        # Validate all prompt files exist on init
        self._validate_prompts()

    def _validate_prompts(self):
        """Validate all required prompt files exist."""
        missing = []
        for provider, filename in self.PROVIDER_PROMPTS.items():
            filepath = self.prompts_dir / filename
            if not filepath.exists():
                missing.append(f"{provider}: {filepath}")

        if missing:
            logger.warning(f"Missing prompt files: {missing}")
            # Don't raise on init, just warn - files might be created later

    def build(
        self,
        provider: str,
        brand_context: str = "",
        language: str = "zh",
    ) -> str:
        """
        Build complete system prompt for a provider.

        Args:
            provider: Provider name (chatgpt, claude, deepseek, doubao)
            brand_context: Brand summary to include in the prompt
            language: Response language code (zh, en, ja, ko)

        Returns:
            Complete system prompt string

        Raises:
            PromptBuilderError: If provider unknown or prompt file missing
        """
        # Validate provider
        if provider not in self.PROVIDER_PROMPTS:
            raise PromptBuilderError(
                f"Unknown provider: {provider}. "
                f"Supported: {list(self.PROVIDER_PROMPTS.keys())}"
            )

        # Load base prompt for the specified language (with cache)
        base_prompt = self._load_base_prompt(provider, language)

        # Build complete prompt
        parts = [base_prompt]

        # Add brand context if provided
        if brand_context and brand_context.strip():
            context_label = self._get_context_label(language)
            parts.append(f"{context_label}\n{brand_context}")

        return "\n\n".join(parts)

    def _load_base_prompt(self, provider: str, language: str = "zh") -> str:
        """
        Load base prompt from file with language support.

        Tries to load language-specific version first (e.g., polling_chatgpt_zh.md),
        falls back to default version (e.g., polling_chatgpt.md) if not found.

        Args:
            provider: Provider name
            language: Language code (zh, en, ja, ko)

        Returns:
            Base prompt content

        Raises:
            PromptBuilderError: If prompt file not found or unreadable
        """
        base_filename = self.PROVIDER_PROMPTS[provider]
        # Build language-specific filename: polling_chatgpt.md -> polling_chatgpt_zh.md
        name_without_ext = base_filename.rsplit('.', 1)[0]
        lang_filename = f"{name_without_ext}_{language}.md"

        # Cache key includes language
        cache_key = f"{provider}_{language}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try language-specific file first
        lang_filepath = self.prompts_dir / lang_filename
        if lang_filepath.exists():
            try:
                content = self._loader.load(lang_filename)
                self._cache[cache_key] = content
                logger.debug(f"Loaded language-specific prompt: {lang_filename}")
                return content
            except PromptLoaderError:
                pass  # Fall through to default

        # Fall back to default file
        try:
            content = self._loader.load(base_filename)
            self._cache[cache_key] = content
            logger.debug(f"Loaded default prompt: {base_filename}")
            return content
        except PromptLoaderError as e:
            raise PromptBuilderError(
                f"Failed to load prompt for {provider}: {e}. "
                f"Expected file: {self.prompts_dir / base_filename}"
            ) from e

    def _get_context_label(self, language: str) -> str:
        """Get context section label for language."""
        labels = {
            "zh": "相关背景信息：",
            "en": "Relevant context:",
            "ja": "関連情報：",
            "ko": "관련 정보:",
            "fr": "Contexte pertinent :",
            "de": "Relevanter Kontext:",
            "es": "Contexto relevante:",
        }
        return labels.get(language, labels["en"])

    def get_all_prompts(self, language: str = "zh") -> Dict[str, str]:
        """
        Get all base prompts for a specific language.

        Args:
            language: Language code (zh, en, ja, ko)

        Returns:
            Dictionary mapping provider names to their base prompts

        Raises:
            PromptBuilderError: If any prompt file is missing
        """
        prompts = {}
        for provider in self.PROVIDER_PROMPTS:
            prompts[provider] = self._load_base_prompt(provider, language)
        return prompts

    def clear_cache(self):
        """Clear the prompt cache to force reload on next access."""
        self._cache.clear()

    @property
    def supported_providers(self) -> list:
        """List of supported provider names."""
        return list(self.PROVIDER_PROMPTS.keys())
