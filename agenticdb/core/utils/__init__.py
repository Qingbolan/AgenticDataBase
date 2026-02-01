# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Core utilities for GEO-SCOPE.
"""
from typing import Any, Dict, List, Optional
import re
import json
from pathlib import Path

from .oracle import Oracle, OracleError


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON content from text that may contain markdown code blocks."""
    text = text.strip()

    # Try to find JSON in code block
    if "```" in text:
        lines = text.split("\n")
        in_block = False
        json_lines = []
        for line in lines:
            if line.startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            return "\n".join(json_lines)

    # Try direct parse
    if text.startswith("[") or text.startswith("{"):
        return text

    return None


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    # Oracle LLM client
    "Oracle",
    "OracleError",
    # Utility functions
    "slugify",
    "safe_json_loads",
    "extract_json_from_text",
    "chunk_list",
    "ensure_dir",
    "merge_dicts",
]
