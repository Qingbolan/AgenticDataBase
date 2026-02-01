# coding: utf-8
from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def extract_mentions(text: str, targets: List[str]) -> Dict[str, bool]:
    """
    Very lightweight mention detector.
    - Case-insensitive
    - Simple substring match
    """
    t = _normalize(text)
    out: Dict[str, bool] = {}
    for x in targets:
        key = (x or "").strip()
        if not key:
            continue
        out[key] = _normalize(key) in t
    return out


def first_mention_rank(text: str, targets: List[str]) -> Dict[str, int]:
    """
    Rank is 1-based sentence index of first mention. -1 if not found.
    """
    sentences = re.split(r"[.!?。！？]+", text or "")
    out: Dict[str, int] = {t: -1 for t in targets if (t or "").strip()}
    for idx, s in enumerate(sentences):
        s_norm = _normalize(s)
        for t in list(out.keys()):
            if out[t] != -1:
                continue
            if _normalize(t) in s_norm:
                out[t] = idx + 1
    return out


