"""Lightweight rule-based sentiment for headlines as a fallback to LLM extraction."""
from __future__ import annotations

import re

POS = {
    "beat", "beats", "soars", "surge", "rally", "record", "growth", "strong",
    "upgrade", "raises", "bullish", "wins", "approved",
}
NEG = {
    "miss", "misses", "plunge", "tumble", "lawsuit", "investigation", "downgrade",
    "cut", "warns", "bearish", "fraud", "halts", "denies", "recall",
}

_TOK = re.compile(r"[A-Za-z']+")


def naive_sentiment(text: str | None) -> float:
    """Returns sentiment in [-1, 1] from a headline. Crude lexicon-based fallback."""
    if not text:
        return 0.0
    toks = [t.lower() for t in _TOK.findall(text)]
    pos = sum(1 for t in toks if t in POS)
    neg = sum(1 for t in toks if t in NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)
