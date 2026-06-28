"""Near-duplicate headline filtering — dependency-free.

Wire stories get syndicated verbatim across dozens of outlets; exact-key dedup
(what the old fetcher did) keeps them all and over-weights whatever the wires
happened to push that day.  This collapses near-duplicates per ticker using a
token-set Jaccard similarity, which needs no embeddings / external libs and is
plenty for headline-level dedup.

``dedup_articles`` keeps the *first* occurrence of each cluster (callers pass
articles newest-first when recency should win).
"""
from __future__ import annotations

import re
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an and or of to for in on at by with from as is are was be this that "
    "its it s inc corp co ltd plc said says new update".split()
)


def _tokens(text: str | None) -> frozenset[str]:
    if not text:
        return frozenset()
    return frozenset(t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def dedup_articles(
    articles: Sequence[dict],
    threshold: float = 0.90,
    title_key: str = "title",
) -> list[dict]:
    """Drop near-duplicate articles (per ticker) by title Jaccard >= threshold.

    Dedup is scoped within a ticker so the same headline mentioning two names is
    kept for each.  Order is preserved; the first article in a cluster wins.
    """
    kept: list[dict] = []
    # token-sets of survivors, grouped by ticker so cross-ticker pairs never merge
    survivor_tokens: dict[str, list[frozenset[str]]] = {}
    for art in articles:
        tk = (art.get("ticker") or "_").upper()
        toks = _tokens(art.get(title_key))
        bucket = survivor_tokens.setdefault(tk, [])
        if any(jaccard(toks, prev) >= threshold for prev in bucket):
            continue
        bucket.append(toks)
        kept.append(art)
    return kept
