"""Point-in-time RAG over stored news/filings — relevance before the LLM.

The old flow sent *every* headline straight to the LLM, so noise (ambiguous-
ticker false positives, syndicated wire dupes) cost tokens and polluted the
signal.  This adds a thin retrieval layer:

  * Index the stored events for a ticker, **filtered to ``known_at <= as_of``**
    so retrieval is leakage-safe (you can only retrieve what was known then).
  * Rank by relevance to a query (default: the ticker + a finance lexicon) and
    return the top-k snippets to feed the LLM as grounded context.

Default backend is **sklearn TF-IDF** (already a dependency — works out of the
box).  If ``sentence-transformers`` is installed, set ``backend="embed"`` for
semantic retrieval with a local model (BGE-style); it degrades to TF-IDF if the
import fails.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

_DEFAULT_QUERY_TERMS = (
    "earnings guidance revenue margin outlook downgrade upgrade lawsuit "
    "regulatory acquisition demand growth risk forecast"
)


def _as_naive_utc(dt) -> datetime:
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
    return datetime.utcnow()


def _tfidf_rank(docs: list[str], query: str, k: int) -> list[tuple[int, float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    mat = vec.fit_transform(docs + [query])
    sims = linear_kernel(mat[-1], mat[:-1]).ravel()
    order = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


def _embed_rank(docs: list[str], query: str, k: int) -> list[tuple[int, float]] | None:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except Exception:
        return None
    try:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        emb = model.encode(docs + [query], normalize_embeddings=True)
        sims = emb[:-1] @ emb[-1]
        order = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]
    except Exception as e:
        logger.debug(f"embedding retrieval failed, using TF-IDF: {e}")
        return None


def retrieve_ticker_news(
    events: pl.DataFrame,
    ticker: str,
    as_of: datetime | None = None,
    k: int = 6,
    query: str | None = None,
    backend: str = "tfidf",
) -> list[dict[str, Any]]:
    """Return the top-k most relevant, point-in-time-safe events for a ticker."""
    if events is None or events.is_empty():
        return []
    ticker = ticker.upper()
    as_of = as_of or datetime.utcnow()
    as_of_naive = _as_naive_utc(as_of)

    sub = (
        events.explode("tickers")
        .filter(pl.col("tickers") == ticker)
        .with_columns(pl.col("known_at").dt.replace_time_zone(None).alias("_ka"))
        .filter(pl.col("_ka") <= as_of_naive)
        .sort("_ka", descending=True)
    )
    if sub.is_empty():
        return []

    rows = sub.to_dicts()
    docs = [f"{r.get('summary') or ''}. {(r.get('content') or '')[:600]}" for r in rows]
    q = query or f"{ticker} {_DEFAULT_QUERY_TERMS}"

    ranked = None
    if backend == "embed":
        ranked = _embed_rank(docs, q, k)
    if ranked is None:
        try:
            ranked = _tfidf_rank(docs, q, k)
        except Exception as e:
            logger.debug(f"TF-IDF retrieval failed for {ticker}: {e}")
            ranked = [(i, 0.0) for i in range(min(k, len(rows)))]  # recency fallback

    out = []
    for idx, score in ranked:
        r = rows[idx]
        out.append({
            "known_at": str(r.get("known_at")),
            "source": r.get("source"),
            "summary": r.get("summary"),
            "content": (r.get("content") or "")[:600],
            "source_url": r.get("source_url"),
            "relevance": round(score, 4),
        })
    return out


def build_context_block(retrieved: list[dict[str, Any]], max_chars: int = 2400) -> str:
    """Concatenate retrieved snippets into a compact LLM context block."""
    parts: list[str] = []
    used = 0
    for i, r in enumerate(retrieved, start=1):
        snippet = f"{i}. [{r.get('known_at','')[:10]}] {r.get('summary','')}"
        body = r.get("content") or ""
        if body:
            snippet += f" — {body[:300]}"
        if used + len(snippet) > max_chars:
            break
        parts.append(snippet)
        used += len(snippet)
    return "\n".join(parts)
