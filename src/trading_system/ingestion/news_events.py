"""News and event ingestion. Returns rows in the structured event schema.

Backend order (configurable via ``cfg['news']['backends']``):
  1. Finnhub company-news  — ticker-tagged, free 60 req/min, cleanest micro signal
  2. Google News RSS       — broad free-text breadth, full-body extraction
  3. NewsAPI               — headline fallback when a key is present

For each ticker the first backend that returns rows wins, so Finnhub handles the
covered names and Google RSS fills gaps (ETFs, foreign listings, niche tickers).
Near-duplicate wire stories are collapsed per ticker before scoring.

The structured rows feed downstream event features and the DeepSeek apprehension
scorer.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import polars as pl
import requests

from ..utils import get_logger
from ..features.sentiment import naive_sentiment
from .dedup import dedup_articles
from .google_news_fetcher import collect_google_news_articles

logger = get_logger(__name__)


EVENT_SCHEMA = {
    "event_id": pl.Utf8,
    "source": pl.Utf8,
    "source_url": pl.Utf8,
    "published_at": pl.Datetime("us", "UTC"),
    "known_at": pl.Datetime("us", "UTC"),
    "tickers": pl.List(pl.Utf8),
    "sectors": pl.List(pl.Utf8),
    "event_type": pl.Utf8,
    "sentiment": pl.Float64,
    "confidence": pl.Float64,
    "novelty": pl.Float64,
    "magnitude": pl.Float64,
    "time_horizon": pl.Utf8,
    "summary": pl.Utf8,
    "content": pl.Utf8,
    "risk_flags": pl.List(pl.Utf8),
}

# Per-backend baseline confidence: ticker-tagged sources are trusted more than
# free-text matches, before any LLM enrichment overwrites these.
_BACKEND_CONFIDENCE = {"finnhub": 0.6, "google_news": 0.45, "newsapi": 0.4}


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def _rows_to_events(articles: list[dict], now: datetime) -> pl.DataFrame:
    """Map generic backend article dicts → EVENT_SCHEMA rows."""
    rows = []
    for art in articles:
        title = (art.get("title") or "")[:500]
        content = (art.get("content") or "")[:8000]
        backend = art.get("backend", art.get("source", "news"))
        rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "source": backend,
                "source_url": art.get("source_url", ""),
                "published_at": art.get("published_at") or now,
                "known_at": now,
                "tickers": [(art.get("ticker") or "UNKNOWN").upper()],
                "sectors": [],
                "event_type": "news",
                "sentiment": naive_sentiment(f"{title} {content[:500]}"),
                "confidence": _BACKEND_CONFIDENCE.get(backend, 0.45),
                "novelty": 0.5,
                "magnitude": 0.0,
                "time_horizon": "1d",
                "summary": title,
                "content": content,
                "risk_flags": ["unverified"],
            }
        )
    return pl.DataFrame(rows, schema=EVENT_SCHEMA) if rows else _empty_events()


# ─────────────────────────────────────────────────────────────────────────────
# Individual backends (each returns a list of generic article dicts)
# ─────────────────────────────────────────────────────────────────────────────

def _backend_finnhub(
    tickers: list[str], days: int, max_per_ticker: int,
    cache_dir: Path | None, cache_hours: float,
) -> list[dict]:
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return []
    from .finnhub_news import collect_finnhub_articles
    return collect_finnhub_articles(
        tickers, api_key, days=days, max_per_ticker=max_per_ticker,
        cache_dir=cache_dir, cache_hours=cache_hours,
    )


def _backend_newsdata(
    tickers: list[str], max_per_ticker: int,
    cache_dir: Path | None, cache_hours: float,
) -> list[dict]:
    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        return []
    from .newsdata_news import collect_newsdata_articles
    return collect_newsdata_articles(
        tickers, api_key, max_per_ticker=min(max_per_ticker, 10),
        cache_dir=cache_dir, cache_hours=cache_hours,
    )


def _backend_google(tickers: list[str], days: int, max_per_ticker: int) -> list[dict]:
    arts = collect_google_news_articles(tickers, days=days, max_urls_per_ticker=max_per_ticker)
    for a in arts:
        a.setdefault("backend", "google_news")
    return arts


def _backend_newsapi(tickers: list[str], days: int, api_key: str | None) -> list[dict]:
    api_key = api_key or os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return []
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    from_date = (now - timedelta(days=days)).date().isoformat()
    for t in tickers:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": t, "from": from_date, "language": "en",
                        "pageSize": 25, "apiKey": api_key},
                timeout=20,
            )
            r.raise_for_status()
            for art in r.json().get("articles", []):
                published = art.get("publishedAt")
                try:
                    pdt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except Exception:
                    pdt = now
                rows.append({
                    "ticker": t.upper(),
                    "title": (art.get("title") or "")[:500],
                    "content": (art.get("description") or "")[:8000],
                    "source_url": art.get("url", ""),
                    "published_at": pdt,
                    "backend": "newsapi",
                })
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed for {t}: {e}")
    return rows


def fetch_news(
    tickers: Iterable[str],
    api_key: str | None = None,
    days: int = 7,
    backends: list[str] | None = None,
    cache_dir: Path | None = None,
    cache_hours: float = 6.0,
    dedup_cosine: float = 0.90,
    max_per_ticker: int = 25,
) -> pl.DataFrame:
    """Fetch recent news as structured, deduplicated events.

    Backends are tried in order; per ticker the first backend that yields rows
    wins, so we don't double-count the same name across sources.  Near-duplicate
    headlines are collapsed before mapping to the event schema.
    """
    tickers = [t.upper() for t in tickers]
    backends = backends or ["finnhub", "newsdata", "google_news", "newsapi"]
    now = datetime.now(timezone.utc)

    collected: dict[str, dict] = {}  # ticker -> article dict list (keep first backend that fills it)
    per_ticker: dict[str, list[dict]] = {t: [] for t in tickers}

    for backend in backends:
        pending = [t for t in tickers if not per_ticker[t]]
        if not pending:
            break
        try:
            if backend == "finnhub":
                arts = _backend_finnhub(pending, days, max_per_ticker, cache_dir, cache_hours)
            elif backend == "newsdata":
                arts = _backend_newsdata(pending, max_per_ticker, cache_dir, cache_hours)
            elif backend == "google_news":
                arts = _backend_google(pending, days, max_per_ticker)
            elif backend == "newsapi":
                arts = _backend_newsapi(pending, days, api_key)
            else:
                logger.warning(f"Unknown news backend '{backend}' — skipping")
                continue
        except Exception as e:
            logger.warning(f"News backend '{backend}' failed (non-fatal): {e}")
            continue

        for a in arts:
            tk = (a.get("ticker") or "UNKNOWN").upper()
            if tk in per_ticker:
                per_ticker[tk].append(a)
        filled = sum(1 for t in pending if per_ticker[t])
        if arts:
            logger.info(f"news[{backend}]: {len(arts)} articles, filled {filled}/{len(pending)} tickers")

    # Flatten newest-first per ticker, dedup near-duplicates
    all_articles: list[dict] = []
    for t in tickers:
        arts = sorted(
            per_ticker[t],
            key=lambda a: a.get("published_at") or now,
            reverse=True,
        )
        all_articles.extend(arts)

    deduped = dedup_articles(all_articles, threshold=dedup_cosine)
    dropped = len(all_articles) - len(deduped)
    if dropped:
        logger.info(f"news dedup: dropped {dropped} near-duplicate headlines")

    if not deduped:
        return _empty_events()
    return _rows_to_events(deduped, now)
