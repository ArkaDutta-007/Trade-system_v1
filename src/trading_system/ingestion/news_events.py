"""News and event ingestion. Returns rows in the structured event schema.

Primary backend: Google News RSS + full article-body fetcher.
Fallback backend: NewsAPI headlines when Google RSS yields nothing.

The structured rows are then fed into downstream event features and the
DeepSeek apprehension scorer.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable

import polars as pl
import requests

from ..utils import get_logger
from ..features.sentiment import naive_sentiment
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


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def _fetch_newsapi_headlines(
    tickers: Iterable[str],
    api_key: str | None = None,
    days: int = 7,
) -> pl.DataFrame:
    api_key = api_key or os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return _empty_events()

    rows = []
    now = datetime.now(timezone.utc)
    from_date = (now - timedelta(days=days)).date().isoformat()
    for t in tickers:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": t,
                    "from": from_date,
                    "language": "en",
                    "pageSize": 25,
                    "apiKey": api_key,
                },
                timeout=20,
            )
            r.raise_for_status()
            for art in r.json().get("articles", []):
                published = art.get("publishedAt")
                try:
                    pdt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except Exception:
                    pdt = now
                title = (art.get("title") or "")[:500]
                rows.append(
                    {
                        "event_id": str(uuid.uuid4()),
                        "source": "newsapi",
                        "source_url": art.get("url", ""),
                        "published_at": pdt,
                        "known_at": now,
                        "tickers": [t.upper()],
                        "sectors": [],
                        "event_type": "news",
                        "sentiment": naive_sentiment(title),
                        "confidence": 0.4,
                        "novelty": 0.5,
                        "magnitude": 0.0,
                        "time_horizon": "1d",
                        "summary": title,
                        "content": (art.get("description") or "")[:8000],
                        "risk_flags": ["unverified"],
                    }
                )
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed for {t}: {e}")

    return pl.DataFrame(rows, schema=EVENT_SCHEMA) if rows else _empty_events()


def fetch_news(
    tickers: Iterable[str],
    api_key: str | None = None,
    days: int = 7,
) -> pl.DataFrame:
    """Fetch recent news as structured events.

    Backend order:
      1. Google News RSS + full article-body extraction (primary)
      2. NewsAPI headline fetch (fallback)
    """
    now = datetime.now(timezone.utc)
    rows = []

    try:
        articles = collect_google_news_articles(tickers, days=days, max_urls_per_ticker=10)
        for art in articles:
            published_at = art.get("published_at") or now
            title = (art.get("title") or "")[:500]
            content = (art.get("content") or "")[:8000]
            rows.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "source": "google_news",
                    "source_url": art.get("source_url", ""),
                    "published_at": published_at,
                    "known_at": now,
                    "tickers": [art.get("ticker", "UNKNOWN").upper()],
                    "sectors": [],
                    "event_type": "news",
                    "sentiment": naive_sentiment(f"{title} {content[:500]}"),
                    "confidence": 0.45,
                    "novelty": 0.5,
                    "magnitude": 0.0,
                    "time_horizon": "1d",
                    "summary": title,
                    "content": content,
                    "risk_flags": ["unverified"],
                }
            )
    except Exception as e:
        logger.warning(f"Google News fetch failed, falling back to NewsAPI: {e}")

    if not rows:
        fallback = _fetch_newsapi_headlines(tickers, api_key=api_key, days=days)
        if not fallback.is_empty():
            return fallback
        return _empty_events()

    return pl.DataFrame(rows, schema=EVENT_SCHEMA)
