"""News and event ingestion. Returns rows in the structured event schema."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Iterable

import polars as pl
import requests

from ..utils import get_logger

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
    "risk_flags": pl.List(pl.Utf8),
}


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def fetch_news(
    tickers: Iterable[str],
    api_key: str | None = None,
    days: int = 7,
) -> pl.DataFrame:
    """Pull news from NewsAPI if key is available; otherwise return empty event frame.

    The output adheres to EVENT_SCHEMA. Sentiment/confidence/etc. are 0.0 placeholders;
    use features.sentiment or an LLM extractor to populate them downstream.
    """
    api_key = api_key or os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.info("NEWSAPI_KEY not set. Returning empty event frame.")
        return _empty_events()

    rows = []
    now = datetime.now(timezone.utc)
    for t in tickers:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": t, "from": (now.replace(hour=0)).date().isoformat(),
                        "language": "en", "pageSize": 25, "apiKey": api_key},
                timeout=20,
            )
            r.raise_for_status()
            for art in r.json().get("articles", []):
                published = art.get("publishedAt")
                try:
                    pdt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except Exception:
                    pdt = now
                rows.append(
                    {
                        "event_id": str(uuid.uuid4()),
                        "source": "news",
                        "source_url": art.get("url", ""),
                        "published_at": pdt,
                        "known_at": now,
                        "tickers": [t.upper()],
                        "sectors": [],
                        "event_type": "news",
                        "sentiment": 0.0,
                        "confidence": 0.5,
                        "novelty": 0.5,
                        "magnitude": 0.0,
                        "time_horizon": "1d",
                        "summary": (art.get("title") or "")[:500],
                        "risk_flags": ["unverified"],
                    }
                )
        except Exception as e:
            logger.warning(f"News fetch failed for {t}: {e}")

    if not rows:
        return _empty_events()
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)
