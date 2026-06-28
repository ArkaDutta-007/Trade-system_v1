"""Finnhub company-news backend — ticker-tagged, free 60 req/min.

Why this exists
---------------
The Google-News RSS backend queries by *free text* (the ticker string), so for
ambiguous symbols (``T``, ``GE``, ``CME``, ``CL``) it returns large volumes of
false positives that then poison the sentiment/apprehension features.  Finnhub's
``company-news`` endpoint is keyed by *symbol*, so every row is genuinely about
the company — a much cleaner micro signal.

Endpoint
--------
``GET https://finnhub.io/api/v1/company-news?symbol=AAPL&from=YYYY-MM-DD&to=YYYY-MM-DD&token=KEY``

Returns a JSON list of objects::

    {"category","datetime"(unix s),"headline","id","image","related","source","summary","url"}

The output rows here are plain dicts that ``news_events._rows_to_events`` maps
into the project EVENT_SCHEMA.  Results are disk-cached per (ticker, day-window)
so re-runs within ``cache_hours`` touch no network.

Falls back silently (returns ``[]``) on any auth/network error so the caller can
move to the next backend — the Finnhub key being absent or invalid never breaks
the pipeline.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests

from ..utils import get_logger

logger = get_logger(__name__)

_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_MIN_INTERVAL_S = 1.05  # stay under 60 req/min on the free tier (one call/ticker)
_last_call_ts = 0.0


def _throttle() -> None:
    """Crude client-side rate limiter: ~1 call/sec keeps us under 60/min."""
    global _last_call_ts
    dt = time.monotonic() - _last_call_ts
    if dt < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - dt)
    _last_call_ts = time.monotonic()


def _cache_path(cache_dir: Path | None, ticker: str, frm: str, to: str) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"finnhub_{ticker.upper()}_{frm}_{to}.json"


def fetch_finnhub_ticker(
    ticker: str,
    api_key: str,
    days: int = 7,
    max_items: int = 25,
    cache_dir: Path | None = None,
    cache_hours: float = 6.0,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch recent company news for one ticker. Returns generic article dicts."""
    now = datetime.now(timezone.utc)
    frm = (now - timedelta(days=days)).date().isoformat()
    to = now.date().isoformat()

    cache_file = _cache_path(cache_dir, ticker, frm, to)
    if cache_file and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
        if age_h <= cache_hours:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass  # corrupt cache → refetch

    sess = session or requests
    try:
        _throttle()
        r = sess.get(
            _FINNHUB_NEWS_URL,
            params={"symbol": ticker.upper(), "from": frm, "to": to, "token": api_key},
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        logger.debug(f"Finnhub news failed for {ticker}: {e}")
        return []

    if not isinstance(raw, list):
        # Finnhub returns {"error": "..."} on bad key / rate limit
        logger.debug(f"Finnhub non-list response for {ticker}: {str(raw)[:120]}")
        return []

    rows: list[dict] = []
    for art in raw[:max_items]:
        ts = art.get("datetime")
        try:
            published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else now
        except Exception:
            published_at = now
        headline = (art.get("headline") or "").strip()
        if not headline:
            continue
        rows.append(
            {
                "ticker": ticker.upper(),
                "title": headline[:500],
                "content": (art.get("summary") or "")[:8000],
                "source_url": art.get("url", ""),
                "publisher_name": art.get("source", "finnhub"),
                "published_at": published_at,
                "backend": "finnhub",
            }
        )

    if cache_file:
        try:
            cache_file.write_text(json.dumps(rows, default=str))
        except Exception as e:
            logger.debug(f"finnhub cache write failed for {ticker}: {e}")
    return rows


def collect_finnhub_articles(
    tickers: Iterable[str],
    api_key: str,
    days: int = 7,
    max_per_ticker: int = 25,
    cache_dir: Path | None = None,
    cache_hours: float = 6.0,
) -> list[dict]:
    """Fetch company news for many tickers (sequential — respects the rate limit)."""
    out: list[dict] = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": "trading-system/news (research)"})
    for t in tickers:
        out.extend(
            fetch_finnhub_ticker(
                t, api_key, days=days, max_items=max_per_ticker,
                cache_dir=cache_dir, cache_hours=cache_hours, session=sess,
            )
        )
    return out
