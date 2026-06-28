"""NewsData.io backend — free tier ~200 credits/day (10 articles each).

The key supplied for this project (``pub_...`` prefix) is a NewsData.io key, so
this is the backend that actually consumes it.  NewsData's free ``/latest``
endpoint is keyword (free-text) search — not symbol-tagged like Finnhub — so we
query ``"<ticker> stock"`` under ``category=business`` to cut the worst of the
ambiguous-ticker noise, then rely on near-dup dedup + downstream LLM enrichment.

Free-tier constraints handled here:
  * ``size`` is capped at 10 on the free plan; one ticker = one credit.
  * no date range on free (``/archive`` is paid) — we just take latest.
  * ``content`` is paywalled on free; we fall back to ``description``.

Results are disk-cached per (ticker, day) so a re-run the same day spends no
credits.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from ..utils import get_logger

logger = get_logger(__name__)

_NEWSDATA_URL = "https://newsdata.io/api/1/latest"
_PAYWALL_SENTINEL = "ONLY AVAILABLE IN PAID PLANS"


def _parse_pubdate(s: str | None, now: datetime) -> datetime:
    if not s:
        return now
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return now


def fetch_newsdata_ticker(
    ticker: str,
    api_key: str,
    max_items: int = 10,
    cache_dir: Path | None = None,
    cache_hours: float = 6.0,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch latest business news for one ticker. Returns generic article dicts."""
    now = datetime.now(timezone.utc)

    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"newsdata_{ticker.upper()}_{now.date().isoformat()}.json"
        if cache_file.exists():
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
            if age_h <= cache_hours:
                try:
                    return json.loads(cache_file.read_text())
                except Exception:
                    pass

    sess = session or requests
    try:
        r = sess.get(
            _NEWSDATA_URL,
            params={
                "apikey": api_key,
                "q": f"{ticker} stock",
                "language": "en",
                "category": "business",
            },
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.debug(f"NewsData fetch failed for {ticker}: {e}")
        return []

    if payload.get("status") != "success":
        logger.debug(f"NewsData error for {ticker}: {str(payload.get('results'))[:120]}")
        return []

    rows: list[dict] = []
    for art in (payload.get("results") or [])[:max_items]:
        title = (art.get("title") or "").strip()
        if not title:
            continue
        content = art.get("content") or ""
        if not content or _PAYWALL_SENTINEL in content:
            content = art.get("description") or ""
        rows.append({
            "ticker": ticker.upper(),
            "title": title[:500],
            "content": content[:8000],
            "source_url": art.get("link", ""),
            "publisher_name": art.get("source_id", "newsdata"),
            "published_at": _parse_pubdate(art.get("pubDate"), now),
            "backend": "newsdata",
        })

    if cache_file:
        try:
            cache_file.write_text(json.dumps(rows, default=str))
        except Exception as e:
            logger.debug(f"newsdata cache write failed for {ticker}: {e}")
    return rows


def collect_newsdata_articles(
    tickers: Iterable[str],
    api_key: str,
    max_per_ticker: int = 10,
    cache_dir: Path | None = None,
    cache_hours: float = 6.0,
    max_tickers: int = 180,
) -> list[dict]:
    """Fetch latest news for many tickers (one credit each; capped to protect the quota)."""
    from ..utils import track

    out: list[dict] = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": "trading-system/news (research)"})
    tickers = list(tickers)
    if len(tickers) > max_tickers:
        logger.info(f"NewsData: capping {len(tickers)} tickers → {max_tickers} to protect daily credits")
        tickers = tickers[:max_tickers]
    for t in track(tickers, "newsdata news"):
        out.extend(
            fetch_newsdata_ticker(
                t, api_key, max_items=max_per_ticker,
                cache_dir=cache_dir, cache_hours=cache_hours, session=sess,
            )
        )
    return out
