"""Wikipedia pageview history — daily retail-attention proxy per ticker, 2015→now.

Daily views of a company's English-Wikipedia article are a clean, free proxy for
**retail attention**: spikes lead/accompany the retail-driven part of volume and
volatility, and — unlike price-derived attention — they're an independent signal.
The Wikimedia REST API serves per-article daily counts back to 2015-07 with no key.

Point-in-time by construction: ``views[D]`` is the traffic *on day D*, known by
that day's close, used to predict the D→D+h forward return. Cached per ticker
(one JSON each) and updated incrementally (recent tail only) on re-runs.

Output silver table: ``data/silver/wiki_history.parquet`` — one row per
(ticker, date) with ``wiki_views``. :mod:`features.wiki_features` turns it into a
causal abnormal-attention z-score + momentum.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import polars as pl
import requests

from ..utils import get_logger

logger = get_logger(__name__)

_PAGEVIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}"
)
_SEARCH = "https://en.wikipedia.org/w/api.php"
_MIN_START = date(2015, 7, 1)          # Wikimedia pageviews coverage floor
_UA = {"User-Agent": "trading-system/research (attention backfill)"}
_THROTTLE_S = 0.2
_throttle_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    with _throttle_lock:
        now = time.monotonic()
        wait = max(0.0, _THROTTLE_S - (now - _last_call))
        _last_call = max(now, _last_call + _THROTTLE_S)
    if wait:
        time.sleep(wait)


def resolve_wiki_title(name: str, sess: requests.Session) -> str | None:
    """Best-matching English-Wikipedia article title for a company name."""
    try:
        _throttle()
        r = sess.get(_SEARCH, params={
            "action": "query", "list": "search", "srsearch": name,
            "srlimit": 1, "format": "json",
        }, timeout=30)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except Exception as e:
        logger.debug(f"wiki title resolve failed for {name!r}: {e}")
        return None


def _fetch_range(title: str, start: date, end: date, sess: requests.Session,
                 retries: int = 3) -> list[dict]:
    url = _PAGEVIEWS.format(
        title=quote(title.replace(" ", "_"), safe=""),
        start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"),
    )
    for attempt in range(retries):
        try:
            _throttle()
            r = sess.get(url, timeout=60)
            if r.status_code == 404:
                return []                      # article/timespan has no data
            if r.status_code in (429, 502, 503):
                time.sleep(1.5 * (attempt + 1)); continue
            r.raise_for_status()
            rows = []
            for it in r.json().get("items", []):
                ts = it.get("timestamp", "")[:8]      # YYYYMMDD00
                if len(ts) == 8:
                    rows.append({"date": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}",
                                 "views": int(it.get("views", 0))})
            return rows
        except Exception as e:
            if attempt == retries - 1:
                logger.debug(f"wiki pageviews failed for {title!r}: {e}")
    return []


def fetch_wiki_ticker(
    ticker: str, title: str | None, start: date, end: date,
    cache_dir: Path | None = None, session: requests.Session | None = None,
    overlap_days: int = 5,
) -> list[dict]:
    """Incrementally fetch/refresh one ticker's daily pageviews → [{date, views}]."""
    ticker = ticker.upper()
    cache_file = None
    cached: dict[str, dict] = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"wiki_{ticker}.json"
        if cache_file.exists():
            try:
                cached = {r["date"]: r for r in json.loads(cache_file.read_text())}
            except Exception:
                cached = {}
    if not title:
        return sorted(cached.values(), key=lambda r: r["date"])

    sess = session or requests.Session()
    sess.headers.update(_UA)

    ranges: list[tuple[date, date]] = []
    if not cached:
        ranges.append((start, end))
    else:
        have = sorted(cached)
        cmin, cmax = date.fromisoformat(have[0]), date.fromisoformat(have[-1])
        if start < cmin:
            ranges.append((start, cmin))
        tail = max(start, cmax - timedelta(days=overlap_days))
        if end > tail:
            ranges.append((tail, end))

    fetched = 0
    for s, e in ranges:
        if s >= e:
            continue
        for r in _fetch_range(title, s, e, sess):
            cached[r["date"]] = r; fetched += 1

    merged = sorted(cached.values(), key=lambda r: r["date"])
    if cache_file is not None and (fetched or not cache_file.exists()):
        try:
            cache_file.write_text(json.dumps(merged))
        except Exception:
            pass
    return merged


def collect_wiki_history(
    tickers: Iterable[str],
    names: dict[str, str],
    start: str | date = _MIN_START,
    end: str | date | None = None,
    cache_dir: Path | None = None,
    workers: int = 4,
) -> pl.DataFrame:
    """Backfill Wikipedia daily pageviews for many tickers → (ticker, date, wiki_views).

    ``names`` maps ticker → company name (e.g. from the SEC map); each is resolved
    to an article title once and the resolution is cached alongside the series.
    """
    tickers = [t.upper() for t in tickers]
    start = date.fromisoformat(start) if isinstance(start, str) else start
    start = max(start, _MIN_START)
    end = (date.fromisoformat(end) if isinstance(end, str) else end) or date.today()

    title_cache: dict[str, str] = {}
    tc_file = None
    if cache_dir is not None:
        tc_file = Path(cache_dir) / "wiki_titles.json"
        if tc_file.exists():
            try:
                title_cache = json.loads(tc_file.read_text())
            except Exception:
                title_cache = {}

    from ..utils import parallel_map

    def _one(t: str):
        sess = requests.Session(); sess.headers.update(_UA)
        title = title_cache.get(t)
        if title is None:
            title = resolve_wiki_title(names.get(t, t), sess) or ""
            title_cache[t] = title            # cache even "" (unresolved) to skip next time
        recs = fetch_wiki_ticker(t, title or None, start, end,
                                 cache_dir=cache_dir, session=sess)
        return [{"ticker": t, **r} for r in recs]

    results = parallel_map(_one, tickers, workers=max(1, min(workers, 6)),
                           description="Wiki pageviews")
    if tc_file is not None:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            tc_file.write_text(json.dumps(title_cache))
        except Exception:
            pass

    flat = [r for sub in results if sub for r in sub]
    if not flat:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date, "wiki_views": pl.Int64})
    df = (
        pl.DataFrame(flat)
        .rename({"views": "wiki_views"})
        .with_columns(pl.col("date").str.to_date())
        .unique(subset=["ticker", "date"])
        .sort(["ticker", "date"])
    )
    logger.info(f"Wiki history: {df.height:,} ticker-days over {df['ticker'].n_unique()} "
                f"tickers ({start} → {end})")
    return df
