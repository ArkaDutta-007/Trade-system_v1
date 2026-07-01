"""GDELT historical news backfill — daily tone + attention per ticker, 2017→now.

The recent-fetch backends (Finnhub/NewsData/Google) only give the last few days,
so news-derived features were empty across ~all of the training panel. GDELT's
DOC 2.0 API returns a **full daily time series** in a single call per company —
average media *tone* (sentiment) and article *volume* (attention) — going back
to 2017. That makes news a real, leakage-free trained feature:

  * point-in-time by construction — tone[D] is the tone of coverage *on day D*,
    known by that day's close, used to predict the D→D+h forward return;
  * cached per ticker (one JSON each) so a re-run costs no network;
  * company-name queries (from the SEC ticker→name map) with an English filter,
    so ambiguous symbols don't pull unrelated coverage.

Output silver table: ``data/silver/gdelt_history.parquet`` — one row per
(ticker, date) with ``gdelt_tone`` (avg tone) and ``gdelt_vol`` (article count).
"""
from __future__ import annotations

import json
import random
import threading
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl
import requests

from ..utils import get_logger

logger = get_logger(__name__)

_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_MIN_START = date(2017, 1, 1)          # GDELT DOC 2.0 full-text coverage floor
_UA = {"User-Agent": "trading-system/research (news backfill)"}
_THROTTLE_S = 1.2                       # GDELT rate-limits; keep ≥1.2s between calls
_throttle_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    """Thread-safe minimum spacing between GDELT calls.

    Reserving the next slot under a lock (then sleeping outside the HTTP) keeps
    calls ≥1.2s apart *without bursting*, so concurrent workers don't trip GDELT's
    rate limiter (429 → slow backoff) — the HTTP round-trips still overlap.
    """
    global _last_call
    with _throttle_lock:
        now = time.monotonic()
        wait = max(0.0, _THROTTLE_S - (now - _last_call))
        _last_call = max(now, _last_call + _THROTTLE_S)
    if wait:
        time.sleep(wait)


def company_names(user_agent: str = "research research@example.com") -> dict[str, str]:
    """ticker → company name from the SEC map (for clean GDELT queries)."""
    try:
        r = requests.get(_SEC_TICKERS, headers={"User-Agent": user_agent}, timeout=30)
        r.raise_for_status()
        return {row["ticker"].upper(): row["title"] for row in r.json().values()}
    except Exception as e:
        logger.warning(f"SEC name map unavailable ({e}); GDELT will query raw tickers.")
        return {}


def _clean_name(name: str) -> str:
    n = name.strip().strip('"')
    for suf in (" INC", " CORP", " CO", " LTD", " PLC", " CORPORATION", " COMPANY",
                " INCORPORATED", " HOLDINGS", " GROUP", " CLASS A", " CLASS C", " /DE/"):
        if n.upper().endswith(suf):
            n = n[: -len(suf)].strip()
    return n


def _gdelt_timeline(query: str, start: date, end: date, mode: str,
                    session: requests.Session, retries: int = 6) -> list[dict]:
    params = {
        "query": query, "mode": mode, "format": "json", "timelinesmooth": "0",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d000000"),
    }
    for attempt in range(retries):
        try:
            _throttle()
            r = session.get(_DOC_URL, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                # exponential backoff + jitter so a burst of workers doesn't re-collide
                time.sleep(min(20.0, 1.5 * 2 ** attempt) + random.uniform(0, 1.0))
                continue
            r.raise_for_status()
            tl = r.json().get("timeline", [])
            return tl[0]["data"] if tl else []
        except Exception as e:
            if attempt == retries - 1:
                logger.debug(f"GDELT {mode} failed for {query!r}: {e}")
            else:
                time.sleep(1.0 + random.uniform(0, 1.0))
    return []


def _fetch_range(query: str, start: date, end: date, sess: requests.Session) -> list[dict]:
    """Fetch one date range's daily [{date, tone, vol}] from GDELT (2 calls)."""
    if start >= end:
        return []
    tone = _gdelt_timeline(query, start, end, "timelinetone", sess)
    vol = _gdelt_timeline(query, start, end, "timelinevolraw", sess)
    vol_by_day = {d["date"][:8]: d.get("value", 0) for d in vol}
    rows = []
    for d in tone:
        day = d["date"][:8]  # YYYYMMDD
        rows.append({
            "date": f"{day[:4]}-{day[4:6]}-{day[6:8]}",
            "tone": round(float(d.get("value", 0.0)), 4),
            "vol": int(vol_by_day.get(day, 0)),
        })
    return rows


def fetch_gdelt_ticker(
    ticker: str, name: str | None, start: date, end: date,
    cache_dir: Path | None = None, session: requests.Session | None = None,
    overlap_days: int = 14,
) -> list[dict]:
    """Incrementally fetch/refresh one ticker's daily tone+volume history.

    The per-ticker JSON is the **persistent store**. Each run fetches only what's
    missing and merges it in — never the whole 2017→now range again:
      * **new days**: re-fetch the recent tail (last stored day − ``overlap_days``
        → now), which also captures GDELT's late-indexed articles for recent days;
      * **front gap**: if ``start`` is earlier than what's stored, fetch just the
        missing older slice and prepend.
    Newly-fetched days overwrite the stored ones for the same date.
    """
    ticker = ticker.upper()
    cache_file = None
    cached_by_date: dict[str, dict] = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"gdelt_{ticker}.json"
        if cache_file.exists():
            try:
                cached_by_date = {r["date"]: r for r in json.loads(cache_file.read_text())}
            except Exception:
                cached_by_date = {}
    if not name:
        return sorted(cached_by_date.values(), key=lambda r: r["date"])

    sess = session or requests.Session()
    sess.headers.update(_UA)
    query = f'"{_clean_name(name)}" sourcelang:english'

    # Decide the minimal sub-ranges to fetch this run.
    ranges: list[tuple[date, date]] = []
    if not cached_by_date:
        ranges.append((start, end))
    else:
        have = sorted(cached_by_date)
        cmin, cmax = date.fromisoformat(have[0]), date.fromisoformat(have[-1])
        if start < cmin:                                   # extend older history
            ranges.append((start, cmin))
        from datetime import timedelta
        tail = max(start, cmax - timedelta(days=overlap_days))  # new days + late-indexed
        if end > tail:
            ranges.append((tail, end))

    fetched = 0
    for s, e in ranges:
        for r in _fetch_range(query, s, e, sess):
            cached_by_date[r["date"]] = r      # new overwrites old for the same date
            fetched += 1

    merged = sorted(cached_by_date.values(), key=lambda r: r["date"])
    if cache_file is not None and (fetched or not cache_file.exists()):
        try:
            cache_file.write_text(json.dumps(merged))
        except Exception:
            pass
    return merged


def collect_gdelt_history(
    tickers: Iterable[str],
    start: str | date = _MIN_START,
    end: str | date | None = None,
    cache_dir: Path | None = None,
    names: dict[str, str] | None = None,
    workers: int = 8,
    min_coverage: float = 0.9,
    max_rounds: int = 8,
) -> pl.DataFrame:
    """Backfill GDELT daily tone + volume for many tickers → long-form frame.

    Retries the still-uncovered tickers each round until ``min_coverage`` of them
    have data (or coverage plateaus). Returns columns: ticker, date, gdelt_tone,
    gdelt_vol.
    """
    tickers = [t.upper() for t in tickers]
    start = date.fromisoformat(start) if isinstance(start, str) else start
    start = max(start, _MIN_START)
    end = (date.fromisoformat(end) if isinstance(end, str) else end) or date.today()
    names = names if names is not None else company_names()

    from .backfill_util import collect_until_covered, BackfillLedger

    def _one(t: str):
        sess = requests.Session(); sess.headers.update(_UA)
        recs = fetch_gdelt_ticker(t, names.get(t), start, end, cache_dir=cache_dir, session=sess)
        return [{"ticker": t, **r} for r in recs]

    def _load_cached(t: str):                    # name=None → read cache only, no network
        return [{"ticker": t, **r}
                for r in fetch_gdelt_ticker(t, None, start, end, cache_dir=cache_dir)]

    ledger = BackfillLedger(Path(cache_dir) / "_progress.json", end) if cache_dir else None

    # GDELT is rate-limited (a global throttle spaces calls); more workers overlap
    # the timeline downloads, and the coverage loop reclaims rate-limited misses.
    flat = collect_until_covered(
        tickers, _one, source="GDELT", workers=max(1, min(workers, 12)),
        min_coverage=min_coverage, max_rounds=max_rounds,
        ledger=ledger, load_cached=_load_cached,
    )
    if not flat:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date,
                                    "gdelt_tone": pl.Float64, "gdelt_vol": pl.Int64})
    df = (
        pl.DataFrame(flat)
        .rename({"tone": "gdelt_tone", "vol": "gdelt_vol"})
        .with_columns(pl.col("date").str.to_date())
        .unique(subset=["ticker", "date"])
        .sort(["ticker", "date"])
    )
    logger.info(f"GDELT history: {df.height} ticker-days over "
                f"{df['ticker'].n_unique()} tickers ({start} → {end})")
    return df
