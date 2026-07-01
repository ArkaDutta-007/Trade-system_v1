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
                    session: requests.Session, retries: int = 3) -> list[dict]:
    params = {
        "query": query, "mode": mode, "format": "json", "timelinesmooth": "0",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d000000"),
    }
    for attempt in range(retries):
        try:
            _throttle()
            r = session.get(_DOC_URL, params=params, timeout=60)
            if r.status_code in (429, 502, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            tl = r.json().get("timeline", [])
            return tl[0]["data"] if tl else []
        except Exception as e:
            if attempt == retries - 1:
                logger.debug(f"GDELT {mode} failed for {query!r}: {e}")
    return []


def fetch_gdelt_ticker(
    ticker: str, name: str | None, start: date, end: date,
    cache_dir: Path | None = None, session: requests.Session | None = None,
    cache_days: float = 24 * 30,
) -> list[dict]:
    """Fetch one ticker's daily [{date, tone, vol}] history (cached per ticker)."""
    ticker = ticker.upper()
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"gdelt_{ticker}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) / 3600 <= cache_days:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass
    if not name:
        return []
    sess = session or requests.Session()
    sess.headers.update(_UA)
    query = f'"{_clean_name(name)}" sourcelang:english'
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
    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(rows))
        except Exception:
            pass
    return rows


def collect_gdelt_history(
    tickers: Iterable[str],
    start: str | date = _MIN_START,
    end: str | date | None = None,
    cache_dir: Path | None = None,
    names: dict[str, str] | None = None,
    workers: int = 4,
) -> pl.DataFrame:
    """Backfill GDELT daily tone + volume for many tickers → long-form frame.

    Returns columns: ticker, date, gdelt_tone, gdelt_vol.
    """
    tickers = [t.upper() for t in tickers]
    start = date.fromisoformat(start) if isinstance(start, str) else start
    start = max(start, _MIN_START)
    end = (date.fromisoformat(end) if isinstance(end, str) else end) or date.today()
    names = names if names is not None else company_names()

    from ..utils import parallel_map

    def _one(t: str):
        sess = requests.Session(); sess.headers.update(_UA)
        recs = fetch_gdelt_ticker(t, names.get(t), start, end, cache_dir=cache_dir, session=sess)
        return [{"ticker": t, **r} for r in recs]

    # GDELT is rate-limited — keep concurrency low; cache makes re-runs free.
    results = parallel_map(_one, tickers, workers=max(1, min(workers, 6)),
                           description="GDELT news")
    flat = [r for sub in results if sub for r in sub]
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
