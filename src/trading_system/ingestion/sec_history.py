"""SEC EDGAR filing history — point-in-time filing events per ticker, deep history.

The GDELT tone series only reaches back to 2017 and covers ~29% of the panel, so
news-derived features were sparse across most of the training window. SEC filings
close that gap from the *event* side: the submissions API returns **every** filing
a company has ever made (form type + filing date), and every established issuer's
history predates 2010 — so this covers essentially the whole panel.

Filing dates are public the day the document is accepted, so trailing counts are
leakage-free features known by that day's close:

  * all-forms count  → disclosure intensity (quiet vs. busy periods);
  * 8-K count        → material corporate events (guidance, M&A, management);
  * Form-4 count     → insider transaction activity;
  * days-since-last  → filing recency.

Output silver table: ``data/silver/sec_history.parquet`` — one row per filing
(ticker, date, form). :mod:`features.sec_features` turns it into rolling counts.

Cached per ticker (one JSON each): the historical shards are immutable, so a
re-run only refetches the *recent* submissions block and merges new filings in.
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

_SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SUBMISSIONS_SHARD = "https://data.sec.gov/submissions/{name}"
# SEC asks for a descriptive UA with contact; fair-access limit is 10 req/s.
_UA = {"User-Agent": "trading-system research (compliance@example.com)"}
_THROTTLE_S = 0.15
_throttle_lock = threading.Lock()
_last_call = 0.0

# Forms worth keeping as distinct signals (others are folded into the all-count).
_KEEP_PREFIXES = ("8-K", "10-K", "10-Q", "4", "3", "5", "SC 13", "DEF 14A", "6-K")


def _throttle() -> None:
    global _last_call
    with _throttle_lock:
        now = time.monotonic()
        wait = max(0.0, _THROTTLE_S - (now - _last_call))
        _last_call = max(now, _last_call + _THROTTLE_S)
    if wait:
        time.sleep(wait)


def company_ciks(user_agent: str = _UA["User-Agent"]) -> dict[str, str]:
    """ticker → 10-digit zero-padded CIK from the SEC map."""
    try:
        r = requests.get(_SEC_TICKERS, headers={"User-Agent": user_agent}, timeout=30)
        r.raise_for_status()
        return {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in r.json().values()}
    except Exception as e:
        logger.warning(f"SEC CIK map unavailable ({e}); SEC features will be empty.")
        return {}


def _get_json(url: str, sess: requests.Session, retries: int = 5):
    import random
    for attempt in range(retries):
        try:
            _throttle()
            r = sess.get(url, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(15.0, 1.5 * 2 ** attempt) + random.uniform(0, 0.5))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.debug(f"SEC fetch failed {url}: {e}")
            else:
                time.sleep(0.5 + random.uniform(0, 0.5))
    return None


def _rows_from_recent(recent: dict) -> list[dict]:
    forms = recent.get("form", []) or []
    fdates = recent.get("filingDate", []) or []
    out = []
    for form, d in zip(forms, fdates):
        if not d:
            continue
        out.append({"date": d, "form": (form or "").strip()})
    return out


def fetch_ticker_filings(
    ticker: str, cik: str | None, cache_dir: Path | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Incrementally fetch one issuer's filing history → [{date, form}].

    First run pulls the recent block **and** every historical shard; later runs
    only refetch the recent block (shards never change) and merge new filings.
    """
    ticker = ticker.upper()
    cache_file = None
    cached: dict[tuple[str, str], dict] = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"sec_{ticker}.json"
        if cache_file.exists():
            try:
                cached = {(r["date"], r["form"]): r for r in json.loads(cache_file.read_text())}
            except Exception:
                cached = {}
    if not cik:
        return sorted(cached.values(), key=lambda r: r["date"])

    sess = session or requests.Session()
    sess.headers.update(_UA)
    had_cache = bool(cached)

    main = _get_json(_SUBMISSIONS.format(cik=int(cik)), sess)
    fetched = 0
    if main:
        filings = main.get("filings", {}) or {}
        for r in _rows_from_recent(filings.get("recent", {}) or {}):
            key = (r["date"], r["form"])
            if key not in cached:
                cached[key] = r; fetched += 1
        # Historical shards only on the first build (they're immutable).
        if not had_cache:
            for shard in filings.get("files", []) or []:
                name = shard.get("name")
                if not name:
                    continue
                sj = _get_json(_SUBMISSIONS_SHARD.format(name=name), sess)
                if sj:
                    for r in _rows_from_recent(sj):
                        key = (r["date"], r["form"])
                        if key not in cached:
                            cached[key] = r; fetched += 1

    merged = sorted(cached.values(), key=lambda r: r["date"])
    if cache_file is not None and (fetched or not cache_file.exists()):
        try:
            cache_file.write_text(json.dumps(merged))
        except Exception:
            pass
    return merged


def collect_sec_history(
    tickers: Iterable[str],
    cache_dir: Path | None = None,
    ciks: dict[str, str] | None = None,
    workers: int = 10,
    min_coverage: float = 0.9,
    max_rounds: int = 6,
) -> pl.DataFrame:
    """Backfill SEC filing history for many tickers → long-form (ticker, date, form).

    Retries the uncovered tickers until ``min_coverage`` (ETFs with no CIK never
    resolve and are dropped once coverage plateaus).
    """
    tickers = [t.upper() for t in tickers]
    ciks = ciks if ciks is not None else company_ciks()

    from .backfill_util import collect_until_covered, BackfillLedger
    from datetime import date as _date

    def _one(t: str):
        sess = requests.Session(); sess.headers.update(_UA)
        recs = fetch_ticker_filings(t, ciks.get(t), cache_dir=cache_dir, session=sess)
        return [{"ticker": t, **r} for r in recs]

    def _load_cached(t: str):                    # cik=None → read cache only, no network
        return [{"ticker": t, **r} for r in fetch_ticker_filings(t, None, cache_dir=cache_dir)]

    ledger = BackfillLedger(Path(cache_dir) / "_progress.json", _date.today()) if cache_dir else None

    flat = collect_until_covered(
        tickers, _one, source="SEC", workers=max(1, min(workers, 16)),
        min_coverage=min_coverage, max_rounds=max_rounds,
        ledger=ledger, load_cached=_load_cached,
    )
    if not flat:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date, "form": pl.Utf8})
    df = (
        pl.DataFrame(flat)
        .with_columns(pl.col("date").str.to_date(strict=False))
        .drop_nulls("date")
        .unique(subset=["ticker", "date", "form"])
        .sort(["ticker", "date"])
    )
    logger.info(f"SEC history: {df.height:,} filings over {df['ticker'].n_unique()} tickers "
                f"({df['date'].min()} → {df['date'].max()})")
    return df
