"""Massive (formerly Polygon.io) REST ingestion — whole-market EOD bars + reference data.

Why this exists
---------------
yfinance is scraped, per-ticker and flaky (ADR timezone glitches, in-progress
bars, silent gaps).  Massive is a real REST API whose *grouped daily* endpoint
returns **every US stock's bar for one date in a single call**, and whose
reference endpoints give splits, dividends, fundamentals, ticker metadata and
ticker-tagged news with LLM sentiment.  Whole-market coverage also lets research
build point-in-time, survivorship-free universes (``ts bias-check`` measured
+12.8pp CAGR of survivorship bias in the curated universe).

Plan constraints (Basic / free tier, checked 2026-09)
-----------------------------------------------------
* **5 requests / minute** — enforced *client-side, cross-process* by a
  file-locked sliding window (``RateLimiter``) so cron jobs and ad-hoc shells
  can never trip the server limit together.
* **EOD only** — the day's grouped bar appears after the close; the 05:15 cron
  therefore picks up *yesterday*, exactly like yfinance did.
* **2 years of history** — so Massive cannot replace the 1995→ training
  history.  ``ingest_universe_spliced`` keeps yfinance for the deep history and
  rescales its adjusted series to be continuous with the Massive window at the
  seam (per ticker), then hands the pipeline the usual ``bronze/ohlcv_daily``.

Endpoints (paths verified against ``massive-com/client-python``)
----------------------------------------------------------------
    GET /v2/aggs/grouped/locale/us/market/stocks/{date}     all tickers, one day
    GET /v2/aggs/ticker/{t}/range/1/day/{from}/{to}          one ticker, range
    GET /v3/reference/tickers[?active=…]                     ticker directory
    GET /v3/reference/tickers/{t}                            overview (SIC, mcap…)
    GET /v3/reference/splits, /v3/reference/dividends        corporate actions
    GET /vX/reference/financials?ticker=…                    fundamentals
    GET /v2/reference/news[?ticker=…]                        tagged news + insights
    GET /v1/marketstatus/upcoming                            holidays

Auth: ``Authorization: Bearer $MASSIVE_API_KEY`` (``POLYGON_API_KEY`` accepted).
Pagination: follow ``next_url`` verbatim (it carries the cursor).

Adjustment maths
----------------
Grouped bars are cached **unadjusted** (immutable truth, so the cache is
forever-valid); adjustments are recomputed at build time from the splits and
dividends tables, CRSP-style:

    split factor  s_t = ∏_{splits d > t}   (from_d / to_d)             (prices ×, volume ÷)
    div   factor  f_t = ∏_{ex-dates d > t} (1 − cash_d / close_{d−1})  (raw units, unit-free)
    close      = raw_close · s_t                    (split-adjusted, yfinance "Close")
    adj_close  = close · f_t                        (split+dividend, yfinance "Adj Close")

Everything network-facing is disk-cached under ``data/raw/massive/`` (gitignored);
bronze outputs land in ``data/bronze/massive/``.  No new dependencies.
"""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import polars as pl
import requests

from ..utils import get_logger

logger = get_logger(__name__)

MASSIVE_BASE = "https://api.massive.com"
_ENV_KEYS = ("MASSIVE_API_KEY", "POLYGON_API_KEY")
DEFAULT_RPM = 5
WINDOW_S = 61.0            # server window is 60 s; 1 s slack for clock skew (≈4.9 req/min sustained)
HISTORY_YEARS = 2          # free-tier depth
_DAY = 86_400.0
_MAX_PAGES_DEFAULT = 200

OHLCV_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


class MassiveError(RuntimeError):
    """Non-retryable API error (4xx other than 429/404)."""


class MassiveAuthError(MassiveError):
    """401/403 — bad or missing key. Never retried."""


class MassiveEntitlementError(MassiveError):
    """403 — the plan does not include this endpoint. Not key-specific; the crawler
    parks the endpoint family for a week instead of burning calls on it."""


class MassiveNotPublished(MassiveError):
    """A dated resource the plan *does* include but has not released yet (the free plan answers 403 for
    yesterday's grouped bars until ~04:00 UTC). Retried soon; never parks the endpoint family."""
    retry_s = 1800.0


class MassiveNotReady(RuntimeError):
    """The local Massive cache is empty/too stale to build from — run `ts massive backfill`."""


def api_keys() -> list[str]:
    """Every configured key, deduplicated: ``MASSIVE_API_KEY``, ``MASSIVE_API_KEY2..9``, ``POLYGON_API_KEY``.

    Each key has its own 5/min allowance, so N keys ≈ N×5 req/min (the client
    round-robins across them and retires any that 401).
    """
    names = ["MASSIVE_API_KEY"] + [f"MASSIVE_API_KEY{i}" for i in range(2, 10)] + ["POLYGON_API_KEY"]
    out: list[str] = []
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and v not in out:
            out.append(v)
    return out


def api_key() -> str | None:
    ks = api_keys()
    return ks[0] if ks else None


def is_configured() -> bool:
    return bool(api_keys())


def _key_fp(key: str) -> str:
    """Short non-reversible fingerprint used to name per-key limiter files (never the key itself)."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def normalize_ticker(t: str) -> str:
    """Massive writes share classes as ``BRK.B``; the repo/yfinance use ``BRK-B``."""
    return t.strip().upper().replace(".", "-")


def api_ticker(t: str) -> str:
    """Inverse of :func:`normalize_ticker` for URL paths/params (``BRK-B`` → ``BRK.B``; the API 400s on the hyphen)."""
    return t.strip().upper().replace("-", ".")


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter — file-locked sliding window shared by every process on the box
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Allow at most ``rpm`` acquisitions per ``window_s`` seconds, machine-wide.

    Timestamps of recent calls live in one small JSON file guarded by
    ``fcntl.flock``; the lock is released while sleeping so waiters don't
    serialise behind each other.  ``clock``/``sleep`` are injectable for tests.
    """

    def __init__(self, path: Path, rpm: int = DEFAULT_RPM, window_s: float = WINDOW_S,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.path = Path(path)
        self.rpm = max(1, int(rpm))
        self.window_s = float(window_s)
        self.clock = clock
        self.sleep = sleep
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.waited_s = 0.0

    def _read(self, fh) -> list[float]:
        fh.seek(0)
        raw = fh.read()
        if not raw:
            return []
        try:
            return [float(x) for x in json.loads(raw)]
        except Exception:
            return []

    def _write(self, fh, stamps: list[float]) -> None:
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(stamps))
        fh.flush()

    def try_acquire(self) -> float:
        """Non-blocking: take a slot now and return 0.0, or return seconds until one frees."""
        with open(self.path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                now = self.clock()
                stamps = sorted(t for t in self._read(fh) if now - t < self.window_s)
                if len(stamps) < self.rpm:
                    stamps.append(now)
                    self._write(fh, stamps)
                    return 0.0
                return max(stamps[0] + self.window_s - now, 0.05)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def acquire(self) -> float:
        """Block until a slot is free; returns seconds waited."""
        waited = 0.0
        while True:
            wait = self.try_acquire()
            if wait == 0.0:
                self.waited_s += waited
                return waited
            self.sleep(wait)
            waited += wait


# ─────────────────────────────────────────────────────────────────────────────
# HTTP client — cache, pagination, retries
# ─────────────────────────────────────────────────────────────────────────────

class MassiveClient:
    """Thin REST client: bearer auth, sliding-window throttle, gz-JSON cache, ``next_url`` paging."""

    def __init__(self, key: str | None = None, base: str = MASSIVE_BASE,
                 cache_dir: Path | None = None, rpm: int = DEFAULT_RPM,
                 session: Any = None, timeout: float = 30.0,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 5, keys: Iterable[str] | None = None):
        ks = [k for k in (keys or ([key] if key else api_keys())) if k]
        if not ks:
            raise MassiveAuthError("MASSIVE_API_KEY is not set (add it to .env)")
        self.keys = list(dict.fromkeys(ks))
        self.key = self.keys[0]                       # back-compat
        self.base = base.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw/massive")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.clock = clock
        self.sleep = sleep
        self.max_retries = max_retries
        # one sliding window PER KEY (each key has its own 5/min at the server)
        self.limiters = [RateLimiter(self.cache_dir / f".ratelimit-{_key_fp(k)}.json", rpm=rpm, clock=clock, sleep=sleep)
                         for k in self.keys]
        self.limiter = self.limiters[0]              # back-compat
        self.dead = [False] * len(self.keys)         # 401 → retired for this process
        self.cooldown = [0.0] * len(self.keys)       # 429 → don't use before this clock time
        self.key_calls = [0] * len(self.keys)
        self._rr = 0
        self._waited_extra = 0.0
        self.calls = 0            # network requests made by this instance
        self.cache_hits = 0

    @property
    def waited_s(self) -> float:
        return sum(l.waited_s for l in self.limiters) + self._waited_extra

    @property
    def live_keys(self) -> int:
        return sum(1 for d in self.dead if not d)

    def _pick_key(self) -> int:
        """Round-robin over live, non-cooling keys; take the first with a free slot, else sleep
        until the earliest slot/cooldown frees. Raises when every key has been retired."""
        n = len(self.keys)
        while True:
            now = self.clock()
            best: float | None = None
            for j in range(n):
                i = (self._rr + j) % n
                if self.dead[i]:
                    continue
                if self.cooldown[i] > now:
                    best = min(best if best is not None else 1e9, self.cooldown[i] - now)
                    continue
                w = self.limiters[i].try_acquire()
                if w == 0.0:
                    self._rr = (i + 1) % n
                    return i
                best = min(best if best is not None else 1e9, w)
            if best is None:
                raise MassiveAuthError("every Massive API key has been retired (401) — check .env")
            wait = max(best, 0.05)
            self.sleep(wait)
            self._waited_extra += wait

    # ---- cache -------------------------------------------------------------
    def _cache_file(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json.gz"

    def cache_read(self, key: str, ttl_s: float | None) -> dict | None:
        """``ttl_s=None`` → immutable (any age ok); ``0`` → always refetch."""
        p = self._cache_file(key)
        if ttl_s == 0 or not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            return None
        if ttl_s is not None and self.clock() - float(doc.get("fetched_at", 0)) > ttl_s:
            return None
        return doc

    def cache_write(self, key: str, doc: dict) -> None:
        p = self._cache_file(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(doc, fh)
        os.replace(tmp, p)

    def cached(self, key: str) -> bool:
        return self._cache_file(key).exists()

    def cache_age(self, key: str) -> float | None:
        """Seconds since the cache file was written (mtime), or None if absent. No JSON parse."""
        try:
            return max(time.time() - self._cache_file(key).stat().st_mtime, 0.0)
        except FileNotFoundError:
            return None

    def _count_call(self) -> None:
        """Append to a per-day counter so `ts massive status` can show budget use."""
        self.calls += 1
        try:
            day = datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y%m%d")
            p = self.cache_dir / f".calls-{day}"
            with open(p, "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                fh.write("1")
        except Exception:
            pass

    # ---- transport -----------------------------------------------------------
    def _request(self, url: str, params: dict | None) -> dict:
        attempt = 0
        while True:
            i = self._pick_key()
            headers = {"Authorization": f"Bearer {self.keys[i]}", "Accept": "application/json"}
            self._count_call()
            self.key_calls[i] += 1
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise MassiveError(f"network error after {attempt} tries: {e}") from e
                self.sleep(min(2.0 ** attempt, 30.0))
                continue
            status = resp.status_code
            if status == 200:
                return resp.json()
            if status == 401:                        # this key is bad; others may be fine
                self.dead[i] = True
                logger.error(f"massive: key #{i + 1} rejected (401) — retired for this process")
                attempt += 1
                continue                             # _pick_key raises once all keys are dead
            if status == 403:
                raise MassiveEntitlementError(f"403 (not in plan) from {url.split('?')[0]}: {resp.text[:200]}")
            if status == 404:
                return {"results": [], "status": "NOT_FOUND"}
            attempt += 1
            if attempt > self.max_retries:
                raise MassiveError(f"{status} after {attempt} tries: {resp.text[:300]}")
            if status == 429:                        # cool THIS key only; another key may serve at once
                ra = resp.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else WINDOW_S
                self.cooldown[i] = self.clock() + wait
                logger.warning(f"massive: 429 on key #{i + 1}, cooling it {wait:.0f}s (attempt {attempt})")
                continue
            if status >= 500:
                self.sleep(min(2.0 ** attempt, 30.0))
                continue
            raise MassiveError(f"{status} from {url.split('?')[0]}: {resp.text[:300]}")

    def get(self, path: str, params: dict | None = None, *, cache_key: str | None = None,
            ttl_s: float | None = None, paginate: bool = True,
            max_pages: int = _MAX_PAGES_DEFAULT, result_key: str = "results") -> dict:
        """GET ``path`` (all pages) → ``{"results": [...], "meta": {...}, "pages": n, "fetched_at": ts}``.

        A dict-valued ``results`` (ticker overview) is wrapped in a one-element list.
        """
        if cache_key:
            hit = self.cache_read(cache_key, ttl_s)
            if hit is not None:
                self.cache_hits += 1
                return hit
        url = self.base + path
        results: list = []
        meta: dict = {}
        pages = 0
        while url:
            payload = self._request(url, params)
            params = None                       # next_url already carries the cursor
            pages += 1
            r = payload.get(result_key)
            if isinstance(r, list):
                results.extend(r)
            elif r is not None:
                results.append(r)
            meta = {k: v for k, v in payload.items() if k not in (result_key, "next_url")}
            url = payload.get("next_url") if paginate else None
            if url and pages >= max_pages:
                logger.warning(f"massive: stopping pagination at {pages} pages for {path}")
                break
        doc = {"results": results, "meta": meta, "pages": pages,
               "fetched_at": self.clock(), "path": path}
        if cache_key:
            self.cache_write(cache_key, doc)
        return doc

    # ---- endpoints -----------------------------------------------------------
    def grouped_daily(self, d: date, *, include_otc: bool = True, ttl_s: float | None = None) -> dict:
        return self.get(f"/v2/aggs/grouped/locale/us/market/stocks/{d.isoformat()}",
                        {"adjusted": "false", "include_otc": "true" if include_otc else "false"},
                        cache_key=f"grouped/{d.isoformat()}", ttl_s=ttl_s, paginate=False)

    def aggs(self, ticker: str, start: date, end: date, *, adjusted: bool = False) -> dict:
        return self.get(f"/v2/aggs/ticker/{api_ticker(ticker)}/range/1/day/{start.isoformat()}/{end.isoformat()}",
                        {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": 50000},
                        cache_key=f"aggs/{ticker}_{start.isoformat()}_{end.isoformat()}_{int(adjusted)}",
                        ttl_s=_DAY)

    def tickers(self, *, active: bool = True, ttl_s: float = 7 * _DAY) -> dict:
        return self.get("/v3/reference/tickers",
                        {"market": "stocks", "active": "true" if active else "false",
                         "limit": 1000, "sort": "ticker", "order": "asc"},
                        cache_key=f"reference/tickers_{'active' if active else 'delisted'}", ttl_s=ttl_s)

    def ticker_details(self, ticker: str, *, ttl_s: float = 30 * _DAY) -> dict:
        return self.get(f"/v3/reference/tickers/{api_ticker(ticker)}", cache_key=f"details/{normalize_ticker(ticker)}",
                        ttl_s=ttl_s, paginate=False)

    def splits(self, gte: date, lte: date, *, ttl_s: float | None) -> dict:
        return self.get("/v3/reference/splits",
                        {"execution_date.gte": gte.isoformat(), "execution_date.lte": lte.isoformat(),
                         "limit": 1000, "sort": "execution_date", "order": "asc"},
                        cache_key=f"splits/{gte.isoformat()}_{lte.isoformat()}", ttl_s=ttl_s)

    def dividends(self, gte: date, lte: date, *, ttl_s: float | None) -> dict:
        return self.get("/v3/reference/dividends",
                        {"ex_dividend_date.gte": gte.isoformat(), "ex_dividend_date.lte": lte.isoformat(),
                         "limit": 1000, "sort": "ex_dividend_date", "order": "asc"},
                        cache_key=f"dividends/{gte.isoformat()}_{lte.isoformat()}", ttl_s=ttl_s)

    def dividends_ticker(self, ticker: str, *, ttl_s: float = 180 * _DAY) -> dict:
        """Complete dividend history for one ticker (the reference layer is not time-capped)."""
        return self.get("/v3/reference/dividends",
                        {"ticker": api_ticker(ticker), "limit": 1000, "sort": "ex_dividend_date", "order": "asc"},
                        cache_key=f"dividends_ticker/{normalize_ticker(ticker)}", ttl_s=ttl_s, max_pages=5)

    def splits_ticker(self, ticker: str, *, ttl_s: float = 180 * _DAY) -> dict:
        return self.get("/v3/reference/splits",
                        {"ticker": api_ticker(ticker), "limit": 1000, "sort": "execution_date", "order": "asc"},
                        cache_key=f"splits_ticker/{normalize_ticker(ticker)}", ttl_s=ttl_s, max_pages=2)

    def short_interest_ticker(self, ticker: str, *, ttl_s: float = 14 * _DAY) -> dict:
        """Bi-monthly short interest back to 2017-12."""
        return self.get("/stocks/v1/short-interest",
                        {"ticker": api_ticker(ticker), "settlement_date.gte": "2000-01-01", "limit": 50000},
                        cache_key=f"short_interest_ticker/{normalize_ticker(ticker)}", ttl_s=ttl_s, max_pages=2)

    def financials(self, ticker: str, *, ttl_s: float = 7 * _DAY) -> dict:
        return self.get("/vX/reference/financials",
                        {"ticker": api_ticker(ticker), "limit": 100, "sort": "filing_date", "order": "desc"},
                        cache_key=f"financials/{normalize_ticker(ticker)}", ttl_s=ttl_s, max_pages=3)

    def news(self, *, ticker: str | None = None, gte: datetime | None = None,
             lte: datetime | None = None, limit: int = 1000, max_pages: int = 10,
             cache_key: str | None = None, ttl_s: float | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit, "sort": "published_utc", "order": "asc"}
        if ticker:
            params["ticker"] = api_ticker(ticker)
        if gte:
            params["published_utc.gte"] = gte.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if lte:
            params["published_utc.lt"] = lte.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.get("/v2/reference/news", params, cache_key=cache_key, ttl_s=ttl_s, max_pages=max_pages)

    def exchanges(self, *, ttl_s: float = 30 * _DAY) -> dict:
        return self.get("/v3/reference/exchanges", {"asset_class": "stocks", "locale": "us"},
                        cache_key="reference/exchanges", ttl_s=ttl_s, paginate=False)

    def ticker_types(self, *, ttl_s: float = 30 * _DAY) -> dict:
        return self.get("/v3/reference/tickers/types", {"asset_class": "stocks", "locale": "us"},
                        cache_key="reference/ticker_types", ttl_s=ttl_s, paginate=False)

    def ticker_events(self, ticker: str, *, ttl_s: float = 30 * _DAY) -> dict:
        """Symbol-change history (point-in-time ticker mapping)."""
        return self.get(f"/vX/reference/tickers/{api_ticker(ticker)}/events", cache_key=f"events/{normalize_ticker(ticker)}",
                        ttl_s=ttl_s, paginate=False)

    def related_companies(self, ticker: str, *, ttl_s: float = 30 * _DAY) -> dict:
        return self.get(f"/v1/related-companies/{api_ticker(ticker)}", cache_key=f"related/{normalize_ticker(ticker)}",
                        ttl_s=ttl_s, paginate=False)

    def ipos(self, *, ttl_s: float = _DAY) -> dict:
        return self.get("/vX/reference/ipos", {"limit": 1000, "order": "desc", "sort": "listing_date"},
                        cache_key="reference/ipos", ttl_s=ttl_s, max_pages=10)

    def short_interest(self, gte: date, *, ttl_s: float = _DAY) -> dict:
        return self.get("/stocks/v1/short-interest", {"settlement_date.gte": gte.isoformat(), "limit": 50000},
                        cache_key=f"short_interest/{gte.isoformat()}", ttl_s=ttl_s, max_pages=4)

    def short_volume(self, d: date, *, ttl_s: float | None) -> dict:
        return self.get("/stocks/v1/short-volume", {"date": d.isoformat(), "limit": 50000},
                        cache_key=f"short_volume/{d.isoformat()}", ttl_s=ttl_s, max_pages=2)

    def short_volume_ticker(self, ticker: str, *, ttl_s: float = 30 * _DAY) -> dict:
        """Daily FINRA short-volume history for one ticker (data set starts 2024-02)."""
        return self.get("/stocks/v1/short-volume",
                        {"ticker": api_ticker(ticker), "limit": 50000, "sort": "date.asc"},
                        cache_key=f"short_volume_ticker/{normalize_ticker(ticker)}", ttl_s=ttl_s, max_pages=2)

    def aggs_minute(self, ticker: str, start: date, end: date, *, ttl_s: float | None = None) -> dict:
        """Unadjusted 1-minute bars (pre/regular/post sessions) for [start, end] — one doc per calendar
        month (≈18k rows for a mega-cap, one page).  The free plan serves the trailing 2 years only and
        answers 403 outside it, so callers clip ``start`` to :meth:`MassiveStore.history_start`."""
        return self.get(f"/v2/aggs/ticker/{api_ticker(ticker)}/range/1/minute/{start.isoformat()}/{end.isoformat()}",
                        {"adjusted": "false", "sort": "asc", "limit": 50000},
                        cache_key=f"aggs_minute/{normalize_ticker(ticker)}/{start:%Y-%m}", ttl_s=ttl_s, max_pages=4)

    def fed(self, series: str, *, ttl_s: float = _DAY) -> dict:
        """``treasury-yields`` | ``inflation`` | ``inflation-expectations`` (Fed data set)."""
        return self.get(f"/fed/v1/{series}", {"limit": 5000, "sort": "date.desc"},
                        cache_key=f"fed/{series}", ttl_s=ttl_s, max_pages=10)

    def holidays(self, *, ttl_s: float = 7 * _DAY) -> dict:
        return self._whole("/v1/marketstatus/upcoming", "reference/holidays", ttl_s)

    def _whole(self, path: str, cache_key: str, ttl_s: float | None) -> dict:
        """Endpoints that return a bare JSON list (no ``results`` wrapper)."""
        hit = self.cache_read(cache_key, ttl_s)
        if hit is not None:
            self.cache_hits += 1
            return hit
        payload = self._request(self.base + path, None)
        results = payload if isinstance(payload, list) else payload.get("results", [])
        doc = {"results": results, "meta": {}, "pages": 1, "fetched_at": self.clock(), "path": path}
        self.cache_write(cache_key, doc)
        return doc


# ─────────────────────────────────────────────────────────────────────────────
# Pure transforms (unit-tested, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

_BAR_SCHEMA = {"date": pl.Date, "ticker": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
               "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
               "vwap": pl.Float64, "n_trades": pl.Int64, "otc": pl.Boolean}


def grouped_to_frame(d: date, results: list[dict]) -> pl.DataFrame:
    """Grouped-daily JSON rows (``T,o,h,l,c,v,vw,n,otc``) → tidy RAW bars for ``d``."""
    if not results:
        return pl.DataFrame(schema=_BAR_SCHEMA)
    rows = []
    for r in results:
        t = r.get("T")
        if not t or r.get("c") is None:
            continue
        rows.append({"date": d, "ticker": normalize_ticker(t),
                     "open": float(r.get("o") or r["c"]), "high": float(r.get("h") or r["c"]),
                     "low": float(r.get("l") or r["c"]), "close": float(r["c"]),
                     "volume": float(r.get("v") or 0.0),
                     "vwap": float(r["vw"]) if r.get("vw") is not None else None,
                     "n_trades": int(r["n"]) if r.get("n") is not None else None,
                     "otc": bool(r.get("otc", False))})
    return pl.DataFrame(rows, schema=_BAR_SCHEMA) if rows else pl.DataFrame(schema=_BAR_SCHEMA)


def aggs_to_frame(ticker: str, results: list[dict]) -> pl.DataFrame:
    """Per-ticker range aggregates (``t`` ms epoch) → RAW bars."""
    rows = []
    for r in results or []:
        if r.get("c") is None or r.get("t") is None:
            continue
        d = datetime.fromtimestamp(int(r["t"]) / 1000, tz=timezone.utc).date()
        rows.append({"date": d, "ticker": normalize_ticker(ticker),
                     "open": float(r.get("o") or r["c"]), "high": float(r.get("h") or r["c"]),
                     "low": float(r.get("l") or r["c"]), "close": float(r["c"]),
                     "volume": float(r.get("v") or 0.0),
                     "vwap": float(r["vw"]) if r.get("vw") is not None else None,
                     "n_trades": int(r["n"]) if r.get("n") is not None else None, "otc": False})
    return pl.DataFrame(rows, schema=_BAR_SCHEMA) if rows else pl.DataFrame(schema=_BAR_SCHEMA)


_MINUTE_SCHEMA = {"ts": pl.Int64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                  "vwap": pl.Float64, "volume": pl.Float64, "trades": pl.Int64}
MINUTE_COLS = ["ts", "date", "session", "ticker", "open", "high", "low", "close", "vwap", "volume", "trades"]


def minute_to_frame(ticker: str, results: list[dict]) -> pl.DataFrame:
    """``/v2/aggs/ticker/{t}/range/1/minute`` rows → tidy bars with a UTC ``ts`` (bar open, exact to the
    minute), the New-York trading ``date`` and a ``session`` tag (pre < 09:30 ≤ regular < 16:00 ≤ post).
    Prices are as traded (unadjusted); apply ``splits.parquet`` factors if you need a continuous series."""
    rows = [{"ts": int(r["t"]), "open": r.get("o"), "high": r.get("h"), "low": r.get("l"), "close": r.get("c"),
             "vwap": r.get("vw"), "volume": r.get("v"), "trades": r.get("n")} for r in results or [] if r.get("t") is not None]
    if not rows:
        return pl.DataFrame(schema={**{"ts": pl.Datetime("us", "UTC"), "date": pl.Date, "session": pl.Utf8, "ticker": pl.Utf8},
                                    **{k: v for k, v in _MINUTE_SCHEMA.items() if k != "ts"}}).select(MINUTE_COLS)
    et = pl.col("ts").dt.convert_time_zone("America/New_York")
    mins = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)     # hour() is i8 → would overflow
    return (pl.DataFrame(rows, schema=_MINUTE_SCHEMA)
              .with_columns(pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC"),
                            ticker=pl.lit(normalize_ticker(ticker)))
              .with_columns(date=et.dt.date(),
                            session=pl.when(mins < 570).then(pl.lit("pre")).when(mins < 960).then(pl.lit("regular"))
                                      .otherwise(pl.lit("post")))
              .unique(subset=["ts"], keep="last").sort("ts").select(MINUTE_COLS))


def splits_frame(results: list[dict]) -> pl.DataFrame:
    rows = [{"ticker": normalize_ticker(r["ticker"]), "execution_date": date.fromisoformat(r["execution_date"][:10]),
             "split_from": float(r["split_from"]), "split_to": float(r["split_to"]), "id": str(r.get("id", ""))}
            for r in results or [] if r.get("ticker") and r.get("execution_date")
            and r.get("split_from") and r.get("split_to")]
    schema = {"ticker": pl.Utf8, "execution_date": pl.Date, "split_from": pl.Float64,
              "split_to": pl.Float64, "id": pl.Utf8}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def dividends_frame(results: list[dict]) -> pl.DataFrame:
    def _d(v):
        return date.fromisoformat(v[:10]) if v else None
    rows = [{"ticker": normalize_ticker(r["ticker"]), "ex_dividend_date": _d(r.get("ex_dividend_date")),
             "cash_amount": float(r.get("cash_amount") or 0.0), "currency": r.get("currency"),
             "dividend_type": r.get("dividend_type"), "frequency": int(r.get("frequency") or 0),
             "declaration_date": _d(r.get("declaration_date")), "record_date": _d(r.get("record_date")),
             "pay_date": _d(r.get("pay_date")), "id": str(r.get("id", ""))}
            for r in results or [] if r.get("ticker") and r.get("ex_dividend_date")]
    schema = {"ticker": pl.Utf8, "ex_dividend_date": pl.Date, "cash_amount": pl.Float64,
              "currency": pl.Utf8, "dividend_type": pl.Utf8, "frequency": pl.Int64,
              "declaration_date": pl.Date, "record_date": pl.Date, "pay_date": pl.Date, "id": pl.Utf8}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def apply_corporate_actions(bars: pl.DataFrame, splits: pl.DataFrame | None,
                            dividends: pl.DataFrame | None) -> pl.DataFrame:
    """RAW bars → split-adjusted OHLC/volume + ``adj_close`` (split + dividend), CRSP-style.

    ``bars``: date, ticker, open, high, low, close, volume (+ any extras, kept).
    Returns the same rows with ``raw_close``, ``split_factor``, ``div_factor``,
    adjusted ``open/high/low/close/volume`` and ``adj_close``.
    """
    if bars.is_empty():
        return bars
    out = bars.with_columns(pl.col("close").alias("raw_close")).sort("date")

    # -- splits: s_t = ∏_{exec_date > t} from/to, matched via asof(forward) on exec_date-1 ≥ t
    if splits is not None and not splits.is_empty():
        s = (splits.filter((pl.col("split_from") > 0) & (pl.col("split_to") > 0))
                   .with_columns((pl.col("split_from") / pl.col("split_to")).alias("_r"))
                   .sort(["ticker", "execution_date"])
                   .with_columns(pl.col("_r").cum_prod(reverse=True).over("ticker").alias("split_factor"),
                                 pl.col("execution_date").dt.offset_by("-1d").alias("_key"))
                   .select("ticker", "_key", "split_factor").sort("_key"))
        out = out.join_asof(s, left_on="date", right_on="_key", by="ticker", strategy="forward", check_sortedness=False).drop("_key", strict=False)
    if "split_factor" not in out.columns:
        out = out.with_columns(pl.lit(1.0).alias("split_factor"))
    out = out.with_columns(pl.col("split_factor").fill_null(1.0))
    out = out.with_columns([(pl.col(c) * pl.col("split_factor")).alias(c) for c in ("open", "high", "low", "close")]
                           + [(pl.col("volume") / pl.col("split_factor")).alias("volume")])

    # -- dividends: f_d = 1 − cash / raw_close_{last bar < ex_date}; f_t = ∏_{d > t} f_d
    if dividends is not None and not dividends.is_empty():
        dv = (dividends.filter(pl.col("cash_amount") > 0)
                       .with_columns(pl.col("ex_dividend_date").dt.offset_by("-1d").alias("_key"))
                       .sort("_key"))
        prev = dv.join_asof(out.select("ticker", "date", "raw_close").sort("date"),
                            left_on="_key", right_on="date", by="ticker", strategy="backward", check_sortedness=False)
        prev = (prev.filter(pl.col("raw_close").is_not_null() & (pl.col("raw_close") > 0))
                    .with_columns((1.0 - pl.col("cash_amount") / pl.col("raw_close")).alias("_f"))
                    .with_columns(pl.when((pl.col("_f") > 0) & (pl.col("_f") <= 1)).then(pl.col("_f")).otherwise(1.0).alias("_f"))
                    .group_by(["ticker", "ex_dividend_date"]).agg(pl.col("_f").product().alias("_f"))
                    .sort(["ticker", "ex_dividend_date"])
                    .with_columns(pl.col("_f").cum_prod(reverse=True).over("ticker").alias("div_factor"),
                                  pl.col("ex_dividend_date").dt.offset_by("-1d").alias("_key"))
                    .select("ticker", "_key", "div_factor").sort("_key"))
        out = out.join_asof(prev, left_on="date", right_on="_key", by="ticker", strategy="forward", check_sortedness=False).drop("_key", strict=False)
    if "div_factor" not in out.columns:
        out = out.with_columns(pl.lit(1.0).alias("div_factor"))
    out = out.with_columns(pl.col("div_factor").fill_null(1.0))
    out = out.with_columns((pl.col("close") * pl.col("div_factor")).alias("adj_close"))
    return out.sort(["ticker", "date"])


def splice_history(deep: pl.DataFrame, recent: pl.DataFrame) -> pl.DataFrame:
    """Per ticker: ``deep`` rows strictly before ``recent`` starts, rescaled at the seam, then ``recent``.

    The seam is the earliest date present in both; deep OHLC/volume are scaled by
    ``recent.close/deep.close`` and deep ``adj_close`` by ``recent.adj_close/deep.adj_close``
    there, so both price series are continuous across the join.  Tickers present
    on only one side pass through unchanged.
    """
    cols = OHLCV_COLS
    if recent.is_empty():
        return deep.select([c for c in cols if c in deep.columns]).sort(["ticker", "date"])
    if deep.is_empty():
        return recent.select(cols).sort(["ticker", "date"])
    deep = deep.select(cols).with_columns(pl.col("date").cast(pl.Date))
    recent = recent.select(cols).with_columns(pl.col("date").cast(pl.Date))
    r0 = recent.group_by("ticker").agg(pl.col("date").min().alias("_r0"))
    seam = (deep.join(recent, on=["ticker", "date"], suffix="_r").sort(["ticker", "date"])
                .group_by("ticker", maintain_order=True).first()
                .select("ticker",
                        (pl.col("close_r") / pl.col("close")).alias("_px"),
                        (pl.col("adj_close_r") / pl.col("adj_close")).alias("_adj")))
    d2 = (deep.join(r0, on="ticker", how="left")
              .filter(pl.col("_r0").is_null() | (pl.col("date") < pl.col("_r0")))
              .join(seam, on="ticker", how="left")
              .with_columns(pl.col("_px").fill_null(1.0).fill_nan(1.0), pl.col("_adj").fill_null(1.0).fill_nan(1.0))
              .with_columns([(pl.col(c) * pl.col("_px")).alias(c) for c in ("open", "high", "low", "close")]
                            + [(pl.col("adj_close") * pl.col("_adj")).alias("adj_close"),
                               (pl.col("volume") / pl.col("_px")).alias("volume")])
              .select(cols))
    return pl.concat([d2, recent], how="vertical_relaxed").sort(["ticker", "date"])


def flatten_financials(results: list[dict]) -> pl.DataFrame:
    """``/vX/reference/financials`` rows → one wide row per filing; every ``*.value`` leaf kept."""
    rows = []
    for r in results or []:
        row = {"ticker": normalize_ticker(r.get("tickers", [None])[0] or ""), "cik": r.get("cik"),
               "company_name": r.get("company_name"), "fiscal_year": r.get("fiscal_year"),
               "fiscal_period": r.get("fiscal_period"), "timeframe": r.get("timeframe"),
               "start_date": r.get("start_date"), "end_date": r.get("end_date"),
               "filing_date": r.get("filing_date"), "acceptance_datetime": r.get("acceptance_datetime"),
               "source_filing_url": r.get("source_filing_url"), "source_filing_file_url": r.get("source_filing_file_url")}
        for stmt, items in (r.get("financials") or {}).items():
            if not isinstance(items, dict):
                continue
            for name, leaf in items.items():
                if isinstance(leaf, dict) and "value" in leaf:
                    try:
                        row[f"{stmt}__{name}"] = float(leaf["value"])
                    except (TypeError, ValueError):
                        pass
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    for c in ("start_date", "end_date", "filing_date"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False))
    return df


def flatten_news(results: list[dict]) -> pl.DataFrame:
    """News articles → tidy (article × ticker) rows with per-ticker LLM sentiment when present."""
    rows = []
    for a in results or []:
        pub = a.get("published_utc")
        tickers = [normalize_ticker(t) for t in (a.get("tickers") or []) if t]
        insights = {normalize_ticker(i.get("ticker", "")): i for i in (a.get("insights") or []) if i.get("ticker")}
        base = {"article_id": a.get("id"), "published_utc": pub, "title": a.get("title"),
                "description": a.get("description"), "author": a.get("author"),
                "publisher": (a.get("publisher") or {}).get("name"), "article_url": a.get("article_url"),
                "keywords": list(a.get("keywords") or [])}
        for t in tickers or ["UNKNOWN"]:
            ins = insights.get(t, {})
            sent = ins.get("sentiment")
            rows.append({**base, "ticker": t,
                         "sentiment": {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(sent),
                         "sentiment_label": sent, "sentiment_reasoning": ins.get("sentiment_reasoning")})
    schema = {"article_id": pl.Utf8, "published_utc": pl.Utf8, "title": pl.Utf8, "description": pl.Utf8,
              "author": pl.Utf8, "publisher": pl.Utf8, "article_url": pl.Utf8, "keywords": pl.List(pl.Utf8),
              "ticker": pl.Utf8, "sentiment": pl.Float64, "sentiment_label": pl.Utf8, "sentiment_reasoning": pl.Utf8}
    if not rows:
        return pl.DataFrame(schema=schema)
    return (pl.DataFrame(rows, schema=schema)
              .with_columns(pl.col("published_utc").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False, time_zone="UTC"))
              .unique(subset=["article_id", "ticker"], keep="first"))


def flatten_details(results: list[dict]) -> pl.DataFrame:
    keep = ["ticker", "name", "market", "locale", "primary_exchange", "type", "active", "currency_name",
            "cik", "composite_figi", "share_class_figi", "sic_code", "sic_description", "market_cap",
            "share_class_shares_outstanding", "weighted_shares_outstanding", "total_employees",
            "list_date", "homepage_url", "description", "delisted_utc", "last_updated_utc"]
    rows = [{k: r.get(k) for k in keep} for r in results or [] if r.get("ticker")]
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.with_columns(pl.col("ticker").str.strip_chars().str.to_uppercase().str.replace_all(".", "-", literal=True))
    for c in ("market_cap", "share_class_shares_outstanding", "weighted_shares_outstanding", "total_employees"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    return df


def flat_rows(results: list[dict], extra: dict | None = None) -> list[dict]:
    """Flatten nested dicts one level per ``__`` (any depth); lists become JSON strings.
    Used for the long tail of reference tables where we keep *every* field."""
    out = []
    for r in results or []:
        row: dict = dict(extra or {})

        def walk(prefix: str, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}{k}__", v)
            elif isinstance(obj, list):
                row[prefix[:-2]] = json.dumps(obj)
            else:
                row[prefix[:-2]] = obj
        walk("", r if isinstance(r, dict) else {"value": r})
        out.append(row)
    return out


def flatten_events(ticker: str, results: list[dict]) -> list[dict]:
    """``/vX/reference/tickers/{t}/events`` → one row per event (ticker changes etc.)."""
    rows = []
    for r in results or []:
        for ev in r.get("events") or []:
            change = ev.get("ticker_change") or {}
            rows.append({"ticker": normalize_ticker(ticker), "name": r.get("name"), "cik": r.get("cik"),
                         "composite_figi": r.get("composite_figi"), "date": ev.get("date"),
                         "type": ev.get("type"), "new_ticker": normalize_ticker(change.get("ticker", "")) or None})
    return rows


def business_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-month (first, last) pairs covering [start, end]."""
    out = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        out.append((cur, nxt - timedelta(days=1)))
        cur = nxt
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Store — orchestrates crawl + build; every step is resumable and cache-first
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrawlSummary:
    calls: int = 0
    cache_hits: int = 0
    fetched_days: int = 0
    empty_days: int = 0
    pending_days: int = 0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "fetched_days": self.fetched_days,
                "empty_days": self.empty_days, "pending_days": self.pending_days, **self.details}


class MassiveStore:
    """Cache-first crawler + builder. ``raw_dir`` = gz-JSON cache, ``bronze_dir`` = parquet outputs."""

    def __init__(self, raw_dir: Path, bronze_dir: Path, client: MassiveClient | None = None,
                 rpm: int = DEFAULT_RPM, include_otc: bool = True, history_years: int = HISTORY_YEARS,
                 today: date | None = None):
        self.raw_dir = Path(raw_dir)
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.rpm = rpm
        self.include_otc = include_otc
        self.history_years = history_years
        self._today_fixed = today

    @property
    def today(self) -> date:
        """UTC date, evaluated per call so a long-running crawler rolls over at midnight (tests freeze it)."""
        return self._today_fixed or datetime.now(timezone.utc).date()

    @classmethod
    def from_config(cls, cfg, client: MassiveClient | None = None) -> "MassiveStore":
        m = (cfg.get("data", {}) or {}).get("massive", {}) or {}
        raw = cfg.path("data_raw") / "massive"
        bronze = cfg.path("data_bronze") / "massive"
        store = cls(raw, bronze, client=client, rpm=int(m.get("rpm", DEFAULT_RPM)),
                    include_otc=bool(m.get("include_otc", True)),
                    history_years=int(m.get("history_years", HISTORY_YEARS)))
        store._base = m.get("base_url", MASSIVE_BASE)
        return store

    @property
    def client(self) -> MassiveClient:
        if self._client is None:
            self._client = MassiveClient(base=getattr(self, "_base", MASSIVE_BASE),
                                         cache_dir=self.raw_dir, rpm=self.rpm)
        return self._client

    # ---- paths ------------------------------------------------------------------
    @property
    def raw_bars_path(self) -> Path:      return self.bronze_dir / "bars_raw.parquet"
    @property
    def ohlcv_all_path(self) -> Path:     return self.bronze_dir / "ohlcv_all.parquet"
    @property
    def splits_path(self) -> Path:        return self.bronze_dir / "splits.parquet"
    @property
    def dividends_path(self) -> Path:     return self.bronze_dir / "dividends.parquet"
    @property
    def tickers_path(self) -> Path:       return self.bronze_dir / "tickers.parquet"
    @property
    def details_path(self) -> Path:       return self.bronze_dir / "details.parquet"
    @property
    def financials_path(self) -> Path:    return self.bronze_dir / "financials.parquet"
    @property
    def news_path(self) -> Path:          return self.bronze_dir / "news.parquet"
    @property
    def deep_path(self) -> Path:          return self.bronze_dir / "ohlcv_deep.parquet"
    @property
    def minute_dir(self) -> Path:         return self.bronze_dir / "bars_minute"      # one parquet per ticker

    # ---- intraday (universe only) ----------------------------------------------
    def minute_windows(self) -> list[tuple[date, date]]:
        """Calendar-month windows covering the plan's 2-year intraday window, oldest first; the first one
        is clipped to :meth:`history_start` so we never ask for a day the plan would 403."""
        lo = self.history_start()
        return [(max(a, lo), min(b, self.today)) for a, b in month_windows(lo, self.today)]

    def build_minute_bars(self, tickers: Iterable[str] | None = None, force: bool = False) -> dict[str, int]:
        """``bronze/massive/bars_minute/{ticker}.parquet`` from the ``aggs_minute/{ticker}/{YYYY-MM}`` docs.
        Incremental: a ticker is rebuilt only when one of its month docs is newer than its parquet."""
        root = self.raw_dir / "aggs_minute"
        out: dict[str, int] = {}
        if not root.exists():
            return out
        want = {normalize_ticker(t) for t in tickers} if tickers is not None else None
        self.minute_dir.mkdir(parents=True, exist_ok=True)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            t = d.name
            if want is not None and t not in want:
                continue
            docs = sorted(d.glob("*.json.gz"))
            if not docs:
                continue
            target = self.minute_dir / f"{t}.parquet"
            if not force and target.exists() and target.stat().st_mtime >= max(p.stat().st_mtime for p in docs):
                continue
            frames = []
            for p in docs:
                doc = self.client.cache_read(f"aggs_minute/{t}/{p.name[:-8]}", None)
                if doc and doc["results"]:
                    frames.append(minute_to_frame(t, doc["results"]))
            if not frames:
                continue
            df = pl.concat(frames).unique(subset=["ts"], keep="last").sort("ts")
            tmp = target.with_suffix(".tmp")
            df.write_parquet(tmp, compression="zstd")
            os.replace(tmp, target)
            out[t] = df.height
        return out

    def minute_summary(self) -> dict | None:
        files = sorted(self.minute_dir.glob("*.parquet")) if self.minute_dir.exists() else []
        if not files:
            return None
        try:
            lf = pl.scan_parquet([str(p) for p in files])
            agg = lf.select(pl.len().alias("rows"), pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last")).collect()
            return {"rows": int(agg["rows"][0]), "tickers": len(files),
                    "first": str(agg["first"][0])[:16], "last": str(agg["last"][0])[:16],
                    "age_h": round((time.time() - max(p.stat().st_mtime for p in files)) / 3600, 1)}
        except Exception:
            return {"rows": -1, "tickers": len(files)}

    def build_deep_prices(self, tickers: Iterable[str], start: str = "1970-01-01",
                          fetch: Callable[..., pl.DataFrame] | None = None) -> pl.DataFrame:
        """Continuous multi-decade daily bars for ``tickers``: yfinance from ``start`` (the plan caps
        Massive bars at 2y — verified) spliced per ticker onto the Massive window, which is
        authoritative where it exists. Written to ``ohlcv_deep.parquet`` with a ``source`` column."""
        if fetch is None:
            from .market_data import fetch_ohlcv as fetch
        tickers = list(dict.fromkeys(normalize_ticker(t) for t in tickers))
        deep = fetch(tickers, start=start, end=None, progress=False)
        if deep.is_empty():
            raise RuntimeError("yfinance deep fetch returned nothing")
        deep = deep.select([c for c in OHLCV_COLS if c in deep.columns])
        if "adj_close" not in deep.columns:
            deep = deep.with_columns(pl.col("close").alias("adj_close"))
        recent = pl.DataFrame()
        if self.ohlcv_all_path.exists():
            recent = (pl.read_parquet(self.ohlcv_all_path, columns=OHLCV_COLS)
                        .filter(pl.col("ticker").is_in(tickers)))
        out = splice_history(deep, recent)
        if not recent.is_empty():
            r0 = recent.group_by("ticker").agg(pl.col("date").min().alias("_r0"))
            out = (out.join(r0, on="ticker", how="left")
                      .with_columns(pl.when(pl.col("_r0").is_not_null() & (pl.col("date") >= pl.col("_r0")))
                                      .then(pl.lit("massive")).otherwise(pl.lit("yfinance")).alias("source"))
                      .drop("_r0"))
        else:
            out = out.with_columns(pl.lit("yfinance").alias("source"))
        out = out.with_columns(pl.col("volume").round(0).cast(pl.Int64, strict=False)).sort(["ticker", "date"])
        out.write_parquet(self.deep_path, compression="zstd")
        logger.info(f"massive: ohlcv_deep {out.height:,} rows · {out['ticker'].n_unique()} tickers · "
                    f"{out['date'].min()} → {out['date'].max()}")
        return out

    def history_start(self) -> date:
        return self.today - timedelta(days=365 * self.history_years - 7)

    # ---- grouped daily bars -------------------------------------------------------
    def cached_days(self) -> list[date]:
        d = self.raw_dir / "grouped"
        if not d.exists():
            return []
        out = []
        for p in d.glob("*.json.gz"):
            try:
                out.append(date.fromisoformat(p.name[:10]))
            except ValueError:
                pass
        return sorted(out)

    def _grouped_ttl(self, d: date) -> float | None:
        """Past days are immutable; the last few may not be published yet → short TTL when empty."""
        return None if d < self.today - timedelta(days=3) else 2 * 3600

    def fetch_day(self, d: date) -> int:
        """Ensure the grouped bar for ``d`` is cached; returns the row count (0 = holiday/not yet)."""
        ttl = self._grouped_ttl(d)
        doc = self.client.cache_read(f"grouped/{d.isoformat()}", ttl)
        if doc is None or (not doc["results"] and ttl is not None):
            doc = self.client.grouped_daily(d, include_otc=self.include_otc, ttl_s=ttl)
        return len(doc["results"])

    def missing_days(self, start: date | None = None, end: date | None = None) -> list[date]:
        start = start or self.history_start()
        end = end or self.today
        have = set(self.cached_days())
        # recent empty days are re-tried (may simply not be published yet)
        recent_empty = set()
        for d in have:
            if d >= self.today - timedelta(days=3):
                doc = self.client.cache_read(f"grouped/{d.isoformat()}", None)
                if doc is not None and not doc["results"]:
                    recent_empty.add(d)
        return [d for d in business_days(start, end) if d not in have or d in recent_empty]

    def crawl_grouped(self, start: date | None = None, end: date | None = None,
                      max_calls: int | None = None, progress: Callable[[date, int], None] | None = None) -> CrawlSummary:
        """Fetch every missing business day in [start, end], newest first (so daily use is fast)."""
        s = CrawlSummary()
        todo = sorted(self.missing_days(start, end), reverse=True)
        c0 = self.client.calls
        for d in todo:
            if max_calls is not None and self.client.calls - c0 >= max_calls:
                break
            n = self.fetch_day(d)
            s.fetched_days += 1
            if n == 0:
                s.empty_days += 1
            if progress:
                progress(d, n)
        s.pending_days = len(self.missing_days(start, end))
        s.calls = self.client.calls - c0
        s.cache_hits = self.client.cache_hits
        return s

    # ---- reference / corporate actions -----------------------------------------
    def crawl_corporate_actions(self, start: date | None = None) -> int:
        """Splits + dividends by calendar month; past months cached 30d, current month 1d."""
        start = start or self.history_start()
        c0 = self.client.calls
        for lo, hi in month_windows(start, self.today):
            ttl = _DAY if hi >= self.today - timedelta(days=7) else 30 * _DAY
            self.client.splits(lo, hi, ttl_s=ttl)
            self.client.dividends(lo, hi, ttl_s=ttl)
        return self.client.calls - c0

    def crawl_tickers(self, include_delisted: bool = False) -> int:
        c0 = self.client.calls
        self.client.tickers(active=True)
        if include_delisted:
            self.client.tickers(active=False, ttl_s=30 * _DAY)
        return self.client.calls - c0

    def crawl_details(self, tickers: Iterable[str], max_calls: int | None = None) -> int:
        c0 = self.client.calls
        for t in tickers:
            if max_calls is not None and self.client.calls - c0 >= max_calls:
                break
            self.client.ticker_details(t)
        return self.client.calls - c0

    def crawl_financials(self, tickers: Iterable[str], max_calls: int | None = None) -> int:
        c0 = self.client.calls
        for t in tickers:
            if max_calls is not None and self.client.calls - c0 >= max_calls:
                break
            self.client.financials(t)
        return self.client.calls - c0

    def crawl_news_backfill(self, tickers: Iterable[str], start: date | None = None,
                            max_calls: int | None = None, ttl_s: float = 60 * _DAY) -> int:
        """Per-ticker news since ``start`` (one call/ticker/1000 articles). Stable cache key
        (``news/ticker/<T>``) so the rolling history window doesn't trigger daily refetches;
        ``crawl_news_recent`` keeps the last two weeks fresh in between."""
        start = start or self.history_start()
        gte = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        c0 = self.client.calls
        for t in tickers:
            if max_calls is not None and self.client.calls - c0 >= max_calls:
                break
            self.client.news(ticker=t, gte=gte, cache_key=f"news/ticker/{t}", ttl_s=ttl_s, max_pages=5)
        return self.client.calls - c0

    def crawl_news_deep(self, ticker: str, ttl_s: float = 365 * _DAY, max_pages: int = 60) -> int:
        """EVERY article Massive has for the ticker (archive starts 2017-04): one call per 1000."""
        c0 = self.client.calls
        self.client.news(ticker=ticker, gte=datetime(2000, 1, 1, tzinfo=timezone.utc),
                         cache_key=f"news/ticker_deep/{ticker}", ttl_s=ttl_s, max_pages=max_pages)
        return self.client.calls - c0

    def crawl_news_recent(self, ticker: str, days: int = 14, ttl_s: float = _DAY) -> int:
        """Rolling per-ticker refresh (last ``days``); merged with the one-off backfill at build time."""
        c0 = self.client.calls
        gte = datetime.now(timezone.utc) - timedelta(days=days)
        gte = gte.replace(hour=0, minute=0, second=0, microsecond=0)
        self.client.news(ticker=ticker, gte=gte, cache_key=f"news/ticker_recent/{ticker}", ttl_s=ttl_s, max_pages=2)
        return self.client.calls - c0

    def holiday_dates(self) -> set[date]:
        doc = self.client.cache_read("reference/holidays", None)
        out = set()
        for h in (doc or {}).get("results", []) or []:
            try:
                if h.get("status") == "closed" and h.get("exchange") in (None, "NYSE", "NASDAQ"):
                    out.add(date.fromisoformat(str(h.get("date"))[:10]))
            except ValueError:
                pass
        return out

    def crawl_news_daily(self, days: int = 3, max_pages: int = 6) -> int:
        """Whole-market news, one cache entry per UTC day (yesterday..today refreshed every 6 h)."""
        c0 = self.client.calls
        for i in range(days, -1, -1):
            d = self.today - timedelta(days=i)
            ttl = None if d < self.today - timedelta(days=1) else 6 * 3600
            gte = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            self.client.news(gte=gte, lte=gte + timedelta(days=1), cache_key=f"news/daily/{d.isoformat()}",
                             ttl_s=ttl, max_pages=max_pages)
        return self.client.calls - c0

    # ---- build ---------------------------------------------------------------------
    def build_raw_bars(self) -> pl.DataFrame:
        """Fold every cached grouped day into ``bars_raw.parquet`` (incremental: only new days parsed)."""
        have = pl.read_parquet(self.raw_bars_path) if self.raw_bars_path.exists() else pl.DataFrame(schema=_BAR_SCHEMA)
        done = set(have["date"].unique().to_list()) if have.height else set()
        new_frames = []
        for d in self.cached_days():
            if d in done:
                continue
            doc = self.client.cache_read(f"grouped/{d.isoformat()}", None)
            if doc and doc["results"]:
                new_frames.append(grouped_to_frame(d, doc["results"]))
        if new_frames:
            have = pl.concat([have] + new_frames, how="vertical_relaxed")
        cutoff = self.history_start() - timedelta(days=31)
        have = have.filter(pl.col("date") >= cutoff).unique(subset=["date", "ticker"], keep="last").sort(["ticker", "date"])
        have.write_parquet(self.raw_bars_path, compression="zstd")
        return have

    def build_corporate_actions(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Monthly whole-market windows (2y) ∪ per-ticker full histories (universe/extended)."""
        sp, dv = [], []
        for lo, hi in month_windows(self.history_start() - timedelta(days=31), self.today):
            s = self.client.cache_read(f"splits/{lo.isoformat()}_{hi.isoformat()}", None)
            d = self.client.cache_read(f"dividends/{lo.isoformat()}_{hi.isoformat()}", None)
            if s:
                sp.append(splits_frame(s["results"]))
            if d:
                dv.append(dividends_frame(d["results"]))
        for _, doc in self._docs_in("splits_ticker"):
            sp.append(splits_frame(doc["results"]))
        for _, doc in self._docs_in("dividends_ticker"):
            dv.append(dividends_frame(doc["results"]))
        splits = pl.concat(sp, how="vertical_relaxed").unique(subset=["ticker", "execution_date"]) if sp else splits_frame([])
        divs = pl.concat(dv, how="vertical_relaxed").unique(subset=["ticker", "ex_dividend_date", "cash_amount"]) if dv else dividends_frame([])
        splits.write_parquet(self.splits_path, compression="zstd")
        divs.write_parquet(self.dividends_path, compression="zstd")
        return splits, divs

    def build_ohlcv_all(self) -> pl.DataFrame:
        """RAW bars + corporate actions → ``ohlcv_all.parquet`` (whole market, adjusted)."""
        bars = self.build_raw_bars()
        if bars.is_empty():
            raise MassiveNotReady("no grouped days cached — run `ts massive backfill`")
        splits, divs = self.build_corporate_actions()
        adj = apply_corporate_actions(bars, splits, divs)
        adj.write_parquet(self.ohlcv_all_path, compression="zstd")
        logger.info(f"massive: ohlcv_all {adj.height:,} rows · {adj['ticker'].n_unique():,} tickers · "
                    f"{adj['date'].min()}→{adj['date'].max()}")
        return adj

    def build_reference(self, tickers: Iterable[str] | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        frames = []
        for kind in ("active", "delisted"):
            doc = self.client.cache_read(f"reference/tickers_{kind}", None)
            if doc and doc["results"]:
                frames.append(flatten_details(doc["results"]))
        if frames:
            tk = pl.concat(frames, how="diagonal_relaxed").unique(subset=["ticker"], keep="first")
            tk.write_parquet(self.tickers_path, compression="zstd")
            out["tickers"] = tk.height
        det, fin, news = [], [], []
        details_dir, fin_dir, news_dir = self.raw_dir / "details", self.raw_dir / "financials", self.raw_dir / "news"
        for p in sorted(details_dir.glob("*.json.gz")) if details_dir.exists() else []:
            doc = self.client.cache_read(f"details/{p.name[:-8]}", None)
            if doc and doc["results"]:
                det.append(flatten_details(doc["results"]))
        for p in sorted(fin_dir.glob("*.json.gz")) if fin_dir.exists() else []:
            doc = self.client.cache_read(f"financials/{p.name[:-8]}", None)
            if doc and doc["results"]:
                fin.append(flatten_financials(doc["results"]))
        for p in sorted(news_dir.rglob("*.json.gz")) if news_dir.exists() else []:
            rel = p.relative_to(self.raw_dir).as_posix()[:-8]
            doc = self.client.cache_read(rel, None)
            if doc and doc["results"]:
                news.append(flatten_news(doc["results"]))
        if det:
            df = pl.concat(det, how="diagonal_relaxed").unique(subset=["ticker"], keep="last")
            df.write_parquet(self.details_path, compression="zstd"); out["details"] = df.height
        if fin:
            df = (pl.concat(fin, how="diagonal_relaxed")
                    .unique(subset=["ticker", "fiscal_year", "fiscal_period", "timeframe"], keep="last")
                    .sort(["ticker", "filing_date"]))
            df.write_parquet(self.financials_path, compression="zstd"); out["financials"] = df.height
        if news:
            df = (pl.concat(news, how="diagonal_relaxed").unique(subset=["article_id", "ticker"], keep="last")
                    .sort(["ticker", "published_utc"]))
            df.write_parquet(self.news_path, compression="zstd"); out["news"] = df.height
        out.update(self.build_extra_tables())
        return out

    def _docs_in(self, folder: str):
        d = self.raw_dir / folder
        if not d.exists():
            return
        for p in sorted(d.glob("*.json.gz")):
            doc = self.client.cache_read(f"{folder}/{p.name[:-8]}", None)
            if doc:
                yield p.name[:-8], doc

    def build_extra_tables(self) -> dict[str, int]:
        """The long tail: events, related, ipos, fed_*, short_interest, short_volume, exchanges, ticker_types."""
        out: dict[str, int] = {}

        def write(name: str, rows: list[dict], subset: list[str] | None = None):
            if not rows:
                return
            df = pl.DataFrame(rows, infer_schema_length=None)
            if subset and all(c in df.columns for c in subset):
                df = df.unique(subset=subset, keep="last")
            df.write_parquet(self.bronze_dir / f"{name}.parquet", compression="zstd")
            out[name] = df.height

        write("events", [r for t, doc in self._docs_in("events") for r in flatten_events(t, doc["results"])],
              ["ticker", "date", "type"])
        write("related", [{"ticker": normalize_ticker(t), "related": normalize_ticker(r.get("ticker", ""))}
                          for t, doc in self._docs_in("related") for r in doc["results"] if r.get("ticker")],
              ["ticker", "related"])
        for name, folder in (("ipos", "reference"), ("exchanges", "reference"), ("ticker_types", "reference")):
            doc = self.client.cache_read(f"reference/{name}", None)
            if doc and doc["results"]:
                write(name, flat_rows(doc["results"]))
        for series, doc in self._docs_in("fed"):
            write(f"fed_{series.replace('-', '_')}", flat_rows(doc["results"]), ["date"])
        write("short_interest", [r for folder in ("short_interest", "short_interest_ticker")
                                 for _, doc in self._docs_in(folder) for r in flat_rows(doc["results"])],
              ["ticker", "settlement_date"])
        write("short_volume", [r for folder in ("short_volume", "short_volume_ticker")
                               for _, doc in self._docs_in(folder) for r in flat_rows(doc["results"])],
              ["ticker", "date"])
        return out

    # ---- status ----------------------------------------------------------------------
    def calls_today(self) -> int:
        p = self.raw_dir / f".calls-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        try:
            return p.stat().st_size
        except FileNotFoundError:
            return 0

    def status(self) -> dict:
        days = self.cached_days()
        st: dict[str, Any] = {
            "configured": is_configured(), "raw_dir": str(self.raw_dir), "bronze_dir": str(self.bronze_dir),
            "grouped_days": len(days), "first_day": days[0].isoformat() if days else None,
            "last_day": days[-1].isoformat() if days else None,
            "pending_days": len(self.missing_days()) if is_configured() else None,
            "calls_today": self.calls_today(), "history_start": self.history_start().isoformat(),
        }
        st["keys"] = len(api_keys())
        extra = sorted(p.stem for p in self.bronze_dir.glob("*.parquet")
                       if p.stem not in {"ohlcv_all", "bars_raw", "splits", "dividends", "tickers", "details", "financials", "news"})
        for name, p in [("ohlcv_all", self.ohlcv_all_path), ("splits", self.splits_path),
                        ("dividends", self.dividends_path), ("tickers", self.tickers_path),
                        ("details", self.details_path), ("financials", self.financials_path),
                        ("news", self.news_path)] + [(n, self.bronze_dir / f"{n}.parquet") for n in extra]:
            st[name] = None
            if p.exists():
                try:
                    lf = pl.scan_parquet(p)
                    st[name] = {"rows": lf.select(pl.len()).collect().item(),
                                "age_h": round((time.time() - p.stat().st_mtime) / 3600, 1)}
                except Exception:
                    st[name] = {"rows": -1}
        st["bars_minute"] = self.minute_summary()
        return st


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry points
# ─────────────────────────────────────────────────────────────────────────────

def _deep_history(cfg, tickers: list[str], cache_dir: Path, refresh_days: float, end: date) -> pl.DataFrame:
    """yfinance history for the pre-Massive era, cached per universe and refreshed weekly."""
    from .market_data import fetch_ohlcv
    name = (cfg.get("universe", {}) or {}).get("name", "universe")
    p = cache_dir / f"yf_deep_{name}.parquet"
    fresh = p.exists() and (time.time() - p.stat().st_mtime) < refresh_days * _DAY
    if fresh:
        df = pl.read_parquet(p)
        if set(tickers) <= set(df["ticker"].unique().to_list()):
            return df
    try:
        df = fetch_ohlcv(tickers, start=cfg["data"]["start_date"], end=end + timedelta(days=1), progress=False)
        if not df.is_empty():
            p.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(p, compression="zstd")
            return df
    except Exception as e:
        logger.warning(f"massive: yfinance deep-history refresh failed ({e}); using cached copy")
    return pl.read_parquet(p) if p.exists() else pl.DataFrame()


def update(cfg, universe_tickers: Iterable[str] | None = None, max_calls: int | None = None,
           news_days: int = 3, progress: Callable[[str], None] | None = None) -> dict:
    """Daily incremental: new grouped days → corp actions → whole-market news → rebuild tables.

    Bounded by ``max_calls`` (config ``data.massive.update_max_calls``, default 40 ≈ 8 min).
    """
    m = (cfg.get("data", {}) or {}).get("massive", {}) or {}
    max_calls = max_calls if max_calls is not None else int(m.get("update_max_calls", 40))
    store = MassiveStore.from_config(cfg)
    say = progress or (lambda s: logger.info(s))
    c0 = store.client.calls
    days = store.crawl_grouped(max_calls=max_calls, progress=lambda d, n: say(f"  grouped {d}: {n:,} bars"))
    say(f"grouped: fetched {days.fetched_days} day(s), {days.pending_days} still pending")
    left = max_calls - (store.client.calls - c0)
    if left > 0:
        store.crawl_corporate_actions(start=store.today - timedelta(days=45))
    left = max_calls - (store.client.calls - c0)
    if left > 0:
        store.crawl_news_daily(days=news_days)
    tables: dict = {}
    if store.cached_days():
        ohlcv = store.build_ohlcv_all()
        tables["ohlcv_all"] = ohlcv.height
    tables.update(store.build_reference(universe_tickers))
    return {"calls": store.client.calls - c0, "cache_hits": store.client.cache_hits,
            "grouped": days.as_dict(), "tables": tables, "pending_days": days.pending_days}


def ingest_universe_spliced(cfg, out: Path | None = None, min_recent_days: int = 60,
                            max_calls: int | None = 0) -> Path:
    """Build ``bronze/ohlcv_daily.parquet`` = yfinance deep history ⊕ Massive window (per ticker).

    ``max_calls`` > 0 lets this call top up missing grouped days itself; the daily
    pipeline runs ``ts massive update`` first, so the default is cache-only.
    Raises ``MassiveNotReady`` when the Massive window is too thin to trust.
    """
    from .market_data import sanitize_ohlcv
    m = (cfg.get("data", {}) or {}).get("massive", {}) or {}
    store = MassiveStore.from_config(cfg)
    if max_calls:
        store.crawl_grouped(max_calls=max_calls)
    if store.ohlcv_all_path.exists() and not store.missing_days():
        allbars = pl.read_parquet(store.ohlcv_all_path)
    else:
        allbars = store.build_ohlcv_all()
    n_days = allbars["date"].n_unique()
    if n_days < min_recent_days:
        raise MassiveNotReady(f"only {n_days} grouped days cached (< {min_recent_days}) — run `ts massive backfill`")
    last = allbars["date"].max()
    if last < store.today - timedelta(days=7):
        raise MassiveNotReady(f"massive window ends {last} (> 7 days stale) — run `ts massive update`")

    tickers = [normalize_ticker(t) for t in cfg["universe"]["tickers"]]
    recent = allbars.filter(pl.col("ticker").is_in(tickers)).select(OHLCV_COLS)
    covered = set(recent["ticker"].unique().to_list())
    missing = sorted(set(tickers) - covered)
    if missing:
        logger.warning(f"massive: {len(missing)} universe tickers absent from the grouped feed "
                       f"(yfinance-only): {', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}")
    # per-ticker threshold: names with only a few Massive days (new listings) stay yfinance-only for now
    counts = recent.group_by("ticker").agg(pl.len().alias("n"))
    thin = set(counts.filter(pl.col("n") < 20)["ticker"].to_list())
    if thin:
        recent = recent.filter(~pl.col("ticker").is_in(thin))

    deep = _deep_history(cfg, tickers, store.raw_dir, float(m.get("deep_refresh_days", 7)), last)
    df = sanitize_ohlcv(splice_history(deep, recent))
    if df.is_empty():
        raise RuntimeError("massive: spliced frame is empty")
    df = df.with_columns(pl.col("volume").round(0).cast(pl.Int64, strict=False))
    out = out or (cfg.path("data_bronze") / "ohlcv_daily.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    src = df.join(recent.select("ticker", "date").with_columns(pl.lit(True).alias("_m")), on=["ticker", "date"], how="left")
    n_m = int(src["_m"].fill_null(False).sum())
    logger.info(f"massive: wrote {df.height:,} rows → {out} ({n_m:,} from Massive, {df.height - n_m:,} yfinance deep; "
                f"{len(covered) - len(thin)}/{len(tickers)} tickers on Massive)")
    return out


def build_liquid_universe(store: MassiveStore, *, min_price: float = 5.0, min_dollar_vol: float = 20e6,
                          top: int = 600, lookback: int = 63, types: tuple[str, ...] = ("CS", "ADRC")) -> pl.DataFrame:
    """Rank the whole market by trailing-``lookback`` median dollar volume (point-in-time-able).

    Requires ``ohlcv_all.parquet``; uses ``tickers.parquet`` (if present) to keep
    common stock / ADRs only and exclude ETFs, warrants, units, preferreds.
    """
    if not store.ohlcv_all_path.exists():
        raise MassiveNotReady("ohlcv_all.parquet missing — run `ts massive backfill`")
    px = pl.read_parquet(store.ohlcv_all_path).filter(~pl.col("otc").fill_null(False))
    last = px["date"].max()
    dates = sorted(px["date"].unique().to_list())[-lookback:]
    win = px.filter(pl.col("date").is_in(dates))
    stats = (win.group_by("ticker").agg([
                pl.len().alias("n_days"),
                (pl.col("close") * pl.col("volume")).median().alias("dollar_vol"),
                pl.col("close").last().alias("last_close"),
                pl.col("date").max().alias("last_date")])
               .filter((pl.col("n_days") >= int(0.9 * len(dates))) & (pl.col("last_date") == last)
                       & (pl.col("last_close") >= min_price) & (pl.col("dollar_vol") >= min_dollar_vol)))
    if store.tickers_path.exists():
        ref = pl.read_parquet(store.tickers_path).select("ticker", "type", "name", "primary_exchange")
        stats = stats.join(ref, on="ticker", how="left").filter(pl.col("type").is_in(list(types)) | pl.col("type").is_null())
    return stats.sort("dollar_vol", descending=True).head(top)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous crawler — keeps the 5 req/min budget busy, most valuable work first
# ─────────────────────────────────────────────────────────────────────────────

_EQUITY_TYPES = ("CS", "ADRC")
BLOCK_TTL_S = 7 * _DAY            # how long a 403'd endpoint family is parked


@dataclass
class Task:
    tier: int
    name: str
    run: Callable[[], int]          # returns network calls made
    family: str = ""                # entitlement family (parked together on 403)


class Crawler:
    """Priority-scheduled crawl loop. Planning is cache-only (mtime ages), so every
    network call is real work.  Tiers (re-checked before *each* call):

      0  today-critical: newest grouped days once published, current-month corp
         actions, whole-market news for today/yesterday
      1  grouped-day backfill (newest first) — the 2-year whole-market panel
      2  directory + calendar + macro: active tickers (7d), holidays, exchanges,
         ticker types, monthly splits/dividends history (30d), delisted directory
         (30d), IPOs, Fed series, short interest / short volume (1d)
      3  universe depth: overview (30d), fundamentals (7d), EVERY news article
         (365d) + rolling 14d news (1d), full dividend/split history (180d), short
         interest (14d) + short volume (30d) history, ticker events + related (30d)
      4  universe INTRADAY: unadjusted 1-minute bars (pre/regular/post) for the
         plan's 2-year window, one doc per ticker-month; closed months are
         immutable, the open month refreshes daily
      5  EXTENDED depth — the top-``extended_top`` (1000) US common stocks/ADRs by
         trailing-63d dollar volume get the tier-3 treatment (no minute bars)
      6  whole-market depth, by liquidity rank: overview (45d), fundamentals
         (14d), rolling news (30d)
      7  QA: Massive's own split-adjusted series per universe+extended ticker
         (7d) so `ts massive status` can report the adjustment-maths discrepancy

    A 403 parks that endpoint *family* for a week (marker under ``.blocked/``)
    instead of retrying it; a 401 retires that key; SIGTERM finishes the current
    call, writes state and exits 0.
    """

    TTL = {"news_deep": 365 * _DAY, "actions_ticker": 180 * _DAY, "short_interest_ticker": 14 * _DAY,
           "short_volume_ticker": 30 * _DAY, "minute_open": _DAY, "deep_prices": 7 * _DAY,
           "tickers_active": 7 * _DAY, "tickers_delisted": 30 * _DAY, "holidays": 7 * _DAY,
           "static": 30 * _DAY, "daily": _DAY,
           "actions_current": _DAY, "actions_past": 30 * _DAY, "news_daily_recent": 6 * 3600,
           "details_universe": 30 * _DAY, "financials_universe": 7 * _DAY, "news_universe": _DAY,
           "news_backfill": 60 * _DAY, "events": 30 * _DAY,
           "details_market": 45 * _DAY, "financials_market": 14 * _DAY, "news_market": 30 * _DAY,
           "aggs_qa": 7 * _DAY}
    PUBLISH_LAG_H = 4.25            # grouped bar for D is tried from D+1 04:15 UTC — the free plan releases it
                                    # at ~04:00 UTC and answers 403 before that (measured 2026-09-22..26)
    ONCE_MAX_TIER = 5               # --once drains up to and including this tier (…extended)

    def __init__(self, store: MassiveStore, universe_tickers: Iterable[str], state_path: Path | None = None,
                 rebuild_every_s: float = 900.0, log: Callable[[str], None] | None = None,
                 extended_top: int = 1000, deep_start: str = "1970-01-01", deep_prices: bool = True):
        self.store = store
        self.universe = [normalize_ticker(t) for t in universe_tickers]
        self.extended_top = int(extended_top)
        self.deep_start = deep_start
        self.deep_prices = deep_prices
        self._deep_thread = None
        self.state_path = state_path or (store.raw_dir / "crawler_state.json")
        self.rebuild_every_s = rebuild_every_s
        self.log = log or (lambda m: logger.info(m))
        self._market: list[str] = []
        self._market_at = 0.0
        self._dirty_ohlcv = False
        self._dirty_ref = False
        self._dirty_minute = False
        self._last_build = 0.0
        self.stop = False
        self.tier_calls: dict[str, int] = {}
        self.started = time.time()
        self.blocked_dir = store.raw_dir / ".blocked"
        self._skip_until: dict[str, float] = {}      # task name → clock time (failure backoff)
        self._fail_count: dict[str, int] = {}

    # ---- helpers ----------------------------------------------------------------
    def _age(self, key: str) -> float | None:
        return self.store.client.cache_age(key)

    def _stale(self, key: str, ttl: float) -> bool:
        a = self._age(key)
        return a is None or a > ttl

    def _publish_ready(self, d: date) -> bool:
        return datetime.now(timezone.utc) >= datetime(d.year, d.month, d.day, tzinfo=timezone.utc) \
            + timedelta(days=1, hours=self.PUBLISH_LAG_H)

    def blocked(self, family: str) -> bool:
        try:
            return time.time() - (self.blocked_dir / family).stat().st_mtime < BLOCK_TTL_S
        except FileNotFoundError:
            return False

    def block(self, family: str, reason: str) -> None:
        self.blocked_dir.mkdir(parents=True, exist_ok=True)
        (self.blocked_dir / family).write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n")
        self.log(f"family '{family}' not in plan — parked for {BLOCK_TTL_S / _DAY:.0f} days")

    def blocked_families(self) -> list[str]:
        if not self.blocked_dir.exists():
            return []
        return sorted(p.name for p in self.blocked_dir.iterdir() if self.blocked(p.name))

    def market_tickers(self) -> list[str]:
        """Active CS/ADRC tickers ranked by trailing-63d median dollar volume (universe first)."""
        if self._market and time.time() - self._market_at < 3600:
            return self._market
        ranked: list[str] = []
        try:
            if self.store.tickers_path.exists():
                ref = pl.read_parquet(self.store.tickers_path)
                if "type" in ref.columns:
                    ref = ref.filter(pl.col("type").is_in(list(_EQUITY_TYPES)))
                if "active" in ref.columns:
                    ref = ref.filter(pl.col("active").fill_null(True))
                eq = set(ref["ticker"].to_list())
                if self.store.ohlcv_all_path.exists():
                    px = pl.read_parquet(self.store.ohlcv_all_path, columns=["date", "ticker", "close", "volume", "otc"])
                    dates = sorted(px["date"].unique().to_list())[-63:]
                    dv = (px.filter(pl.col("date").is_in(dates) & ~pl.col("otc").fill_null(False))
                            .group_by("ticker").agg((pl.col("close") * pl.col("volume")).median().alias("dv"))
                            .sort("dv", descending=True))
                    ranked = [t for t in dv["ticker"].to_list() if t in eq]
                    ranked += sorted(eq - set(ranked))
                else:
                    ranked = sorted(eq)
        except Exception as e:
            logger.warning(f"crawler: market ranking failed ({e}); using universe only")
        seen = set(self.universe)
        self._market = self.universe + [t for t in ranked if t not in seen]
        self._market_at = time.time()
        self._write_extended_universe(ranked)
        return self._market

    def extended_tickers(self) -> list[str]:
        """Top-N liquid names beyond the universe (needs the directory + ≥63 days of bars)."""
        mk = self.market_tickers()
        uni = set(self.universe)
        return [t for t in mk if t not in uni][: max(self.extended_top - len(uni), 0)]

    def _write_extended_universe(self, ranked: list[str]) -> None:
        """Gitignored YAML of the top-N by liquidity — promote with `ts massive universe` when wanted."""
        if not ranked:
            return
        try:
            px_days = pl.scan_parquet(self.store.ohlcv_all_path).select(pl.col("date").n_unique()).collect().item()
            if px_days < 63:
                return
            import yaml
            uni = set(self.universe)
            top = self.universe + [t for t in ranked if t not in uni]
            top = top[: self.extended_top]
            doc = {"name": "universe_extended", "benchmark": "SPY",
                   "description": f"Universe + top US common stocks/ADRs by trailing-63d median dollar volume "
                                  f"(Massive grouped bars, {self.store.today}); {len(top)} names, refreshed hourly by the crawler.",
                   "required": self.universe, "additions": [t for t in top if t not in uni]}
            out = self.store.bronze_dir / "universe_extended.yaml"
            out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        except Exception as e:
            logger.debug(f"crawler: extended universe yaml skipped: {e}")

    def _mark(self, tier: int, ohlcv: bool = False, ref: bool = False,
              minute: bool = False) -> Callable[[Callable[[], int]], Callable[[], int]]:
        def wrap(fn):
            def run():
                n = fn()
                self.tier_calls[str(tier)] = self.tier_calls.get(str(tier), 0) + n
                if n:
                    self._dirty_ohlcv |= ohlcv
                    self._dirty_ref |= ref
                    self._dirty_minute |= minute
                return n
            return run
        return wrap

    # ---- planning ---------------------------------------------------------------
    def _depth_tasks(self, tier: int, tickers: Iterable[str], *, details_ttl: float, fin_ttl: float,
                     news_recent_ttl: float, backfill: bool, events: bool, start: date):
        """``backfill=True`` (universe / extended) = full depth: EVERY news article (2017→), complete
        dividend/split history, short interest (2017→), ticker events, related companies.
        ``backfill=False`` (rest of market) = overview, fundamentals, rolling 30d news."""
        st, c, T = self.store, self.store.client, self.TTL
        for t in tickers:
            if not self.blocked("details") and self._stale(f"details/{t}", details_ttl):
                yield Task(tier, f"details {t}", self._mark(tier, ref=True)(lambda t=t: self._calls(lambda: c.ticker_details(t, ttl_s=0))), "details")
            if not self.blocked("financials") and self._stale(f"financials/{t}", fin_ttl):
                yield Task(tier, f"financials {t}", self._mark(tier, ref=True)(lambda t=t: self._calls(lambda: c.financials(t, ttl_s=0))), "financials")
            if not self.blocked("news"):
                if backfill and self._stale(f"news/ticker_deep/{t}", T["news_deep"]):
                    yield Task(tier, f"news-deep {t}", self._mark(tier, ref=True)(lambda t=t: st.crawl_news_deep(t, ttl_s=0)), "news")
                elif self._stale(f"news/ticker_recent/{t}", news_recent_ttl):
                    days = 14 if backfill else 30
                    yield Task(tier, f"news-recent {t}", self._mark(tier, ref=True)(lambda t=t, days=days: st.crawl_news_recent(t, days=days, ttl_s=0)), "news")
            if backfill:
                if not self.blocked("dividends") and self._stale(f"dividends_ticker/{t}", T["actions_ticker"]):
                    yield Task(tier, f"dividends-deep {t}", self._mark(tier, ohlcv=True)(lambda t=t: self._calls(lambda: c.dividends_ticker(t, ttl_s=0))), "dividends")
                if not self.blocked("splits") and self._stale(f"splits_ticker/{t}", T["actions_ticker"]):
                    yield Task(tier, f"splits-deep {t}", self._mark(tier, ohlcv=True)(lambda t=t: self._calls(lambda: c.splits_ticker(t, ttl_s=0))), "splits")
                if not self.blocked("short_interest") and self._stale(f"short_interest_ticker/{t}", T["short_interest_ticker"]):
                    yield Task(tier, f"short-interest-deep {t}", self._mark(tier, ref=True)(lambda t=t: self._calls(lambda: c.short_interest_ticker(t, ttl_s=0))), "short_interest")
                if not self.blocked("short_volume") and self._stale(f"short_volume_ticker/{t}", T["short_volume_ticker"]):
                    yield Task(tier, f"short-volume-deep {t}", self._mark(tier, ref=True)(lambda t=t: self._calls(lambda: c.short_volume_ticker(t, ttl_s=0))), "short_volume")
            if events:
                if not self.blocked("events") and self._stale(f"events/{t}", T["events"]):
                    yield Task(tier, f"events {t}", self._mark(tier, ref=True)(lambda t=t: self._calls(lambda: c.ticker_events(t, ttl_s=0))), "events")
                if not self.blocked("related") and self._stale(f"related/{t}", T["events"]):
                    yield Task(tier, f"related {t}", self._mark(tier, ref=True)(lambda t=t: self._calls(lambda: c.related_companies(t, ttl_s=0))), "related")

    def _minute_stale(self, ticker: str, lo: date, hi: date) -> bool:
        """A month doc is complete once fetched after its last day has published (D+1 + lag);
        the still-open month is refreshed daily."""
        age = self._age(f"aggs_minute/{ticker}/{lo:%Y-%m}")
        if age is None:
            return True
        complete_at = datetime(hi.year, hi.month, hi.day, tzinfo=timezone.utc) + timedelta(days=1, hours=self.PUBLISH_LAG_H)
        now = datetime.now(timezone.utc)
        if now < complete_at:
            return age > self.TTL["minute_open"]
        return now - timedelta(seconds=age) < complete_at

    def _minute_tasks(self, tier: int, tickers: Iterable[str]):
        """One task per stale (ticker, month), newest month first within a ticker."""
        if self.blocked("aggs_minute"):
            return
        c = self.store.client
        windows = self.store.minute_windows()
        for t in tickers:
            for i, (lo, hi) in reversed(list(enumerate(windows))):
                if not self._minute_stale(t, lo, hi):
                    continue

                def run(t=t, lo=lo, hi=hi, edge=(i == 0)):
                    try:
                        return self._calls(lambda: c.aggs_minute(t, lo, hi, ttl_s=0))
                    except MassiveEntitlementError as e:
                        if edge:                     # oldest month may sit just outside the 2y window:
                            raise MassiveError(f"plan edge: {e}") from e   # back off this month, don't park the family
                        raise
                yield Task(tier, f"minute {t} {lo:%Y-%m}", self._mark(tier, minute=True)(run), "aggs_minute")

    # ---- deep prices (yfinance, no Massive budget) ------------------------------------
    def maybe_deep_prices(self, force: bool = False) -> bool:
        """Weekly, in a background thread so the Massive loop never idles: multi-decade bars for
        universe + extended (`build_deep_prices`). Returns True when a build was started."""
        if not self.deep_prices:
            return False
        if self._deep_thread is not None and self._deep_thread.is_alive():
            return False
        p = self.store.deep_path
        fresh = p.exists() and (time.time() - p.stat().st_mtime) < self.TTL["deep_prices"]
        if fresh and not force:
            return False
        tickers = self.universe + self.extended_tickers()
        import threading

        def _run():
            try:
                self.log(f"deep prices: yfinance {self.deep_start}→ for {len(tickers)} tickers (background)")
                df = self.store.build_deep_prices(tickers, start=self.deep_start)
                self.log(f"deep prices: {df.height:,} rows · {df['ticker'].n_unique()} tickers · {df['date'].min()} → {df['date'].max()}")
            except Exception as e:
                logger.warning(f"crawler: deep prices failed: {e}")
        self._deep_thread = threading.Thread(target=_run, name="massive-deep-prices", daemon=True)
        self._deep_thread.start()
        return True

    def due_tasks(self):
        st, c, T = self.store, self.store.client, self.TTL
        today = st.today
        hol = st.holiday_dates()
        start = st.history_start()

        # tier 0 — freshest bars, current corp actions, today's news
        for d in sorted(st.missing_days(today - timedelta(days=7), today), reverse=True):
            if d in hol or not self._publish_ready(d) or self.blocked("grouped"):
                continue
            yield Task(0, f"grouped {d}", self._mark(0, ohlcv=True)(lambda d=d: self._fetch_day_calls(d)), "grouped")
        for lo, hi in month_windows(today - timedelta(days=45), today):
            ttl = T["actions_current"] if hi >= today - timedelta(days=7) else T["actions_past"]
            for kind in ("splits", "dividends"):
                key = f"{kind}/{lo.isoformat()}_{hi.isoformat()}"
                if not self.blocked(kind) and self._stale(key, ttl):
                    fn = c.splits if kind == "splits" else c.dividends
                    yield Task(0, f"{kind} {lo:%Y-%m}", self._mark(0, ohlcv=True)(lambda fn=fn, lo=lo, hi=hi: self._calls(lambda: fn(lo, hi, ttl_s=0))), kind)
        if not self.blocked("news"):
            for i in (0, 1, 2):
                d = today - timedelta(days=i)
                ttl = T["news_daily_recent"] if i <= 1 else None
                key = f"news/daily/{d.isoformat()}"
                if (ttl is None and self._age(key) is None) or (ttl is not None and self._stale(key, ttl)):
                    yield Task(0, f"news-daily {d}", self._mark(0, ref=True)(lambda d=d, ttl=ttl: self._calls(
                        lambda: c.news(gte=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
                                       lte=datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=1),
                                       cache_key=f"news/daily/{d.isoformat()}", ttl_s=ttl, max_pages=6))), "news")

        # tier 1 — 2-year grouped backfill, newest first
        for d in sorted(st.missing_days(), reverse=True):
            if d in hol or not self._publish_ready(d) or self.blocked("grouped"):
                continue
            yield Task(1, f"grouped {d}", self._mark(1, ohlcv=True)(lambda d=d: self._fetch_day_calls(d)), "grouped")

        # tier 2 — directory, calendar, corp-action history, macro, short data
        if self._stale("reference/tickers_active", T["tickers_active"]):
            yield Task(2, "tickers active", self._mark(2, ref=True)(lambda: self._calls(lambda: c.tickers(active=True, ttl_s=0))), "tickers")
        if self._stale("reference/holidays", T["holidays"]):
            yield Task(2, "holidays", self._mark(2)(lambda: self._calls(lambda: c.holidays(ttl_s=0))), "holidays")
        for name, fn in (("exchanges", c.exchanges), ("ticker_types", c.ticker_types)):
            if not self.blocked(name) and self._stale(f"reference/{name}", T["static"]):
                yield Task(2, name, self._mark(2, ref=True)(lambda fn=fn: self._calls(lambda: fn(ttl_s=0))), name)
        for lo, hi in month_windows(start - timedelta(days=31), today):
            for kind in ("splits", "dividends"):
                key = f"{kind}/{lo.isoformat()}_{hi.isoformat()}"
                if not self.blocked(kind) and self._stale(key, T["actions_past"]):
                    fn = c.splits if kind == "splits" else c.dividends
                    yield Task(2, f"{kind} {lo:%Y-%m}", self._mark(2, ohlcv=True)(lambda fn=fn, lo=lo, hi=hi: self._calls(lambda: fn(lo, hi, ttl_s=0))), kind)
        if self._stale("reference/tickers_delisted", T["tickers_delisted"]):
            yield Task(2, "tickers delisted", self._mark(2, ref=True)(lambda: self._calls(lambda: c.tickers(active=False, ttl_s=0))), "tickers")
        if not self.blocked("ipos") and self._stale("reference/ipos", T["daily"]):
            yield Task(2, "ipos", self._mark(2, ref=True)(lambda: self._calls(lambda: c.ipos(ttl_s=0))), "ipos")
        for series in ("treasury-yields", "inflation", "inflation-expectations"):
            if not self.blocked("fed") and self._stale(f"fed/{series}", T["daily"]):
                yield Task(2, f"fed {series}", self._mark(2, ref=True)(lambda series=series: self._calls(lambda: c.fed(series, ttl_s=0))), "fed")
        si_start = today - timedelta(days=45)
        if not self.blocked("short_interest") and self._stale(f"short_interest/{si_start.isoformat()}", T["daily"]):
            yield Task(2, "short interest", self._mark(2, ref=True)(lambda: self._calls(lambda: c.short_interest(si_start, ttl_s=0))), "short_interest")
        if not self.blocked("short_volume"):
            for d in sorted(business_days(today - timedelta(days=10), today - timedelta(days=1)), reverse=True):
                if d in hol:
                    continue
                if self._age(f"short_volume/{d.isoformat()}") is None:
                    yield Task(2, f"short volume {d}", self._mark(2, ref=True)(lambda d=d: self._calls(lambda: c.short_volume(d, ttl_s=None))), "short_volume")

        # tier 3 — universe depth
        yield from self._depth_tasks(3, self.universe, details_ttl=T["details_universe"], fin_ttl=T["financials_universe"],
                                     news_recent_ttl=T["news_universe"], backfill=True, events=True, start=start)

        # tier 4 — universe intraday: 1-minute bars for the 2-year window
        yield from self._minute_tasks(4, self.universe)

        # tier 5 — extended (top-N liquid) depth, same treatment as the universe (minus minute bars)
        ext = self.extended_tickers()
        yield from self._depth_tasks(5, ext, details_ttl=T["details_universe"], fin_ttl=T["financials_universe"],
                                     news_recent_ttl=T["news_universe"], backfill=True, events=True, start=start)

        # tier 6 — whole market by liquidity rank (lighter TTLs)
        skip = set(self.universe) | set(ext)
        yield from self._depth_tasks(6, (t for t in self.market_tickers() if t not in skip),
                                     details_ttl=T["details_market"], fin_ttl=T["financials_market"],
                                     news_recent_ttl=T["news_market"], backfill=False, events=False, start=start)

        # tier 7 — QA: Massive's split-adjusted series vs ours (universe + extended)
        if not self.blocked("aggs"):
            for t in self.universe + ext:
                key = f"aggs_qa/{t}"
                if self._stale(key, T["aggs_qa"]):
                    yield Task(7, f"aggs-qa {t}", self._mark(7)(lambda t=t, key=key: self._calls(
                        lambda: c.get(f"/v2/aggs/ticker/{api_ticker(t)}/range/1/day/{start.isoformat()}/{today.isoformat()}",
                                      {"adjusted": "true", "sort": "asc", "limit": 50000}, cache_key=key, ttl_s=0))), "aggs")

    def _calls(self, fn: Callable[[], Any]) -> int:
        c0 = self.store.client.calls
        fn()
        return self.store.client.calls - c0

    def _fetch_day_calls(self, d: date) -> int:
        try:
            return self._calls(lambda: self.store.fetch_day(d))
        except MassiveEntitlementError as e:
            # a 403 on a day a few sessions old is "not released to this plan yet", not "not in plan":
            # retry in 30 min instead of parking the whole grouped family (which, before 2026-09-26, the
            # grouped tasks did not even honour — they retried ~10×/min for hours every night)
            if (self.store.today - d).days <= 5:
                raise MassiveNotPublished(f"grouped {d} not released yet: {str(e)[:80]}") from e
            raise

    # ---- execution -----------------------------------------------------------------
    FAIL_BACKOFF_S = (6 * 3600, 24 * 3600, 7 * _DAY)   # 1st, 2nd, 3rd+ failure of the same task

    def next_task(self) -> Task | None:
        now = time.time()
        for t in self.due_tasks():
            if self._skip_until.get(t.name, 0.0) > now:
                continue
            return t
        return None

    def _note_failure(self, task: Task, err: Exception) -> None:
        if isinstance(err, MassiveNotPublished):
            self._skip_until[task.name] = time.time() + err.retry_s
            logger.info(f"crawler: {task.name} not released yet — retry in {err.retry_s / 60:.0f} min")
            return
        n = self._fail_count.get(task.name, 0) + 1
        self._fail_count[task.name] = n
        back = self.FAIL_BACKOFF_S[min(n, len(self.FAIL_BACKOFF_S)) - 1]
        self._skip_until[task.name] = time.time() + back
        logger.warning(f"crawler: {task.name} failed ({n}×): {str(err)[:160]} — retry in {back / 3600:.0f}h")

    def maybe_rebuild(self, force: bool = False) -> None:
        if not (force or time.time() - self._last_build >= self.rebuild_every_s):
            return
        if self._dirty_ohlcv or force:
            try:
                self.store.build_ohlcv_all()
                self._dirty_ohlcv = False
            except MassiveNotReady:
                pass
            except Exception as e:
                logger.warning(f"crawler: ohlcv build failed: {e}")
        if self._dirty_ref or force:
            try:
                n = self.store.build_reference(self.universe)
                self.log(f"reference tables rebuilt: {n}")
                self._dirty_ref = False
            except Exception as e:
                logger.warning(f"crawler: reference build failed: {e}")
        if self._dirty_minute or force:
            try:
                n = self.store.build_minute_bars()
                if n:
                    self.log(f"minute bars rebuilt: {len(n)} tickers, {sum(n.values()):,} rows")
                self._dirty_minute = False
            except Exception as e:
                logger.warning(f"crawler: minute-bar build failed: {e}")
        self._last_build = time.time()

    def write_state(self, current: str | None, idle: bool = False) -> None:
        c = self.store.client
        st = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pid": os.getpid(),
              "current": current, "idle": idle, "calls_session": c.calls, "cache_hits": c.cache_hits,
              "tier_calls": self.tier_calls, "uptime_h": round((time.time() - self.started) / 3600, 2),
              "waited_s": round(c.waited_s, 1), "calls_today": self.store.calls_today(),
              "keys": len(c.keys), "keys_live": c.live_keys, "key_calls": c.key_calls,
              "blocked": self.blocked_families(), "extended_top": self.extended_top,
              "failing": sorted(k for k, v in self._skip_until.items() if v > time.time())[:20],
              "deep_prices_running": bool(self._deep_thread is not None and self._deep_thread.is_alive())}
        try:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1))
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    def run(self, once: bool = False, max_calls: int | None = None, idle_sleep: float = 60.0) -> int:
        """Loop until stopped. Returns network calls made."""
        import signal

        def _stop(signum, frame):
            self.log(f"signal {signum} — finishing current call")
            self.stop = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _stop)
            except ValueError:
                pass                                # not main thread (tests)
        c0 = self.store.client.calls
        last_tier: int | None = None
        while not self.stop:
            if max_calls is not None and self.store.client.calls - c0 >= max_calls:
                break
            task = self.next_task()
            if task is None or (once and task.tier > self.ONCE_MAX_TIER):
                self.maybe_rebuild(force=self._dirty_ohlcv or self._dirty_ref)
                self.write_state(None, idle=True)
                if once:
                    break
                time.sleep(idle_sleep)
                continue
            if task.tier != last_tier:
                self.log(f"tier {task.tier} → {task.name}")
                last_tier = task.tier
            self.write_state(task.name)
            try:
                task.run()
            except MassiveAuthError:
                raise
            except MassiveEntitlementError as e:
                self.block(task.family or task.name.split()[0], str(e)[:160])
            except MassiveError as e:
                self._note_failure(task, e)
            except Exception as e:                   # never let one bad payload stop the loop
                self._note_failure(task, e)
            self.maybe_rebuild()
            self.maybe_deep_prices()
        self.maybe_rebuild(force=True)
        if self._deep_thread is not None and self._deep_thread.is_alive():
            self._deep_thread.join(timeout=600)
        self.write_state(None, idle=True)
        return self.store.client.calls - c0


def crawler_state(store: MassiveStore) -> dict | None:
    p = store.raw_dir / "crawler_state.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def adjustment_qa(store: MassiveStore, max_tickers: int = 2000) -> pl.DataFrame | None:
    """Compare our split-adjusted ``close`` against Massive's own ``adjusted=true`` bars (tier-7 cache).

    Returns per-ticker ``max_abs_dev`` (relative) and ``n`` overlapping days, or
    None when nothing is cached yet.  A large deviation means a split we are
    missing (or one Massive applies that we don't) — investigate before trusting
    that name's history.
    """
    qa_dir = store.raw_dir / "aggs_qa"
    if not qa_dir.exists() or not store.ohlcv_all_path.exists():
        return None
    ours = pl.read_parquet(store.ohlcv_all_path, columns=["date", "ticker", "close"])
    rows = []
    for p in sorted(qa_dir.glob("*.json.gz"))[:max_tickers]:
        t = p.name[:-8]
        doc = store.client.cache_read(f"aggs_qa/{t}", None)
        if not doc or not doc["results"]:
            continue
        theirs = aggs_to_frame(t, doc["results"]).select("date", pl.col("close").alias("their_close"))
        j = ours.filter(pl.col("ticker") == t).join(theirs, on="date", how="inner")
        if j.height:
            dev = ((j["close"] / j["their_close"]) - 1.0).abs()
            rows.append({"ticker": t, "n": j.height, "max_abs_dev": float(dev.max()), "mean_abs_dev": float(dev.mean())})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"ticker": pl.Utf8, "n": pl.Int64, "max_abs_dev": pl.Float64, "mean_abs_dev": pl.Float64})
