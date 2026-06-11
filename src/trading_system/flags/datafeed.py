"""Resilient macro/market data feed for the flag lookups.

The flag board must never hang or hard-fail on a flaky network. This module
provides one robust entry point — ``fred_series`` — with a three-tier strategy:

  1. Official FRED API   (api.stlouisfed.org, needs free FRED_API_KEY)
  2. Keyless fredgraph    (fred.stlouisfed.org/graph/fredgraph.csv)
  3. Disk cache           (data/silver/macro_cache/<series>.parquet, last-good)

Whichever tier succeeds first wins; a success refreshes the cache. Monthly
(CPI) and daily (fed funds) series are cached for ``cache_ttl_hours`` so the
common case touches no network at all — the board stays instant and a FRED
outage degrades to a clearly-labelled "cached" reading instead of UNKNOWN.

Some networks block one FRED host but not the other (observed: the CSV host
times out while the API host responds in <0.5s), so trying both matters.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; guard just in case
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

from ..utils import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = (4.0, 12.0)  # (connect, read) seconds — fail fast, fall back
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=2, connect=2, read=1, backoff_factor=0.4,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
        s.headers.update({"User-Agent": "trading-system/flags (research)"})
        _SESSION = s
    return _SESSION


@dataclass
class SeriesResult:
    df: pl.DataFrame             # columns: date, value (sorted ascending)
    source: str                 # fred-api | fred-csv | cache | cache-stale
    as_of: str                  # latest observation date
    from_cache: bool
    age_hours: float | None = None
    error: str | None = None


def _cache_dir(silver: Path | None) -> Path | None:
    if silver is None:
        return None
    d = Path(silver) / "macro_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fred_api(sid: str, api_key: str, timeout) -> pl.DataFrame:
    params = {
        "series_id": sid,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2015-01-01",
    }
    r = _session().get(FRED_API_URL, params=params, timeout=timeout)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = [(o["date"], o["value"]) for o in obs if o.get("value") not in (".", None, "")]
    if not rows:
        raise RuntimeError(f"FRED API returned no observations for {sid}")
    return (
        pl.DataFrame({"date": [a for a, _ in rows], "value": [b for _, b in rows]})
        .with_columns(pl.col("date").str.to_date(), pl.col("value").cast(pl.Float64))
        .sort("date")
    )


def _fred_csv(sid: str, timeout) -> pl.DataFrame:
    r = _session().get(FRED_CSV_URL.format(sid=sid), timeout=timeout)
    r.raise_for_status()
    df = pl.read_csv(io.BytesIO(r.content), null_values=["."])
    date_col, val_col = df.columns[0], df.columns[1]
    return (
        df.rename({date_col: "date", val_col: "value"})
        .with_columns(pl.col("date").cast(pl.Date), pl.col("value").cast(pl.Float64))
        .drop_nulls()
        .sort("date")
    )


def fred_series(
    sid: str,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    cache_ttl_hours: float = 12.0,
    timeout=DEFAULT_TIMEOUT,
    force: bool = False,
) -> SeriesResult:
    """Fetch a FRED series with cache → API → CSV → stale-cache fallback."""
    cdir = _cache_dir(cache_dir)
    cache_path = (cdir / f"{sid}.parquet") if cdir else None

    # 1. fresh cache → no network
    if cache_path and cache_path.exists() and not force:
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_h <= cache_ttl_hours:
            df = pl.read_parquet(cache_path)
            return SeriesResult(df, "cache", str(df["date"][-1]), True, round(age_h, 1))

    # 2. network: API host first (often reachable when CSV host is blocked)
    api_key = api_key or os.environ.get("FRED_API_KEY")
    df, source, err = None, None, None
    if api_key:
        try:
            df, source = _fred_api(sid, api_key, timeout), "fred-api"
        except Exception as e:
            err = f"api: {e}"
            logger.debug(f"FRED API failed for {sid}: {e}")
    if df is None:
        try:
            df, source = _fred_csv(sid, timeout), "fred-csv"
        except Exception as e:
            err = f"{err + '; ' if err else ''}csv: {e}"
            logger.debug(f"FRED CSV failed for {sid}: {e}")

    if df is not None and len(df):
        if cache_path:
            try:
                df.write_parquet(cache_path, compression="zstd")
            except Exception as e:
                logger.debug(f"macro cache write failed for {sid}: {e}")
        return SeriesResult(df, source, str(df["date"][-1]), False)

    # 3. stale cache as last resort
    if cache_path and cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
        df = pl.read_parquet(cache_path)
        logger.warning(f"FRED {sid} unreachable; using cached value ({age_h:.0f}h old)")
        return SeriesResult(df, "cache-stale", str(df["date"][-1]), True, round(age_h, 1), err)

    raise RuntimeError(f"FRED {sid} unavailable and no cache present ({err})")
