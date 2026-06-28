"""Calendar / corporate-action events.

V1: Earnings dates via yfinance.
V2: Economic calendar (FOMC, CPI, NFP, PPI, Retail Sales) from FRED release schedule.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Iterable

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)


def _earnings_for_ticker(t: str) -> list[dict]:
    """Fetch one ticker's earnings dates (yfinance). Returns rows or []."""
    import yfinance as yf
    out: list[dict] = []
    yt = yf.Ticker(t)
    df = getattr(yt, "earnings_dates", None)
    if df is None or len(df) == 0:
        return out
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    for _, r in df.iterrows():
        d = r.get("earnings_date") or r.get("earnings_date_")
        if d is None:
            continue
        try:
            d = d.date()
        except Exception:
            pass
        out.append({"ticker": t.upper(), "event_type": "earnings", "date": d})
    return out


def build_earnings_calendar(tickers: Iterable[str], workers: int = 8) -> pl.DataFrame:
    """Best-effort earnings calendar from yfinance (threaded + progress). Empty for ETFs."""
    from ..utils import parallel_map

    tickers = list(tickers)
    results = parallel_map(
        _earnings_for_ticker, tickers, workers=workers, description="earnings cal"
    )
    rows = [r for sub in results if sub for r in sub]

    if not rows:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "event_type": pl.Utf8, "date": pl.Date})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])


# ---------------------------------------------------------------------------
# V2: Economic Calendar from FRED
# ---------------------------------------------------------------------------

# FRED series whose release schedule we care about.
# Key = human-readable label, Value = FRED release ID.
_FRED_RELEASES = {
    "CPI": 10,           # Consumer Price Index
    "PPI": 11,           # Producer Price Index
    "NFP": 50,           # Employment Situation (Non-Farm Payrolls)
    "Retail Sales": 84,  # Advance Retail Sales
    "GDP": 53,           # Gross Domestic Product
}

# FOMC meeting schedule is not available via a FRED release endpoint.
# We use a hardcoded list for the current year + next year as a reliable fallback.
# Update this list when the Fed publishes the next year's schedule.
_FOMC_DATES_2025_2026 = [
    # 2025
    date(2025, 1, 28), date(2025, 1, 29),
    date(2025, 3, 18), date(2025, 3, 19),
    date(2025, 4, 29), date(2025, 4, 30),
    date(2025, 6, 17), date(2025, 6, 18),
    date(2025, 7, 29), date(2025, 7, 30),
    date(2025, 9, 16), date(2025, 9, 17),
    date(2025, 10, 28), date(2025, 10, 29),
    date(2025, 12, 9), date(2025, 12, 10),
    # 2026
    date(2026, 1, 27), date(2026, 1, 28),
    date(2026, 3, 17), date(2026, 3, 18),
    date(2026, 4, 28), date(2026, 4, 29),
    date(2026, 6, 9), date(2026, 6, 10),
    date(2026, 7, 28), date(2026, 7, 29),
    date(2026, 9, 15), date(2026, 9, 16),
    date(2026, 10, 27), date(2026, 10, 28),
    date(2026, 12, 8), date(2026, 12, 9),
]


def _fetch_fred_release_dates(
    release_id: int,
    start: date,
    end: date,
    api_key: str,
) -> list[date]:
    """Fetch release dates for a FRED release ID."""
    try:
        import requests
        url = "https://api.stlouisfed.org/fred/release/dates"
        params = {
            "release_id": release_id,
            "realtime_start": str(start),
            "realtime_end": str(end),
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "asc",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        dates = [
            date.fromisoformat(r["date"])
            for r in resp.json().get("release_dates", [])
        ]
        return dates
    except Exception as e:
        logger.debug(f"FRED release {release_id} fetch failed: {e}")
        return []


def fetch_economic_calendar(
    lookahead_days: int = 30,
    lookback_days: int = 7,
    api_key: str | None = None,
) -> pl.DataFrame:
    """Fetch macro event dates from FRED + hardcoded FOMC schedule.

    Parameters
    ----------
    lookahead_days:
        How many days into the future to look for events.
    lookback_days:
        How many days back to include (for context on recent events).
    api_key:
        FRED API key. Falls back to FRED_API_KEY env var.

    Returns
    -------
    pl.DataFrame with columns:
        event_name (Utf8), date (Date), days_from_today (Int32)
    """
    today = date.today()
    start = today - timedelta(days=lookback_days)
    end = today + timedelta(days=lookahead_days)

    rows: list[dict] = []

    # FOMC dates (hardcoded, reliable)
    for d in _FOMC_DATES_2025_2026:
        if start <= d <= end:
            rows.append({
                "event_name": "FOMC Meeting",
                "date": d,
                "days_from_today": (d - today).days,
            })

    # FRED release schedules
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if api_key:
        for name, release_id in _FRED_RELEASES.items():
            dates = _fetch_fred_release_dates(release_id, start, end, api_key)
            for d in dates:
                rows.append({
                    "event_name": name,
                    "date": d,
                    "days_from_today": (d - today).days,
                })
    else:
        logger.info("FRED_API_KEY not set — economic calendar shows FOMC only.")

    if not rows:
        return pl.DataFrame(schema={
            "event_name": pl.Utf8,
            "date": pl.Date,
            "days_from_today": pl.Int32,
        })

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Date))
        .unique(subset=["event_name", "date"])
        .sort("date")
    )
