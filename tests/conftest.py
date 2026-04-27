"""Shared fixtures: synthetic OHLCV that doesn't require network access."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest


def _gen_prices(tickers, n_days=600, start="2020-01-01", seed=42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start_dt = dt.date.fromisoformat(start)
    dates = [start_dt + dt.timedelta(days=i) for i in range(n_days)]
    # business-day filter
    dates = [d for d in dates if d.weekday() < 5]
    rows = []
    for t in tickers:
        # Per-ticker drift / vol
        mu = rng.uniform(0.0001, 0.0006)
        sigma = rng.uniform(0.008, 0.02)
        rets = rng.normal(mu, sigma, len(dates))
        px = 100 * np.cumprod(1 + rets)
        opens = px * (1 + rng.normal(0, 0.001, len(px)))
        highs = np.maximum(px, opens) * (1 + np.abs(rng.normal(0, 0.003, len(px))))
        lows = np.minimum(px, opens) * (1 - np.abs(rng.normal(0, 0.003, len(px))))
        vols = rng.integers(1_000_000, 5_000_000, len(px))
        for d, o, h, l, c, v in zip(dates, opens, highs, lows, px, vols):
            rows.append(
                {"date": d, "ticker": t, "open": float(o), "high": float(h),
                 "low": float(l), "close": float(c), "adj_close": float(c),
                 "volume": int(v)}
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pl.DataFrame:
    return _gen_prices(["SPY", "QQQ", "XLK", "XLF", "XLE"])


@pytest.fixture(scope="session")
def small_universe():
    return ["SPY", "QQQ", "XLK", "XLF", "XLE"]
