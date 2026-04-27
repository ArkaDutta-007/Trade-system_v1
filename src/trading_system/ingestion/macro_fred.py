"""FRED macro data ingestion. Requires FRED_API_KEY for full coverage."""
from __future__ import annotations

import os
from datetime import date
from typing import Iterable

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

DEFAULT_SERIES = {
    "DGS10": "ust_10y",
    "DGS2": "ust_2y",
    "DGS3MO": "ust_3m",
    "FEDFUNDS": "fed_funds",
    "CPIAUCSL": "cpi",
    "UNRATE": "unrate",
    "T10Y2Y": "yield_curve_10y2y",
    "VIXCLS": "vix",
    "BAMLH0A0HYM2": "hy_oas",
}


def fetch_fred_series(
    series: Iterable[str] | None = None,
    start: str | date = "2000-01-01",
    api_key: str | None = None,
) -> pl.DataFrame:
    """Fetch FRED time series. Returns long-form (date, series_id, value)."""
    series_ids = list(series) if series else list(DEFAULT_SERIES.keys())
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set. Skipping FRED ingestion.")
        return pl.DataFrame(schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64})

    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames: list[pl.DataFrame] = []
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start)
            if s is None or len(s) == 0:
                continue
            df = pl.DataFrame(
                {
                    "date": [d.date() for d in s.index.to_pydatetime()],
                    "series_id": [sid] * len(s),
                    "value": s.values.astype(float).tolist(),
                }
            )
            frames.append(df)
        except Exception as e:
            logger.warning(f"FRED fetch failed for {sid}: {e}")

    if not frames:
        return pl.DataFrame(schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64})
    return pl.concat(frames).sort(["series_id", "date"])
