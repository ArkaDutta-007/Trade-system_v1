"""Macro-level features — the market backdrop the model was previously blind to.

The system already *fetches* rich macro series but never fed their **levels** to
the model (only event-proximity flags, which were themselves disabled in the live
pipeline).  This module pulls daily FRED series through the resilient
``flags.datafeed.fred_series`` path (API → keyless CSV → disk cache) and turns
each into three point-in-time-safe features:

  * ``<name>``         — the level, forward-filled onto every trading day
  * ``<name>_chg_20d`` — 20-trading-day change (regime momentum)
  * ``<name>_z_252``   — 1-year rolling z-score (how stretched vs its own history)

Everything is keyed on ``date`` and broadcast across tickers via a left join, so
a single macro row applies to the whole cross-section that day.  Forward-fill
(never back-fill) keeps it leakage-safe: a value known at date *t* is only ever
used on *t* or later.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..flags.datafeed import fred_series
from ..utils import get_logger

logger = get_logger(__name__)

# FRED id -> output feature name.  All daily series so the join aligns cleanly.
DEFAULT_MACRO_SERIES: dict[str, str] = {
    "DGS10": "macro_ust_10y",
    "T10Y2Y": "macro_yield_curve",
    "VIXCLS": "macro_vix",
    "BAMLH0A0HYM2": "macro_hy_oas",
    "DFF": "macro_fed_funds",
}


def build_macro_features(
    series_map: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    cache_ttl_hours: float = 12.0,
) -> pl.DataFrame:
    """Build a date-indexed macro feature frame. Empty frame if nothing fetched.

    Returns columns: ``date`` + one level/chg/z triple per requested series.
    Failures degrade gracefully (a series that can't be fetched is skipped).
    """
    series_map = series_map or DEFAULT_MACRO_SERIES
    frames: list[pl.DataFrame] = []

    for sid, name in series_map.items():
        try:
            res = fred_series(sid, cache_dir=cache_dir, api_key=api_key, cache_ttl_hours=cache_ttl_hours)
        except Exception as e:
            logger.warning(f"macro series {sid} unavailable: {e}")
            continue
        df = res.df.select(["date", "value"]).rename({"value": name}).sort("date")
        frames.append(df)

    if not frames:
        return pl.DataFrame(schema={"date": pl.Date})

    # Outer-join all series onto a shared daily calendar, forward-fill the levels
    macro = frames[0]
    for df in frames[1:]:
        macro = macro.join(df, on="date", how="full", coalesce=True)
    macro = macro.sort("date")

    value_cols = [c for c in macro.columns if c != "date"]
    macro = macro.with_columns([pl.col(c).forward_fill() for c in value_cols])

    # Derived: 20d change + 1y rolling z-score per series
    derived: list[pl.Expr] = []
    for c in value_cols:
        derived.append((pl.col(c) - pl.col(c).shift(20)).alias(f"{c}_chg_20d"))
        mean_252 = pl.col(c).rolling_mean(window_size=252, min_samples=60)
        std_252 = pl.col(c).rolling_std(window_size=252, min_samples=60)
        derived.append(
            pl.when(std_252 > 0)
            .then((pl.col(c) - mean_252) / std_252)
            .otherwise(0.0)
            .alias(f"{c}_z_252")
        )
    macro = macro.with_columns(derived)
    return macro


def join_macro_features(features: pl.DataFrame, macro: pl.DataFrame | None) -> pl.DataFrame:
    """Left-join macro features onto the feature matrix by date, forward-filling.

    Forward-fill (sorted by date within each ticker) handles trading days that
    don't have a fresh macro print (weekends/holidays differ across series).
    Remaining leading nulls are zero-filled so the model never sees NaN.
    """
    if macro is None or macro.is_empty():
        return features

    macro_cols = [c for c in macro.columns if c != "date"]
    out = features.join(macro, on="date", how="left").sort(["ticker", "date"])
    out = out.with_columns([pl.col(c).forward_fill().over("ticker") for c in macro_cols])
    out = out.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in macro_cols])
    return out
