"""Programmatic data-quality checks. Can be wrapped in a Great Expectations suite later."""
from __future__ import annotations

import polars as pl


class DataQualityError(AssertionError):
    pass


def run_ohlcv_checks(df: pl.DataFrame) -> dict:
    """Return a dict of check_name -> bool (True = pass)."""
    results: dict[str, bool] = {}
    if df.is_empty():
        raise DataQualityError("OHLCV frame is empty.")

    # 1. Required columns
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    results["has_required_columns"] = required.issubset(set(df.columns))

    # 2. No duplicates per (date, ticker)
    dupes = df.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
    results["no_duplicate_date_ticker"] = dupes.is_empty()

    # 3. OHLC sanity
    bad = df.filter(
        (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("high"))
    )
    results["ohlc_sanity"] = bad.is_empty()

    # 4. Non-negative volume
    results["volume_non_negative"] = df.filter(pl.col("volume") < 0).is_empty()

    # 5. Positive prices
    results["prices_positive"] = (
        df.filter(
            (pl.col("open") <= 0) | (pl.col("close") <= 0)
            | (pl.col("high") <= 0) | (pl.col("low") <= 0)
        ).is_empty()
    )

    # 6. Missingness threshold
    miss = df.null_count().row(0)
    n = len(df)
    results["missing_under_5pct"] = all(m / n < 0.05 for m in miss)

    return results


def run_event_checks(events: pl.DataFrame) -> dict:
    results = {}
    if events.is_empty():
        return {"non_empty": False}
    results["non_empty"] = True
    results["known_at_after_published"] = (
        events.filter(pl.col("known_at") < pl.col("published_at")).is_empty()
    )
    results["sentiment_in_range"] = (
        events.filter((pl.col("sentiment") < -1.0) | (pl.col("sentiment") > 1.0)).is_empty()
    )
    return results
