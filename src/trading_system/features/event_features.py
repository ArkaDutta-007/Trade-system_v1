"""Aggregate event-schema rows into per (date, ticker) features.

Critical: uses `known_at` (not `published_at`) so backtests are point-in-time safe.
"""
from __future__ import annotations

import polars as pl


def aggregate_events_to_daily(events: pl.DataFrame) -> pl.DataFrame:
    """Explode event tickers and aggregate per (date, ticker)."""
    if events.is_empty():
        return pl.DataFrame(schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "event_count": pl.Int64,
            "event_sentiment_mean": pl.Float64,
            "event_magnitude_mean": pl.Float64,
            "event_novelty_max": pl.Float64,
        })

    df = (
        events
        .with_columns(date=pl.col("known_at").dt.date())
        .explode("tickers")
        .rename({"tickers": "ticker"})
    )
    return (
        df.group_by(["date", "ticker"])
        .agg(
            event_count=pl.len(),
            event_sentiment_mean=pl.col("sentiment").mean(),
            event_magnitude_mean=pl.col("magnitude").mean(),
            event_novelty_max=pl.col("novelty").max(),
        )
        .sort(["ticker", "date"])
    )
