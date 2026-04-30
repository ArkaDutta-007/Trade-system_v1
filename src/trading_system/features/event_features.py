"""Aggregate event-schema rows into per (date, ticker) features.

Critical: uses `known_at` (not `published_at`) so backtests are point-in-time safe.

Features produced
-----------------
Base (per-day aggregates):
  event_count            : number of events known on that day
  event_sentiment_mean   : mean headline sentiment [-1, 1]
  event_magnitude_mean   : mean magnitude [0, 1]
  event_novelty_max      : max novelty [0, 1]
  risk_flag_count        : total risk flags across all headlines that day

Exponentially-weighted rolling sentiment (λ=0.85):
  sent_decay_3d          : EW sentiment over 3 trading days
  sent_decay_7d          : EW sentiment over 7 trading days
  sent_decay_14d         : EW sentiment over 14 trading days

Derived:
  sent_momentum          : sent_decay_3d - sent_decay_7d
                           > 0  → sentiment improving (fear fading)
                           < 0  → sentiment deteriorating (fear building)
"""
from __future__ import annotations

from typing import Sequence

import polars as pl

# Exponential decay factor — each extra day back is discounted by this factor
_LAMBDA = 0.85

_EMPTY_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "event_count": pl.Int64,
    "event_sentiment_mean": pl.Float64,
    "event_magnitude_mean": pl.Float64,
    "event_novelty_max": pl.Float64,
    "risk_flag_count": pl.Int64,
    "sent_decay_3d": pl.Float64,
    "sent_decay_7d": pl.Float64,
    "sent_decay_14d": pl.Float64,
    "sent_momentum": pl.Float64,
}


def _ew_decay(
    daily: pl.DataFrame,
    windows: Sequence[int] = (3, 7, 14),
    lam: float = _LAMBDA,
) -> pl.DataFrame:
    """Add exponentially-weighted rolling sentiment columns.

    For each window N and each (ticker, date) row:

        sent_decay_Nd = Σ_{i=0}^{N-1} λ^i * s_{t-i}  /  Σ_{i=0}^{N-1} λ^i * 1[has data]

    `daily` must have columns: date (pl.Date), ticker (pl.Utf8),
    event_sentiment_mean (pl.Float64).  Rows with no events on a day have
    sentiment 0.0 (already filled by the caller).
    """
    # Build precomputed weight vectors
    weight_vecs: dict[int, list[float]] = {}
    for n in windows:
        weight_vecs[n] = [lam ** i for i in range(n)]

    results = []
    for ticker, grp in daily.group_by("ticker"):
        grp = grp.sort("date")
        sents = grp["event_sentiment_mean"].to_list()
        n_rows = len(sents)
        decay_cols: dict[int, list[float]] = {n: [] for n in windows}

        for idx in range(n_rows):
            for n in windows:
                weights = weight_vecs[n]
                num = 0.0
                denom = 0.0
                for lag in range(n):
                    j = idx - lag
                    if j < 0:
                        break
                    w = weights[lag]
                    num += w * sents[j]
                    denom += w
                val = num / denom if denom > 0 else 0.0
                decay_cols[n].append(val)

        for n in windows:
            grp = grp.with_columns(
                pl.Series(f"sent_decay_{n}d", decay_cols[n], dtype=pl.Float64)
            )
        results.append(grp)

    return pl.concat(results, how="diagonal") if results else daily


def aggregate_events_to_daily(events: pl.DataFrame) -> pl.DataFrame:
    """Explode event tickers and aggregate per (date, ticker).

    Returns a DataFrame with all columns in _EMPTY_SCHEMA.
    """
    if events.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    df = (
        events
        .with_columns(date=pl.col("known_at").dt.date())
        .explode("tickers")
        .rename({"tickers": "ticker"})
    )

    # Per-day raw aggregates
    daily = (
        df.group_by(["date", "ticker"])
        .agg(
            event_count=pl.len(),
            event_sentiment_mean=pl.col("sentiment").mean(),
            event_magnitude_mean=pl.col("magnitude").mean(),
            event_novelty_max=pl.col("novelty").max(),
            risk_flag_count=pl.col("risk_flags").list.len().sum(),
        )
        .sort(["ticker", "date"])
    )

    # Exponentially-weighted rolling sentiment
    daily = _ew_decay(daily, windows=[3, 7, 14], lam=_LAMBDA)

    # Sentiment momentum: short-term minus medium-term decay
    daily = daily.with_columns(
        sent_momentum=(pl.col("sent_decay_3d") - pl.col("sent_decay_7d"))
    )

    return daily
