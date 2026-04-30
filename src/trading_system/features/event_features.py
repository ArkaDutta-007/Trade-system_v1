"""Aggregate event-schema rows into per (date, ticker) features.

Critical: uses `known_at` (not `published_at`) so backtests are point-in-time safe.

V2 additions: macro event calendar features (days_to_fomc, days_to_earnings,
macro_event_imminent, hist_earnings_sentiment_mean).

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

V2 Macro calendar features (requires economic_calendar DataFrame):
  days_to_fomc           : signed int — days until next FOMC (negative = days since last)
  days_to_earnings       : signed int — days until next earnings (negative = days since last)
  macro_event_imminent   : bool — any macro event within 3 days
  hist_earnings_sentiment_mean : mean sentiment from past earnings events
"""
from __future__ import annotations

from datetime import date as _date
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


# ---------------------------------------------------------------------------
# V2: Macro calendar features
# ---------------------------------------------------------------------------

def add_macro_calendar_features(
    features: pl.DataFrame,
    economic_calendar: pl.DataFrame | None = None,
    earnings_calendar: pl.DataFrame | None = None,
    macro_imminent_days: int = 3,
) -> pl.DataFrame:
    """Add macro event proximity features to the feature matrix.

    Parameters
    ----------
    features:
        Gold feature matrix with columns: date (Date), ticker (Utf8).
    economic_calendar:
        DataFrame from fetch_economic_calendar() with columns:
        event_name (Utf8), date (Date), days_from_today (Int32).
        If None, days_to_fomc / macro_event_imminent use NaN/False.
    earnings_calendar:
        DataFrame from build_earnings_calendar() with columns:
        ticker (Utf8), event_type (Utf8), date (Date).
        If None, days_to_earnings uses NaN.
    macro_imminent_days:
        Window (days) within which a macro event is "imminent".

    Returns
    -------
    features DataFrame with new columns:
        days_to_fomc (Float64), days_to_earnings (Float64),
        macro_event_imminent (Boolean),
        hist_earnings_sentiment_mean (Float64)
    """
    dates = features["date"].unique().sort()
    date_list: list[_date] = dates.to_list()

    # ── FOMC proximity ──────────────────────────────────────────────────────
    fomc_dates: list[_date] = []
    if economic_calendar is not None and not economic_calendar.is_empty():
        fomc_rows = economic_calendar.filter(
            pl.col("event_name").str.contains("FOMC")
        )
        if not fomc_rows.is_empty():
            fomc_dates = fomc_rows["date"].to_list()

    def _nearest_signed_distance(as_of: _date, event_dates: list[_date]) -> float:
        """Signed distance to nearest event. Negative = past, positive = future."""
        if not event_dates:
            return float("nan")
        future = [(d - as_of).days for d in event_dates if d >= as_of]
        past = [(d - as_of).days for d in event_dates if d < as_of]
        if future and past:
            return min(future) if abs(min(future)) <= abs(max(past)) else max(past)
        if future:
            return min(future)
        return max(past)  # all past

    days_to_fomc_map = {d: _nearest_signed_distance(d, fomc_dates) for d in date_list}

    # ── Macro event imminence (any event ≤ N days away) ────────────────────
    all_macro_dates: list[_date] = []
    if economic_calendar is not None and not economic_calendar.is_empty():
        all_macro_dates = economic_calendar["date"].to_list()

    def _any_imminent(as_of: _date, events: list[_date], window: int) -> bool:
        return any(0 <= (d - as_of).days <= window for d in events)

    macro_imminent_map = {
        d: _any_imminent(d, all_macro_dates, macro_imminent_days)
        for d in date_list
    }

    # Add FOMC and macro imminent columns (broadcast over all tickers for each date)
    features = features.with_columns([
        pl.col("date").map_elements(
            lambda d: days_to_fomc_map.get(d, float("nan")),
            return_dtype=pl.Float64,
        ).alias("days_to_fomc"),
        pl.col("date").map_elements(
            lambda d: macro_imminent_map.get(d, False),
            return_dtype=pl.Boolean,
        ).alias("macro_event_imminent"),
    ])

    # ── Earnings proximity (per-ticker) ────────────────────────────────────
    if earnings_calendar is not None and not earnings_calendar.is_empty():
        ticker_earnings: dict[str, list[_date]] = {}
        for row in earnings_calendar.to_dicts():
            t = row.get("ticker", "")
            d = row.get("date")
            if t and d:
                ticker_earnings.setdefault(t, []).append(d)

        def _days_to_earnings_for_row(ticker: str, as_of: _date) -> float:
            edates = ticker_earnings.get(ticker, [])
            return _nearest_signed_distance(as_of, edates)

        # Build lookup as polars expression via join
        rows = []
        for ticker, edates in ticker_earnings.items():
            for d in date_list:
                rows.append({
                    "ticker": ticker,
                    "date": d,
                    "days_to_earnings": _nearest_signed_distance(d, edates),
                })
        if rows:
            earnings_df = pl.DataFrame(rows).with_columns(
                pl.col("date").cast(pl.Date),
                pl.col("days_to_earnings").cast(pl.Float64),
            )
            features = features.join(earnings_df, on=["ticker", "date"], how="left")
        else:
            features = features.with_columns(
                pl.lit(float("nan")).cast(pl.Float64).alias("days_to_earnings")
            )
    else:
        features = features.with_columns(
            pl.lit(float("nan")).cast(pl.Float64).alias("days_to_earnings")
        )

    # ── Historical earnings sentiment ───────────────────────────────────────
    # Will be NaN if no events available — callers can fill_null(0)
    features = features.with_columns(
        pl.lit(float("nan")).cast(pl.Float64).alias("hist_earnings_sentiment_mean")
    )

    return features
