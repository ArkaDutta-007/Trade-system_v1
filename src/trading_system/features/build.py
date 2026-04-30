"""Assemble the full feature matrix from OHLCV + optional event/macro inputs."""
from __future__ import annotations

import polars as pl

from .technical import compute_technical_features
from .regimes import compute_regime_features
from .event_features import aggregate_events_to_daily


def add_targets(df: pl.DataFrame, horizons: tuple[int, ...] = (5, 20)) -> pl.DataFrame:
    """Add forward returns. NEVER use these as features."""
    px = pl.col("adj_close")
    df = df.sort(["ticker", "date"])
    for h in horizons:
        df = df.with_columns(
            ((px.shift(-h).over("ticker") / px) - 1).alias(f"forward_return_{h}d")
        )
    return df


def build_feature_matrix(
    ohlcv: pl.DataFrame,
    events: pl.DataFrame | None = None,
    apprehension: pl.DataFrame | None = None,
    benchmark: str = "SPY",
    horizons: tuple[int, ...] = (5, 20),
) -> pl.DataFrame:
    """End-to-end feature build. Output is one row per (ticker, date).

    Parameters
    ----------
    ohlcv:
        Bronze OHLCV parquet.
    events:
        Optional events DataFrame (EVENT_SCHEMA from ingestion).
        Produces: event_count, event_sentiment_mean, event_magnitude_mean,
        event_novelty_max, risk_flag_count, sent_decay_3d, sent_decay_7d,
        sent_decay_14d, sent_momentum.
    apprehension:
        Optional DataFrame from compute_apprehension_scores().
        Produces: apprehension_score, outlook, apprehension_drivers.
    benchmark:
        Benchmark ticker for regime features.
    horizons:
        Forward-return horizons to attach as targets.
    """
    feat = compute_technical_features(ohlcv)
    feat = compute_regime_features(feat, benchmark=benchmark)
    feat = add_targets(feat, horizons=horizons)

    if events is not None and not events.is_empty():
        ev = aggregate_events_to_daily(events)
        feat = feat.join(ev, on=["date", "ticker"], how="left").with_columns(
            pl.col("event_count").fill_null(0),
            pl.col("event_sentiment_mean").fill_null(0.0),
            pl.col("event_magnitude_mean").fill_null(0.0),
            pl.col("event_novelty_max").fill_null(0.0),
            pl.col("risk_flag_count").fill_null(0),
            pl.col("sent_decay_3d").fill_null(0.0),
            pl.col("sent_decay_7d").fill_null(0.0),
            pl.col("sent_decay_14d").fill_null(0.0),
            pl.col("sent_momentum").fill_null(0.0),
        )

    if apprehension is not None and not apprehension.is_empty():
        app = apprehension.select(["date", "ticker", "apprehension_score", "outlook"])
        feat = feat.join(app, on=["date", "ticker"], how="left").with_columns(
            pl.col("apprehension_score").fill_null(0.0),
            pl.col("outlook").fill_null("stable"),
        )

    return feat
