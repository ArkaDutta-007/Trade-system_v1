"""Assemble the full feature matrix from OHLCV + optional event/macro inputs.

V2: Integrates economic calendar (FOMC, CPI, NFP) and earnings calendar
as additional features: days_to_fomc, days_to_earnings, macro_event_imminent.
"""
from __future__ import annotations

import polars as pl

from .technical import compute_technical_features
from .regimes import compute_regime_features
from .event_features import aggregate_events_to_daily, add_macro_calendar_features
from .macro import join_macro_features
from .extended_features import compute_extended_features
from .nonlinear_panel import compute_nonlinear_features
from .rmt import compute_rmt_features


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
    economic_calendar: pl.DataFrame | None = None,
    earnings_calendar: pl.DataFrame | None = None,
    macro_features: pl.DataFrame | None = None,
    benchmark: str = "SPY",
    horizons: tuple[int, ...] = (5, 20),
    add_macro_features: bool = True,
    add_extended_features: bool = True,
    add_nonlinear_features: bool = True,
    nonlinear_deep: bool = False,
    nonlinear_parallel: bool = True,
    nonlinear_jobs: int | None = None,
    add_rmt_features: bool = True,
    add_text_features: bool = False,
    text_cache_dir=None,
    gdelt: pl.DataFrame | None = None,
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
    economic_calendar:
        V2: Optional DataFrame from fetch_economic_calendar().
        Produces: days_to_fomc, macro_event_imminent.
    earnings_calendar:
        V2: Optional DataFrame from build_earnings_calendar().
        Produces: days_to_earnings.
    benchmark:
        Benchmark ticker for regime features.
    horizons:
        Forward-return horizons to attach as targets.
    add_macro_features:
        V2: If True and calendars are provided, adds macro proximity features.
    """
    feat = compute_technical_features(ohlcv)
    if add_extended_features:
        feat = compute_extended_features(feat, benchmark=benchmark)
    feat = compute_regime_features(feat, benchmark=benchmark)
    feat = add_targets(feat, horizons=horizons)

    # Event features are left NULL where there's no news (not 0). "No coverage" is
    # not "neutral sentiment", and 0-filling a mostly-empty column both feeds the
    # models a near-constant recency artifact and fools resolve_reserve's coverage
    # gate. Consistent with the text/GDELT features: null → dropped unless dense.
    # (The recent-fetch events still power the live decision layer via events.parquet.)
    if events is not None and not events.is_empty():
        ev = aggregate_events_to_daily(events)
        feat = feat.join(ev, on=["date", "ticker"], how="left")

    if apprehension is not None and not apprehension.is_empty():
        app = apprehension.select(["date", "ticker", "apprehension_score", "outlook"])
        feat = feat.join(app, on=["date", "ticker"], how="left").with_columns(
            pl.col("outlook").fill_null("stable"),   # categorical, not a trained feature
        )

    # V2: Macro + earnings calendar features
    if add_macro_features and (economic_calendar is not None or earnings_calendar is not None):
        feat = add_macro_calendar_features(
            feat,
            economic_calendar=economic_calendar,
            earnings_calendar=earnings_calendar,
        )
        feat = feat.with_columns(
            pl.col("days_to_fomc").fill_nan(999.0),
            pl.col("days_to_earnings").fill_nan(999.0),
            pl.col("macro_event_imminent").fill_null(False),
            pl.col("hist_earnings_sentiment_mean").fill_nan(0.0),
        )

    # Macro *levels* (yields, curve, VIX, HY OAS, fed funds) joined by date.
    if macro_features is not None and not macro_features.is_empty():
        feat = join_macro_features(feat, macro_features)

    # Nonlinear-dynamics fingerprint (chaos/fractal/entropy/early-warning) — causal,
    # strided, per ticker.  `nonlinear_deep` also computes the heavier O(W²) tier.
    if add_nonlinear_features:
        feat = compute_nonlinear_features(
            feat, deep=nonlinear_deep, parallel=nonlinear_parallel, n_jobs=nonlinear_jobs
        )

    # RMT cross-sectional denoising (systematic-risk fraction + market-mode beta).
    if add_rmt_features:
        feat = compute_rmt_features(feat)

    # GDELT historical news tone/attention — dense back to 2017, point-in-time.
    if gdelt is not None and not gdelt.is_empty():
        from .gdelt_features import compute_gdelt_features
        feat = compute_gdelt_features(feat, gdelt)

    # FinBERT news-text sentiment (optional; needs transformers + events).
    if add_text_features and events is not None and not events.is_empty():
        from .text_features import compute_text_features
        feat = compute_text_features(feat, events, cache_dir=text_cache_dir)

    return feat
