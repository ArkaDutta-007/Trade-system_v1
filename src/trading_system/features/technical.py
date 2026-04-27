"""Technical / price-based features. Computed per ticker, point-in-time safe."""
from __future__ import annotations

import numpy as np
import polars as pl


def _per_ticker(df: pl.DataFrame, exprs: list[pl.Expr]) -> pl.DataFrame:
    return df.sort(["ticker", "date"]).with_columns(
        [e.over("ticker") for e in exprs]
    )


def compute_technical_features(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """Add momentum, volatility, volume, and breakout features.

    Input columns required: date, ticker, open, high, low, close, adj_close, volume.
    All features use only past values (shift before any forward-looking op).
    """
    if ohlcv.is_empty():
        return ohlcv

    px = pl.col("adj_close")
    df = ohlcv.sort(["ticker", "date"])
    df = df.with_columns(
        ret_1d=px.pct_change().over("ticker"),
        log_ret_1d=(px / px.shift(1).over("ticker")).log(),
    )

    # Momentum windows
    for w in (5, 10, 20, 60, 120):
        df = df.with_columns(
            (px / px.shift(w).over("ticker") - 1).alias(f"mom_{w}d"),
        )
    # 12-1 momentum: skip the most recent month
    df = df.with_columns(
        ((px.shift(21).over("ticker") / px.shift(252).over("ticker")) - 1).alias("mom_12m1m"),
    )

    # Moving averages and gaps
    for w in (10, 20, 50, 200):
        df = df.with_columns(
            px.rolling_mean(window_size=w).over("ticker").alias(f"sma_{w}"),
        )
        df = df.with_columns(
            ((px / pl.col(f"sma_{w}")) - 1).alias(f"sma_gap_{w}"),
        )

    # Realized vol & drawdown
    for w in (10, 20, 60):
        df = df.with_columns(
            (pl.col("log_ret_1d").rolling_std(window_size=w).over("ticker") * np.sqrt(252)).alias(
                f"vol_{w}d"
            )
        )

    df = df.with_columns(
        rolling_max_60=px.rolling_max(window_size=60).over("ticker"),
    ).with_columns(
        ((px / pl.col("rolling_max_60")) - 1).alias("dd_from_high_60"),
    )

    # ATR (using high-low range proxy on log scale via TR)
    high, low, close = pl.col("high"), pl.col("low"), pl.col("close")
    prev_close = close.shift(1).over("ticker")
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    df = df.with_columns(tr=tr).with_columns(
        atr_14=pl.col("tr").rolling_mean(window_size=14).over("ticker"),
    )

    # Volume / liquidity
    df = df.with_columns(
        dollar_volume=(close * pl.col("volume")),
    ).with_columns(
        rel_vol_20=(pl.col("volume") / pl.col("volume").rolling_mean(window_size=20).over("ticker")),
        avg_dollar_volume_20=pl.col("dollar_volume").rolling_mean(window_size=20).over("ticker"),
    )

    # Breakout distance
    df = df.with_columns(
        rolling_max_20=high.rolling_max(window_size=20).over("ticker"),
        rolling_min_20=low.rolling_min(window_size=20).over("ticker"),
    ).with_columns(
        breakout_20=((close - pl.col("rolling_max_20")) / pl.col("rolling_max_20")),
        breakdown_20=((close - pl.col("rolling_min_20")) / pl.col("rolling_min_20")),
    )

    # RSI(14)
    delta = px.diff().over("ticker")
    up = pl.when(delta > 0).then(delta).otherwise(0.0)
    down = pl.when(delta < 0).then(-delta).otherwise(0.0)
    df = df.with_columns(
        roll_up=up.rolling_mean(window_size=14).over("ticker"),
        roll_down=down.rolling_mean(window_size=14).over("ticker"),
    ).with_columns(
        rsi_14=(100 - 100 / (1 + (pl.col("roll_up") / pl.col("roll_down"))))
    )

    return df.drop(["tr", "rolling_max_60", "rolling_max_20", "rolling_min_20",
                    "dollar_volume", "roll_up", "roll_down"])
