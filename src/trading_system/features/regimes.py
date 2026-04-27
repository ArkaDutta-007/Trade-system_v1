"""Regime / cross-sectional features."""
from __future__ import annotations

import polars as pl


def compute_regime_features(features: pl.DataFrame, benchmark: str = "SPY") -> pl.DataFrame:
    """Add bull/bear, vol regime, and cross-sectional rank features.

    Requires `features` to include date, ticker, adj_close, vol_20d, mom_20d, sma_200.
    """
    if features.is_empty():
        return features

    df = features.sort(["ticker", "date"])

    # Per-ticker regime: above 200d SMA = bull
    if "sma_200" in df.columns:
        df = df.with_columns(
            bull_regime=(pl.col("adj_close") > pl.col("sma_200")).cast(pl.Int8),
        )

    if "vol_20d" in df.columns:
        df = df.with_columns(
            high_vol_regime=(
                pl.col("vol_20d") > pl.col("vol_20d").rolling_median(window_size=252).over("ticker")
            ).cast(pl.Int8)
        )

    # Cross-sectional momentum rank within date
    if "mom_20d" in df.columns:
        df = df.with_columns(
            mom_20d_rank=pl.col("mom_20d").rank("ordinal").over("date"),
        ).with_columns(
            mom_20d_rank=(pl.col("mom_20d_rank") - 1)
            / (pl.col("mom_20d_rank").max().over("date") - 1).clip(lower_bound=1)
        )

    # Benchmark-relative excess return (vs SPY)
    bench = (
        df.filter(pl.col("ticker") == benchmark)
        .select(["date", pl.col("ret_1d").alias("bench_ret_1d")])
    )
    df = df.join(bench, on="date", how="left").with_columns(
        excess_ret_1d=(pl.col("ret_1d") - pl.col("bench_ret_1d")),
    )

    return df
