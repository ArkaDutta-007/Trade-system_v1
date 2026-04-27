"""Per-ticker PnL attribution from a backtest."""
from __future__ import annotations

import polars as pl


def attribute_pnl(weights_used: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Compute per-(date, ticker) contribution to portfolio return.

    weights_used: wide frame [date, t1, t2, ...] of executed weights.
    prices: long frame [date, ticker, adj_close].
    """
    rets = (
        prices.sort(["ticker", "date"])
        .with_columns(ret=pl.col("adj_close").pct_change().over("ticker"))
        .select(["date", "ticker", "ret"])
    )
    long_w = weights_used.unpivot(index="date", variable_name="ticker", value_name="weight")
    df = long_w.join(rets, on=["date", "ticker"], how="left").drop_nulls("ret")
    df = df.with_columns(contribution=(pl.col("weight") * pl.col("ret")))
    by_ticker = df.group_by("ticker").agg(
        total_contribution=pl.col("contribution").sum(),
        days_held=(pl.col("weight").abs() > 1e-6).sum(),
    ).sort("total_contribution", descending=True)
    return by_ticker
