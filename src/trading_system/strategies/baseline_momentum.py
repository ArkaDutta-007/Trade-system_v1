"""Baseline momentum and benchmark strategies."""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .base import StrategyMeta


@dataclass
class BuyAndHold:
    benchmark: str = "SPY"
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(name="buy_and_hold", description="100% benchmark"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        return (
            features.select("date").unique()
            .with_columns(ticker=pl.lit(self.benchmark), weight=pl.lit(1.0))
            .sort("date")
        )


@dataclass
class MovingAverageCrossover:
    fast: int = 50
    slow: int = 200
    benchmark: str = "SPY"
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(name="ma_crossover", description="Long benchmark when fast > slow"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        f = features.filter(pl.col("ticker") == self.benchmark).sort("date")
        f = f.with_columns(
            fast=pl.col("adj_close").rolling_mean(self.fast),
            slow=pl.col("adj_close").rolling_mean(self.slow),
        )
        return f.select(
            "date",
            pl.lit(self.benchmark).alias("ticker"),
            pl.when(pl.col("fast") > pl.col("slow")).then(1.0).otherwise(0.0).alias("weight"),
        )


@dataclass
class MomentumRotation:
    """Top-k cross-sectional momentum over `lookback` days, equally weighted."""
    lookback: int = 126  # ~6 months
    top_k: int = 4
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(name="momentum_rotation", description="Top-k momentum, monthly"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = f"mom_{self.lookback}d" if f"mom_{self.lookback}d" in features.columns else "mom_120d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        # Rank and select top_k per date
        df = df.with_columns(
            rk=pl.col(col).rank(method="ordinal", descending=True).over("date")
        )
        df = df.with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"]).sort(["date", "ticker"])

        # Sparse rebalance: only update weights every `rebalance_days`
        dates = df.select("date").unique().sort("date").with_row_index("i")
        keep = dates.filter((pl.col("i") % self.rebalance_days) == 0).select("date")
        df_rb = df.join(keep, on="date", how="inner")

        # Forward-fill weights between rebalance dates
        all_dates = features.select("date").unique().sort("date")
        all_pairs = all_dates.join(df.select("ticker").unique(), how="cross")
        out = all_pairs.join(df_rb, on=["date", "ticker"], how="left").sort(["ticker", "date"])
        out = out.with_columns(
            weight=pl.col("weight").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return out
