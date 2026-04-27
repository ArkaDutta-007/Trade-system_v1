"""ML-driven cross-sectional ranker.

Wraps a fitted prediction frame (date, ticker, score) into a top-k weighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .base import StrategyMeta


@dataclass
class MLRankerStrategy:
    predictions: pl.DataFrame  # columns: date, ticker, score
    top_k: int = 4
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(name="ml_ranker", description="Top-k by ML score"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        preds = self.predictions.select(["date", "ticker", "score"])
        df = preds.with_columns(
            rk=pl.col("score").rank(method="ordinal", descending=True).over("date")
        )
        df = df.with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])

        # Rebalance every N days; forward-fill in-between.
        dates = df.select("date").unique().sort("date").with_row_index("i")
        keep = dates.filter((pl.col("i") % self.rebalance_days) == 0).select("date")
        df_rb = df.join(keep, on="date", how="inner")

        all_dates = features.select("date").unique().sort("date")
        all_pairs = all_dates.join(df.select("ticker").unique(), how="cross")
        out = all_pairs.join(df_rb, on=["date", "ticker"], how="left").sort(["ticker", "date"])
        out = out.with_columns(
            weight=pl.col("weight").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return out
