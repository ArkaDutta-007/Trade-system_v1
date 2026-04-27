"""Mean-reversion strategy: long after a sharp drop in a high-quality regime."""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .base import StrategyMeta


@dataclass
class MeanReversionAfterDrop:
    drop_threshold: float = -0.03   # 3% down day
    hold_days: int = 5
    require_bull_regime: bool = True
    weight_per_position: float = 0.10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="mean_reversion",
        description="Long after large 1-day drop, hold N days",
    ))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        cond = pl.col("ret_1d") < self.drop_threshold
        if self.require_bull_regime and "bull_regime" in df.columns:
            cond = cond & (pl.col("bull_regime") == 1)

        df = df.with_columns(entry=cond.cast(pl.Int8))
        # Stay long for hold_days after entry
        df = df.with_columns(
            in_trade=pl.col("entry")
            .rolling_sum(window_size=self.hold_days, min_periods=1)
            .over("ticker")
            .clip(0, 1)
        )
        return df.select(
            "date", "ticker",
            (pl.col("in_trade").cast(pl.Float64) * self.weight_per_position).alias("weight"),
        )
