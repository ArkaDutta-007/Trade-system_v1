"""Event-driven strategy: small positive tilt on positive-sentiment, novel events."""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .base import StrategyMeta


@dataclass
class EventDrivenStrategy:
    sentiment_threshold: float = 0.3
    novelty_threshold: float = 0.5
    hold_days: int = 3
    weight_per_position: float = 0.05
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(name="event_driven", description="Sentiment + novelty tilt"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "event_sentiment_mean" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.sort(["ticker", "date"])
        entry = (
            (pl.col("event_sentiment_mean") > self.sentiment_threshold)
            & (pl.col("event_novelty_max") > self.novelty_threshold)
        ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry")
            .rolling_sum(window_size=self.hold_days, min_periods=1)
            .over("ticker")
            .clip(0, 1)
        )
        return df.select(
            "date", "ticker",
            (pl.col("in_trade").cast(pl.Float64) * self.weight_per_position).alias("weight"),
        )
