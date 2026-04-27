"""Strategy contract.

A strategy maps a feature matrix into a target weight per (date, ticker).
Weights are bounded, sum to <= max_gross_exposure, and are point-in-time:
weight at date `t` is decided using info up to `t` (close-of-day).

The backtest engine applies the configured signal_delay_days when executing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


SignalFrame = pl.DataFrame  # columns: date, ticker, weight


@dataclass
class StrategyMeta:
    name: str
    description: str
    universe: list[str] | None = None


class Strategy(Protocol):
    meta: StrategyMeta

    def generate_signals(self, features: pl.DataFrame) -> SignalFrame: ...
