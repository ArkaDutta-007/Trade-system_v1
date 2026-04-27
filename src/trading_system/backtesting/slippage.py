"""Slippage and transaction-cost model.

Cost applied to each rebalance is:
    cost_t = turnover_t * (commission_bps + slippage_bps + spread_bps/2) / 10_000
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    spread_bps: float = 1.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.spread_bps / 2.0

    def cost_for_turnover(self, turnover: float) -> float:
        return turnover * self.total_bps / 10_000.0


def apply_slippage(returns_series, turnover_series, cost_model: CostModel):
    """Subtract per-period cost from returns_series (Polars Series or Numpy)."""
    return returns_series - turnover_series * (cost_model.total_bps / 10_000.0)
