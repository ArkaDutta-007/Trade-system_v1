"""Slippage and transaction-cost model.

Cost applied to each rebalance is:
    cost_t = turnover_t * (commission_bps + slippage_bps + spread_bps/2) / 10_000
             + impact_coeff_bps * turnover_t^1.5 / 10_000

The second term is a square-root market-impact model (impact per unit traded
grows with the square root of trade size, so total cost grows with
turnover^1.5). ``impact_coeff_bps = 0`` (default) reproduces the old flat
model exactly; 10–20 is a reasonable range for liquid US large caps.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    impact_coeff_bps: float = 0.0   # square-root impact; 0 = legacy flat costs

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.spread_bps / 2.0

    def cost_for_turnover(self, turnover: float) -> float:
        impact = self.impact_coeff_bps * turnover ** 1.5
        return (turnover * self.total_bps + impact) / 10_000.0


def apply_slippage(returns_series, turnover_series, cost_model: CostModel):
    """Subtract per-period cost from returns_series (Polars Series or Numpy)."""
    flat = turnover_series * (cost_model.total_bps / 10_000.0)
    impact = (turnover_series ** 1.5) * (cost_model.impact_coeff_bps / 10_000.0)
    return returns_series - flat - impact
