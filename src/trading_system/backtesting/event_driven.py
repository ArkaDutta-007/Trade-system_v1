"""Lightweight event-driven simulator with explicit fills, for execution realism checks.

Vectorized backtest is the workhorse; this module is for verifying that order policies
(stop-loss, limits) reproduce vectorized PnL within tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Fill:
    date: object
    ticker: str
    qty: float
    price: float
    side: str  # "buy" | "sell"


def simulate_fills_from_weights(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    initial_cash: float = 100_000.0,
    cost_bps: float = 4.0,
) -> tuple[pl.DataFrame, list[Fill]]:
    """Convert target weights into share orders, simulate at next-day open."""
    px = prices.select(["date", "ticker", "open", "adj_close"]).sort(["ticker", "date"])
    w = weights.select(["date", "ticker", "weight"]).sort(["ticker", "date"])

    df = px.join(w, on=["date", "ticker"], how="left").with_columns(
        weight=pl.col("weight").fill_null(0.0)
    )

    fills: list[Fill] = []
    cash = initial_cash
    holdings: dict[str, float] = {}
    equity_rows = []

    for (d,), block in df.group_by("date", maintain_order=True):
        # Target dollar per ticker
        prices_d = dict(zip(block["ticker"].to_list(), block["adj_close"].to_list()))
        opens_d = dict(zip(block["ticker"].to_list(), block["open"].to_list()))
        weights_d = dict(zip(block["ticker"].to_list(), block["weight"].to_list()))

        equity = cash + sum(holdings.get(t, 0) * prices_d.get(t, 0.0) for t in holdings)
        for t, target_w in weights_d.items():
            target_dollars = target_w * equity
            cur_qty = holdings.get(t, 0.0)
            px_open = opens_d.get(t) or prices_d.get(t) or 0.0
            if px_open <= 0:
                continue
            target_qty = target_dollars / px_open
            delta = target_qty - cur_qty
            if abs(delta * px_open) < 1.0:
                continue
            cost = abs(delta * px_open) * (cost_bps / 10_000.0)
            cash -= delta * px_open + cost
            holdings[t] = target_qty
            fills.append(Fill(d, t, delta, px_open, "buy" if delta > 0 else "sell"))

        equity = cash + sum(holdings.get(t, 0) * prices_d.get(t, 0.0) for t in holdings)
        equity_rows.append({"date": d, "cash": cash, "equity": equity})

    eq = pl.DataFrame(equity_rows).sort("date")
    return eq, fills
