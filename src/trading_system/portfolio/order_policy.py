"""Translate target weights into orders given current holdings + equity."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    ticker: str
    qty: float
    side: str  # "buy" | "sell"
    notional: float


def weights_to_orders(
    target_weights: dict[str, float],
    holdings: dict[str, float],
    prices: dict[str, float],
    equity: float,
    min_notional: float = 1.0,
) -> list[Order]:
    orders: list[Order] = []
    universe = set(target_weights) | set(holdings)
    for t in universe:
        target_dollars = target_weights.get(t, 0.0) * equity
        px = prices.get(t)
        if px is None or px <= 0:
            continue
        target_qty = target_dollars / px
        cur_qty = holdings.get(t, 0.0)
        delta = target_qty - cur_qty
        notional = abs(delta * px)
        if notional < min_notional:
            continue
        orders.append(
            Order(
                ticker=t,
                qty=delta,
                side="buy" if delta > 0 else "sell",
                notional=notional,
            )
        )
    return orders
