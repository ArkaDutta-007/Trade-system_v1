"""bt adapter. Use for portfolio-level allocation backtests (e.g. monthly rebalancing)."""
from __future__ import annotations

import polars as pl


def _require():
    try:
        import bt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bt not installed. Install with `pip install -e \".[backtest]\"`."
        ) from e


def bt_backtest(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    name: str = "strategy",
    initial_cash: float = 100_000.0,
):
    """Run a `bt` backtest using user-supplied target weights.

    Returns a bt Result object so the caller can call `.display()` or
    `.stats` for institutional-style reports.
    """
    _require()
    import bt

    px = (
        prices.select(["date", "ticker", "adj_close"]).sort(["ticker", "date"]).to_pandas()
        .pivot(index="date", columns="ticker", values="adj_close")
    )
    w = (
        weights.select(["date", "ticker", "weight"]).sort(["ticker", "date"]).to_pandas()
        .pivot(index="date", columns="ticker", values="weight")
        .reindex(index=px.index, columns=px.columns).fillna(0.0)
    )
    strat = bt.Strategy(
        name,
        [
            bt.algos.RunOnDate(*w.index),
            bt.algos.SelectAll(),
            bt.algos.WeighTarget(w),
            bt.algos.Rebalance(),
        ],
    )
    test = bt.Backtest(strat, px, initial_capital=initial_cash)
    return bt.run(test)
