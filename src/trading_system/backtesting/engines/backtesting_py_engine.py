"""backtesting.py adapter. Use for quick single-asset strategy prototyping."""
from __future__ import annotations

import polars as pl


def _require():
    try:
        from backtesting import Backtest  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "backtesting not installed. Install with `pip install -e \".[backtest]\"`."
        ) from e


def backtesting_py_backtest(
    prices: pl.DataFrame,
    ticker: str,
    strategy_class,
    cash: float = 100_000.0,
    commission: float = 0.0002,
):
    """Run a single-asset backtesting.py simulation.

    Args:
        prices: long-form OHLCV frame with [date, ticker, open, high, low, close, volume].
        ticker: which ticker to test.
        strategy_class: a class deriving from backtesting.Strategy.
        cash: initial cash.
        commission: round-trip commission rate.
    """
    _require()
    from backtesting import Backtest

    pdf = (
        prices.filter(pl.col("ticker") == ticker)
        .sort("date")
        .select(["date", "open", "high", "low", "close", "volume"])
        .to_pandas()
        .set_index("date")
        .rename(columns=str.capitalize)
    )
    bt = Backtest(pdf, strategy_class, cash=cash, commission=commission, exclusive_orders=True)
    return bt.run()
