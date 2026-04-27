"""vectorbt adapter. Use for fast parameter sweeps over signal grids."""
from __future__ import annotations

import polars as pl


def _require():
    try:
        import vectorbt as vbt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "vectorbt not installed. Install with `pip install -e \".[backtest]\"`."
        ) from e


def vectorbt_backtest(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    initial_cash: float = 100_000.0,
    fees: float = 0.0001,
    slippage: float = 0.0002,
):
    """Run a portfolio backtest in vectorbt. Returns a vbt.Portfolio object."""
    _require()
    import vectorbt as vbt

    px = (
        prices.select(["date", "ticker", "adj_close"]).sort(["ticker", "date"]).to_pandas()
        .pivot(index="date", columns="ticker", values="adj_close")
    )
    w = (
        weights.select(["date", "ticker", "weight"]).sort(["ticker", "date"]).to_pandas()
        .pivot(index="date", columns="ticker", values="weight")
        .reindex(index=px.index, columns=px.columns).fillna(0.0)
    )

    pf = vbt.Portfolio.from_orders(
        close=px,
        size=w,
        size_type="targetpercent",
        fees=fees,
        slippage=slippage,
        init_cash=initial_cash,
        cash_sharing=True,
        group_by=True,
        freq="1D",
    )
    return pf
