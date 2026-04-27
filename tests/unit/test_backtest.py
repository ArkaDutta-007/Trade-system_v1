import polars as pl
import pytest

from trading_system.backtesting import run_vectorized_backtest, compute_metrics
from trading_system.backtesting.slippage import CostModel
from trading_system.strategies import BuyAndHold


def test_buy_and_hold_matches_benchmark_within_costs(synthetic_ohlcv):
    bah = BuyAndHold(benchmark="SPY")
    weights = bah.generate_signals(synthetic_ohlcv)
    res = run_vectorized_backtest(
        synthetic_ohlcv, weights, cost=CostModel(0, 0, 0),
        max_position_weight=1.0, max_gross_exposure=1.0,
    )
    spy = synthetic_ohlcv.filter(pl.col("ticker") == "SPY").sort("date")
    spy_ret = spy["adj_close"].pct_change().drop_nulls()
    # Day 0 of backtest has weight=0 due to signal_delay; compare day 1+ to SPY day 1+.
    strat_ret = res.daily["net_ret"].slice(1).to_numpy()
    expected = spy_ret.to_numpy()[1:len(strat_ret) + 1]
    diffs = abs(strat_ret - expected[:len(strat_ret)])
    assert diffs.max() < 1e-9


def test_costs_reduce_returns(synthetic_ohlcv):
    bah = BuyAndHold(benchmark="SPY")
    weights = bah.generate_signals(synthetic_ohlcv)
    no_cost = run_vectorized_backtest(synthetic_ohlcv, weights, cost=CostModel(0, 0, 0))
    high_cost = run_vectorized_backtest(synthetic_ohlcv, weights, cost=CostModel(50, 50, 50))
    assert no_cost.daily["net_ret"].sum() >= high_cost.daily["net_ret"].sum()


def test_max_position_weight_respected(synthetic_ohlcv):
    """Verify the engine clips per-ticker weights."""
    n_dates = synthetic_ohlcv["date"].n_unique()
    weights = synthetic_ohlcv.select("date", "ticker").unique().with_columns(
        weight=pl.lit(1.0)  # absurdly large
    )
    res = run_vectorized_backtest(
        synthetic_ohlcv, weights, max_position_weight=0.10, max_gross_exposure=1.0
    )
    used = res.weights_used.drop("date").to_numpy()
    assert used.max() <= 0.10 + 1e-9
