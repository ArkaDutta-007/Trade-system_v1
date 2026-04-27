"""Vectorized backtest engine.

Inputs:
    prices  -- long DataFrame with columns [date, ticker, adj_close]
    weights -- long DataFrame with columns [date, ticker, weight]
    cost    -- CostModel
    signal_delay_days -- shift weights forward by N days before applying

Outputs a BacktestResult with daily strategy returns, turnover, equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .slippage import CostModel


@dataclass
class BacktestResult:
    daily: pl.DataFrame  # date, gross_ret, turnover, net_ret, equity
    weights_used: pl.DataFrame
    benchmark_ret: pl.DataFrame | None = None

    def returns(self) -> pl.Series:
        return self.daily["net_ret"]

    def equity(self) -> pl.Series:
        return self.daily["equity"]


def _pivot(df: pl.DataFrame, value: str) -> pl.DataFrame:
    return df.pivot(values=value, index="date", on="ticker", aggregate_function="first").sort("date")


def run_vectorized_backtest(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    cost: CostModel | None = None,
    signal_delay_days: int = 1,
    initial_cash: float = 100_000.0,
    max_gross_exposure: float = 1.0,
    max_position_weight: float = 0.20,
    benchmark: str | None = None,
) -> BacktestResult:
    if cost is None:
        cost = CostModel()

    # Compute per-ticker daily returns
    px = prices.select(["date", "ticker", "adj_close"]).sort(["ticker", "date"])
    px = px.with_columns(ret=(pl.col("adj_close").pct_change().over("ticker"))).drop_nulls("ret")

    # Pivot to wide
    ret_wide = _pivot(px, "ret").fill_null(0.0)
    w_wide = _pivot(weights, "weight").fill_null(0.0)

    # Align dates and tickers
    common_dates = sorted(set(ret_wide["date"]) & set(w_wide["date"]))
    ret_wide = ret_wide.filter(pl.col("date").is_in(common_dates)).sort("date")
    w_wide = w_wide.filter(pl.col("date").is_in(common_dates)).sort("date")

    tickers = [c for c in ret_wide.columns if c != "date" and c in w_wide.columns]
    if not tickers:
        raise ValueError("No overlapping tickers between prices and weights.")

    R = ret_wide.select(tickers).to_numpy()
    W = w_wide.select(tickers).to_numpy()

    # Cap individual weights, then enforce gross exposure cap
    W = np.clip(W, -max_position_weight, max_position_weight)
    gross = np.abs(W).sum(axis=1, keepdims=True)
    scale = np.where(gross > max_gross_exposure, max_gross_exposure / np.maximum(gross, 1e-12), 1.0)
    W = W * scale

    # Apply signal delay: signal computed at t executes at t+delay
    if signal_delay_days > 0:
        W = np.vstack([np.zeros((signal_delay_days, W.shape[1])), W[:-signal_delay_days]])

    # Daily strategy gross return: dot of weights (held over t->t+1) with returns
    # Convention: W[t] is the weight held during the return R[t].
    gross_ret = (W * R).sum(axis=1)

    # Turnover = sum(|W[t] - W[t-1]|)
    turnover = np.zeros_like(gross_ret)
    turnover[0] = np.abs(W[0]).sum()
    turnover[1:] = np.abs(W[1:] - W[:-1]).sum(axis=1)

    cost_drag = turnover * (cost.total_bps / 10_000.0)
    net_ret = gross_ret - cost_drag
    equity = initial_cash * np.cumprod(1.0 + net_ret)

    daily = pl.DataFrame(
        {
            "date": ret_wide["date"],
            "gross_ret": gross_ret,
            "turnover": turnover,
            "cost": cost_drag,
            "net_ret": net_ret,
            "equity": equity,
        }
    )

    bench_df = None
    if benchmark and benchmark in tickers:
        bench_idx = tickers.index(benchmark)
        bench_df = pl.DataFrame({"date": ret_wide["date"], "ret": R[:, bench_idx]})

    weights_used = pl.DataFrame({"date": ret_wide["date"]}).hstack(
        pl.DataFrame({t: W[:, i] for i, t in enumerate(tickers)})
    )

    return BacktestResult(daily=daily, weights_used=weights_used, benchmark_ret=bench_df)
