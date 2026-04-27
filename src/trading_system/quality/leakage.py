"""Leakage diagnostics. Each function returns a Sharpe ratio so the caller can
compare against the baseline Sharpe.

Convention: a SUSPICIOUS signal is one that improves when shifted forward
(uses future information) or fails to degrade when delayed.
"""
from __future__ import annotations

import polars as pl

from ..backtesting.metrics import sharpe
from ..backtesting.slippage import CostModel
from ..backtesting.vectorized import run_vectorized_backtest


def _bt_sharpe(prices: pl.DataFrame, weights: pl.DataFrame, **kw) -> float:
    res = run_vectorized_backtest(prices, weights, cost=CostModel(), **kw)
    return sharpe(res.daily["net_ret"])


def shift_features_test(prices: pl.DataFrame, weights: pl.DataFrame, shift: int = 1) -> dict:
    """Shift weights backwards (peeking into the future) by `shift` days.

    If Sharpe IMPROVES, weights are leaking future info.
    """
    base = _bt_sharpe(prices, weights, signal_delay_days=1)
    leak_w = weights.sort(["ticker", "date"]).with_columns(
        weight=pl.col("weight").shift(-shift).over("ticker").fill_null(0.0)
    )
    leaked = _bt_sharpe(prices, leak_w, signal_delay_days=1)
    return {"base_sharpe": base, "shifted_sharpe": leaked, "delta": leaked - base,
            "leak_suspect": (leaked - base) > 0.3}


def signal_delay_test(prices: pl.DataFrame, weights: pl.DataFrame) -> dict:
    """Delay signal by extra day; performance should degrade but not collapse."""
    base = _bt_sharpe(prices, weights, signal_delay_days=1)
    delayed = _bt_sharpe(prices, weights, signal_delay_days=3)
    return {"base_sharpe": base, "delayed_sharpe": delayed,
            "robust": (delayed > 0) and (base - delayed) < 0.5 * abs(base + 1e-6) + 1.0}


def label_shuffle_test(prices: pl.DataFrame, weights: pl.DataFrame, seed: int = 0) -> dict:
    """Shuffle weights across dates; Sharpe should be ~0."""
    w = weights.with_columns(
        weight=pl.col("weight").shuffle(seed=seed)
    )
    return {"shuffled_sharpe": _bt_sharpe(prices, w, signal_delay_days=1)}
