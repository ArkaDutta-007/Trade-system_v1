#!/usr/bin/env python3
"""Quick daily backtest of the CURRENT model predictions — digest-friendly.

Reads data/gold/predictions.parquet (walk-forward OOS scores, refreshed by the
weekly retrain), trades top-20 equal-weight through the repo's vectorized
backtester (flat costs), and prints a compact block for the morning digest:
full-history metrics, the risk-overlay version, and the trailing 63-day
(≈1 quarter) health check so model decay is visible within days, not months.

Runtime ~30-60s. Read-only on the repo except nothing — writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_repo_to_path, ops_root  # noqa: E402

REPO = add_repo_to_path()
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402
from trading_system.backtesting import compute_metrics, run_vectorized_backtest  # noqa: E402
from trading_system.backtesting.slippage import CostModel  # noqa: E402

from backtest import build_weights, overlay  # noqa: E402  (research modules)

PREDS = REPO / "data/gold/predictions.parquet"
ANN = 252


def fmt(m: dict) -> str:
    return (f"CAGR {m['CAGR']:+.1%} · Sharpe {m['Sharpe']:.2f} · "
            f"MaxDD {m['MaxDrawdown']:.1%} · Calmar {m.get('Calmar', 0):.2f} · "
            f"hit {m['HitRate']:.0%}")


def main() -> int:
    if not PREDS.exists():
        print("no data/gold/predictions.parquet — run the weekly retrain "
              "(~/trade-ops/weekly_retrain.sh) or `ts train` first.")
        return 0
    oos = pl.read_parquet(PREDS)
    if oos.is_empty():
        print("predictions.parquet is empty — model needs retraining.")
        return 0
    prices = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet")
    feat = pl.read_parquet(REPO / "data/gold/features.parquet")
    vol = feat.select(["date", "ticker", "vol_20d"])

    w = build_weights(oos, vol, "equal")
    res = run_vectorized_backtest(
        prices, w, cost=CostModel(1.0, 2.0, 1.0), signal_delay_days=1,
        initial_cash=10_000.0, max_gross_exposure=1.0,
        max_position_weight=0.10, benchmark="SPY")
    net = res.daily["net_ret"].to_numpy()
    dates = res.daily["date"].to_list()

    full = compute_metrics(net, turnover=res.daily["turnover"].to_numpy())
    ov = compute_metrics(overlay(net))

    print(f"Model P&L (top-20 equal-wt, OOS {dates[0]} → {dates[-1]}, after costs)")
    print(f"  raw     : {fmt(full)}")
    print(f"  overlay : {fmt(ov)}   (15% vol-target + dd-throttle)")

    if len(net) >= 63:
        q = net[-63:]
        qret = float(np.prod(1 + q) - 1)
        qsharpe = float(np.mean(q) / (np.std(q, ddof=1) + 1e-12) * np.sqrt(ANN))
        flag = "OK" if qsharpe > 0 else "⚠ DECAY?"
        print(f"  last 63d: ret {qret:+.1%} · Sharpe {qsharpe:.2f}  [{flag}]")

    # today's top-10 names by score, for the brief
    last_day = oos["date"].max()
    top = (oos.filter(pl.col("date") == last_day)
              .sort("score", descending=True).head(10))
    names = ", ".join(top["ticker"].to_list())
    print(f"  top-10 today ({last_day}): {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
