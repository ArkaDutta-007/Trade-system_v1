"""Streamlit dashboard. Run via `ts dashboard` or `streamlit run scripts/dashboard.py`."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from trading_system.backtesting import compute_metrics, run_vectorized_backtest, summarize
from trading_system.backtesting.slippage import CostModel
from trading_system.config import get_config
from trading_system.features import build_feature_matrix
from trading_system.strategies import (
    BuyAndHold, MomentumRotation, MeanReversionAfterDrop, MovingAverageCrossover,
)

st.set_page_config(page_title="Trading System", layout="wide")
st.title("Trading System — Research Dashboard")

cfg = get_config()
bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
if not bronze.exists():
    st.warning(f"No data yet. Run: `ts ingest`")
    st.stop()

ohlcv = pl.read_parquet(bronze)
features = build_feature_matrix(ohlcv, benchmark=cfg["universe"]["benchmark"])

st.sidebar.header("Strategy")
choice = st.sidebar.selectbox(
    "Strategy",
    ["buy_and_hold", "ma_crossover", "momentum_rotation", "mean_reversion"],
)
top_k = st.sidebar.slider("Top K (momentum)", 1, 8, 4)
lookback = st.sidebar.slider("Lookback (days)", 20, 252, 126)

if choice == "buy_and_hold":
    strat = BuyAndHold(benchmark=cfg["universe"]["benchmark"])
elif choice == "ma_crossover":
    strat = MovingAverageCrossover(benchmark=cfg["universe"]["benchmark"])
elif choice == "mean_reversion":
    strat = MeanReversionAfterDrop()
else:
    strat = MomentumRotation(lookback=lookback, top_k=top_k)

weights = strat.generate_signals(features)
cost = CostModel(
    commission_bps=cfg["backtest"]["commission_bps"],
    slippage_bps=cfg["backtest"]["slippage_bps"],
    spread_bps=cfg["backtest"]["spread_bps"],
)
res = run_vectorized_backtest(
    ohlcv, weights, cost=cost,
    signal_delay_days=cfg["backtest"]["signal_delay_days"],
    benchmark=cfg["universe"]["benchmark"],
)
metrics = compute_metrics(
    res.daily["net_ret"].to_numpy(),
    turnover=res.daily["turnover"].to_numpy(),
    benchmark=res.benchmark_ret["ret"].to_numpy() if res.benchmark_ret is not None else None,
)

cols = st.columns(4)
cols[0].metric("CAGR", f"{metrics['CAGR']:.2%}")
cols[1].metric("Sharpe", f"{metrics['Sharpe']:.2f}")
cols[2].metric("Max Drawdown", f"{metrics['MaxDrawdown']:.2%}")
cols[3].metric("Annual Vol", f"{metrics['AnnualVol']:.2%}")

st.subheader("Equity curve")
st.line_chart(res.daily.select(["date", "equity"]).to_pandas().set_index("date"))

st.subheader("Holdings over time")
w = res.weights_used.to_pandas().set_index("date")
st.area_chart(w)

st.subheader("Metrics")
st.code(summarize(metrics))

# Daily report (if present)
reports = sorted(Path(cfg.path("reports")).glob("daily_*.json"))
if reports:
    st.subheader(f"Latest daily report: {reports[-1].name}")
    st.json(json.loads(reports[-1].read_text()))
