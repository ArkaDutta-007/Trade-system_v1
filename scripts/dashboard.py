"""Streamlit dashboard. Run via `ts dashboard` or `streamlit run scripts/dashboard.py`."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pandas as pd
import streamlit as st

from trading_system.backtesting import compute_metrics, run_vectorized_backtest, summarize
from trading_system.backtesting.slippage import CostModel
from trading_system.config import get_config
from trading_system.features import build_feature_matrix
from trading_system.strategies import STRATEGY_REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stMetric > div { font-size: 0.95rem; }
    [data-testid="stSidebarNav"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config & data
# ─────────────────────────────────────────────────────────────────────────────
cfg = get_config()
bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
if not bronze.exists():
    st.error("No OHLCV data found. Run `ts ingest` first.")
    st.stop()

@st.cache_data(show_spinner="Loading features…", ttl=3600)
def _load_features():
    ohlcv = pl.read_parquet(bronze)
    gold = cfg.path("data_gold") / "features.parquet"
    if gold.exists():
        features = pl.read_parquet(gold)
    else:
        features = build_feature_matrix(ohlcv, benchmark=cfg["universe"]["benchmark"])
    return ohlcv, features

ohlcv, features = _load_features()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/48/combo-chart.png", width=40)
    st.title("Trade System")
    st.caption(f"Universe: {cfg['universe']['name']} · {len(cfg['universe']['tickers'])} tickers")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Strategy Backtest", "🔍 Stock Screener", "📋 Decision Reports",
         "🌐 Universe Overview", "💼 Paper Portfolio", "🧠 Model Comparison",
         "⚙️ Strategy Catalog"],
        label_visibility="collapsed",
    )
    st.divider()

    # Strategy selector (used by backtest page)
    strategy_names = sorted(STRATEGY_REGISTRY.keys())
    chosen_strategy = st.selectbox("Strategy", strategy_names, index=strategy_names.index("momentum_rotation"))

    st.subheader("Backtest params")
    top_k = st.slider("Top K positions", 1, 20, 6)
    rebal = st.slider("Rebalance (days)", 1, 63, 10)
    commission = st.slider("Commission (bps)", 0.0, 10.0, float(cfg["backtest"]["commission_bps"]))
    slippage = st.slider("Slippage (bps)", 0.0, 10.0, float(cfg["backtest"]["slippage_bps"]))

# ─────────────────────────────────────────────────────────────────────────────
# Page: Strategy Backtest
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Strategy Backtest":
    st.header("📊 Strategy Backtest")

    @st.cache_data(show_spinner="Running backtest…", ttl=300)
    def _run_backtest(strat_name: str, _top_k: int, _rebal: int, _comm: float, _slip: float):
        cls = STRATEGY_REGISTRY[strat_name]
        # Pass common params where the strategy accepts them
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        if "top_k" in sig.parameters:
            kwargs["top_k"] = _top_k
        if "rebalance_days" in sig.parameters:
            kwargs["rebalance_days"] = _rebal
        strat = cls(**kwargs)
        weights = strat.generate_signals(features)
        cost = CostModel(commission_bps=_comm, slippage_bps=_slip, spread_bps=cfg["backtest"]["spread_bps"])
        return run_vectorized_backtest(
            ohlcv, weights, cost=cost,
            signal_delay_days=cfg["backtest"]["signal_delay_days"],
            benchmark=cfg["universe"]["benchmark"],
            initial_cash=cfg["backtest"]["initial_cash"],
            max_position_weight=cfg["backtest"]["max_position_weight"],
            max_gross_exposure=cfg["backtest"]["max_gross_exposure"],
        )

    res = _run_backtest(chosen_strategy, top_k, rebal, commission, slippage)
    metrics = compute_metrics(
        res.daily["net_ret"].to_numpy(),
        turnover=res.daily["turnover"].to_numpy(),
        benchmark=res.benchmark_ret["ret"].to_numpy() if res.benchmark_ret is not None else None,
    )

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
    c2.metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
    c3.metric("Sortino", f"{metrics.get('Sortino', 0):.2f}")
    c4.metric("Max Drawdown", f"{metrics.get('MaxDrawdown', 0):.2%}")
    c5.metric("Annual Vol", f"{metrics.get('AnnualVol', 0):.2%}")
    alpha_val = metrics.get("Alpha") or metrics.get("alpha")
    c6.metric("Alpha vs SPY", f"{alpha_val:.2%}" if alpha_val is not None else "n/a")

    st.divider()

    # Equity curve tab
    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Drawdown", "Holdings", "Full Metrics"])

    with tab1:
        eq = res.daily.select(["date", "equity"]).to_pandas().set_index("date")
        if res.benchmark_ret is not None:
            bm = res.benchmark_ret.to_pandas().set_index("date")
            bm["bm_equity"] = (1 + bm["ret"]).cumprod() * cfg["backtest"]["initial_cash"]
            eq = eq.join(bm[["bm_equity"]], how="left")
            eq.columns = [chosen_strategy, "SPY (benchmark)"]
        st.line_chart(eq, use_container_width=True)

    with tab2:
        eq_arr = res.daily["equity"].to_numpy()
        running_max = pd.Series(eq_arr).cummax()
        drawdown = (pd.Series(eq_arr) / running_max - 1)
        dd_df = pd.DataFrame({"Drawdown": drawdown.values}, index=res.daily["date"].to_numpy())
        st.area_chart(dd_df, color="#d62728", use_container_width=True)

    with tab3:
        w = res.weights_used.to_pandas().set_index("date")
        st.area_chart(w, use_container_width=True)
        st.caption(f"Avg turnover: {res.daily['turnover'].mean():.2%}/day")

    with tab4:
        st.code(summarize(metrics))

    # Multi-strategy comparison
    st.divider()
    st.subheader("Quick Compare: run multiple strategies")
    compare_strats = st.multiselect(
        "Select strategies to compare",
        strategy_names,
        default=["momentum_rotation", "dual_momentum_absolute", "fractal_momentum",
                 "min_vol_portfolio", "adaptive_regime_blend"][:min(5, len(strategy_names))],
    )
    if st.button("▶ Run Comparison") and compare_strats:
        rows = []
        prog = st.progress(0)
        for i, sn in enumerate(compare_strats):
            prog.progress((i + 1) / len(compare_strats), text=f"Running {sn}…")
            try:
                r2 = _run_backtest(sn, top_k, rebal, commission, slippage)
                m2 = compute_metrics(
                    r2.daily["net_ret"].to_numpy(),
                    turnover=r2.daily["turnover"].to_numpy(),
                    benchmark=r2.benchmark_ret["ret"].to_numpy() if r2.benchmark_ret is not None else None,
                )
                rows.append({
                    "Strategy": sn,
                    "CAGR": f"{m2.get('CAGR', 0):.2%}",
                    "Sharpe": f"{m2.get('Sharpe', 0):.2f}",
                    "MaxDD": f"{m2.get('MaxDrawdown', 0):.2%}",
                    "Ann Vol": f"{m2.get('AnnualVol', 0):.2%}",
                    "Sortino": f"{m2.get('Sortino', 0):.2f}",
                })
            except Exception as e:
                rows.append({"Strategy": sn, "CAGR": "error", "Sharpe": str(e)[:40],
                             "MaxDD": "", "Ann Vol": "", "Sortino": ""})
        prog.empty()
        st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Stock Screener
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Stock Screener":
    st.header("🔍 Stock Screener")

    latest = features.filter(pl.col("date") == features["date"].max())

    col_filter, col_result = st.columns([1, 3])
    with col_filter:
        st.subheader("Filters")
        min_mom = st.slider("Min 20d Momentum %", -30, 50, 0) / 100
        max_vol = st.slider("Max Realized Vol % (ann)", 10, 100, 60) / 100
        max_dd = st.slider("Max Drawdown from 60d High %", -50, 0, -5) / 100
        rsi_range = st.slider("RSI(14) range", 0, 100, (20, 80))
        min_adv = st.number_input("Min Avg $ Volume ($M)", 0, 50000, 100) * 1e6
        sort_by = st.selectbox("Sort by", ["mom_20d", "mom_60d", "vol_20d", "rsi_14", "dd_from_high_60"])
        asc = st.checkbox("Ascending", value=False)

    with col_result:
        filtered = latest
        if "mom_20d" in filtered.columns:
            filtered = filtered.filter(pl.col("mom_20d") >= min_mom)
        if "vol_20d" in filtered.columns:
            filtered = filtered.filter(pl.col("vol_20d") <= max_vol)
        if "dd_from_high_60" in filtered.columns:
            filtered = filtered.filter(pl.col("dd_from_high_60") >= max_dd)
        if "rsi_14" in filtered.columns:
            filtered = filtered.filter(
                (pl.col("rsi_14") >= rsi_range[0]) & (pl.col("rsi_14") <= rsi_range[1])
            )
        if "avg_dollar_volume_20" in filtered.columns:
            filtered = filtered.filter(pl.col("avg_dollar_volume_20") >= min_adv)

        display_cols = ["ticker", "adj_close", "ret_1d", "mom_5d", "mom_20d", "mom_60d",
                        "vol_20d", "rsi_14", "atr_14", "dd_from_high_60", "rel_vol_20"]
        available = [c for c in display_cols if c in filtered.columns]

        if sort_by in filtered.columns:
            filtered = filtered.sort(sort_by, descending=not asc)

        st.caption(f"{len(filtered)} stocks match filters (universe date: {features['date'].max()})")

        df_display = filtered.select(available).to_pandas()
        # Format percentages
        for c in ["ret_1d", "mom_5d", "mom_20d", "mom_60d", "vol_20d", "dd_from_high_60"]:
            if c in df_display.columns:
                df_display[c] = df_display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True, height=500)

    # Quick chart for selected ticker
    st.divider()
    st.subheader("Price Chart")
    ticker_sel = st.selectbox("Ticker", sorted(ohlcv["ticker"].unique().to_list()))
    if ticker_sel:
        px_df = ohlcv.filter(pl.col("ticker") == ticker_sel).sort("date").select(
            ["date", "adj_close", "volume"]
        ).to_pandas().set_index("date")
        st.line_chart(px_df[["adj_close"]], use_container_width=True)

        feat_row = latest.filter(pl.col("ticker") == ticker_sel)
        if not feat_row.is_empty():
            st.subheader(f"{ticker_sel} — Latest Signals")
            sig_cols = ["mom_5d", "mom_20d", "mom_60d", "rsi_14", "vol_20d",
                        "sma_gap_50", "sma_gap_200", "breakout_20", "dd_from_high_60"]
            sig_avail = {c: feat_row[c][0] for c in sig_cols if c in feat_row.columns}
            c1, c2, c3 = st.columns(3)
            items = list(sig_avail.items())
            for i, (k, v) in enumerate(items):
                col = [c1, c2, c3][i % 3]
                if isinstance(v, float):
                    col.metric(k, f"{v:.2%}" if abs(v) < 10 else f"{v:.2f}")
                else:
                    col.metric(k, str(v))

# ─────────────────────────────────────────────────────────────────────────────
# Page: Decision Reports
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Decision Reports":
    st.header("📋 Decision Reports")

    reports_dir = cfg.path("reports") / "decisions"
    md_files = sorted(reports_dir.glob("*.md"), reverse=True) if reports_dir.exists() else []
    json_files = sorted(reports_dir.glob("*.json"), reverse=True) if reports_dir.exists() else []

    if not md_files:
        st.info("No decision reports yet. Run `ts analyze MSFT` to generate one.")
    else:
        report_names = [f.name for f in md_files]
        selected = st.selectbox("Select report", report_names)
        sel_path = reports_dir / selected

        col_report, col_json = st.columns([2, 1])

        with col_report:
            st.markdown(sel_path.read_text())

        with col_json:
            json_name = selected.replace(".md", ".json")
            json_path = reports_dir / json_name
            if json_path.exists():
                data = json.loads(json_path.read_text())
                stance = data.get("stance", "HOLD")
                color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(stance, "⚪")
                st.subheader(f"{color} {data.get('ticker')} — {stance}")
                st.metric("Confidence", f"{data.get('confidence', 0):.0%}")
                st.metric("5d Forecast", f"{(data.get('forecast_5d') or 0)*100:.2f}%")
                st.metric("20d Forecast", f"{(data.get('forecast_20d') or 0)*100:.2f}%")
                st.metric("Score Source", data.get("score_source", "n/a"))
                with st.expander("Raw JSON"):
                    st.json(data)

        # Summary table of all reports
        st.divider()
        st.subheader("All Reports Summary")
        summary_rows = []
        for jf in json_files[:50]:
            try:
                d = json.loads(jf.read_text())
                summary_rows.append({
                    "Ticker": d.get("ticker"), "Date": d.get("as_of"),
                    "Stance": d.get("stance"), "Confidence": f"{d.get('confidence', 0):.0%}",
                    "5d Fcst": f"{(d.get('forecast_5d') or 0)*100:.2f}%",
                    "20d Fcst": f"{(d.get('forecast_20d') or 0)*100:.2f}%",
                    "Source": d.get("score_source"),
                })
            except Exception:
                pass
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Universe Overview
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌐 Universe Overview":
    st.header("🌐 Universe Overview")

    u = cfg["universe"]
    st.markdown(f"**Universe:** `{u['name']}` · **Benchmark:** `{u['benchmark']}` · **Total:** {len(u['tickers'])} tickers")

    latest = features.filter(pl.col("date") == features["date"].max())

    tab_heat, tab_rank, tab_regime = st.tabs(["Momentum Heatmap", "Cross-Section Rank", "Regime Stats"])

    with tab_heat:
        if "mom_20d" in latest.columns:
            heat = latest.select(["ticker", "mom_20d"]).sort("mom_20d", descending=True).to_pandas()
            heat["color"] = heat["mom_20d"].apply(lambda x: "🟢" if x > 0.05 else ("🔴" if x < -0.05 else "🟡"))
            heat["mom_20d_pct"] = heat["mom_20d"].map(lambda x: f"{x:.2%}")
            st.dataframe(heat[["ticker", "color", "mom_20d_pct"]].rename(columns={"mom_20d_pct": "20d Mom"}),
                         use_container_width=True, height=500)
        else:
            st.info("Run `ts features` to build the feature matrix.")

    with tab_rank:
        rank_cols = [c for c in ["mom_5d", "mom_20d", "mom_60d", "vol_20d", "rsi_14"] if c in latest.columns]
        if rank_cols:
            rank_df = latest.select(["ticker"] + rank_cols).to_pandas().set_index("ticker")
            for c in rank_cols:
                rank_df[c] = pd.to_numeric(rank_df[c], errors="coerce")
            st.dataframe(
                rank_df.sort_values("mom_20d", ascending=False).style.background_gradient(
                    subset=[c for c in rank_cols if c != "vol_20d"], cmap="RdYlGn"
                ).background_gradient(subset=["vol_20d"] if "vol_20d" in rank_df.columns else [], cmap="RdYlGn_r"),
                use_container_width=True, height=500,
            )

    with tab_regime:
        if "bull_regime" in latest.columns:
            n_bull = int(latest["bull_regime"].sum())
            n_high_vol = int(latest["high_vol_regime"].sum()) if "high_vol_regime" in latest.columns else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Bull Regime Stocks", f"{n_bull}/{len(latest)}")
            c2.metric("High Vol Regime Stocks", f"{n_high_vol}/{len(latest)}")
            c3.metric("Data as of", str(features["date"].max()))

            # Histogram of momentum distribution
            if "mom_20d" in latest.columns:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 3))
                mom = latest["mom_20d"].drop_nulls().to_numpy()
                ax.hist(mom, bins=30, color="#4c72b0", edgecolor="white", alpha=0.85)
                ax.axvline(0, color="red", lw=1.5, linestyle="--")
                ax.set_xlabel("20d Momentum")
                ax.set_ylabel("Count")
                ax.set_title("Universe Momentum Distribution")
                ax.set_facecolor("#0e1117")
                fig.patch.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Run `ts features` to see regime stats.")

# ─────────────────────────────────────────────────────────────────────────────
# Page: Paper Portfolio
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💼 Paper Portfolio":
    st.header("💼 Paper Portfolio")
    st.caption("Simulated portfolio driven by ensemble ML signals · $100k starting capital")

    equity_log_path = cfg.path("data_gold") / "paper_equity_log.parquet"
    journal_path = cfg.path("data_gold") / "paper_portfolio_journal.json"

    if not equity_log_path.exists():
        st.info(
            "No paper portfolio history yet.\n\n"
            "Run `ts paper-trade --backfill` to replay history from model predictions, "
            "or `ts paper-trade` for today's live decisions."
        )
    else:
        eq_log = pl.read_parquet(equity_log_path).sort("date")
        eq_pd = eq_log.to_pandas().set_index("date")

        # ── Summary KPIs ───────────────────────────────────────────────────
        start_eq = float(eq_pd["equity"].iloc[0])
        end_eq = float(eq_pd["equity"].iloc[-1])
        total_return = (end_eq - start_eq) / start_eq
        peak = float(eq_pd["equity"].max())
        max_dd = float(((eq_pd["equity"] / eq_pd["equity"].cummax()) - 1).min())
        n_days = len(eq_pd)
        cagr = (end_eq / start_eq) ** (252 / max(n_days, 1)) - 1 if n_days > 1 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        color = "normal" if total_return >= 0 else "inverse"
        c1.metric("Equity", f"${end_eq:,.0f}", delta=f"{total_return:.2%}")
        c2.metric("CAGR", f"{cagr:.2%}")
        c3.metric("Max Drawdown", f"{max_dd:.2%}")
        c4.metric("Trading Days", str(n_days))
        c5.metric("Peak Equity", f"${peak:,.0f}")

        st.divider()

        # ── Equity curve vs SPY ────────────────────────────────────────────
        tab_eq, tab_dd, tab_horizons, tab_holdings = st.tabs(
            ["Equity Curve", "Drawdown", "Horizon PnL", "Holdings"]
        )

        with tab_eq:
            bench_col = {}
            bm_ticker = cfg["universe"].get("benchmark", "SPY")
            bm_ohlcv = ohlcv.filter(pl.col("ticker") == bm_ticker).sort("date")
            if not bm_ohlcv.is_empty():
                bm_pd = bm_ohlcv.select(["date", "adj_close"]).to_pandas().set_index("date")
                bm_pd["adj_close"] = bm_pd["adj_close"].astype(float)
                bm_start = bm_pd.loc[bm_pd.index >= eq_pd.index[0], "adj_close"]
                if len(bm_start):
                    bm_eq = (bm_start / bm_start.iloc[0]) * start_eq
                    bench_col[bm_ticker] = bm_eq

            plot_df = eq_pd[["equity"]].rename(columns={"equity": "Paper Portfolio"})
            for bm_name, bm_series in bench_col.items():
                plot_df = plot_df.join(bm_series.rename(bm_name), how="left")
            st.line_chart(plot_df, use_container_width=True)

        with tab_dd:
            dd_series = (eq_pd["equity"] / eq_pd["equity"].cummax() - 1)
            st.area_chart(pd.DataFrame({"Drawdown": dd_series}), color="#d62728", use_container_width=True)

        with tab_horizons:
            from trading_system.execution.paper_portfolio import PaperPortfolio
            portfolio = PaperPortfolio(
                journal_path=journal_path,
                equity_log_path=equity_log_path,
            )
            horizons = portfolio.compute_pnl_horizons()
            h_rows = []
            for label, val in horizons.items():
                h_rows.append({"Horizon": label, "Return": f"{val*100:.2f}%" if val is not None else "—"})
            st.table(pd.DataFrame(h_rows).set_index("Horizon"))

        with tab_holdings:
            if journal_path.exists():
                import json as _json
                jdata = _json.loads(journal_path.read_text())
                holdings = {t: q for t, q in jdata.get("holdings", {}).items() if q > 0.001}
                latest_prices = {
                    r["ticker"]: float(r["adj_close"])
                    for r in features.filter(pl.col("date") == features["date"].max()).to_dicts()
                    if r.get("adj_close")
                }
                if holdings:
                    hold_rows = []
                    for ticker, qty in sorted(holdings.items()):
                        px = latest_prices.get(ticker, 0.0)
                        hold_rows.append({
                            "Ticker": ticker,
                            "Qty": f"{qty:.2f}",
                            "Price": f"${px:.2f}",
                            "Value": f"${qty * px:,.0f}",
                        })
                    st.dataframe(pd.DataFrame(hold_rows), use_container_width=True)
                else:
                    st.info("No open positions.")

# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Model Comparison":
    st.header("🧠 Model Comparison")
    st.caption("14 base models + 3 ensemble variants · IC / MAE / R² across walk-forward folds")

    model_comp_path = cfg.path("reports") / "model_comparison.json"
    if not model_comp_path.exists():
        st.info("No model comparison data yet. Run `ts train` to generate it.")
    else:
        import json as _json
        comp = _json.loads(model_comp_path.read_text())
        agg_rows = comp.get("aggregated", [])
        per_fold_rows = comp.get("per_fold", [])

        if agg_rows:
            agg_df = pd.DataFrame(agg_rows).sort_values("ic_mean", ascending=False)

            # ── KPIs for best model ──────────────────────────────────────────
            best_row = agg_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Model", best_row["model"])
            c2.metric("Best IC (mean)", f"{best_row['ic_mean']:.4f}")
            c3.metric("Best R²", f"{best_row['r2_mean']:.4f}")

            st.divider()

            # ── Aggregated comparison table ──────────────────────────────────
            st.subheader("Aggregated Metrics (all folds)")
            disp = agg_df.copy()
            for col in ["ic_mean", "ic_std", "mae_mean", "r2_mean", "weight_mean"]:
                if col in disp.columns:
                    disp[col] = disp[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
            st.dataframe(disp.rename(columns={
                "model": "Model", "ic_mean": "IC (mean)", "ic_std": "IC (std)",
                "mae_mean": "MAE", "r2_mean": "R²", "weight_mean": "Blend Weight"
            }).set_index("Model"), use_container_width=True)

            # ── IC bar chart ─────────────────────────────────────────────────
            st.subheader("IC by Model (mean across folds)")
            ic_df = pd.DataFrame(agg_rows).sort_values("ic_mean", ascending=True)
            ic_series = ic_df.set_index("model")["ic_mean"]
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in ic_series]
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, max(4, len(ic_series) * 0.35)))
            ax.barh(ic_series.index, ic_series.values, color=colors)
            ax.axvline(0, color="white", lw=0.8, linestyle="--")
            ax.set_xlabel("Spearman IC (mean)")
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            st.pyplot(fig, use_container_width=True)

        if per_fold_rows:
            st.divider()
            st.subheader("Per-Fold IC Evolution")
            fold_df = pd.DataFrame(per_fold_rows)
            if "fold" in fold_df.columns and "model" in fold_df.columns and "ic" in fold_df.columns:
                pivot = fold_df.pivot_table(index="fold", columns="model", values="ic", aggfunc="mean")
                # Only show ensemble and a few top models
                ensemble_cols = [c for c in pivot.columns if "ensemble" in c]
                top_base = (
                    fold_df.groupby("model")["ic"].mean()
                    .sort_values(ascending=False)
                    .index[:5].tolist()
                )
                show_cols = list(dict.fromkeys(ensemble_cols + top_base))
                show_cols = [c for c in show_cols if c in pivot.columns]
                st.line_chart(pivot[show_cols], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Strategy Catalog
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Strategy Catalog":
    st.header("⚙️ Strategy Catalog")
    st.caption(f"{len(STRATEGY_REGISTRY)} strategies registered")

    categories = {
        "📈 Trend-Following": [
            "buy_and_hold", "ma_crossover", "momentum_rotation",
            "dual_momentum_absolute", "trend_ema_cross", "triple_ma",
            "mom_vol_filter", "breakout_20d_high", "turtle_donchian",
            "adaptive_trend_regime", "roc_momentum", "weekly_mom_rotation",
            "time_series_momentum",
        ],
        "🔄 Mean Reversion": [
            "mean_reversion", "rsi_oversold_bounce", "bollinger_reversion",
            "overnight_gap_fill", "sma_stretch_reversion", "hf_mean_reversion",
            "drawdown_bounce", "low_rsi_mom_combo",
        ],
        "⚡ Volatility / Risk": [
            "min_vol_portfolio", "vol_breakout", "risk_parity_vol_target",
            "vix_contrarian", "atr_position_sizing",
        ],
        "🔢 Factor Combinations": [
            "mom_quality_combo", "momentum_value_mix", "low_vol_momentum",
            "sector_rotation_proxy", "trend_plus_reversion", "cross_sectional_rankz",
        ],
        "🏛️ Regime / Macro Adaptive": [
            "regime_switching_momentum", "high_vol_cash", "trend_strength_filter",
            "mkt_cap_weighted_mom", "yield_curve_adaptive",
        ],
        "📐 Statistical": [
            "relative_momentum_spread", "cross_sectional_reversal",
            "variance_ratio_pairs", "momentum_anomaly_filter",
        ],
        "🧬 Novel Composites": [
            "momentum_volume_surge", "acceleration_momentum", "adaptive_regime_blend",
            "fractal_momentum", "vol_normalised_momentum", "event_momentum_confluence",
            "contra_momentum_high_vol", "liquidity_weighted_momentum", "multi_hold_blend",
        ],
        "🤖 ML / Event": ["ml_ranker", "event_driven", "event_momentum_confluence"],
    }

    for cat, strats in categories.items():
        with st.expander(cat, expanded=False):
            for name in strats:
                cls = STRATEGY_REGISTRY.get(name)
                if cls is None:
                    continue
                desc = getattr(getattr(cls, "meta", None), "description", "") or ""
                # Try to get description from default instance
                try:
                    inst = cls()
                    desc = inst.meta.description
                except Exception:
                    pass
                st.markdown(f"**`{name}`** — {desc}")

    st.divider()
    st.subheader("API Keys Status")
    import os
    keys = {
        "FRED_API_KEY": os.environ.get("FRED_API_KEY", ""),
        "NEWSAPI_KEY": os.environ.get("NEWSAPI_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
    }
    for k, v in keys.items():
        status = "✅ Set" if v else "❌ Not set"
        st.markdown(f"**{k}**: {status}")

