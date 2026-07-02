"""Streamlit dashboard. Run via `ts dashboard` or `streamlit run scripts/dashboard.py`.

Primary pages:
  🎯 Trade Signals     — latest BUY/SELL/HOLD predictions for every ticker
  📈 Future Predictions — forward model forecasts (verifiable via future backtest)
  🔍 Stock Screener    — filter by momentum, vol, RSI
  📋 Decision Reports  — full per-ticker analysis reports

Portfolio:
  💰 Invest Planner    — budget → gated, sized, hold-horizon buy plan
  💼 Paper Simulation  — clean forward paper trading (reset, fresh $10k)
  📒 Calibration Ledger — the system's scored track record (hit/coverage/IC)
  ⚡ Live Prices       — quasi-realtime price overlay
  🤖 Agent Analysis    — ReAct thought chain + SHAP waterfall per ticker

Research (internal):
  🌐 Universe Overview · 📊 Strategy Backtest · 🧠 Model Comparison · ⚙️ Strategy Catalog
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pandas as pd
import streamlit as st

from trading_system.backtesting import compute_metrics, run_vectorized_backtest, summarize
from trading_system.backtesting.slippage import CostModel
from trading_system.config import get_config
from trading_system.features import build_feature_matrix
from trading_system.strategies import STRATEGY_REGISTRY

# Top-level plotly/matplotlib imports (moved inside page handlers for speed; caught gracefully)
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner look + signal badges
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stMetric > div { font-size: 0.95rem; }
    .signal-table td, .signal-table th {
        padding: 6px 12px;
        font-size: 0.9rem;
    }
    .signal-table tr:nth-child(even) { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config & data
# ─────────────────────────────────────────────────────────────────────────────
try:
    cfg = get_config()
except Exception as _cfg_err:
    st.error(f"Failed to load config: {_cfg_err}")
    st.info("Make sure `configs/default.yaml` exists and is valid YAML.")
    st.stop()

try:
    bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
    if not bronze.exists():
        st.error("No OHLCV data found. Run `ts ingest` first.")
        st.stop()
except Exception as _bronze_err:
    st.warning(f"Could not check OHLCV path: {_bronze_err}. Some pages may fail.")

# ── Lazy-load helpers (only called for pages that need them) ──────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_ohlcv():
    """Load raw OHLCV — needed by most pages."""
    return pl.read_parquet(bronze)

@st.cache_data(show_spinner=False, ttl=3600)
def _load_features():
    """Build or load feature matrix — only called by pages that need signals."""
    ohlcv = _load_ohlcv()
    gold = cfg.path("data_gold") / "features.parquet"
    if gold.exists():
        return pl.read_parquet(gold)
    return build_feature_matrix(ohlcv, benchmark=cfg["universe"]["benchmark"])

@st.cache_data(show_spinner=False, ttl=300)
def _load_decisions():
    """Return dict[ticker → latest decision JSON] from reports/decisions/."""
    decisions_dir = cfg.path("reports") / "decisions"
    if not decisions_dir.exists():
        return {}
    latest: dict[str, dict] = {}
    for jf in sorted(decisions_dir.glob("*.json"), reverse=True):
        t = jf.name.split("_")[0]
        if t not in latest:
            try:
                latest[t] = json.loads(jf.read_text())
            except Exception:
                pass
    return latest

@st.cache_data(show_spinner=False, ttl=900)
def _load_events_for(ticker: str):
    """Return a list of {date, summary, sentiment, source} events for a ticker."""
    ev_path = cfg.path("data_silver") / "events.parquet"
    if not ev_path.exists():
        return []
    try:
        ev = pl.read_parquet(ev_path)
        sub = (
            ev.explode("tickers")
            .filter(pl.col("tickers") == ticker.upper())
            .with_columns(d=pl.col("known_at").dt.date())
            .select(["d", "summary", "sentiment", "source"])
            .sort("d", descending=True)
        )
        return sub.to_dicts()
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=1800)
def _compute_bounds_cached(ticker: str):
    """Calibrated price bounds for a ticker (quantile bundle → MC fallback)."""
    try:
        from trading_system.decision.bounds import compute_bounds
        ohlcv = _load_ohlcv()
        features = _load_features()
        last_date = features["date"].max()
        row = features.filter((pl.col("ticker") == ticker.upper()) & (pl.col("date") == last_date))
        if row.is_empty():
            return None
        last_price = float(row["adj_close"][0]) if "adj_close" in row.columns else 0.0
        dec = _load_decisions().get(ticker.upper(), {})
        score = float(dec.get("forecast_5d") or 0.0)
        return compute_bounds(cfg, ticker, features, ohlcv, last_price, score)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=600)
def _last_close_prices():
    """Return dict[ticker → last adj_close] from feature matrix."""
    feats = _load_features()
    return {
        r["ticker"]: float(r["adj_close"])
        for r in feats.filter(pl.col("date") == feats["date"].max()).to_dicts()
        if r.get("adj_close")
    }

@st.cache_data(show_spinner=False, ttl=900)
def _fetch_yf_news(ticker: str) -> list[dict]:
    """Fetch latest news for a ticker via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).news
        return info[:10] if info else []
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/48/combo-chart.png", width=40)
    st.title("Trade System")
    st.caption(f"Universe: {cfg['universe']['name']} · {len(cfg['universe']['tickers'])} tickers")
    st.divider()

    # Grouped, two-level navigation — 14 pages collapsed into 5 intuitive sections
    NAV_GROUPS = {
        "🚦 Desk": ["🚦 Trading Desk", "🧭 Playbook", "⚡ Live Prices"],
        "🔭 Research": ["🔭 Stock Analysis", "🔍 Stock Screener",
                        "🌐 Universe Overview", "🎯 Trade Signals"],
        "🔮 Forecasts": ["📈 Future Predictions", "📋 Decision Reports", "🤖 Agent Analysis"],
        "🧠 Models": ["🧠 Model Comparison", "📊 Strategy Backtest", "⚙️ Strategy Catalog"],
        "💼 Portfolio": ["💰 Invest Planner", "💼 Paper Simulation", "📒 Calibration Ledger"],
    }
    section = st.selectbox("Section", list(NAV_GROUPS), key="nav_section_main")
    page = st.radio("Page", NAV_GROUPS[section], label_visibility="collapsed", key="nav_page")
    st.divider()

    # Backtest params — only shown when on the backtest page
    strategy_names = sorted(STRATEGY_REGISTRY.keys())
    if page == "📊 Strategy Backtest":
        chosen_strategy = st.selectbox("Strategy", strategy_names,
                                       index=strategy_names.index("momentum_rotation"),
                                       key="bt_strategy")
        st.subheader("Backtest params")
        top_k = st.slider("Top K positions", 1, 20, 6, key="bt_top_k")
        rebal = st.slider("Rebalance (days)", 1, 63, 10, key="bt_rebal")
        commission = st.slider("Commission (bps)", 0.0, 10.0, float(cfg["backtest"]["commission_bps"]), key="bt_commission")
        slippage = st.slider("Slippage (bps)", 0.0, 10.0, float(cfg["backtest"]["slippage_bps"]), key="bt_slippage")
    # else: backtest defaults only needed in the backtest page block below

# ─────────────────────────────────────────────────────────────────────────────
# Page: Trading Desk + Playbook  (V3 — live flags + playbook engine)
# ─────────────────────────────────────────────────────────────────────────────
if page == "🚦 Trading Desk":
    from scripts import desk
    desk.render_trading_desk(cfg)

elif page == "🧭 Playbook":
    from scripts import desk
    desk.render_playbook(cfg)

elif page == "💰 Invest Planner":
    from scripts import invest_desk
    invest_desk.render_invest(cfg)

elif page == "📒 Calibration Ledger":
    from scripts import invest_desk
    invest_desk.render_ledger(cfg)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Trade Signals  (PRIMARY PAGE)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Trade Signals":
    features = _load_features()
    st.header("🎯 Trade Signals")
    st.caption(
        "Latest BUY / SELL / HOLD for every ticker in the universe · "
        "powered by ensemble ML model · run `ts analyze TICKER` or `ts run` to refresh"
    )

    reports_dir = cfg.path("reports") / "decisions"
    json_files = sorted(reports_dir.glob("*.json"), reverse=True) if reports_dir.exists() else []

    if not json_files:
        st.info(
            "No decision reports found.  \n"
            "Run `ts analyze AAPL` for one stock, or `ts run` for the full universe."
        )
    else:
        # ── Load latest signal per ticker ────────────────────────────────────
        latest_signals: dict[str, dict] = {}
        for jf in json_files:
            ticker = jf.name.split("_")[0]
            if ticker not in latest_signals:
                try:
                    latest_signals[ticker] = json.loads(jf.read_text())
                except Exception:
                    pass

        # Last-close prices from feature matrix as fallback
        last_close = {
            r["ticker"]: float(r["adj_close"])
            for r in features.filter(pl.col("date") == features["date"].max()).to_dicts()
            if r.get("adj_close")
        }

        # ── Build summary DataFrame ──────────────────────────────────────────
        rows = []
        for ticker, d in sorted(latest_signals.items()):
            rows.append({
                "Ticker": ticker,
                "Signal": d.get("stance", "HOLD"),
                "Confidence": d.get("confidence", 0),
                "5d Forecast": d.get("forecast_5d", 0) or 0,
                "20d Forecast": d.get("forecast_20d", 0) or 0,
                "Last Close": last_close.get(ticker),
                "Model": (d.get("score_source") or "").replace("ensemble:", ""),
                "As Of": d.get("as_of", ""),
            })

        sig_df = pd.DataFrame(rows)

        # ── KPI banner ───────────────────────────────────────────────────────
        buy_n = int((sig_df["Signal"] == "BUY").sum())
        hold_n = int((sig_df["Signal"] == "HOLD").sum())
        sell_n = int((sig_df["Signal"] == "SELL").sum())
        as_of_latest = sig_df["As Of"].max() if not sig_df.empty else "—"

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🟢 BUY", buy_n)
        col2.metric("🟡 HOLD", hold_n)
        col3.metric("🔴 SELL", sell_n)
        col4.metric("Total Tickers", len(sig_df))
        col5.metric("Signals As Of", as_of_latest)

        st.divider()

        # ── Filter + sort controls ───────────────────────────────────────────
        cf1, cf2, cf3 = st.columns([1, 1, 2])
        with cf1:
            stance_filter = st.multiselect(
                "Filter by Signal", ["BUY", "HOLD", "SELL"],
                default=["BUY", "HOLD", "SELL"], key="ts_stance"
            )
        with cf2:
            min_conf = st.slider(
                "Min Confidence", 0.0, 1.0, 0.0, 0.05, format="%.0f%%", key="ts_conf"
            )
        with cf3:
            sort_col = st.selectbox(
                "Sort by",
                ["Confidence ↓", "5d Forecast ↓", "20d Forecast ↓", "Ticker ↑"],
                key="ts_sort",
            )

        # Apply filters
        filtered = sig_df[sig_df["Signal"].isin(stance_filter)].copy()
        filtered = filtered[filtered["Confidence"] >= min_conf]
        sort_map = {
            "Confidence ↓": ("Confidence", False),
            "5d Forecast ↓": ("5d Forecast", False),
            "20d Forecast ↓": ("20d Forecast", False),
            "Ticker ↑": ("Ticker", True),
        }
        sk, sa = sort_map.get(sort_col, ("Confidence", False))
        filtered = filtered.sort_values(sk, ascending=sa)

        # ── Styled HTML signal table ─────────────────────────────────────────
        def _badge(s: str) -> str:
            bg = {"BUY": "#1a4731", "SELL": "#4a1010", "HOLD": "#3d3b00"}.get(s, "#333")
            fg = {"BUY": "#4ade80", "SELL": "#f87171", "HOLD": "#facc15"}.get(s, "#ccc")
            return (
                f'<span style="background:{bg};color:{fg};padding:2px 10px;'
                f'border-radius:4px;font-weight:700;font-size:0.85rem">{s}</span>'
            )

        disp = filtered.copy()
        disp["Signal"] = disp["Signal"].apply(_badge)
        disp["Confidence"] = disp["Confidence"].map(lambda x: f"{x:.0%}")
        disp["5d Forecast"] = disp["5d Forecast"].map(lambda x: f"{x*100:+.2f}%")
        disp["20d Forecast"] = disp["20d Forecast"].map(lambda x: f"{x*100:+.2f}%")
        disp["Last Close"] = disp["Last Close"].map(
            lambda x: f"${x:.2f}" if pd.notna(x) else "—"
        )

        html_table = disp[
            ["Ticker", "Signal", "Confidence", "5d Forecast", "20d Forecast",
             "Last Close", "As Of"]
        ].to_html(escape=False, index=False, classes="signal-table")
        st.write(html_table, unsafe_allow_html=True)

        st.caption(f"{len(filtered)} tickers shown · {len(sig_df)} total in universe")

        # ── Drill-down for a single ticker ───────────────────────────────────
        st.divider()
        st.subheader("📋 Single Stock Detail")
        st.caption("For full candlestick charts, forecasts, news & SHAP → go to **🔭 Stock Analysis**")
        sel_ticker = st.selectbox(
            "Select ticker", sorted(latest_signals.keys()), key="ts_drill_ticker"
        )
        if sel_ticker and sel_ticker in latest_signals:
            d = latest_signals[sel_ticker]
            stance = d.get("stance", "HOLD")
            color_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(stance, "⚪")
            st.subheader(f"{color_icon} {sel_ticker} — {stance}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence", f"{d.get('confidence', 0):.0%}")
            c2.metric("5d Forecast", f"{(d.get('forecast_5d') or 0)*100:+.2f}%")
            c3.metric("20d Forecast", f"{(d.get('forecast_20d') or 0)*100:+.2f}%")
            c4.metric("As Of", d.get("as_of", ""))

            if d.get("rationale"):
                with st.expander("Model Rationale", expanded=True):
                    st.markdown(d["rationale"])

            rpt_path = d.get("report_path")
            if rpt_path and Path(rpt_path).exists():
                with st.expander("Full Decision Report (Markdown)", expanded=False):
                    st.markdown(Path(rpt_path).read_text())

# ─────────────────────────────────────────────────────────────────────────────
# Page: Future Predictions  (NEW PAGE)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Future Predictions":
    features = _load_features()
    st.header("📈 Future Predictions")
    st.caption(
        "Forward model forecasts · verify against live prices at each horizon date · "
        "run `ts future-predict` to generate"
    )

    fp_dir = cfg.project_root / "future_predict"
    fp_files = sorted(fp_dir.glob("*/forecast.json"), reverse=True) if fp_dir.exists() else []

    if not fp_files:
        st.info(
            "No future predictions found.  \n"
            "Run `ts future-predict` to generate the next forward forecast."
        )
    else:
        dates_available = [fp.parent.name for fp in fp_files]
        sel_date = st.selectbox("Prediction batch", dates_available, key="fp_date")
        fp_data = json.loads((fp_dir / sel_date / "forecast.json").read_text())

        # ── Metadata row ────────────────────────────────────────────────────
        model_name = fp_data.get("model", "unknown")
        prices_date = fp_data.get("prices_as_of", "")
        budget = fp_data.get("budget", 0)
        all_preds = fp_data.get("all_predictions", [])
        positions = fp_data.get("portfolio", {}).get("positions", [])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction Date", fp_data.get("prediction_date", sel_date))
        col2.metric("Model", fp_data.get("best_variant", model_name))
        col3.metric("Simulation Budget", f"${budget:,.0f}")
        col4.metric("Tickers Ranked", len(all_preds))

        # ── Horizon verification dates ───────────────────────────────────────
        horizons = fp_data.get("horizons", {})
        if horizons:
            horizon_txt = "  ·  ".join(f"**{k}** → {v}" for k, v in horizons.items())
            st.info(
                f"📅 **Verification dates:** {horizon_txt}\n\n"
                "Run `ts backtest` after each date to verify prediction accuracy."
            )

        st.divider()

        # ── Simulated portfolio positions ────────────────────────────────────
        if positions:
            st.subheader("📌 Model-Selected Positions")
            st.caption(
                f"Top-scored tickers with entry prices as of **{prices_date}**.  "
                "This is a **forward simulation** — no real trades were executed."
            )

            # Compare entry vs last-close
            last_close = {
                r["ticker"]: float(r["adj_close"])
                for r in features.filter(pl.col("date") == features["date"].max()).to_dicts()
                if r.get("adj_close")
            }

            pos_rows = []
            for pos in positions:
                t = pos["ticker"]
                entry = pos["entry_price"]
                current = last_close.get(t)
                chg = (current / entry - 1) if current and entry else None
                pos_rows.append({
                    "Ticker": t,
                    "Score": f"{pos['score']:.5f}",
                    "Entry Price": f"${entry:.2f}",
                    "Allocated": f"${pos['allocated']:,.0f}",
                    "Shares": f"{pos['shares']:.4f}",
                    "Last Close": f"${current:.2f}" if current else "—",
                    "Move Since Entry": (
                        f"{chg*100:+.2f}%" if chg is not None else "—"
                    ),
                })

            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

            total_alloc = sum(p["allocated"] for p in positions)
            cash_res = fp_data.get("portfolio", {}).get("cash_reserved", 0)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Deployed Capital", f"${total_alloc:,.0f}")
            sc2.metric("Cash Reserved", f"${cash_res:,.0f}")
            sc3.metric("Positions", len(positions))

        st.divider()

        # ── Full ranked signal table ─────────────────────────────────────────
        if all_preds:
            st.subheader("🏆 Full Ranked Signal Table")
            st.caption(f"{len(all_preds)} tickers ranked by model score · features as of {fp_data.get('features_as_of', '')}")

            pred_rows = []
            for rank, p in enumerate(all_preds, 1):
                pred_rows.append({
                    "Rank": rank,
                    "Ticker": p["ticker"],
                    "Signal": p["stance"],
                    "Score": f"{p['score']:.5f}",
                    "Entry Price": f"${p['entry_price']:.2f}",
                })
            pred_df = pd.DataFrame(pred_rows)

            pf1, pf2 = st.columns([1, 3])
            with pf1:
                stance_opts = sorted(pred_df["Signal"].unique().tolist())
                sel_stances = st.multiselect(
                    "Filter", stance_opts, default=stance_opts, key="fp_stance"
                )
            pred_df_show = pred_df[pred_df["Signal"].isin(sel_stances)]

            def _signal_color(val: str) -> str:
                return {
                    "BUY": "color: #4ade80",
                    "SELL": "color: #f87171",
                    "HOLD": "color: #facc15",
                }.get(val, "")

            st.dataframe(
                pred_df_show.style.applymap(_signal_color, subset=["Signal"]),
                use_container_width=True,
                height=450,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Page: Strategy Backtest
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Strategy Backtest":
    ohlcv = _load_ohlcv()
    features = _load_features()
    st.header("📊 Strategy Backtest")

    # Sidebar widgets define these; provide safe defaults for first-run edge case
    chosen_strategy = locals().get("chosen_strategy", "momentum_rotation")
    top_k = locals().get("top_k", 6)
    rebal = locals().get("rebal", 10)
    commission = locals().get("commission", float(cfg["backtest"]["commission_bps"]))
    slippage = locals().get("slippage", float(cfg["backtest"]["slippage_bps"]))

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
        key="bt_compare_strats",
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
    ohlcv = _load_ohlcv()
    features = _load_features()
    st.header("🔍 Stock Screener")

    latest = features.filter(pl.col("date") == features["date"].max())

    col_filter, col_result = st.columns([1, 3])
    with col_filter:
        st.subheader("Filters")
        min_mom = st.slider("Min 20d Momentum %", -30, 50, 0, key="ss_min_mom") / 100
        max_vol = st.slider("Max Realized Vol % (ann)", 10, 100, 60, key="ss_max_vol") / 100
        max_dd = st.slider("Max Drawdown from 60d High %", -50, 0, -5, key="ss_max_dd") / 100
        rsi_range = st.slider("RSI(14) range", 0, 100, (20, 80), key="ss_rsi_range")
        min_adv = st.number_input("Min Avg $ Volume ($M)", 0, 50000, 100, key="ss_min_adv") * 1e6
        sort_by = st.selectbox("Sort by", ["mom_20d", "mom_60d", "vol_20d", "rsi_14", "dd_from_high_60"], key="ss_sort_by")
        asc = st.checkbox("Ascending", value=False, key="ss_asc")

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
    ticker_sel = st.selectbox("Ticker", sorted(ohlcv["ticker"].unique().to_list()), key="ss_ticker")
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
        selected = st.selectbox("Select report", report_names, key="dr_report")
        sel_path = reports_dir / selected

        tab_report, tab_signals, tab_agent = st.tabs(
            ["📄 Full Report", "📊 Signals", "🤖 Agent Chain"]
        )

        with tab_report:
            st.markdown(sel_path.read_text())

        with tab_signals:
            json_name = selected.replace(".md", ".json")
            json_path = reports_dir / json_name
            if json_path.exists():
                data = json.loads(json_path.read_text())
                stance = data.get("stance", "HOLD")
                color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(stance, "⚪")
                st.subheader(f"{color} {data.get('ticker')} — {stance}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Confidence", f"{data.get('confidence', 0):.0%}")
                c2.metric("5d Forecast", f"{(data.get('forecast_5d') or 0)*100:.2f}%")
                c3.metric("20d Forecast", f"{(data.get('forecast_20d') or 0)*100:.2f}%")
                c4.metric("Score Source", data.get("score_source", "n/a"))
                with st.expander("Raw JSON"):
                    st.json(data)
            else:
                st.info("No JSON file for this report.")

        with tab_agent:
            agent_reports_dir = cfg.path("reports") / "agent"
            ticker_name = selected.split("_")[0]
            agent_files = (
                sorted(agent_reports_dir.glob(f"{ticker_name}_*.json"), reverse=True)
                if agent_reports_dir.exists() else []
            )
            if agent_files:
                agent_data = json.loads(agent_files[0].read_text())
                st.caption(f"Task: {agent_data.get('task', '')}  ·  {agent_files[0].name}")
                steps = agent_data.get("steps", [])
                for i, step in enumerate(steps, 1):
                    thought = step.get("thought", "")
                    action = step.get("action", "")
                    obs = step.get("observation", "")
                    with st.expander(f"Step {i}: {action or 'Thought'}", expanded=(i == 1)):
                        if thought:
                            st.markdown(f"**Thought:** {thought}")
                        if action:
                            st.markdown(f"**Action:** `{action}`  →  `{step.get('action_input', '')}`")
                        if obs:
                            st.text_area("Observation", obs, disabled=True, height=120,
                                         key=f"dr_obs_{i}")
                final = agent_data.get("final_answer", "")
                if final:
                    st.success(f"**Final Answer:** {final}")
            else:
                st.info(
                    f"No agent reports for **{ticker_name}** yet.  "
                    f"Run `ts agent-analyze {ticker_name}` or use the 🤖 Agent Analysis page."
                )

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
# Page: Universe Overview  (Research)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌐 Universe Overview":
    ohlcv = _load_ohlcv()
    features = _load_features()
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
# Page: Stock Analysis  (comprehensive per-ticker deep-dive)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔭 Stock Analysis":
    import math
    ohlcv    = _load_ohlcv()
    features = _load_features()
    decisions = _load_decisions()

    st.header("🔭 Stock Analysis")

    tickers_all = sorted(cfg["universe"]["tickers"])
    col_sel, col_period = st.columns([2, 2])
    with col_sel:
        sel = st.selectbox("Select ticker", tickers_all, key="sa_ticker")
    with col_period:
        period_days = st.select_slider(
            "Chart period",
            options=[30, 60, 90, 180, 365],
            value=90,
            format_func=lambda x: f"{x}d",
            key="sa_period",
        )

    # ── OHLCV slice ─────────────────────────────────────────────────────────
    tk_ohlcv = ohlcv.filter(pl.col("ticker") == sel).sort("date")
    if tk_ohlcv.is_empty():
        st.warning(f"No price data for {sel}. Run `ts ingest`.")
    else:
        # plotly already imported at module top

        tk_pd = tk_ohlcv.to_pandas()
        tk_pd["date"] = pd.to_datetime(tk_pd["date"])
        cutoff = tk_pd["date"].max() - pd.Timedelta(days=period_days)
        chart_df = tk_pd[tk_pd["date"] >= cutoff].copy()

        # ── Candlestick + Volume ────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )

        fig.add_trace(go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"] if "open" in chart_df else chart_df["adj_close"],
            high=chart_df["high"] if "high" in chart_df else chart_df["adj_close"],
            low=chart_df["low"]   if "low"  in chart_df else chart_df["adj_close"],
            close=chart_df["adj_close"],
            name=sel,
            increasing_line_color="#4ade80",
            decreasing_line_color="#f87171",
        ), row=1, col=1)

        # MA overlays
        for ma_days, color in [(20, "#facc15"), (50, "#60a5fa"), (200, "#c084fc")]:
            if len(chart_df) >= ma_days:
                ma = chart_df["adj_close"].rolling(ma_days).mean()
                fig.add_trace(go.Scatter(
                    x=chart_df["date"], y=ma,
                    mode="lines", line=dict(color=color, width=1.2),
                    name=f"MA{ma_days}", opacity=0.8,
                ), row=1, col=1)

        # Volume bars
        if "volume" in chart_df.columns:
            vol_colors = [
                "#4ade80" if r["adj_close"] >= (chart_df["adj_close"].iloc[i - 1] if i > 0 else r["adj_close"])
                else "#f87171"
                for i, r in chart_df.reset_index(drop=True).iterrows()
            ]
            fig.add_trace(go.Bar(
                x=chart_df["date"], y=chart_df["volume"],
                marker_color=vol_colors, name="Volume", opacity=0.6,
            ), row=2, col=1)

        # ── Event markers (news overlaid on price) ──────────────────────────
        ev_rows = _load_events_for(sel)
        if ev_rows:
            ev_df = pd.DataFrame(ev_rows)
            ev_df["d"] = pd.to_datetime(ev_df["d"])
            ev_df = ev_df[ev_df["d"] >= cutoff]
            if not ev_df.empty:
                # snap each event to that day's low for marker placement
                low_by_date = chart_df.set_index("date")["low" if "low" in chart_df else "adj_close"]
                def _yfor(d):
                    try:
                        return float(low_by_date.asof(d)) * 0.985
                    except Exception:
                        return float(chart_df["adj_close"].iloc[-1])
                ev_df["y"] = ev_df["d"].map(_yfor)
                ev_df["color"] = ev_df["sentiment"].map(
                    lambda s: "#4ade80" if (s or 0) > 0.05 else ("#f87171" if (s or 0) < -0.05 else "#facc15")
                )
                fig.add_trace(go.Scatter(
                    x=ev_df["d"], y=ev_df["y"], mode="markers",
                    marker=dict(symbol="triangle-up", size=11, color=ev_df["color"],
                                line=dict(width=0.5, color="#0e1117")),
                    name="News",
                    text=ev_df["summary"].str.slice(0, 110),
                    hovertemplate="%{x|%b %d}<br>%{text}<extra></extra>",
                ), row=1, col=1)

        last_price = float(chart_df["adj_close"].iloc[-1])
        fig.update_layout(
            title=f"{sel} — {period_days}d Price",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=10, r=10, t=50, b=10),
            height=500,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#1f2937")
        st.plotly_chart(fig, use_container_width=True)

        # ── Current Signal ───────────────────────────────────────────────────
        st.divider()
        decision = decisions.get(sel)
        if decision:
            stance = decision.get("stance", "HOLD")
            conf   = decision.get("confidence", 0)
            f5     = (decision.get("forecast_5d") or 0) * 100
            f20    = (decision.get("forecast_20d") or 0) * 100
            col_stance_color = {"BUY": "#4ade80", "SELL": "#f87171", "HOLD": "#facc15"}.get(stance, "#9ca3af")
            col_stance_bg    = {"BUY": "#14532d", "SELL": "#450a0a", "HOLD": "#422006"}.get(stance, "#111827")

            st.markdown(
                f'<div style="background:{col_stance_bg};border-left:4px solid {col_stance_color};'
                f'padding:14px 20px;border-radius:6px;margin-bottom:8px">'
                f'<span style="font-size:1.4rem;font-weight:800;color:{col_stance_color}">{stance}</span>'
                f'&nbsp;&nbsp;<span style="color:#d1d5db;font-size:0.9rem">Confidence: {conf:.0%}'
                f' · As of {decision.get("as_of", "")} · Model: '
                f'{(decision.get("score_source") or "").replace("ensemble:", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Current Price",  f"${last_price:.2f}")
            mc2.metric("5d Forecast",    f"{f5:+.2f}%",  delta=f"${last_price*(1+f5/100):.2f} target")
            mc3.metric("20d Forecast",   f"{f20:+.2f}%", delta=f"${last_price*(1+f20/100):.2f} target")
            mc4.metric("Score Source",   (decision.get("score_source") or "n/a").split(":")[-1])
        else:
            st.info(f"No ML decision yet for **{sel}**. Run `ts analyze {sel}` to generate one.")
            mc1, _, _, _ = st.columns(4)
            mc1.metric("Current Price", f"${last_price:.2f}")

        # ── Calibrated Price Bounds (conformal quantiles → MC fallback) ──────
        st.divider()
        st.subheader("📅 Price Bounds — lower / median / upper")

        feat_row = features.filter(
            (pl.col("ticker") == sel) & (pl.col("date") == features["date"].max())
        )
        bounds = _compute_bounds_cached(sel)
        if bounds and bounds.get("horizons"):
            method = bounds.get("method", "")
            cap = f" · target coverage {bounds['target_coverage']:.0%}" if bounds.get("target_coverage") else ""
            iv = f" · 1m IV {bounds['implied_vol_1m']:.0%}" if bounds.get("implied_vol_1m") else ""
            st.caption(f"Method: `{method}`{cap}{iv} — calibrated, asymmetric (not ±2σ).")

            order = ["5d", "1m", "3m", "6m", "12m"]
            hz = bounds["horizons"]
            labels = [o for o in order if o in hz]
            rows = []
            for label in labels:
                h = hz[label]
                p, r = h["price"], h["return"]
                rows.append({
                    "Horizon": label,
                    "Low": f"${p['lo']:.2f}",
                    "Median": f"${p['median']:.2f}",
                    "High": f"${p['hi']:.2f}",
                    "Low %": f"{r['lo']*100:+.1f}%",
                    "Med %": f"{r['median']*100:+.1f}%",
                    "High %": f"{r['hi']*100:+.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            los = [hz[l]["price"]["lo"] for l in labels]
            meds = [hz[l]["price"]["median"] for l in labels]
            his = [hz[l]["price"]["hi"] for l in labels]
            q25 = [hz[l]["price"].get("q25", hz[l]["price"]["median"]) for l in labels]
            q75 = [hz[l]["price"].get("q75", hz[l]["price"]["median"]) for l in labels]

            fan_fig = go.Figure()
            # outer band (lo..hi)
            fan_fig.add_trace(go.Scatter(
                x=labels + labels[::-1], y=his + los[::-1],
                fill="toself", fillcolor="rgba(96,165,250,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="5–95% band",
            ))
            # inner band (q25..q75)
            fan_fig.add_trace(go.Scatter(
                x=labels + labels[::-1], y=q75 + q25[::-1],
                fill="toself", fillcolor="rgba(96,165,250,0.25)",
                line=dict(color="rgba(0,0,0,0)"), name="25–75% band",
            ))
            fan_fig.add_trace(go.Scatter(
                x=labels, y=meds, mode="lines+markers",
                line=dict(color="#60a5fa", width=2), marker=dict(size=7), name="Median",
            ))
            fan_fig.add_hline(y=last_price, line_dash="dash", line_color="#9ca3af",
                              annotation_text="Current price")
            fan_fig.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="white"),
                margin=dict(l=10, r=10, t=30, b=10), height=320, legend=dict(orientation="h"),
            )
            fan_fig.update_yaxes(showgrid=True, gridcolor="#1f2937")
            st.plotly_chart(fan_fig, use_container_width=True)
        else:
            st.info("No bounds yet. Train them with `ts train-intervals` (or they fall back to a Monte-Carlo fan).")

        # ── Technical Indicators ─────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Technical Indicators")
        if not feat_row.is_empty():
            sig_cols = [
                "mom_5d", "mom_20d", "mom_60d", "rsi_14", "vol_20d",
                "sma_gap_50", "sma_gap_200", "breakout_20",
                "dd_from_high_60", "rel_vol_20", "atr_14",
            ]
            available_sigs = {c: feat_row[c][0] for c in sig_cols if c in feat_row.columns}
            ic1, ic2, ic3 = st.columns(3)
            for i, (k, v) in enumerate(available_sigs.items()):
                col = [ic1, ic2, ic3][i % 3]
                if isinstance(v, float):
                    col.metric(k, f"{v:.2%}" if abs(v) < 5 else f"{v:.4f}")
                else:
                    col.metric(k, str(v))
        else:
            st.caption("No feature data. Run `ts features` first.")

        # ── News & Sentiment ─────────────────────────────────────────────────
        st.divider()
        st.subheader("📰 Latest News & Sentiment")
        with st.spinner(f"Fetching news for {sel}…"):
            news_items = _fetch_yf_news(sel)

        if news_items:
            for item in news_items[:8]:
                title   = item.get("title", "")
                pub_ts  = item.get("providerPublishTime", 0)
                pub_str = pd.Timestamp(pub_ts, unit="s").strftime("%b %d, %Y") if pub_ts else ""
                source  = item.get("publisher", "")
                link    = item.get("link", "#")
                # Crude sentiment
                text_lower = title.lower()
                pos_words = ["surge", "beat", "gain", "rise", "growth", "profit", "record", "strong", "up"]
                neg_words = ["fall", "drop", "miss", "cut", "loss", "risk", "down", "concern", "weak"]
                pos_score = sum(1 for w in pos_words if w in text_lower)
                neg_score = sum(1 for w in neg_words if w in text_lower)
                sentiment = "🟢 Positive" if pos_score > neg_score else ("🔴 Negative" if neg_score > pos_score else "⚪ Neutral")
                st.markdown(
                    f"**[{title}]({link})**  \n"
                    f"{sentiment} · {source} · {pub_str}"
                )
        else:
            st.caption("No recent news available via yfinance.")

        # ── Verdict Panel ────────────────────────────────────────────────────
        st.divider()
        st.subheader("⚖️ Verdict")
        if decision:
            rationale = decision.get("rationale", "")
            if rationale:
                stance = decision.get("stance", "HOLD")
                color_map = {"BUY": "#4ade80", "SELL": "#f87171", "HOLD": "#facc15"}
                verdict_color = color_map.get(stance, "#9ca3af")
                st.markdown(
                    f'<div style="border-left:4px solid {verdict_color};padding:12px 16px;'
                    f'background:#111827;border-radius:4px">{rationale}</div>',
                    unsafe_allow_html=True,
                )
            rpt_path = decision.get("report_path")
            if rpt_path and Path(rpt_path).exists():
                with st.expander("Full Model Report", expanded=False):
                    st.markdown(Path(rpt_path).read_text())
        else:
            st.info(f"Run `ts analyze {sel}` to get a full ML verdict.")

        # ── On-demand SHAP ───────────────────────────────────────────────────
        st.divider()
        st.subheader("🔬 SHAP Feature Explanation")
        model_dir = cfg.path("reports") / "models"
        if model_dir.exists():
            shap_top = st.slider("Top N features", 5, 20, 12, key="sa_shap_n")
            if st.button("Compute SHAP Waterfall", key="sa_shap_btn"):
                with st.spinner("Computing SHAP…"):
                    try:
                        from trading_system.monitoring.shap_viz import (
                            compute_shap_waterfall, render_shap_waterfall_fig,
                        )
                        sd = compute_shap_waterfall(model_dir, features, sel, top_n=shap_top)
                        if sd:
                            st.session_state["sa_shap_data"] = sd
                        else:
                            st.warning("SHAP returned no data (model may not be tree-based or ticker missing).")
                    except Exception as e:
                        st.error(f"SHAP error: {e}")
            if "sa_shap_data" in st.session_state:
                from trading_system.monitoring.shap_viz import render_shap_waterfall_fig
                st.pyplot(
                    render_shap_waterfall_fig(st.session_state["sa_shap_data"]),
                    use_container_width=True,
                )
        else:
            st.info("Train a model first (`ts train`) to enable SHAP explanations.")

# ─────────────────────────────────────────────────────────────────────────────
# Page: Paper Simulation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💼 Paper Simulation":
    ohlcv = _load_ohlcv()
    # Build last-close dict from OHLCV instead of full features
    _paper_last_px_pl = ohlcv.group_by("ticker").agg(pl.last("adj_close")).to_dict(as_series=False)
    _paper_last_close = dict(zip(_paper_last_px_pl["ticker"], _paper_last_px_pl["adj_close"]))
    st.header("💼 Paper Simulation")
    st.caption(
        "Forward paper trading simulation · started fresh with $10,000 · "
        "all trades logged from today for future backtest verification"
    )

    equity_log_path = cfg.path("data_gold") / "paper_equity_log.parquet"
    journal_path = cfg.path("data_gold") / "paper_portfolio_journal.json"

    if not equity_log_path.exists():
        st.info(
            "🚀 **Forward simulation started fresh.**  \n"
            "Run `ts paper-trade` daily to log today's ML signals as virtual trades.  \n"
            "Equity history will appear here after the first daily run.",
            icon="📈",
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

        # ── Metric Glossary ────────────────────────────────────────────────
        with st.expander("📖 What do these metrics mean?", expanded=False):
            st.markdown("""
| Metric | What it means |
|--------|---------------|
| **Equity** | Total portfolio value = cash + all open positions at current prices |
| **CAGR** | *Compound Annual Growth Rate* — your annualised % return assuming reinvestment |
| **Max Drawdown** | Worst peak-to-trough loss ever reached, e.g. −25 % means the portfolio once fell 25 % from its highest point |
| **Trading Days** | Number of market days the simulation covers |
| **Peak Equity** | Highest total value ever recorded |
| **1 m / 3 m / 6 m / 1 y** | Rolling return over that look-back period ending today |
| **Unrealized P&L** | Market value of open positions minus what you paid for them |
""")

        c1, c2, c3, c4, c5 = st.columns(5)
        color = "normal" if total_return >= 0 else "inverse"
        c1.metric("Equity", f"${end_eq:,.0f}", delta=f"{total_return:.2%}")
        c2.metric("CAGR", f"{cagr:.2%}")
        c3.metric("Max Drawdown", f"{max_dd:.2%}")
        c4.metric("Trading Days", str(n_days))
        c5.metric("Peak Equity", f"${peak:,.0f}")

        st.divider()

        # ── Investment Simulator ──────────────────────────────────────────────
        st.subheader("💰 Investment Simulator")
        st.caption(
            "Scales the portfolio's actual % return to any starting capital. "
            "This is a view-only estimate — to re-simulate from scratch use `ts paper-trade --backfill`."
        )
        sim_capital = st.slider(
            "Starting capital",
            min_value=100, max_value=50_000, value=1_000, step=100,
            format="$%d",
            key="sim_capital",
        )
        if start_eq and start_eq > 0:
            sim_end = sim_capital * (end_eq / start_eq)
            sim_gain = sim_end - sim_capital
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Starting Capital", f"${sim_capital:,.0f}")
            sc2.metric("Would Be Worth Today", f"${sim_end:,.2f}",
                       delta=f"${sim_gain:+,.2f}")
            sc3.metric("Total Return", f"{total_return:.2%}")
            scaled_eq = eq_pd["equity"] / start_eq * sim_capital
            st.line_chart(
                pd.DataFrame({"Scaled Portfolio ($)": scaled_eq}),
                use_container_width=True,
            )
        else:
            st.info("No equity history to scale.")

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

            plot_df = eq_pd[["equity"]].rename(columns={"equity": "Paper Simulation"})
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

                # V2: Live prices via yfinance fast_info
                col_hdr, col_btn = st.columns([3, 1])
                col_hdr.caption("Holdings as of last portfolio run")
                refresh_live = col_btn.button("🔄 Refresh Live Prices", key="holdings_refresh")

                price_key = "live_holding_prices"
                if refresh_live or price_key not in st.session_state:
                    if holdings:
                        try:
                            from trading_system.ingestion.realtime import live_price_snapshot
                            live_px = live_price_snapshot(list(holdings.keys()))
                            st.session_state[price_key] = live_px
                        except Exception:
                            st.session_state[price_key] = {}
                    else:
                        st.session_state[price_key] = {}

                live_prices = st.session_state.get(price_key, {})
                # Fall back to last OHLCV close when live unavailable
                latest_prices_fb = _paper_last_close

                if holdings:
                    hold_rows = []
                    total_live_value = 0.0
                    for ticker, qty in sorted(holdings.items()):
                        px = live_prices.get(ticker) or latest_prices_fb.get(ticker, 0.0)
                        is_live = ticker in live_prices
                        val = qty * px
                        total_live_value += val
                        hold_rows.append({
                            "Ticker": ticker,
                            "Qty": f"{qty:.2f}",
                            "Price": f"${px:.2f}" + (" 🔴" if not is_live else " 🟢"),
                            "Value": f"${val:,.0f}",
                        })
                    st.dataframe(pd.DataFrame(hold_rows), use_container_width=True)
                    if live_prices:
                        st.caption(f"Total live value: **${total_live_value:,.0f}**  (🟢 = live price  🔴 = last close)")
                    else:
                        st.caption(f"Total estimated value: **${total_live_value:,.0f}**  (⚪ = reference close, not live)")
                else:
                    st.info("No open positions.")

        # ── Portfolio Management ──────────────────────────────────────────────
        st.divider()
        with st.expander("⚙️ Simulation Management (Reset)", expanded=False):
            st.warning(
                "**Resetting is irreversible.** The equity log and all trade history will be deleted. "
                "The simulation restarts from today with a clean slate."
            )
            reset_capital = st.number_input(
                "New starting capital ($)",
                min_value=100, max_value=50_000,
                value=10_000, step=500,
                key="reset_capital_input",
            )
            if st.button("🔄 Reset Simulation", type="secondary", key="reset_pp_btn"):
                from trading_system.execution.paper_portfolio import PaperPortfolio
                pp = PaperPortfolio(
                    journal_path=journal_path,
                    equity_log_path=equity_log_path,
                    initial_cash=float(reset_capital),
                )
                pp.reset(float(reset_capital))
                st.success(f"Simulation reset with **${reset_capital:,.0f}** starting capital. Reload the page to see the empty log.")
                st.cache_data.clear()

# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Model Comparison":
    features = _load_features()
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
            c3.metric("Best R²", f"{best_row.get('r2_mean', 0):.4f}")

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

    # ── V2: SHAP Deep Dive tab ───────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 SHAP Deep Dive")
    model_dir = cfg.path("reports") / "models"
    if not model_dir.exists():
        st.info("No trained models found. Run `ts train` first.")
    else:
        tickers_available = sorted(cfg["universe"]["tickers"])
        shap_ticker = st.selectbox("Ticker for SHAP analysis", tickers_available, key="shap_ticker_mc")
        shap_top_n = st.slider("Top N features", 5, 20, 12, key="shap_top_n_mc")
        if st.button("Compute SHAP", key="run_shap_mc"):
            with st.spinner("Computing SHAP values…"):
                try:
                    from trading_system.monitoring.shap_viz import compute_shap_waterfall, render_shap_waterfall_fig
                    shap_data = compute_shap_waterfall(model_dir, features, shap_ticker, top_n=shap_top_n)
                    if shap_data:
                        st.session_state["shap_data_mc"] = shap_data
                    else:
                        st.warning("SHAP computation returned no data (model may not be a tree model or ticker missing).")
                except Exception as e:
                    st.error(f"SHAP error: {e}")

        if "shap_data_mc" in st.session_state:
            from trading_system.monitoring.shap_viz import render_shap_waterfall_fig
            fig = render_shap_waterfall_fig(st.session_state["shap_data_mc"])
            st.pyplot(fig, use_container_width=True)

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
                # Get description from default instance, fallback to class-level meta
                try:
                    desc = cls().meta.description
                except Exception:
                    desc = getattr(getattr(cls, "meta", None), "description", "") or ""
                st.markdown(f"**`{name}`** — {desc}")

    st.divider()
    st.subheader("API Keys Status")
    keys = {
        "FRED_API_KEY": os.environ.get("FRED_API_KEY", ""),
        "NEWSAPI_KEY": os.environ.get("NEWSAPI_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "OLLAMA_HOST": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    }
    for k, v in keys.items():
        status = "✅ Set" if v else "❌ Not set"
        st.markdown(f"**{k}**: {status}")

# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Prices (V2)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Live Prices":
    ohlcv_for_live = _load_ohlcv()
    st.header("⚡ Live Prices")
    st.caption("Quasi-realtime price overlay · 5-min refresh via yfinance · paper simulation P&L")

    tickers = cfg["universe"]["tickers"]

    # ── Auto-refresh toggle ──────────────────────────────────────────────────
    col_ctrl, col_status = st.columns([2, 2])
    with col_ctrl:
        auto_refresh = st.checkbox("Auto-refresh every 5 min", value=False, key="lt_autorefresh")
        if st.button("🔄 Fetch Live Prices Now", key="lt_fetch"):
            st.session_state["lt_force_refresh"] = True

    # Fetch prices
    if st.session_state.get("lt_force_refresh") or "lt_prices" not in st.session_state:
        with st.spinner("Fetching live prices…"):
            try:
                from trading_system.ingestion.realtime import live_price_snapshot
                prices = live_price_snapshot(tickers)
                st.session_state["lt_prices"] = prices
                st.session_state["lt_ts"] = pd.Timestamp.now().strftime("%H:%M:%S")
            except Exception as e:
                st.error(f"Price fetch failed: {e}")
                prices = {}
        st.session_state.pop("lt_force_refresh", None)
    else:
        prices = st.session_state.get("lt_prices", {})

    with col_status:
        ts = st.session_state.get("lt_ts", "—")
        st.metric("Last Update", ts)
        st.metric("Tickers Fetched", f"{len(prices)}/{len(tickers)}")

    # ── Kill Switch ──────────────────────────────────────────────────────────
    st.divider()
    col_ks, col_mode = st.columns([1, 3])
    with col_ks:
        kill = st.toggle("🛑 KILL SWITCH (halt paper trades)", value=False, key="kill_switch")
    with col_mode:
        if kill:
            st.error("⚠️ Kill switch ACTIVE — all paper trade execution halted.")
        else:
            st.success("✅ Paper trading enabled.")

    # ── Live P&L ─────────────────────────────────────────────────────────────
    journal_path = cfg.path("data_gold") / "paper_portfolio_journal.json"
    if journal_path.exists() and prices:
        import json as _json
        jdata = _json.loads(journal_path.read_text())
        holdings = {t: q for t, q in jdata.get("holdings", {}).items() if q > 0.001}
        avg_costs = jdata.get("avg_cost", {})
        cash = float(jdata.get("cash", 0))

        if holdings:
            st.subheader("Live P&L — Open Positions")
            rows = []
            total_mkt = 0.0
            total_cost = 0.0
            for t, qty in sorted(holdings.items()):
                px = prices.get(t, 0.0)
                cost = avg_costs.get(t, px)
                mkt_val = qty * px
                cost_val = qty * cost
                unrealized = mkt_val - cost_val
                pct = (unrealized / cost_val) if cost_val else 0.0
                total_mkt += mkt_val
                total_cost += cost_val
                rows.append({
                    "Ticker": t, "Qty": f"{qty:.2f}",
                    "Avg Cost": f"${cost:.2f}", "Live Price": f"${px:.2f}",
                    "Mkt Value": f"${mkt_val:,.0f}",
                    "Unrealized PnL": f"${unrealized:+,.0f}",
                    "Return %": f"{pct:.2%}",
                })

            df_pnl = pd.DataFrame(rows)
            st.dataframe(df_pnl, use_container_width=True)

            total_unrealized = total_mkt - total_cost
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Market Value", f"${total_mkt:,.0f}")
            col2.metric("Total Cost Basis", f"${total_cost:,.0f}")
            col3.metric("Unrealized P&L", f"${total_unrealized:+,.0f}",
                        delta=f"{total_unrealized/total_cost:.2%}" if total_cost else None)
            col4.metric("Cash", f"${cash:,.0f}")

            # Donut chart of position weights
            try:
                weights = {t: qty * prices.get(t, 0.0) for t, qty in holdings.items() if prices.get(t)}
                if weights:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(weights.keys()),
                        values=list(weights.values()),
                        hole=0.55,
                        textinfo="label+percent",
                    )])
                    fig.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="white"),
                        margin=dict(l=20, r=20, t=40, b=20),
                        title="Portfolio Weights (live)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.caption("Install plotly>=5.20 for the portfolio donut chart.")
        else:
            st.info("No open positions in paper portfolio.")
    else:
        st.info("Run `ts paper-trade` to start a paper portfolio, then refresh live prices.")

    # ── Live price table ──────────────────────────────────────────────────────
    if prices:
        st.divider()
        st.subheader("Universe Live Prices")
        # Compute change vs last close from OHLCV (avoid _load_features overhead)
        _lc_pl = ohlcv_for_live.group_by("ticker").agg(pl.last("adj_close")).to_dict(as_series=False)
        last_close = dict(zip(_lc_pl["ticker"], _lc_pl["adj_close"]))
        price_rows = []
        for t, px in sorted(prices.items()):
            lc = last_close.get(t)
            chg = (px / lc - 1) if lc else None
            price_rows.append({
                "Ticker": t,
                "Live Price": f"${px:.2f}",
                "Last Close": f"${lc:.2f}" if lc else "—",
                "Chg %": f"{chg:.2%}" if chg is not None else "—",
            })
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, height=400)

    # Auto-refresh logic — rerun every 5 min
    if auto_refresh:
        import time as _time
        _time.sleep(300)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Page: Agent Analysis (V2)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Agent Analysis":
    features = _load_features()
    st.header("🤖 Agent Analysis")
    st.caption("ReAct multi-step reasoning chain · DeepSeek cloud → Ollama fallback")

    tickers_available = sorted(cfg["universe"]["tickers"])
    col_sel, col_cfg = st.columns([2, 2])
    with col_sel:
        agent_ticker = st.selectbox("Ticker", tickers_available, key="agent_ticker")
        run_btn = st.button("▶ Run Agent Analysis", key="run_agent")
    with col_cfg:
        verbose = st.checkbox("Show full thought chain", value=True, key="aa_verbose")
        save_result = st.checkbox("Save result to reports/agent/", value=True, key="aa_save")

    if run_btn:
        with st.spinner(f"Running ReAct agent for {agent_ticker}…"):
            try:
                from trading_system.ingestion.llm_extractor import LLMRouter, _default_router
                from trading_system.agent import TradingAgentOrchestrator

                router = _default_router()
                orch = TradingAgentOrchestrator(cfg=cfg, llm_router=router)
                result = orch.run_ticker_analysis(agent_ticker)

                if save_result:
                    out_dir = cfg.path("reports") / "agent"
                    orch.save_result(result, output_dir=out_dir)

                st.session_state["agent_result"] = result
                st.session_state["agent_ticker_done"] = agent_ticker
            except Exception as e:
                st.error(f"Agent failed: {e}")

    if "agent_result" in st.session_state:
        result = st.session_state["agent_result"]
        done_ticker = st.session_state.get("agent_ticker_done", "")

        st.subheader(f"Results for {done_ticker}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Success", "✅" if result.success else "❌")
        c2.metric("Steps", len(result.steps))
        c3.metric("Backend", result.backend_used or "n/a")

        if result.final_answer:
            st.success(f"**Final Answer:** {result.final_answer}")

        if verbose and result.steps:
            st.divider()
            st.subheader("Reasoning Chain")
            for i, step in enumerate(result.steps, 1):
                icon = "💭" if not step.action else "⚙️"
                label = f"Step {i}: {icon} {step.action or 'Thought'}"
                with st.expander(label, expanded=(i <= 2)):
                    if step.thought:
                        st.markdown(f"**Thought:** {step.thought}")
                    if step.action:
                        st.markdown(f"**Action:** `{step.action}`")
                        st.code(str(step.action_input), language="json")
                    if step.observation:
                        st.text_area("Observation", step.observation, disabled=True,
                                     height=120, key=f"aa_obs_{i}")

        # SHAP waterfall for the analyzed ticker
        st.divider()
        st.subheader("🔍 SHAP Explanation")
        model_dir = cfg.path("reports") / "models"
        if model_dir.exists():
            shap_top = st.slider("Top N features", 5, 20, 12, key="shap_top_agent")
            if st.button("Compute SHAP Waterfall", key="shap_agent_btn"):
                with st.spinner("Computing SHAP…"):
                    try:
                        from trading_system.monitoring.shap_viz import compute_shap_waterfall, render_shap_waterfall_fig
                        sd = compute_shap_waterfall(model_dir, features, done_ticker, top_n=shap_top)
                        if sd:
                            st.session_state["agent_shap"] = sd
                        else:
                            st.warning("SHAP returned no data.")
                    except Exception as e:
                        st.error(f"SHAP error: {e}")

            if "agent_shap" in st.session_state:
                from trading_system.monitoring.shap_viz import render_shap_waterfall_fig
                sfig = render_shap_waterfall_fig(st.session_state["agent_shap"])
                st.pyplot(sfig, use_container_width=True)
        else:
            st.info("No trained models found. Run `ts train` first.")

    # Saved agent reports browser
    st.divider()
    st.subheader("Saved Agent Reports")
    agent_dir = cfg.path("reports") / "agent"
    if agent_dir.exists():
        saved = sorted(agent_dir.glob("*.json"), reverse=True)
        if saved:
            names = [f.name for f in saved]
            selected_ar = st.selectbox("Select saved report", names, key="agent_report_sel")
            if selected_ar:
                ar_data = json.loads((agent_dir / selected_ar).read_text())
                with st.expander("Raw report JSON"):
                    st.json(ar_data)
                steps = ar_data.get("steps", [])
                for i, step in enumerate(steps, 1):
                    with st.expander(f"Step {i}: {step.get('action', 'Thought')}"):
                        if step.get("thought"):
                            st.markdown(f"**Thought:** {step['thought']}")
                        if step.get("action"):
                            st.markdown(f"**Action:** `{step['action']}` → `{step.get('action_input', '')}`")
                        if step.get("observation"):
                            st.text_area("Observation", step["observation"], disabled=True,
                                         height=120, key=f"ar_obs_{i}")
        else:
            st.info("No saved agent reports yet. Run an analysis above.")
    else:
        st.info("No agent reports directory yet.")

