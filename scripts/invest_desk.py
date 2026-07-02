"""💰 Invest Planner + 📒 Calibration Ledger pages for the Streamlit dashboard.

Invest Planner: enter a budget → a gated, sized, hold-horizon-annotated buy
plan (the `ts invest` engine, interactive). Calibration Ledger: the system's
scored track record — every plan position is a falsifiable prediction, and
this page shows how they actually resolved.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False, ttl=900)
def _build_plan(budget: float, top_n: int, max_weight: float,
                min_position: float, use_flags: bool, record: bool):
    from trading_system.config import get_config
    from trading_system.decision.invest import build_invest_plan
    return build_invest_plan(
        get_config(), budget=budget, top_n=top_n, max_weight=max_weight,
        min_position=min_position, use_flags=use_flags, record=record,
    )


def render_invest(cfg) -> None:
    st.header("💰 Invest Planner")
    st.caption(
        "Give it a budget → what to buy, how many shares, and how long to hold. "
        "Conviction blends every committed forecaster horizon (ICIR-weighted); "
        "hold = the calibrated band's best annualized reward-to-downside; every "
        "BUY clears playbook compliance and the composite flag board; sizing is "
        "RMT-cleaned HRP × Kelly. Each position is logged to the decision ledger."
    )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    budget = c1.number_input("Budget ($)", min_value=100.0, value=1000.0, step=100.0)
    top_n = c2.number_input("Max names", min_value=1, max_value=20, value=8)
    max_w = c3.slider("Per-name cap", 0.10, 0.50, 0.25, 0.05)
    use_flags = c4.toggle("Flag gating", value=True,
                          help="Scale deployment by the composite O/F/I/S/C board")
    record = st.toggle("Log plan to decision ledger", value=True,
                       help="Each position becomes a scored, falsifiable prediction")

    if not st.button("Build plan", type="primary"):
        st.info("Set a budget and press **Build plan**. First run may take a minute "
                "(loads features + models).")
        return

    try:
        with st.spinner("Scoring horizons, computing bounds, gating, sizing…"):
            plan = _build_plan(budget, int(top_n), float(max_w), 50.0, use_flags, record)
    except Exception as e:
        st.error(f"Plan failed: {e}")
        st.info("Make sure `ts features`, `ts train-forecast` and `ts train-intervals` "
                "have been run (models_store/ needs committed models).")
        return

    comp = plan.get("composite") or {}
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Budget", f"${plan['budget']:,.0f}")
    m2.metric("Composite", comp.get("color", "n/a"),
              help=comp.get("rationale", ""))
    m3.metric("Deployable", f"${plan['deployable']:,.0f}",
              f"{plan['deployment_fraction']:.0%} of budget")
    m4.metric("Invested", f"${plan['invested']:,.2f}")
    m5.metric("Cash reserve", f"${plan['cash_reserve']:,.2f}")

    if not plan["positions"]:
        st.warning("Nothing clears the gates today — hold cash.")
    else:
        rows = []
        for s in plan["positions"]:
            rows.append({
                "Ticker": s["ticker"], "Dollars": s["dollars"], "Shares": s["shares"],
                "Weight": s["weight"], "Entry": s["entry"],
                "Median target": s["median_target"], "Stretch": s["stretch_target"],
                "Stop": s["stop"], "Hold": s["hold"],
                "Ann. edge": s["annualized_edge"], "R/Downside": s["reward_downside"],
                "Model": f"{s.get('model')} ({s.get('hold_days')}d"
                         f"{', leak-gate FAIL' if s.get('leak_pass') is False else ''})",
                "Timing": s["timing"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={
                "Dollars": st.column_config.NumberColumn(format="$%.0f"),
                "Shares": st.column_config.NumberColumn(format="%.3f"),
                "Weight": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=0.5),
                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                "Median target": st.column_config.NumberColumn(format="$%.2f"),
                "Stretch": st.column_config.NumberColumn(format="$%.2f"),
                "Stop": st.column_config.NumberColumn(format="$%.2f"),
                "Ann. edge": st.column_config.NumberColumn(format="percent"),
            },
        )

        try:
            import plotly.express as px_
            fig = px_.pie(df, values="Dollars", names="Ticker", hole=0.45,
                          title="Tranche allocation")
            cash = plan["cash_reserve"]
            if cash > 1:
                fig2 = px_.pie(
                    pd.concat([df[["Ticker", "Dollars"]],
                               pd.DataFrame([{"Ticker": "CASH", "Dollars": cash}])]),
                    values="Dollars", names="Ticker", hole=0.45,
                    title="Including cash reserve")
                a, b = st.columns(2)
                a.plotly_chart(fig, use_container_width=True)
                b.plotly_chart(fig2, use_container_width=True)
            else:
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        warn = [(s["ticker"], w) for s in plan["positions"] for w in s.get("warnings", [])]
        for tk, w in warn:
            st.warning(f"{tk}: {w}")
        if plan.get("ledger_recorded"):
            st.success(f"Logged {plan['ledger_recorded']} predictions to the decision "
                       "ledger — see 📒 Calibration Ledger.")

    if plan.get("skipped"):
        with st.expander(f"Not bought — {len(plan['skipped'])} names (transparency)"):
            st.dataframe(pd.DataFrame(plan["skipped"]), hide_index=True,
                         use_container_width=True)
    st.caption(plan["note"])


def render_ledger(cfg) -> None:
    from trading_system.monitoring.ledger import (
        calibration_report, load_ledger, resolve_ledger,
    )

    st.header("📒 Calibration Ledger")
    st.caption(
        "The system's scored track record. Every invest-plan position is a "
        "falsifiable prediction; once its hold horizon elapses it is scored "
        "against realised prices — hit rate, band coverage (promised ~90%), and "
        "whether conviction actually ranked outcomes. Trust is earned here."
    )

    if st.button("Resolve matured predictions now"):
        counts = resolve_ledger(cfg)
        st.success(f"Resolved {counts['resolved']} · pending {counts['pending']} "
                   f"· total {counts['total']}")

    rep = calibration_report(cfg)
    if rep["n_predictions"] == 0:
        st.info("Ledger is empty — build a plan in 💰 Invest Planner (or run "
                "`ts invest <budget>`) to start the track record.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions", rep["n_predictions"])
    c2.metric("Resolved", rep["n_resolved"])
    c3.metric("Open", rep["n_pending"])

    if rep["groups"]:
        st.subheader("Calibration by horizon")
        gdf = pd.DataFrame(rep["groups"]).rename(columns={
            "source": "Source", "horizon_days": "Horizon (d)", "n": "N",
            "hit_rate": "Hit rate", "band_coverage": "Band coverage",
            "avg_forecast": "Avg forecast", "avg_realized": "Avg realized",
            "conviction_ic": "Conviction IC",
        })
        st.dataframe(
            gdf, hide_index=True, use_container_width=True,
            column_config={
                "Hit rate": st.column_config.NumberColumn(format="percent"),
                "Band coverage": st.column_config.NumberColumn(
                    format="percent", help="Target ≈90% — the conformal promise"),
                "Avg forecast": st.column_config.NumberColumn(format="percent"),
                "Avg realized": st.column_config.NumberColumn(format="percent"),
            },
        )

    df = load_ledger(cfg)
    if df.is_empty():
        return
    pdf = df.to_pandas()
    resolved = pdf[pdf["realized_return"].notna()].copy() \
        if "realized_return" in pdf.columns else pd.DataFrame()

    if not resolved.empty:
        st.subheader("Forecast vs realized")
        try:
            import plotly.express as px_
            resolved["outcome"] = resolved["hit"].map({True: "hit", False: "miss"})
            fig = px_.scatter(
                resolved, x="forecast_return", y="realized_return",
                color="outcome", hover_name="ticker",
                color_discrete_map={"hit": "#2ca02c", "miss": "#d62728"},
                labels={"forecast_return": "forecast (median band)",
                        "realized_return": "realized"},
            )
            lim = float(max(resolved[["forecast_return", "realized_return"]]
                            .abs().max().max(), 0.05))
            fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                          line=dict(dash="dot", color="gray"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.subheader("All predictions")
    show = [c for c in ["created_at", "as_of", "ticker", "horizon_days", "entry_price",
                        "band_lo", "band_median", "band_hi", "conviction", "dollars",
                        "matured_on", "terminal_price", "realized_return", "hit",
                        "in_band", "model", "composite"] if c in pdf.columns]
    st.dataframe(pdf[show].sort_values("created_at", ascending=False),
                 hide_index=True, use_container_width=True)
