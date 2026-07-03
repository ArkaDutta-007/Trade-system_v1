"""💰 Invest Planner + 🏦 Portfolio Tabs + 🔬 Moonshot Discovery + 📒 Ledger.

Invest Planner: enter a budget → a gated, sized, hold-horizon-annotated buy
plan (the `ts invest` engine, interactive), with an optional capped moonshot
sleeve and one-click booking into a named virtual portfolio. Portfolio Tabs:
money-level scoreboards of those booked decisions, marked to market against a
SPY counterfactual. Moonshot Discovery: the speculative funnel (EDGAR
spinoffs/IPOs + in-universe sleepers, asymmetry-scored). Calibration Ledger:
the system's scored track record — every plan position is a falsifiable
prediction, and this page shows how they actually resolved.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False, ttl=900)
def _build_plan(budget: float, top_n: int, max_weight: float,
                min_position: float, use_flags: bool, record: bool,
                moonshot_frac: float = 0.0):
    from trading_system.config import get_config
    from trading_system.decision.invest import build_invest_plan
    return build_invest_plan(
        get_config(), budget=budget, top_n=top_n, max_weight=max_weight,
        min_position=min_position, use_flags=use_flags, record=record,
        moonshot_frac=moonshot_frac,
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
    c5, c6 = st.columns([2, 2])
    portfolio = c5.text_input("Book into portfolio (optional)", key="inv_portfolio",
                              placeholder="p1")
    moonshot_frac = c6.slider(
        "Moonshot sleeve", 0.0, 0.30, 0.0, 0.05,
        help="speculative discovery sleeve (spinoffs/IPOs/sleepers), "
             "capped and separately ledgered")
    record = st.toggle("Log plan to decision ledger", value=True,
                       help="Each position becomes a scored, falsifiable prediction")

    if not st.button("Build plan", type="primary"):
        st.info("Set a budget and press **Build plan**. First run may take a minute "
                "(loads features + models).")
        return

    try:
        with st.spinner("Scoring horizons, computing bounds, gating, sizing…"):
            plan = _build_plan(budget, int(top_n), float(max_w), 50.0, use_flags,
                               record, float(moonshot_frac))
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

        warn = [(s["ticker"], w) for s in plan["positions"] for w in s.get("warnings", [])]
        for tk, w in warn:
            st.warning(f"{tk}: {w}")

    # ── Moonshot sleeve (speculative, capped, separately ledgered) ───────────
    moonshot = plan.get("moonshot") or []
    if moonshot:
        st.subheader("Moonshot sleeve (speculative)")
        mdf = pd.DataFrame([{
            "Ticker": m["ticker"], "Category": m["category"],
            "Dollars": m["dollars"], "Shares": m["shares"], "Entry": m["entry"],
            "Median target": m["median_target"], "Stretch": m["stretch_target"],
            "Stop": m["stop"], "Score": m["conviction"],
        } for m in moonshot])
        st.dataframe(
            mdf, hide_index=True, use_container_width=True,
            column_config={
                "Dollars": st.column_config.NumberColumn(format="$%.0f"),
                "Shares": st.column_config.NumberColumn(format="%.3f"),
                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                "Median target": st.column_config.NumberColumn(format="$%.2f"),
                "Stretch": st.column_config.NumberColumn(format="$%.2f"),
                "Stop": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        # every sleeve position carries the same honesty caveat — show it once
        seen_warns: set[str] = set()
        for m in moonshot:
            for w in m.get("warnings", []):
                if w not in seen_warns:
                    seen_warns.add(w)
                    st.warning(w)

    # ── Allocation pies: core + moonshot slices (+ cash) ─────────────────────
    alloc = [{"Ticker": s["ticker"], "Dollars": s["dollars"]}
             for s in plan["positions"]]
    alloc += [{"Ticker": f"{m['ticker']} (moonshot)", "Dollars": m["dollars"]}
              for m in moonshot]
    if alloc:
        try:
            import plotly.express as px_
            adf = pd.DataFrame(alloc)
            fig = px_.pie(adf, values="Dollars", names="Ticker", hole=0.45,
                          title="Tranche allocation")
            cash = plan["cash_reserve"]
            if cash > 1:
                fig2 = px_.pie(
                    pd.concat([adf,
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

    if plan.get("ledger_recorded"):
        st.success(f"Logged {plan['ledger_recorded']} predictions to the decision "
                   "ledger — see 📒 Calibration Ledger.")

    # ── Optional booking into a named virtual portfolio ──────────────────────
    # plan_id (= the plan's generated_at) makes booking idempotent: re-pressing
    # Build with an unchanged cached plan skips fills already in the book
    # instead of doubling it.
    portfolio_name = (portfolio or "").strip()
    if portfolio_name and (plan["positions"] or moonshot):
        try:
            from trading_system.execution.tranches import book_fills, load_book
            n_before = len(load_book(cfg, portfolio_name).get("fills", []))
            if plan["positions"]:
                book_fills(cfg, portfolio_name, plan["positions"],
                           as_of=plan["as_of"],
                           spy_close=plan.get("benchmark_close"),
                           source="invest",
                           cash_reserve=float(plan.get("cash_reserve") or 0.0),
                           plan_id=plan.get("generated_at"))
            if moonshot:
                book_fills(cfg, portfolio_name, moonshot,
                           as_of=plan["as_of"],
                           spy_close=plan.get("benchmark_close"),
                           source="moonshot", cash_reserve=0.0,
                           plan_id=plan.get("generated_at"))
            n_new = len(load_book(cfg, portfolio_name).get("fills", [])) - n_before
            if n_new:
                st.success(f"Booked {n_new} fills into portfolio {portfolio_name} — "
                           "see 🏦 Portfolio Tabs.")
            else:
                st.info(f"Plan already booked into {portfolio_name} — no new fills.")
        except Exception as e:
            st.error(f"Booking into portfolio failed: {e}")

    if plan.get("skipped"):
        with st.expander(f"Not bought — {len(plan['skipped'])} names (transparency)"):
            st.dataframe(pd.DataFrame(plan["skipped"]), hide_index=True,
                         use_container_width=True)
    st.caption(plan["note"])


def render_tabs(cfg) -> None:
    from trading_system.execution.tranches import mark_all

    st.header("🏦 Portfolio Tabs")
    st.caption(
        "Named virtual portfolios — money-level scoreboards of `ts invest` "
        "decisions. Every booked fill also records a SPY counterfactual (the "
        "benchmark shares the same dollars would have bought at the same "
        "close), so each tab shows alpha vs 'you could have just bought SPY', "
        "not raw P&L alone. Paper accounting at plan entry closes: it measures "
        "decision quality, not execution quality."
    )

    try:
        with st.spinner("Marking portfolios to market…"):
            marked = [s for s in mark_all(cfg, fetch_missing=True)
                      if s.get("n_fills")]
    except Exception as e:
        st.error(f"Mark-to-market failed: {e}")
        return
    errored = [s for s in marked if s.get("error")]
    summaries = [s for s in marked if not s.get("error")]
    for s in errored:
        st.warning(f"{s['name']}: {s['error']}")
    if not summaries:
        if not errored:
            st.info("No portfolios yet — build a plan in 💰 Invest Planner with "
                    "'Book into portfolio' filled in, or run "
                    "`ts invest 1000 --portfolio p1`.")
        return

    total_cost = sum(s.get("cost") or 0.0 for s in summaries)
    total_value = sum(s.get("value") or 0.0 for s in summaries)
    total_pnl = sum(s.get("pnl") or 0.0 for s in summaries)
    total_alpha = sum(s.get("alpha_vs_spy") or 0.0 for s in summaries)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cost", f"${total_cost:,.2f}")
    m2.metric("Total value", f"${total_value:,.2f}")
    m3.metric("Total P&L", f"${total_pnl:,.2f}",
              f"{total_pnl / total_cost:+.2%}" if total_cost else None)
    m4.metric("Total alpha vs SPY", f"${total_alpha:,.2f}",
              help="Value minus what the same dollars in SPY would be worth")

    try:
        import plotly.graph_objects as go_
        names = [s["name"] for s in summaries]
        fig = go_.Figure(data=[
            go_.Bar(name="Portfolio", x=names,
                    y=[s.get("pnl_pct") for s in summaries]),
            go_.Bar(name="SPY counterfactual", x=names,
                    y=[s.get("spy_pnl_pct") for s in summaries]),
        ])
        fig.update_layout(barmode="group", title="P&L % vs SPY counterfactual",
                          yaxis_tickformat="+.1%", yaxis_title="return")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    for s in summaries:
        pnl_pct = s.get("pnl_pct")
        pct_txt = f" ({pnl_pct:+.2%})" if pnl_pct is not None else ""
        with st.expander(f"{s['name']} — ${s.get('value', 0):,.2f}{pct_txt}"):
            alpha = s.get("alpha_vs_spy")
            st.caption(
                f"cost ${s.get('cost', 0):,.2f} · value ${s.get('value', 0):,.2f} · "
                f"P&L ${s.get('pnl', 0):,.2f} · alpha vs SPY "
                f"{f'${alpha:,.2f}' if alpha is not None else 'n/a (no SPY data)'} · "
                f"as of {s.get('as_of')} · {s['n_fills']} fills"
            )
            pos = s.get("positions") or []
            if pos:
                pdf = pd.DataFrame(pos)
                pdf["Flags"] = ["STOP" if p.get("hit_stop")
                                else "TGT" if p.get("hit_target") else ""
                                for p in pos]
                show = [c for c in ["ticker", "shares", "avg_cost", "last_price",
                                    "cost", "value", "pnl", "pnl_pct", "hold_days",
                                    "stop", "median_target", "first_fill", "Flags"]
                        if c in pdf.columns]
                st.dataframe(
                    pdf[show], hide_index=True, use_container_width=True,
                    column_config={
                        "avg_cost": st.column_config.NumberColumn(format="$%.2f"),
                        "last_price": st.column_config.NumberColumn(format="$%.2f"),
                        "cost": st.column_config.NumberColumn(format="$%.2f"),
                        "value": st.column_config.NumberColumn(format="$%.2f"),
                        "pnl": st.column_config.NumberColumn(format="$%.2f"),
                        "stop": st.column_config.NumberColumn(format="$%.2f"),
                        "median_target": st.column_config.NumberColumn(format="$%.2f"),
                        "pnl_pct": st.column_config.NumberColumn(format="percent"),
                        "Flags": st.column_config.TextColumn(
                            help="STOP = at/below plan stop · TGT = at/above median target"),
                    },
                )
            if s.get("unpriced"):
                st.warning(f"{s['name']}: no price history yet for "
                           f"{', '.join(s['unpriced'])} — their "
                           f"${s.get('unpriced_cost', 0):,.2f} cost is held OUT "
                           "of P&L (unknown ≠ loss) until prices are available.")


@st.cache_data(show_spinner=False, ttl=900)
def _run_discovery(top_n: int, lookback_days: int, record: bool):
    from trading_system.config import get_config
    from trading_system.decision.discover import build_discovery
    return build_discovery(get_config(), top_n=top_n,
                           lookback_days=lookback_days, record=record)


def render_discovery(cfg) -> None:
    st.header("🔬 Moonshot Discovery")
    st.caption(
        "The speculative funnel: EDGAR structural filings (spinoffs, new "
        "listings, IPOs — names the market hasn't finished pricing) plus "
        "in-universe 'sleepers' (bottom-decile dollar volume with attention "
        "igniting), each ranked by the asymmetry of its bootstrapped 12-month "
        "band. Every score component is shown — no black-box picks — and this "
        "sleeve is judged by its own ledger (source='moonshot'), not stories."
    )

    c1, c2, c3 = st.columns([1, 2, 1])
    top = c1.number_input("Top picks", min_value=1, max_value=50, value=12)
    lookback = c2.slider("EDGAR lookback (days)", 30, 365, 120)
    record = c3.toggle("Record to ledger", value=True,
                       help="Log picks as 12-month predictions under "
                            "source='moonshot' so calibration is scored")

    if not st.button("Run discovery", type="primary"):
        st.info("Press **Run discovery** — queries EDGAR full-text search and "
                "screens the gold panel for sleepers (needs network for EDGAR).")
        return

    try:
        with st.spinner("Searching EDGAR, screening sleepers, bootstrapping bands…"):
            plan = _run_discovery(int(top), int(lookback), bool(record))
    except Exception as e:
        st.error(f"Discovery failed: {e}")
        st.info("EDGAR full-text search needs network access; sleeper screening "
                "needs `ts features` output. Try again or check connectivity.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("EDGAR fresh filings", plan.get("n_fresh", 0))
    m2.metric("Sleepers screened", plan.get("n_sleepers", 0))
    m3.metric("As of", plan.get("as_of", "—"))

    picks = plan.get("picks") or []
    if not picks:
        st.warning("No band-able candidates today — check the watchlist below.")
    else:
        rows = []
        for p in picks:
            a = p.get("asym") or {}
            rows.append({
                "Ticker": p["ticker"], "Category": p["category"],
                "Score": p.get("moonshot_score"), "Price": p.get("last_price"),
                "12m lo": a.get("lo"), "12m median": a.get("median"),
                "12m hi": a.get("hi"), "Asym": a.get("asym"),
                "Filed": p.get("filed", ""),
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "12m lo": st.column_config.NumberColumn(format="percent"),
                "12m median": st.column_config.NumberColumn(format="percent"),
                "12m hi": st.column_config.NumberColumn(format="percent"),
                "Asym": st.column_config.NumberColumn(
                    format="%.2f", help="upside(95th) / |downside(5th)| of the "
                                        "bootstrapped 12m terminal band"),
            },
        )
        if plan.get("ledger_recorded"):
            st.success(f"Recorded {plan['ledger_recorded']} moonshot predictions "
                       "to the ledger — see 📒 Calibration Ledger.")

    if plan.get("watchlist"):
        with st.expander(f"Watchlist — {len(plan['watchlist'])} names not yet band-able"):
            st.dataframe(pd.DataFrame(plan["watchlist"]), hide_index=True,
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
