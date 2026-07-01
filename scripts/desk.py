"""Live Trading Desk + Playbook pages for the Streamlit dashboard.

Kept separate from dashboard.py so the V3 flag/playbook engine gets a clean,
self-contained home. Two entry points, both called from dashboard.py:

  render_trading_desk(cfg)  — auto-refreshing flag board, live price strip, and
                              standing-rule alerts evaluated at live prices.
  render_playbook(cfg)      — cycle rules, pre-trade compliance check, and the
                              trade blotter.

The "live" feel comes from st.fragment(run_every=...): only the live block
reruns on the chosen cadence, so prices and flags tick without recomputing the
whole page or blocking interaction.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

_FLAG_BG = {"GREEN": "#14532d", "YELLOW": "#422006", "RED": "#450a0a", "UNKNOWN": "#1f2937"}
_FLAG_FG = {"GREEN": "#4ade80", "YELLOW": "#facc15", "RED": "#f87171", "UNKNOWN": "#9ca3af"}
_STATUS_COLOR = {"TRIGGERED": "#f87171", "NEAR": "#facc15", "AWAITING": "#60a5fa",
                 "OK": "#4ade80", "NO_DATA": "#9ca3af"}


# ── cached data helpers (short TTLs so the fragment stays cheap) ──────────────

@st.cache_data(ttl=10, show_spinner=False)
def _live_prices(tickers: tuple[str, ...]) -> dict:
    from trading_system.ingestion.realtime import live_price_snapshot
    return live_price_snapshot(list(tickers))


@st.cache_resource(show_spinner=False)
def _playbook_portfolio(_cfg):
    from trading_system.playbook import load_playbook, load_portfolio
    return load_playbook(_cfg), load_portfolio(_cfg)


def _snapshot(cfg, refresh: bool = False):
    from trading_system.flags import get_flag_snapshot
    return get_flag_snapshot(cfg, refresh=refresh, max_age_minutes=2.0)


# ── flag board ───────────────────────────────────────────────────────────────

def _render_flag_board(snap) -> None:
    from trading_system.flags import FLAG_ORDER

    comp = snap.composite
    cols = st.columns(5)
    for i, f in enumerate(FLAG_ORDER):
        r = snap.readings[f]
        color = r.color.value
        bg, fg = _FLAG_BG.get(color, "#1f2937"), _FLAG_FG.get(color, "#9ca3af")
        val = "" if r.value is None else (f"{r.value:,}" if isinstance(r.value, (int, float)) else str(r.value))
        stale = " ⚠" if r.stale else ""
        src = "📡" if r.source in ("live", "auto+override") else ("🗄" if r.source == "cache" else "✎")
        cols[i].markdown(
            f'<div style="background:{bg};border-radius:10px;padding:12px 10px;text-align:center">'
            f'<div style="font-size:0.75rem;color:#9ca3af">{f} · {r.name}</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:{fg};line-height:1.3">{color}{stale}</div>'
            f'<div style="font-size:0.95rem;color:#e5e7eb">{val}</div>'
            f'<div style="font-size:0.65rem;color:#6b7280">{src} {r.source}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    cc = _FLAG_FG.get(comp.color.value, "#9ca3af")
    cbg = _FLAG_BG.get(comp.color.value, "#1f2937")
    freeze = ('&nbsp;&nbsp;<span style="background:#450a0a;color:#f87171;padding:2px 10px;'
              'border-radius:6px;font-weight:700">SEMI FREEZE</span>') if comp.semi_freeze else ""
    st.markdown(
        f'<div style="background:{cbg};border-left:5px solid {cc};border-radius:8px;'
        f'padding:12px 18px;margin-top:12px">'
        f'<span style="font-size:1.2rem;font-weight:800;color:{cc}">COMPOSITE: {comp.color.value}</span>'
        f'&nbsp;&nbsp;<span style="color:#d1d5db">deploy <b>{comp.deployment_fraction:.0%}</b> of tranche '
        f'· {comp.n_green}G / {comp.n_yellow}Y / {comp.n_red}R</span>{freeze}'
        f'<div style="color:#9ca3af;font-size:0.85rem;margin-top:4px">{comp.rationale}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for w in comp.data_warnings:
        st.caption(f"⚠ {w}")


def _render_live_prices(playbook, portfolio, prices: dict) -> None:
    held = portfolio.held_symbols
    if not held:
        return
    st.markdown("##### Live holdings")
    cols = st.columns(6)
    for i, t in enumerate(held):
        pos = portfolio.position(t)
        live = prices.get(t)
        ref = pos.last_price if pos else None
        if live is None:
            live = ref
        delta = ((live / ref - 1) * 100) if (live and ref) else None
        is_live = t in prices
        cols[i % 6].metric(
            f"{t}{' 🟢' if is_live else ' ⚪'}",
            f"${live:,.2f}" if live else "—",
            f"{delta:+.2f}%" if delta is not None else None,
        )


def _render_standing_alerts(playbook, portfolio, prices: dict) -> None:
    from trading_system.playbook import evaluate_standing_rules

    checks = evaluate_standing_rules(playbook, portfolio, prices)
    hot = [c for c in checks if c.status in ("TRIGGERED", "NEAR")]
    st.markdown("##### Standing-rule alerts (live)")
    if not hot:
        st.success("No standing rules triggered or near at current prices.")
    for c in hot:
        col = _STATUS_COLOR.get(c.status, "#9ca3af")
        px = f"${c.price:,.2f}" if c.price else "—"
        st.markdown(
            f'<div style="border-left:4px solid {col};background:#111827;padding:8px 14px;'
            f'border-radius:4px;margin-bottom:6px">'
            f'<b style="color:{col}">{c.status}</b> · <b>{c.ticker}</b> {px} → '
            f'<b>{c.action}</b><br><span style="color:#9ca3af;font-size:0.85rem">{c.detail}</span></div>',
            unsafe_allow_html=True,
        )


# ── PAGE: Trading Desk ───────────────────────────────────────────────────────

def render_trading_desk(cfg) -> None:
    st.header("🚦 Trading Desk")
    st.caption("Live O/F/I/S/C flag board · standing-rule alerts at live prices · v2 playbook")

    try:
        playbook, portfolio = _playbook_portfolio(cfg)
    except Exception as e:
        st.error(f"Could not load playbook/portfolio: {e}")
        st.info("Ensure `portfolio and watchlist.json` and `configs/playbook_v2.yaml` exist.")
        return

    ctrl1, ctrl2 = st.columns([1.2, 1])
    with ctrl1:
        interval = st.selectbox("Auto-refresh", ["Off", "5s", "10s", "15s", "30s", "60s"],
                                index=3, key="desk_interval")
    with ctrl2:
        if st.button("🔄 Refresh flags now", key="desk_force"):
            st.session_state["desk_force_flags"] = True
    run_every = None if interval == "Off" else interval

    held = tuple(portfolio.held_symbols)

    @st.fragment(run_every=run_every)
    def live_block():
        force = st.session_state.pop("desk_force_flags", False)
        try:
            snap = _snapshot(cfg, refresh=force)
        except Exception as e:
            st.error(f"Flag snapshot failed: {e}")
            return
        prices = {}
        try:
            prices = _live_prices(held)
        except Exception:
            pass

        st.caption(f"flags as of {snap.as_of[11:19]}Z · {len(prices)}/{len(held)} live prices "
                   f"· refreshed {pd.Timestamp.now().strftime('%H:%M:%S')}")

        _render_flag_board(snap)
        st.divider()
        _render_live_prices(playbook, portfolio, prices)
        st.divider()
        _render_standing_alerts(playbook, portfolio, prices)

    live_block()

    # ── static context below the live block ──────────────────────────────────
    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("##### Concentration vs caps (rule 3.5)")
        snap_prices = {}
        try:
            snap_prices = _live_prices(held)
        except Exception:
            pass
        weights = portfolio.weights(snap_prices)
        cap_rows = []
        for t, w in sorted(weights.items(), key=lambda kv: -kv[1])[:12]:
            cap = playbook.cap_for(t)
            state = "🚫 BARRED" if w >= cap else ("⚠ near" if w >= cap - 1.5 else "ok")
            cap_rows.append({"Ticker": t, "Weight": f"{w:.1f}%", "Cap": f"{cap:.0f}%", "New cash": state})
        st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True, height=320)

    with right:
        st.markdown("##### Catalysts (next 21 days)")
        from datetime import date, timedelta
        today = date.today()
        horizon = today + timedelta(days=21)
        upcoming = [c for c in playbook.catalysts
                    if today <= date.fromisoformat(str(c["date"])) <= horizon]
        if upcoming:
            st.dataframe(
                pd.DataFrame([{"Date": c["date"], "Event": c["event"][:60],
                               "Flags": ", ".join(c.get("flags", [])) or "—"} for c in upcoming]),
                use_container_width=True, hide_index=True, height=320,
            )
        else:
            st.caption("No catalysts in the next 21 days.")


# ── PAGE: Playbook ───────────────────────────────────────────────────────────

def render_playbook(cfg) -> None:
    st.header("🧭 Playbook")
    st.caption("Cycle rules (§4) · pre-trade compliance · blotter")

    try:
        playbook, portfolio = _playbook_portfolio(cfg)
        snap = _snapshot(cfg)
    except Exception as e:
        st.error(f"Could not load playbook: {e}")
        return

    tab_cycles, tab_check, tab_blotter = st.tabs(
        ["📋 Cycle Rules", "✅ Pre-Trade Check", "📒 Blotter"]
    )

    held = tuple(portfolio.held_symbols)
    try:
        prices = _live_prices(held)
    except Exception:
        prices = {}

    with tab_cycles:
        from trading_system.flags.service import load_playbook_raw  # noqa: F401
        from trading_system.playbook import evaluate_cycles
        from trading_system.playbook.briefing import _live_prices as _lp
        all_rules = st.checkbox("Include inactive rules", value=False, key="pb_all")
        try:
            cyc_prices = _lp(playbook, portfolio)
        except Exception:
            cyc_prices = prices
        evals = evaluate_cycles(playbook, portfolio, snap, cyc_prices, include_inactive=all_rules)
        st.caption(snap.summary_line())
        icon = {"FIRES": "✅", "PRICE_GUARD": "⏸", "AWAITING_EVENT": "⏳",
                "BLOCKED_FLAGS": "🚫", "INACTIVE": "·"}
        for e in evals:
            with st.expander(f"{icon.get(e.status, '·')} Rule {e.rule_id} — {e.label}  [{e.status}]",
                             expanded=(e.status == "FIRES")):
                if e.reasons:
                    st.markdown("- " + "\n- ".join(e.reasons))
                for o in e.orders:
                    px = f"${o.price:,.2f}" if o.price else "?"
                    ok = "✓ in range" if o.price_ok else "✗ OUT OF RANGE"
                    verdict = (o.compliance or {}).get("verdict", "")
                    st.markdown(f"  - **{o.ticker}** ${o.dollars:,.0f} · guard {o.guard} · last {px} · "
                                f"{ok}{' · ' + verdict if verdict else ''}")
                if e.note:
                    st.caption(e.note)

    with tab_check:
        st.markdown("Run a proposed order through the compliance gate.")
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", value="ASML", key="chk_t").upper()
        side = c2.selectbox("Side", ["BUY", "SELL"], key="chk_s")
        dollars = c3.number_input("Dollars (0 = full position for SELL)", 0, 100000, 700, 50, key="chk_d")
        if st.button("Check", key="chk_go"):
            from trading_system.playbook import check_trade
            sma50 = {}
            if ticker in playbook.lockout_tickers:
                try:
                    from trading_system.flags.lookups import _yf_history
                    px = _yf_history(ticker, period="6mo").sort("date")
                    sma50[ticker] = float(px["close"].tail(50).mean())
                except Exception:
                    pass
            res = check_trade(ticker, side, float(dollars), playbook, portfolio,
                              snapshot=snap, prices=prices, sma50=sma50)
            (st.success if res.allowed else st.error)(
                f"{res.verdict} — {side} {ticker} " + (f"${dollars:,.0f}" if dollars else "(full)"))
            for v in res.violations:
                st.markdown(f"- 🚫 {v}")
            for w in res.warnings:
                st.markdown(f"- ⚠ {w}")

    with tab_blotter:
        from trading_system.playbook import load_blotter
        df = load_blotter(cfg.path("reports"))
        if df is None or df.is_empty():
            st.info("No trades logged yet. Use `ts log-trade …` or the form below.")
        else:
            st.dataframe(df.to_pandas(), use_container_width=True, hide_index=True)
        with st.expander("➕ Log a fill"):
            f1, f2, f3, f4 = st.columns(4)
            lt = f1.text_input("Ticker", key="lt_t").upper()
            ls = f2.selectbox("Side", ["BUY", "SELL"], key="lt_s")
            lq = f3.number_input("Qty", 0.0, 100000.0, 0.0, key="lt_q")
            lp = f4.number_input("Price", 0.0, 100000.0, 0.0, key="lt_p")
            if st.button("Log trade", key="lt_go") and lq > 0 and lp > 0:
                from trading_system.playbook import log_trade
                pos = portfolio.position(lt)
                basis = pos.average_cost if (pos and ls == "SELL") else None
                row = log_trade(cfg.path("reports"), lt, ls, lq, lp, avg_cost_basis=basis)
                st.success(f"Logged {ls} {lq} {lt} @ {lp}"
                           + (f" · realized {row['realized_pnl']}" if row["realized_pnl"] != "" else ""))
                st.cache_data.clear()
