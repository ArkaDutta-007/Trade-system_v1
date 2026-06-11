"""Morning briefing — the daily pre-market artifact.

One command (`ts brief`) produces what a discretionary PM would want on one
page before the open:

  1. Flag board (O/F/I/S/C) + composite + deployment guidance
  2. Catalyst calendar — what's coming in the next two weeks and which flag it sets
  3. Standing-rule checks vs live prices (triggered / near first)
  4. Cycle rules active today and whether they fire
  5. Portfolio concentration vs caps + drawdown vs the Jun 10 baseline
  6. 2026 NRA tax-shield status
  7. The §7 pre-buy checklist

Written to reports/briefings/brief_YYYYMMDD_HHMM.md (+ .json sidecar).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from ..config import Config
from ..flags.models import FlagSnapshot, FLAG_ORDER
from ..utils import get_logger
from .blotter import blotter_realized
from .cycles import evaluate_cycles
from .loader import Playbook, Portfolio, load_playbook, load_portfolio
from .standing_rules import evaluate_standing_rules
from .tax import shield_status

logger = get_logger(__name__)


def _live_prices(playbook: Playbook, portfolio: Portfolio) -> dict[str, float]:
    """Live quotes for every ticker the briefing touches; JSON snapshot fallback."""
    tickers: set[str] = set(portfolio.held_symbols)
    for rule in playbook.standing_rules:
        tickers |= {t.upper() for t in rule.get("tickers", [])}
    placeholders = ("SEMICAP", "DIVERSIF", "BEST")
    for rule in playbook.cycles:
        for o in rule.get("orders") or []:
            t = str(o["ticker"]).upper()
            if t.isalpha() and not t.startswith(placeholders):
                tickers.add(t)
        tickers |= {t.upper() for t in (rule.get("condition") or {}).get("price_below", {})}

    prices: dict[str, float] = {}
    try:
        from ..ingestion.realtime import live_price_snapshot

        prices = live_price_snapshot(sorted(tickers))
    except Exception as e:
        logger.warning(f"live price snapshot failed, using JSON snapshot prices: {e}")

    for h in portfolio.holdings:
        prices.setdefault(h.symbol, h.last_price)
    for sym, row in portfolio.watchlist.items():
        if row.get("last_price"):
            prices.setdefault(sym, float(row["last_price"]))
    return prices


def build_briefing(
    cfg: Config,
    snapshot: FlagSnapshot,
    playbook: Playbook | None = None,
    portfolio: Portfolio | None = None,
    prices: dict[str, float] | None = None,
    today: date | None = None,
) -> dict:
    playbook = playbook or load_playbook(cfg)
    portfolio = portfolio or load_portfolio(cfg)
    today = today or date.today()
    prices = prices or _live_prices(playbook, portfolio)

    rule_checks = evaluate_standing_rules(playbook, portfolio, prices)
    cycle_evals = evaluate_cycles(playbook, portfolio, snapshot, prices, today=today)

    realized = blotter_realized(cfg.path("reports"), year=int(playbook.tax.get("shield_year", 2026)))
    shield = shield_status(playbook, portfolio, realized)

    horizon = today + timedelta(days=14)
    upcoming = [
        c for c in playbook.catalysts
        if today <= date.fromisoformat(str(c["date"])) <= horizon
    ]

    invested = portfolio.invested_value(prices)
    baseline = playbook.baseline_invested
    drawdown = (1.0 - invested / baseline) if baseline > 0 else 0.0
    weights = portfolio.weights(prices)
    cap_breaches = [
        {"ticker": t, "weight_pct": round(w, 2), "cap_pct": playbook.cap_for(t),
         "barred": w >= playbook.cap_for(t)}
        for t, w in sorted(weights.items(), key=lambda kv: -kv[1])
        if w >= playbook.cap_for(t) - 1.5
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": str(today),
        "flags": snapshot.to_dict(),
        "flag_summary": snapshot.summary_line(),
        "catalysts_next_14d": upcoming,
        "standing_rules": [c.to_dict() for c in rule_checks],
        "cycle_rules": [e.to_dict() for e in cycle_evals],
        "portfolio": {
            "as_of": portfolio.as_of,
            "invested_value": round(invested, 2),
            "cash": round(portfolio.cash, 2),
            "total_value": round(invested + portfolio.cash, 2),
            "baseline_invested": baseline,
            "drawdown_vs_baseline": round(drawdown, 4),
            "semi_freeze_if_drawdown_gt": playbook.drawdown_semi_freeze,
            "top_weights": [
                {"ticker": t, "weight_pct": round(w, 2)}
                for t, w in sorted(weights.items(), key=lambda kv: -kv[1])[:8]
            ],
            "cap_watch": cap_breaches,
        },
        "tax_shield": {
            "year": shield.year,
            "realized_gains": round(shield.realized_gains, 2),
            "realized_losses": round(shield.realized_losses, 2),
            "net_realized": round(shield.net_realized, 2),
            "shield_remaining": round(shield.shield_remaining, 2),
            "all_in_rate": shield.all_in_rate,
            "summary": shield.summary(),
        },
        "prebuy_checklist": playbook.raw.get("prebuy_checklist", []),
    }


def _md_flag_table(brief: dict) -> str:
    lines = ["| Flag | Color | Reading | Source |", "|---|---|---|---|"]
    readings = brief["flags"]["readings"]
    for f in FLAG_ORDER:
        r = readings.get(f)
        if not r:
            continue
        stale = " ⚠️stale" if r.get("stale") else ""
        lines.append(f"| **{f}** ({r['name']}) | {r['color']}{stale} | {r['detail']} | {r['source']} |")
    return "\n".join(lines)


def render_markdown(brief: dict) -> str:
    comp = brief["flags"]["composite"]
    out: list[str] = []
    out.append(f"# Morning Briefing — {brief['as_of_date']}")
    out.append(f"_generated {brief['generated_at']} · playbook v2-nra_\n")

    out.append("## 1 · Flag board")
    out.append(_md_flag_table(brief))
    out.append(f"\n**Composite: {comp['color']}** — {comp['rationale']}")
    if comp.get("data_warnings"):
        out.append("\n> ⚠️ " + "\n> ⚠️ ".join(comp["data_warnings"]))

    out.append("\n## 2 · Catalysts (next 14 days)")
    if brief["catalysts_next_14d"]:
        out.append("| Date | Event | Flags |")
        out.append("|---|---|---|")
        for c in brief["catalysts_next_14d"]:
            out.append(f"| {c['date']} | {c['event']} | {', '.join(c.get('flags', [])) or '—'} |")
    else:
        out.append("_none in window_")

    out.append("\n## 3 · Standing rules (§3)")
    hot = [c for c in brief["standing_rules"] if c["status"] in ("TRIGGERED", "NEAR")]
    rest = [c for c in brief["standing_rules"] if c["status"] not in ("TRIGGERED", "NEAR")]
    if hot:
        out.append("**Action required / watch:**")
        for c in hot:
            icon = "🔴" if c["status"] == "TRIGGERED" else "🟡"
            out.append(f"- {icon} **{c['ticker']}** [{c['status']}] {c['action']} — {c['detail']}")
    out.append("<details><summary>All checks</summary>\n")
    out.append("| Ticker | Status | Rule | Detail |")
    out.append("|---|---|---|---|")
    for c in hot + rest:
        out.append(f"| {c['ticker']} | {c['status']} | {c['action']} | {c['detail']} |")
    out.append("\n</details>")

    out.append("\n## 4 · Cycle playbook (§4)")
    shown = [e for e in brief["cycle_rules"] if e["status"] != "INACTIVE"]
    if not shown:
        out.append("_no cycle rules active today_")
    for e in shown:
        icon = {"FIRES": "✅", "PRICE_GUARD": "⏸", "AWAITING_EVENT": "⏳", "BLOCKED_FLAGS": "🚫"}.get(e["status"], "·")
        out.append(f"\n### {icon} Rule {e['rule_id']} — {e['label']}  `[{e['status']}]`")
        if e["reasons"]:
            out.append("- " + "; ".join(e["reasons"]))
        for o in e["orders"]:
            px = f"{o['price']:.2f}" if o.get("price") else "?"
            ok = "✓ in range" if o["price_ok"] else "✗ OUT OF RANGE — does not fire"
            comp_v = (o.get("compliance") or {}).get("verdict", "")
            comp_s = f" · compliance: {comp_v}" if comp_v else ""
            out.append(f"- **{o['ticker']}** ${o['dollars']:,.0f} (guard {o['guard']}, last {px}) {ok}{comp_s}")
            if o.get("note"):
                out.append(f"  - {o['note']}")
            for v in (o.get("compliance") or {}).get("violations", []):
                out.append(f"  - 🚫 {v}")
        if e.get("note"):
            out.append(f"\n_{e['note']}_")

    p = brief["portfolio"]
    out.append("\n## 5 · Portfolio")
    out.append(
        f"Invested **${p['invested_value']:,.0f}** + cash **${p['cash']:,.0f}** "
        f"= **${p['total_value']:,.0f}** · invested value vs Jun-10 baseline: "
        f"**{-p['drawdown_vs_baseline']:+.1%}** (semi adds suspended at −{p['semi_freeze_if_drawdown_gt']:.0%}, rule 3.4)"
    )
    out.append("\n| Ticker | Weight |")
    out.append("|---|---|")
    for w in p["top_weights"]:
        out.append(f"| {w['ticker']} | {w['weight_pct']:.1f}% |")
    if p["cap_watch"]:
        out.append("\n**Cap watch (rule 3.5 — bar new cash, never sell):**")
        for c in p["cap_watch"]:
            state = "BARRED" if c["barred"] else "approaching"
            out.append(f"- {c['ticker']}: {c['weight_pct']:.1f}% vs cap {c['cap_pct']:.0f}% → {state}")

    out.append("\n## 6 · 2026 NRA tax shield")
    out.append(brief["tax_shield"]["summary"])

    out.append("\n## 7 · Pre-buy chart check (run manually before any order)")
    for i, item in enumerate(brief["prebuy_checklist"], 1):
        out.append(f"{i}. {item}")

    out.append("\n---\n_Research framework, not financial advice. Re-verify every price at order time._")
    return "\n".join(out)


def write_briefing(cfg: Config, brief: dict) -> tuple[Path, Path]:
    out_dir = cfg.path("reports") / "briefings"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md_path = out_dir / f"brief_{stamp}.md"
    json_path = out_dir / f"brief_{stamp}.json"
    md_path.write_text(render_markdown(brief))
    json_path.write_text(json.dumps(brief, indent=2, default=str))
    return md_path, json_path
