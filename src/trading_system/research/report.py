"""Assemble the research artefacts into one markdown write-up.

Reads whatever the stages left in ``reports/research/`` and produces a single
document: the performance table, the robustness corrections, the execution and
cost-ladder comparisons, and the RL verdict.  Nothing is computed here — this
only formats results, so the report can never disagree with the artefacts.

The ordering is deliberate.  Corrections come before headline numbers, because
a table of Sharpe ratios read before its PBO is read is a table that will be
believed.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

__all__ = ["write_report"]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _md(v) -> str:
    """Escape a cell for a markdown table.

    Variant names carry a ``|`` separator (``xgb63|gp35``), which is also the
    markdown column delimiter, so an unescaped name silently splits the row.
    """
    return str(v).replace("|", "\\|")


def _table(rows: list[dict], cols: list[str], sort_by: str = "Sharpe") -> str:
    if not rows:
        return "_(no results)_"
    rows = sorted(rows, key=lambda r: -(r.get(sort_by) or -9))
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---:" if c != "variant" else ":---" for c in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                if c in ("CAGR", "MaxDD", "cost_drag_annual", "unfilled_frac"):
                    v = f"{v:.2%}"
                else:
                    v = f"{v:.3g}"
            cells.append(_md(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def write_report(out_dir: Path) -> Path:
    suite = _load(out_dir / "backtest" / "suite_report.json")
    rl_dirs = sorted((out_dir / "rl").glob("*/rl_report.json")) \
        if (out_dir / "rl").exists() else []

    L: list[str] = []
    L.append("# Causal walk-forward results")
    L.append("")
    L.append(f"Generated {date.today().isoformat()}. Every number below comes from a "
             "simulation in which the model was refit only on data available at the "
             "time, scored the cross-section on that date, and traded the next day "
             "paying per-name spread and square-root impact with a participation cap.")
    L.append("")

    if suite:
        rb = suite.get("robustness", {})
        L.append("## Read this first: the corrections")
        L.append("")
        if "pbo" in rb:
            L.append(f"- **PBO (CSCV): {rb['pbo']:.1%}** across {rb.get('n_configs', '?')} "
                     f"configurations. The in-sample winner's median out-of-sample rank "
                     f"was {rb.get('median_oos_rank', float('nan')):.2f} "
                     "(0.5 would be pure chance).")
        if "spa" in rb:
            s = rb["spa"]
            L.append(f"- **SPA test vs `{_md(s['benchmark'])}`: p = {s['p_value']:.3f}** over "
                     f"{s['n_models']} learned configurations; best was "
                     f"`{_md(s['best_model'])}`. "
                     "This prices the fact that several models were tried before one won.")
        defl = rb.get("deflated", {})
        if defl:
            L.append("- **Deflated Sharpe and bootstrap intervals** (stationary bootstrap, "
                     "21-day mean block, so serial dependence survives the resample):")
            L.append("")
            L.append("| variant | DSR | Sharpe 95% CI | P(Sharpe > 0) |")
            L.append("|:---|---:|:---:|---:|")
            for k, v in sorted(defl.items(), key=lambda kv: -kv[1]["dsr"]):
                L.append(f"| `{_md(k)}` | {v['dsr']:.3f} | "
                         f"[{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}] | {v['p_gt_0']:.2f} |")
        sv = rb.get("survivorship")
        if sv:
            L.append("")
            L.append("### The universe is already inflated")
            L.append("")
            L.append("Before reading any strategy row: an equal weight of *today's* "
                     "universe is not a fair baseline, because the universe was "
                     "chosen after the fact. Measuring that against a real "
                     "survivorship-free index product over the same dates:")
            L.append("")
            L.append("| | CAGR | Sharpe |")
            L.append("|:---|---:|---:|")
            L.append(f"| equal weight of today's universe | "
                     f"{sv['universe_cagr']:.2%} | {sv['universe_sharpe']:.2f} |")
            L.append(f"| {sv['reference']} | {sv['reference_cagr']:.2%} | "
                     f"{sv['reference_sharpe']:.2f} |")
            L.append(f"| **unearned** | **{sv['cagr_bias']:+.2%}** | "
                     f"**{sv['sharpe_bias']:+.2f}** |")
            L.append("")
            L.append("So the benchmark to beat is `universe_ew`, not `spy`. A "
                     "strategy that beats the index but not the universe has "
                     "demonstrated survivorship bias, not skill.")
        L.append("")
        L.append("## Performance")
        L.append("")
        L.append(_table(suite.get("table", []),
                        ["variant", "years", "CAGR", "Sharpe", "MaxDD",
                         "ann_turnover", "cost_drag_annual", "unfilled_frac"]))
        L.append("")
        L.append("`ann_turnover` is one-way turnover per year as a fraction of equity. "
                 "`cost_drag_annual` is execution cost per year as a fraction of the "
                 "equity actually at risk when it was paid, not of the starting cash, "
                 "so it stays comparable between books that compounded differently. "
                 "`unfilled_frac` is the share of intended notional the participation "
                 "cap refused: a large value means the strategy only exists at sizes it "
                 "cannot trade.")
        L.append("")

    for p in rl_dirs:
        rl = _load(p)
        if not rl:
            continue
        L.append(f"## Reinforcement learning — execution policy (`{p.parent.name}`)")
        L.append("")
        L.append("The agent sets four bounded scalars per rebalance (deploy, trade rate, "
                 "no-trade band, concentration). It is trained on the early period and "
                 "evaluated once on the later one, against the Gârleanu-Pedersen "
                 "closed form for the same problem.")
        L.append("")
        ev = rl.get("eval", {})
        if ev:
            L.append("| policy | Sharpe | CAGR | MaxDD | ann turnover | cost drag |")
            L.append("|:---|---:|---:|---:|---:|---:|")
            for k, v in sorted(ev.items(), key=lambda kv: -kv[1].get("sharpe", -9)):
                L.append(f"| `{_md(k)}` | {v.get('sharpe')} | {v.get('cagr', 0):.2%} | "
                         f"{v.get('maxdd', 0):.2%} | {v.get('ann_turnover')} | "
                         f"{v.get('cost_drag', 0):.3%} |")
            L.append("")
            for k, v in ev.items():
                if v.get("mean_action"):
                    L.append(f"- `{k}` mean action: {v['mean_action']}")
        if rl.get("verdict"):
            L.append("")
            L.append(f"**{rl['verdict']}**")
        L.append("")

    L.append("## What these numbers are not")
    L.append("")
    L.append("- **Survivorship.** The universe is today's liquid names. Companies that "
             "were liquid in 2005 and later failed or were acquired are absent, and free "
             "data sources do not serve their price history (yfinance and Stooq both "
             "return nothing for ATVI, TWTR, FRC, SIVB, XLNX, RTN). The size of that "
             "bias is measured above rather than assumed, and it is large. Every CAGR "
             "here is an upper bound, and the bias grows the further back the window "
             "starts. The fix is point-in-time constituents plus delisted prices — CRSP "
             "through a university subscription, or a paid vendor.")
    L.append("- **One market.** US equities, long only, no leverage, no shorting.")
    L.append("- **Costs are a model.** Spreads are estimated from daily bars "
             "(Abdi-Ranaldo), not quoted. The cost ladder shows how much the conclusions "
             "depend on that assumption; treat a result that only survives at 1x costs "
             "as unproven.")
    L.append("")

    out = out_dir / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    return out
