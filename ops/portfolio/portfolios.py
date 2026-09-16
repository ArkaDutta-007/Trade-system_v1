#!/usr/bin/env python3
"""Competing dummy portfolios — long-horizon, side-by-side, real prices.

Purpose: stop guessing which approach works. Run several $10,000 paper books
with *different selection rules* against the same market, mark them daily, and
let 6–12 months of live out-of-sample results settle the argument. This is the
honest complement to backtests (which are survivorship-biased and in-sample-ish
no matter how carefully they're purged).

Books
  spy_benchmark  100% SPY. The bar everything must clear.
  ml_raw         top-10 of the current `ts picks` (raw model score).
                 Deliberately kept as the control so the v2 gates are
                 measured, not assumed.
  ml_v2          top-10 of picks_v2 (quality gate + risk-adjusted + diversified)
  momentum       top-10 by 120d momentum among liquid names — the rule-based
                 sleeve that beat the ML model risk-adjusted in backtests
                 (Sharpe 1.21 vs 1.03)
  blend          50% ml_v2 + 50% momentum, the diversified-across-methods book
  ml_v2_gp       THE RESEARCH WINNER, live (added 2026-09-16). Replicates
                 `xgb63|gp35` from reports/research/REPORT.md: the 63-day
                 forecaster (not 252d), top-20 with a 10% cap, run through
                 picks_v2's gates/shrink/risk-adjust/theme caps, and executed
                 with Gârleanu-Pedersen PARTIAL trading — each month the book
                 moves only 35% of the way from its current weights toward the
                 target. On 21.7 causal years that execution change alone took
                 Sharpe 0.862 → 0.995 and cut turnover 57%. Uses the repo's
                 research.execution.GarleanuPedersenPolicy with signal_decay=0,
                 which is exactly the form the simulator scored.

Design decisions that matter
  * MONTHLY rebalance (not daily). The 2026-07 research showed 3x costs halve
    Sharpe — daily top-k churn is where the edge dies.
  * Equal weight within a book: scores are too noisy to size on (measured).
  * Costs charged on turnover at 4bps + sqrt impact, same model as the repo.
  * Everything is append-only JSON under ~/trade-ops/portfolio/books/ so a
    crash or a rerun can never silently rewrite history.

Usage:
    python3 portfolios.py --init            # create books (once)
    python3 portfolios.py --mark            # mark to market (daily, idempotent)
    python3 portfolios.py --rebalance       # monthly; safe to call daily
    python3 portfolios.py --report          # digest-friendly table
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_repo_to_path, ops_root  # noqa: E402

REPO = add_repo_to_path()
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402

# Live state, not code: lives under ops_root() (~/trade-ops on the RIT box),
# never inside the tracked tree. paths.py documents the precedence.
BOOKS_DIR = ops_root() / "portfolio" / "books"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)

START_CASH = 10_000.0
N_HOLD = 10
COST_BPS = 4.0
IMPACT_BPS = 10.0
BOOKS = ["spy_benchmark", "ml_raw", "ml_v2", "momentum", "blend", "ml_v2_gp"]

# ml_v2_gp — the research winner's configuration (reports/research/REPORT.md,
# variant xgb63|gp35). Kept as a SEPARATE book so the five that have been
# tracking since 2026-09-08 stay an unbroken forward test.
GP_HORIZON = 63           # days: the horizon whose ICIR (3.40) beat 252d's (3.03)
GP_TOP = 20               # research book width; top-10 scored 0.866 vs 0.995
GP_MAX_W = 0.10           # research single-name cap
GP_TRADE_RATE = 0.35      # move 35% of the way toward target each rebalance
# (no signal-decay shrink: the simulator that produced the 0.995 result did not apply one)


# ── price helpers ───────────────────────────────────────────────────────────
def price_frame() -> pl.DataFrame:
    return pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet",
                           columns=["date", "ticker", "adj_close"])


def prices_on(px: pl.DataFrame, d) -> dict[str, float]:
    row = px.filter(pl.col("date") == d)
    return dict(zip(row["ticker"].to_list(), row["adj_close"].to_list()))


def latest_date(px: pl.DataFrame):
    return px["date"].max()


# ── selection rules ─────────────────────────────────────────────────────────
def select(book: str, n: int = N_HOLD) -> list[str]:
    if book == "spy_benchmark":
        return ["SPY"]
    if book == "momentum":
        f = pl.read_parquet(REPO / "data/gold/features.parquet")
        last = f["date"].max()
        d = (f.filter(pl.col("date") == last)
              .filter((pl.col("avg_dollar_volume_20") >= 20e6)
                      & (pl.col("close") >= 5.0))
              .drop_nulls(["mom_120d"])
              .sort("mom_120d", descending=True))
        return d.head(n)["ticker"].to_list()
    if book == "ml_raw":
        from picks_v2 import raw_picks, latest_features
        feats, _ = latest_features()
        df = raw_picks().join(feats, on="ticker", how="left").drop_nulls(["close"])
        return df.sort("score", descending=True).head(n)["ticker"].to_list()
    if book == "ml_v2":
        from picks_v2 import build
        return [p["ticker"] for p in build(n)["picks"]]
    if book == "blend":
        half = max(1, n // 2)
        a = select("ml_v2", half)
        b = [t for t in select("momentum", n) if t not in a][:n - len(a)]
        return a + b
    raise ValueError(book)


def select_weighted(book: str) -> dict[str, float]:
    """Target WEIGHTS (not just names) — needed for partial trading."""
    if book != "ml_v2_gp":
        raise ValueError(book)
    from picks_v2 import build
    plan = build(GP_TOP, GP_HORIZON)
    w = {p["ticker"]: float(p.get("weight") or 0.0) for p in plan["picks"]}
    tot = sum(w.values()) or 1.0
    w = {t: v / tot for t, v in w.items()}
    # research cap is 10%; picks_v2's default is 20% — clip and renormalise
    for _ in range(50):
        over = {t: v for t, v in w.items() if v > GP_MAX_W}
        if not over:
            break
        excess = sum(v - GP_MAX_W for v in over.values())
        for t in over:
            w[t] = GP_MAX_W
        free = {t: v for t, v in w.items() if t not in over}
        fs = sum(free.values()) or 1.0
        for t in free:
            w[t] += excess * free[t] / fs
    return w


# ── book state ──────────────────────────────────────────────────────────────
def book_path(name: str) -> Path:
    return BOOKS_DIR / f"{name}.json"


def load_book(name: str) -> dict:
    p = book_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return {"name": name, "created": None, "cash": START_CASH,
            "holdings": {}, "equity_log": [], "trades": [], "last_rebalance": None}


def save_book(b: dict) -> None:
    book_path(b["name"]).write_text(json.dumps(b, indent=1, default=str))


def equity(b: dict, px: dict[str, float]) -> float:
    v = b["cash"]
    for t, q in b["holdings"].items():
        p = px.get(t)
        if p:
            v += q * p
    return v


def rebalance_book(b: dict, targets: list[str], px: dict[str, float], d) -> None:
    eq = equity(b, px)
    tradeable = [t for t in targets if px.get(t)]
    if not tradeable:
        return
    target_val = eq / len(tradeable)
    new_hold, turnover = {}, 0.0
    for t in tradeable:
        q = target_val / px[t]
        old_val = b["holdings"].get(t, 0.0) * px[t]
        turnover += abs(target_val - old_val)
        new_hold[t] = q
    for t, q in b["holdings"].items():
        if t not in new_hold and px.get(t):
            turnover += q * px[t]
    turn_frac = turnover / eq if eq > 0 else 0.0
    cost = eq * ((COST_BPS * turn_frac) + IMPACT_BPS * turn_frac ** 1.5) / 10_000.0
    eq_after = eq - cost
    scale = eq_after / eq if eq > 0 else 1.0
    b["holdings"] = {t: q * scale for t, q in new_hold.items()}
    b["cash"] = 0.0
    b["last_rebalance"] = str(d)
    b["trades"].append({"date": str(d), "targets": tradeable,
                        "turnover_frac": round(turn_frac, 4), "cost": round(cost, 2)})


def rebalance_book_gp(b: dict, target_w: dict[str, float], px: dict[str, float], d) -> None:
    """Gârleanu-Pedersen partial trading: new_w = cur_w + rate * (target_w - cur_w).

    Costs are charged on the turnover actually traded, same model as the other
    books, so the comparison is only about execution — not about cost
    assumptions.
    """
    import numpy as np

    eq = equity(b, px)
    if eq <= 0:
        return
    names = sorted({t for t in target_w if px.get(t)} | {t for t in b["holdings"] if px.get(t)})
    if not names:
        return
    cur = np.array([b["holdings"].get(t, 0.0) * px[t] / eq for t in names])
    tgt = np.array([target_w.get(t, 0.0) for t in names])
    # The PLAIN form, exactly as research/wfbacktest.py line ~432 applies it:
    #     new = cur + rate * (target - cur)
    # NOT research.execution.GarleanuPedersenPolicy.step(): that method
    # renormalises the result to sum 1, which from all-cash rescales a 35 % move
    # into 100 % deployment and makes "partial trading" a no-op at seeding. The
    # simulator that scored Sharpe 0.995 (and reported mean deploy 0.81) uses
    # the un-normalised form and leaves the remainder in cash. Replicate THAT.
    new = cur + GP_TRADE_RATE * (tgt - cur)
    new = np.clip(new, 0.0, None)
    turn_frac = float(np.abs(new - cur).sum())
    cost = eq * ((COST_BPS * turn_frac) + IMPACT_BPS * turn_frac ** 1.5) / 10_000.0
    eq_after = eq - cost
    b["holdings"] = {t: float(new[i] * eq_after / px[t]) for i, t in enumerate(names) if new[i] > 1e-6}
    b["cash"] = float(eq_after * max(0.0, 1.0 - new.sum()))
    b["last_rebalance"] = str(d)
    b["trades"].append({"date": str(d), "targets": [t for t in names if target_w.get(t, 0) > 0],
                        "turnover_frac": round(turn_frac, 4), "cost": round(cost, 2),
                        "policy": f"GP partial rate={GP_TRADE_RATE}"})


def _rebalance(b: dict, name: str, p: dict[str, float], d) -> None:
    """Dispatch: the GP book trades toward weights; the others equal-weight names."""
    if name == "ml_v2_gp":
        rebalance_book_gp(b, select_weighted(name), p, d)
    else:
        rebalance_book(b, select(name), p, d)


def do_init(px: pl.DataFrame) -> None:
    d = latest_date(px)
    p = prices_on(px, d)
    for name in BOOKS:
        b = load_book(name)
        if b["created"]:
            print(f"  {name}: exists (created {b['created']}) — untouched")
            continue
        b["created"] = str(d)
        _rebalance(b, name, p, d)
        b["equity_log"] = [{"date": str(d), "equity": round(equity(b, p), 2)}]
        save_book(b)
        print(f"  {name}: seeded ${START_CASH:,.0f} → {list(b['holdings'])[:6]}"
              f"{'…' if len(b['holdings']) > 6 else ''}")


def do_mark(px: pl.DataFrame) -> None:
    d = latest_date(px)
    p = prices_on(px, d)
    for name in BOOKS:
        b = load_book(name)
        if not b["created"]:
            continue
        if b["equity_log"] and b["equity_log"][-1]["date"] == str(d):
            b["equity_log"][-1]["equity"] = round(equity(b, p), 2)   # idempotent
        else:
            b["equity_log"].append({"date": str(d), "equity": round(equity(b, p), 2)})
        save_book(b)
    print(f"marked {len(BOOKS)} books to {d}")


def do_rebalance(px: pl.DataFrame, force: bool = False) -> None:
    d = latest_date(px)
    p = prices_on(px, d)
    for name in BOOKS:
        b = load_book(name)
        if not b["created"]:
            continue
        last = b.get("last_rebalance")
        if last and not force:
            prev = datetime.fromisoformat(str(last)).date()
            cur = d if isinstance(d, date) else datetime.fromisoformat(str(d)).date()
            if (cur.year, cur.month) == (prev.year, prev.month):
                continue                     # already rebalanced this month
        _rebalance(b, name, p, d)
        save_book(b)
        print(f"  rebalanced {name} → {list(b['holdings'])[:6]}")


def metrics(log: list[dict]) -> dict:
    if len(log) < 2:
        return {}
    import numpy as np
    eq = np.array([r["equity"] for r in log], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    days = len(eq)
    years = max(days / 252.0, 1e-9)
    total = eq[-1] / eq[0] - 1
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0.08 else total
    vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-12) * np.sqrt(252)) if len(rets) > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    return {"equity": eq[-1], "total_ret": total, "cagr": cagr,
            "vol": vol, "sharpe": sharpe, "maxdd": dd, "days": days}


def do_report() -> str:
    rows, seeded = [], []
    for name in BOOKS:
        b = load_book(name)
        if not b["created"]:
            continue
        seeded.append((name, b))
        m = metrics(b["equity_log"])
        if m:
            rows.append((name, b, m))
    if not seeded:
        return "No books initialised yet — run `portfolios.py --init`."
    if not rows:
        # seeded but <2 marks: show composition so the digest still says something
        L = [f"Dummy portfolios · ${START_CASH:,.0f} each · seeded "
             f"{seeded[0][1]['created']} — awaiting a second mark for P&L.", ""]
        for name, b in seeded:
            L.append(f"  {name:<15} {', '.join(list(b['holdings'])[:8])}")
        return "\n".join(L)
    bench = next((m for n, _, m in rows if n == "spy_benchmark"), None)
    L = [f"Dummy portfolios · ${START_CASH:,.0f} each · monthly rebalance · "
         f"since {rows[0][1]['created']} ({rows[0][2]['days']} sessions)",
         "",
         f"{'Book':<15}{'Equity':>10}{'Total':>9}{'vs SPY':>9}{'Sharpe':>8}"
         f"{'MaxDD':>8}  Holdings"]
    for name, b, m in sorted(rows, key=lambda r: -r[2]["total_ret"]):
        rel = (m["total_ret"] - bench["total_ret"]) * 100 if bench else 0.0
        hold = ",".join(list(b["holdings"])[:5])
        L.append(f"{name:<15}{m['equity']:>10,.0f}{m['total_ret']*100:>8.1f}%"
                 f"{rel:>+8.1f}%{m['sharpe']:>8.2f}{m['maxdd']*100:>7.1f}%  {hold}")
    if rows[0][2]["days"] < 20:
        L.append("")
        L.append("⚠ Too early to judge — these need months, not days. "
                 "Ignore rankings until ~60+ sessions.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--rebalance", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not any([a.init, a.mark, a.rebalance, a.report]):
        a.report = True
    px = price_frame() if (a.init or a.mark or a.rebalance) else None
    if a.init:
        print("initialising books:")
        do_init(px)
    if a.rebalance:
        do_rebalance(px, a.force)
    if a.mark:
        do_mark(px)
    if a.report:
        print(do_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
