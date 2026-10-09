#!/usr/bin/env python3
"""Competing dummy portfolios — long-horizon, side-by-side, real prices.

Purpose: stop guessing which approach works. Run several $10,000 paper books
with *different selection rules* against the same market, mark them daily, and
let 6–12 months of live out-of-sample results settle the argument. This is the
honest complement to backtests (which are survivorship-biased and in-sample-ish
no matter how carefully they're purged).

Books (active)
  spy_benchmark  100% SPY. The bar everything must clear.
  momentum       top-10 by 120d momentum among liquid names — the rule-based
                 control (needs no model; computed straight from the price
                 store since 2026-10-09).
  alpha_v2       THE ALPHA ENGINE v2 book (added 2026-09-21): `ts alpha picks`
                 targets — 1000-name panel, 5/21/63d rank forecasters blended
                 by trailing realised IC from the forecast ledger, gated,
                 sector-capped, ½ equal + ½ inverse-vol weighted, 25% vol brake,
                 200-day trend overlay — traded with Gârleanu-Pedersen partial
                 trading (35% of the gap per monthly rebalance).

Books (retired 2026-10-09 with the legacy pipeline — frozen at their last mark,
history kept, shown in the report's "Retired" block)
  ml_raw         top-10 of the legacy `ts picks` (raw model score)
  ml_v2          top-10 of picks_v2 (gated legacy model)
  blend          50% ml_v2 + 50% momentum
  ml_v2_gp       legacy 63-day forecaster, top-20, GP partial trading
  All four were driven by the legacy 14-model ensemble, which the alpha engine
  superseded on 2026-09-21. At retirement (last mark 2026-10-08, ~5 weeks) the
  three pure-model books trailed SPY by 6.9-7.9% over their own spans; blend
  (half momentum) was 1.9% ahead.

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
BOOKS = ["spy_benchmark", "momentum", "alpha_v2"]
# Legacy-model books, retired with the legacy pipeline: never marked or traded again; their JSON
# (holdings, equity log, trades) stays on disk unchanged and the report shows their final numbers.
RETIRED = {"ml_raw": "2026-10-09", "ml_v2": "2026-10-09", "blend": "2026-10-09", "ml_v2_gp": "2026-10-09"}
ALPHA_TOP = 20            # alpha_v2 book width (its own cap/vol target live in trading_system.alpha.portfolio)
MOM_LOOKBACK = 120        # sessions — the legacy gold feature mom_120d
MOM_MIN_ADV = 20e6        # 20-session average dollar volume
MOM_MIN_PRICE = 5.0
GP_TRADE_RATE = 0.35      # move 35% of the way toward target each rebalance
# (no signal-decay shrink: the simulator that produced the 0.995 result did not apply one)


# ── price helpers ───────────────────────────────────────────────────────────
def price_frame() -> pl.DataFrame:
    """Liquid-universe bronze prices, plus the Massive whole-market table for any name the alpha_v2
    book holds outside that universe (its panel is the top-1000 by liquidity)."""
    px = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet", columns=["date", "ticker", "adj_close"])
    massive = REPO / "data/bronze/massive/ohlcv_all.parquet"
    if massive.exists():
        have = set(px["ticker"].unique().to_list())
        extra = (pl.scan_parquet(massive).select("date", "ticker", "adj_close")
                   .filter(~pl.col("ticker").is_in(list(have)), pl.col("adj_close") > 0)
                   .with_columns(pl.col("date").cast(px.schema["date"])).collect())
        if extra.height:
            px = pl.concat([px, extra.select(px.columns)], how="vertical_relaxed")
    return px


def prices_on(px: pl.DataFrame, d) -> dict[str, float]:
    row = px.filter(pl.col("date") == d)
    return dict(zip(row["ticker"].to_list(), row["adj_close"].to_list()))


def latest_date(px: pl.DataFrame):
    return px["date"].max()


# ── selection rules ─────────────────────────────────────────────────────────
def momentum_rank(px: pl.DataFrame) -> pl.DataFrame:
    """Last date's liquid names ranked by 120-session momentum. Same definitions as the legacy gold
    features it replaces (mom_120d = adj_close / adj_close 120 rows earlier − 1; avg_dollar_volume_20 =
    20-row mean of close × volume), computed from the price store so no feature pipeline is needed."""
    f = (px.sort(["ticker", "date"])
           .with_columns(mom=pl.col("adj_close") / pl.col("adj_close").shift(MOM_LOOKBACK).over("ticker") - 1,
                         adv20=(pl.col("close") * pl.col("volume")).rolling_mean(20).over("ticker")))
    last = f["date"].max()
    return (f.filter(pl.col("date") == last, pl.col("adv20") >= MOM_MIN_ADV, pl.col("close") >= MOM_MIN_PRICE)
             .drop_nulls(["mom"]).sort("mom", descending=True))


def select(book: str, n: int = N_HOLD) -> list[str]:
    if book == "spy_benchmark":
        return ["SPY"]
    if book == "momentum":
        px = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet",
                             columns=["date", "ticker", "close", "adj_close", "volume"])
        return momentum_rank(px).head(n)["ticker"].to_list()
    raise ValueError(f"{book}: not an active equal-weight book")


def select_weighted(book: str) -> dict[str, float]:
    """Target WEIGHTS (not just names) — needed for partial trading."""
    if book != "alpha_v2":
        raise ValueError(f"{book}: not an active weighted book")
    from trading_system.alpha.live import targets_as_dict
    from trading_system.config import get_config
    return targets_as_dict(get_config(str(REPO / "configs/default.yaml")), ALPHA_TOP)


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


DUST_W = 0.005            # a position the model no longer wants and below 0.5% of equity is sold outright


def rebalance_book_gp(b: dict, target_w: dict[str, float], px: dict[str, float], d, rate: float | None = None,
                      note: str = "") -> None:
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
    # Partial trading damps REBALANCING churn. It must not ration the initial deployment: before
    # 2026-10-08 a new book moved 35%/month out of cash, so alpha_v2 was 34% invested after two
    # rebalances and the forward test measured mostly cash. A book with no holdings goes to target.
    if rate is None:
        rate = 1.0 if not b["holdings"] else GP_TRADE_RATE
    new = cur + rate * (tgt - cur)
    new = np.clip(new, 0.0, None)
    new = np.where((tgt <= 0.0) & (new < DUST_W), 0.0, new)   # no fractional leftovers of dropped names
    turn_frac = float(np.abs(new - cur).sum())
    cost = eq * ((COST_BPS * turn_frac) + IMPACT_BPS * turn_frac ** 1.5) / 10_000.0
    eq_after = eq - cost
    b["holdings"] = {t: float(new[i] * eq_after / px[t]) for i, t in enumerate(names) if new[i] > 1e-6}
    b["cash"] = float(eq_after * max(0.0, 1.0 - new.sum()))
    b["last_rebalance"] = str(d)
    b["trades"].append({"date": str(d), "targets": [t for t in names if target_w.get(t, 0) > 0],
                        "turnover_frac": round(turn_frac, 4), "cost": round(cost, 2),
                        "policy": f"GP partial rate={rate}" + (f" · {note}" if note else "")})


def _rebalance(b: dict, name: str, p: dict[str, float], d) -> None:
    """Dispatch: the GP book trades toward weights; the others equal-weight names."""
    if name == "alpha_v2":
        rebalance_book_gp(b, select_weighted(name), p, d)
    else:
        rebalance_book(b, select(name), p, d)


def do_catch_up(px: pl.DataFrame, names: list[str]) -> None:
    """One-off: trade the GP books fully to their current targets (repairs the cash-ramp bug, 2026-10-08)."""
    d = latest_date(px)
    p = prices_on(px, d)
    for name in names:
        b = load_book(name)
        if not b["created"]:
            continue
        before = equity(b, p)
        rebalance_book_gp(b, select_weighted(name), p, d, rate=1.0, note="catch-up: cash-ramp fix 2026-10-08")
        save_book(b)
        inv = 1 - b["cash"] / max(equity(b, p), 1e-9)
        print(f"  {name}: caught up → {len(b['holdings'])} names, {inv:.0%} invested (equity {before:,.0f})")


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


def spy_over(spy_log: list[dict], log: list[dict]) -> float | None:
    """SPY's return over exactly the dates a book has been live (books started on different days)."""
    sp = {r["date"]: r["equity"] for r in spy_log}
    a, z = log[0]["date"], log[-1]["date"]
    return sp[z] / sp[a] - 1 if a in sp and z in sp else None


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
    spy_log = load_book("spy_benchmark")["equity_log"]
    L = [f"Dummy portfolios · ${START_CASH:,.0f} each · monthly rebalance · "
         f"since {rows[0][1]['created']} ({rows[0][2]['days']} sessions)",
         "",
         f"{'Book':<15}{'Equity':>10}{'Total':>9}{'vs SPY':>9}{'Sharpe':>8}"
         f"{'MaxDD':>8}  Holdings"]
    for name, b, m in sorted(rows, key=lambda r: -r[2]["total_ret"]):
        sp = spy_over(spy_log, b["equity_log"])
        rel = f"{(m['total_ret'] - sp) * 100:>+8.1f}%" if sp is not None else f"{'n/a':>9}"
        hold = ",".join(list(b["holdings"])[:5])
        L.append(f"{name:<15}{m['equity']:>10,.0f}{m['total_ret']*100:>8.1f}%"
                 f"{rel}{m['sharpe']:>8.2f}{m['maxdd']*100:>7.1f}%  {hold}")
    L.append("(vs SPY = SPY over the same dates as the book; alpha_v2 started 2026-09-18)")
    if rows[0][2]["days"] < 20:
        L.append("")
        L.append("⚠ Too early to judge — these need months, not days. "
                 "Ignore rankings until ~60+ sessions.")
    retired = []
    for name, when in RETIRED.items():
        b = load_book(name)
        m = metrics(b["equity_log"]) if b["created"] else {}
        if not m:
            continue
        sp = spy_over(spy_log, b["equity_log"])
        rel = f"{(m['total_ret'] - sp) * 100:>+8.1f}%" if sp is not None else f"{'n/a':>9}"
        retired.append(f"  {name:<13}{m['equity']:>10,.0f}{m['total_ret']*100:>8.1f}%{rel}"
                       f"  {b['created']} → {b['equity_log'][-1]['date']} (retired {when})")
    if retired:
        L += ["", "Retired with the legacy model — frozen at their last mark, not traded:"] + retired
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--rebalance", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--catch-up", nargs="*", metavar="BOOK", help="one-off: trade these GP books fully to target")
    a = ap.parse_args()
    if not any([a.init, a.mark, a.rebalance, a.report]):
        a.report = True
    px = price_frame() if (a.init or a.mark or a.rebalance or a.catch_up) else None
    if a.catch_up:
        do_catch_up(px, a.catch_up)
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
