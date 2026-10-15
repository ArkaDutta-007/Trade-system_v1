#!/usr/bin/env python3
"""Apples-to-apples: Vibe-Trading's agent-designed rule vs our models, one backtester, one spec (2026-10-15).

Spec given to Vibe's agent (and applied identically here to every strategy):
  universe   the point-in-time eligible list of each day (≈980 names, failures included)
  test       2025-11-03 → 2026-10-14; rebalance on the first trading day of each month
  ranking    by the strategy's score computed from data up to the PREVIOUS close
  book       top 20, equal weight, long only, held to the next rebalance (rebalance at that day's close)
  costs      10 bps per side on turnover
Strategies: the agent's rule (re-implemented from its reported code, see AGENT_RULE), our current model,
our point-in-time-trained model (RESEARCH §13), and SPY. Results → reports/alpha/lab_2026_10/vibe_apples.json
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import lab_2026_10 as LAB  # noqa: E402

START, END = date(2025, 11, 3), date(2026, 10, 14)
TOP, COST = 20, 0.0010


def agent_rule(px: pl.DataFrame) -> pl.DataFrame:
    """The rule Vibe's agent froze on year 1 (run 20261015_153624_53_8a1d0d, oos_backtest.py): 60-day
    volatility-adjusted momentum = (close[t] / close[t-60] - 1) / (std of the 60 daily returns × √252)."""
    return (px.sort(["ticker", "date"])
              .with_columns(r=pl.col("close").pct_change().over("ticker"))
              .with_columns(score=(pl.col("close") / pl.col("close").shift(60).over("ticker") - 1)
                            / (pl.col("r").rolling_std(60).over("ticker") * 252 ** 0.5))
              .select("date", "ticker", "score"))


def backtest(scores: pl.DataFrame, px: pl.DataFrame, elig: pl.DataFrame) -> dict:
    days = sorted(px.filter((pl.col("date") >= START) & (pl.col("date") <= END))["date"].unique().to_list())
    allp = px.sort(["ticker", "date"]).with_columns(r=pl.col("close") / pl.col("close").shift(1).over("ticker") - 1)
    R = dict(((d, t), r) for d, t, r in allp.select("date", "ticker", "r").iter_rows() if r is not None)
    prev_of = {d: p for p, d in zip(sorted(px["date"].unique().to_list())[:-1], sorted(px["date"].unique().to_list())[1:])}
    rebal = [d for i, d in enumerate(days) if i == 0 or d.month != days[i - 1].month]
    sc = scores.join(elig, on=["date", "ticker"], how="semi").drop_nulls("score")
    by = {d: g for (d,), g in sc.group_by(["date"])}
    w, rets, holds, turn_total = {}, [], [], 0.0
    for d in days:
        r = sum(wt * R.get((d, t), 0.0) for t, wt in w.items()) if w else 0.0    # return earned today on held book
        cost = 0.0
        if d in rebal:
            g = by.get(prev_of.get(d))
            elig_today = set(elig.filter(pl.col("date") == prev_of.get(d))["ticker"].to_list()) if g is None else None
            if g is not None and g.height >= TOP:
                pick = g.sort("score", descending=True).head(TOP)["ticker"].to_list()
                new = {t: 1.0 / TOP for t in pick}
                turn = sum(abs(new.get(t, 0) - w.get(t, 0)) for t in set(new) | set(w))
                cost = turn * COST
                turn_total += turn
                w = new
                holds.append({"date": str(d), "holdings": pick})
        rets.append(r - cost)
    x = np.array(rets)
    eq = np.cumprod(1 + x)
    yrs = len(x) / 252
    return {"total": float(eq[-1] - 1), "ann": float(eq[-1] ** (1 / yrs) - 1), "sharpe": float(x.mean() / (x.std() + 1e-12) * np.sqrt(252)),
            "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min()), "sessions": len(x), "turnover": turn_total,
            "rebalances": holds}


def main():
    long = pl.read_parquet(LAB.cfg.path("data_silver") / "vibe" / "pit_long.parquet").with_columns(pl.col("date").cast(pl.Date))
    px = long.select("date", "ticker", "open", "high", "low", "close", "volume")
    elig = pl.read_csv(Path.home() / ".vibe-trading/data-bridge/pit_universe_eligible.csv", try_parse_dates=True)
    strategies = {"our current model": long.select("date", "ticker", score=pl.col("prod")),
                  "our point-in-time-trained model": long.select("date", "ticker", score=pl.col("pitaug"))}
    strategies["Vibe agent's rule (qwen3.7-flash)"] = agent_rule(px)
    res = {k: backtest(v, px, elig) for k, v in strategies.items()}
    pw, allpx, _ = LAB.pit_panel()
    spy = allpx.filter(pl.col("ticker") == "SPY").select("date", "ticker", close="adj_close").with_columns(pl.col("date").cast(pl.Date))
    s = spy.sort("date").with_columns(r=pl.col("close").pct_change()).filter((pl.col("date") >= START) & (pl.col("date") <= END))
    x = s["r"].to_numpy(); eq = np.cumprod(1 + x)
    res["SPY"] = {"total": float(eq[-1] - 1), "ann": float(eq[-1] ** (252 / len(x)) - 1), "sharpe": float(x.mean() / x.std() * np.sqrt(252)),
                  "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min()), "sessions": len(x)}
    (LAB.OUT / "vibe_apples.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        print(f"{k:<38} total {v['total']:+7.1%}  annual {v['ann']:+7.1%}  Sharpe {v['sharpe']:5.2f}  maxDD {v['maxdd']:+6.1%}"
              + (f"  turnover {v['turnover']:.1f}" if "turnover" in v else ""))


if __name__ == "__main__":
    main()
