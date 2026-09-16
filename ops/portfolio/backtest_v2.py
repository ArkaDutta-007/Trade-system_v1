#!/usr/bin/env python3
"""Historical backtest of the picks-v2 selection rules vs the raw ranking.

The v2 gates were designed from a single live cross-section (2026-09-04). This
answers the only question that matters: would those rules have helped over the
last decade, or are they a story fitted to one day?

Method
  - Start from the walk-forward OOS predictions (data/gold/predictions.parquet,
    honest purged scores from the deployed ensemble).
  - Re-run the *selection* rules historically, month by month, using only data
    available at each rebalance date (features are as-of that date; the sector
    map is static metadata, not a forecast).
  - Compare variants that isolate ONE rule at a time, so any improvement is
    attributable:
        raw            top-k by score                     (today's `ts picks`)
        liq            + price/dollar-volume/vol gates
        liq_ra         + risk adjustment (score / vol^a)
        liq_ra_div     + theme diversification            (full picks v2)
        equal_spy      SPY buy & hold                     (the bar)
  - Monthly rebalance, equal weight, costs = 4bps flat + sqrt impact, matching
    the repo's cost model and the dummy books.

Usage:  python3 backtest_v2.py [--top 10] [--start 2015-01-01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_repo_to_path, ops_root  # noqa: E402

REPO = add_repo_to_path()
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402

from picks_v2 import (MIN_PRICE, MIN_DOLLAR_VOL, MAX_VOL_20D, VOL_ALPHA,  # noqa: E402
                      MAX_PER_THEME)
from sectors import load as load_sectors, theme_of  # noqa: E402

COST_BPS, IMPACT_BPS, ANN = 4.0, 10.0, 252


def load_panel() -> pl.DataFrame:
    preds = REPO / "data/gold/predictions.parquet"
    if not preds.exists():
        raise SystemExit("no data/gold/predictions.parquet — run the retrain first")
    p = pl.read_parquet(preds)
    f = pl.read_parquet(REPO / "data/gold/features.parquet",
                        columns=["date", "ticker", "close", "vol_20d",
                                 "avg_dollar_volume_20"])
    px = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet",
                         columns=["date", "ticker", "adj_close"])
    return p.join(f, on=["date", "ticker"], how="inner"), px


def select_month(day_df: pl.DataFrame, variant: str, top: int, meta: dict) -> list[str]:
    d = day_df
    if variant != "raw":
        d = d.filter((pl.col("close") >= MIN_PRICE)
                     & (pl.col("avg_dollar_volume_20") >= MIN_DOLLAR_VOL)
                     & (pl.col("vol_20d") <= MAX_VOL_20D))
    if d.is_empty():
        return []
    if variant in ("liq_ra", "liq_ra_div"):
        d = d.with_columns(rank_score=pl.col("score")
                           / pl.col("vol_20d").clip(0.05, None) ** VOL_ALPHA)
    else:
        d = d.with_columns(rank_score=pl.col("score"))
    d = d.sort("rank_score", descending=True)
    if variant != "liq_ra_div":
        return d.head(top)["ticker"].to_list()
    picked, themes = [], {}
    for t in d["ticker"].to_list():
        th = theme_of(t, meta)
        if themes.get(th, 0) >= MAX_PER_THEME:
            continue
        themes[th] = themes.get(th, 0) + 1
        picked.append(t)
        if len(picked) >= top:
            break
    return picked


def run_variant(panel: pl.DataFrame, px: pl.DataFrame, variant: str,
                top: int, meta: dict) -> np.ndarray:
    dates = sorted(set(panel["date"].to_list()))
    # month-start rebalance dates
    rebal, seen = [], set()
    for d in dates:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            rebal.append(d)
    px_wide = px.pivot(index="date", on="ticker", values="adj_close").sort("date")
    all_days = px_wide["date"].to_list()
    tick_cols = [c for c in px_wide.columns if c != "date"]
    arr = px_wide.select(tick_cols).to_numpy()
    col = {t: i for i, t in enumerate(tick_cols)}
    day_i = {d: i for i, d in enumerate(all_days)}

    holdings: dict[str, float] = {}
    equity, eq = [], 10_000.0
    prev_val = None
    for i, d in enumerate(all_days):
        if d < dates[0]:
            continue
        row = arr[i]
        # mark
        if holdings:
            val = sum(q * row[col[t]] for t, q in holdings.items()
                      if not np.isnan(row[col[t]]))
            if prev_val:
                eq *= val / prev_val
            prev_val = val
        equity.append(eq)
        if d in rebal:
            day_df = panel.filter(pl.col("date") == d)
            if day_df.is_empty():
                continue
            names = [t for t in select_month(day_df, variant, top, meta)
                     if t in col and not np.isnan(row[col[t]])]
            if not names:
                continue
            old = set(holdings)
            turn = 1.0 if not old else len(set(names) ^ old) / max(len(names | old if isinstance(names, set) else set(names) | old), 1)
            cost = (COST_BPS * turn + IMPACT_BPS * turn ** 1.5) / 10_000.0
            eq *= (1 - cost)
            per = eq / len(names)
            holdings = {t: per / row[col[t]] for t in names}
            prev_val = sum(q * row[col[t]] for t, q in holdings.items())
    return np.array(equity)


def spy_curve(px: pl.DataFrame, start) -> np.ndarray:
    s = px.filter((pl.col("ticker") == "SPY") & (pl.col("date") >= start)).sort("date")
    v = s["adj_close"].to_numpy()
    return 10_000.0 * v / v[0] if len(v) else np.array([])


def stats(eq: np.ndarray) -> dict:
    if len(eq) < 3:
        return {}
    r = np.diff(eq) / eq[:-1]
    yrs = len(eq) / ANN
    peak = np.maximum.accumulate(eq)
    return {"CAGR": (eq[-1] / eq[0]) ** (1 / yrs) - 1,
            "Sharpe": float(np.mean(r) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(ANN)),
            "MaxDD": float(((eq - peak) / peak).min()),
            "Final": eq[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--start", default="2015-01-01")
    a = ap.parse_args()

    panel, px = load_panel()
    panel = panel.filter(pl.col("date") >= pl.lit(a.start).str.to_date())
    meta = load_sectors()
    print(f"panel {panel.height:,} rows · {panel['date'].min()} → {panel['date'].max()}"
          f" · top-{a.top} · monthly rebalance\n")

    out = {}
    for v in ("raw", "liq", "liq_ra", "liq_ra_div"):
        eq = run_variant(panel, px, v, a.top, meta)
        out[v] = stats(eq)
        print(f"  {v:<12} done ({len(eq)} sessions)", flush=True)
    sp = spy_curve(px, panel["date"].min())
    out["spy_hold"] = stats(sp)

    print(f"\n{'variant':<12}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Final $':>11}")
    for k, s in out.items():
        if s:
            print(f"{k:<12}{s['CAGR']:>8.1%}{s['Sharpe']:>9.2f}"
                  f"{s['MaxDD']:>8.1%}{s['Final']:>11,.0f}")
    Path(__file__).parent.joinpath("backtest_v2_report.json").write_text(
        json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
