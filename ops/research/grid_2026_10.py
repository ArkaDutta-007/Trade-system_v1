#!/usr/bin/env python3
"""Book-parameter grid, 2004 → 2026, production scores + production overlay, debiased eligibility.

Knobs: top_k × max_weight × rebalance days × partial-trade rate × eligible universe size. The point is
NOT to pick the best cell — that is how backtests get overfit — but to see whether production sits on a
stable plateau, and to measure the Probability of Backtest Overfitting (CSCV, Bailey et al. 2017) of
choosing by in-sample Sharpe across the grid.
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import lab_2026_10 as LAB  # noqa: E402
from trading_system.alpha import model as M  # noqa: E402
from trading_system.alpha import panel as P  # noqa: E402
from trading_system.alpha.backtest import ColumnAlpha, _weight_fn  # noqa: E402
from trading_system.alpha.portfolio import BookConfig, market_regime_on  # noqa: E402
from trading_system.research.stats import deflated_sharpe, pbo_cscv, sharpe  # noqa: E402
from trading_system.research.wfbacktest import WalkForwardConfig, build_panel, run_walk_forward  # noqa: E402

GRID = {"top_k": [10, 15, 20, 30], "max_w": [0.06, 0.08, 0.12], "reb": [10, 21, 42], "rate": [0.2, 0.35, 0.6, 1.0],
        "elig_top": [300, 500, 800]}


def main():
    pn = P.load_panel(LAB.cfg)
    frame, _ = M.prepare(pn, LAB.BASE_FEATS)
    sc = pl.read_parquet(LAB.cfg.path("data_bronze").parent / "ledger" / "alpha_backtest_scores.parquet")
    comp = LAB.zcomp(sc.filter(pl.col("horizon").is_in(list(LAB.H))))
    pdata = build_panel(P.load_prices(LAB.cfg, start="2002-01-01"))
    sector_of = dict(pn.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pn.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    base = frame.select("date", "ticker", "adj_close", "close", "vol_63", "liq_rank").join(comp, on=["date", "ticker"], how="left")
    feats = {}
    for et in GRID["elig_top"]:
        feats[et] = (base.with_columns(elig=(pl.col("close") >= 5) & (pl.col("liq_rank") <= et) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07))
                         .with_columns(comp=pl.when(pl.col("elig")).then(pl.col("comp").cast(pl.Float64)))
                         .filter(pl.col("comp").is_not_null()).select("date", "ticker", "adj_close", "comp"))
    rows, rets = [], []
    combos = list(itertools.product(*GRID.values()))
    for i, (k, mw, reb, rate, et) in enumerate(combos):
        if mw * k < 1.0:                      # cap cannot hold a fully invested book
            continue
        bk = replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0, top_k=k, max_weight=mw)
        wcfg = WalkForwardConfig(oos_start=date(2004, 1, 2), rebalance_days=reb, retrain_days=10**6, min_train_days=5, horizon=21,
                                 top_k=k, max_weight=mw, min_dollar_volume=0.0, min_price=0.0, partial_trade_rate=rate,
                                 respect_target_gross=True)
        r = run_walk_forward(pdata, feats[et], ["comp"], lambda: ColumnAlpha("comp"), wcfg,
                             weight_fn=_weight_fn(bk, sector_of, regime), progress=False)
        x = r.returns()
        eq = np.cumprod(1 + x); y = len(x) / 252
        rows.append({"top_k": k, "max_w": mw, "reb": reb, "rate": rate, "elig_top": et, "cagr": float(eq[-1] ** (1 / y) - 1),
                     "sharpe": sharpe(x), "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                     "turnover": float(r.daily["turnover"].sum() / y)})
        rets.append(x)
        if i % 20 == 0:
            LAB.log(f"grid {i + 1}/{len(combos)}")
    n = min(len(x) for x in rets)
    R = np.column_stack([x[-n:] for x in rets])
    pbo = pbo_cscv(R, n_blocks=16)
    trials = np.array([sharpe(R[:, j], annualise=False) for j in range(R.shape[1])])
    best = int(np.argmax(trials))
    prod = next(j for j, r in enumerate(rows) if (r["top_k"], r["max_w"], r["reb"], r["rate"], r["elig_top"]) == (20, 0.08, 21, 0.35, 500))
    out = {"rows": rows, "pbo": pbo, "best": rows[best], "production": rows[prod],
           "dsr_best": deflated_sharpe(R[:, best], trials), "dsr_production": deflated_sharpe(R[:, prod], trials)}
    (LAB.OUT / "grid.json").write_text(json.dumps(out, indent=1, default=str))
    LAB.log(f"grid done: {len(rows)} configs · PBO {pbo['pbo']:.2f}")


if __name__ == "__main__":
    main()
