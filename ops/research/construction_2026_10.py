#!/usr/bin/env python3
"""Portfolio-construction study on the CURRENT model's causal scores (no retraining).

Same signal, different books — so any difference is construction, not prediction:
  width        top-20 (production) vs top-30 / top-50
  size tilt    weights ∝ √(dollar volume) inside the picks instead of ½ inverse-vol + ½ equal
  overlays     production (200-day trend × ½, 25% vol brake) vs none
  core         50% SPY + 50% book, rebalanced monthly
  long-short   top-20 long / bottom-20 short, equal weight, dollar-neutral, monthly; 20 bp per unit
               turnover each side + 1%/yr borrow on the short leg (a simple, conservative cost model)

Run on the PIT window (whole-market point-in-time universe) and, when available, the long debiased
history (2009 →, top-500-by-liquidity-that-day eligibility).
"""
from __future__ import annotations

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
from trading_system.alpha import portfolio as PF  # noqa: E402
from trading_system.alpha.backtest import ColumnAlpha  # noqa: E402
from trading_system.alpha.portfolio import BookConfig, market_regime_on  # noqa: E402
from trading_system.research.wfbacktest import WalkForwardConfig, build_panel, run_walk_forward  # noqa: E402

OUT = LAB.OUT


def sqrt_dv_weights(book: BookConfig, sector_of: dict, regime: dict | None):
    """Like production target_weights but sizes the picks ∝ √(ADV) (tilts toward larger, more liquid names)."""
    def fn(scores, cols, cfg):
        sectors = np.array([sector_of.get(t, "unknown") for t in cols["ticker"]])
        w = PF.target_weights(scores, cols, replace(book, inv_vol_blend=0.0), sectors,
                              regime_on=True if regime is None else regime.get(cols.get("date"), True))
        pick = w > 0
        if pick.any():
            g = w.sum()
            s = np.sqrt(np.maximum(np.asarray(cols["adv"], dtype=float), 1.0)) * pick
            w = PF._cap(s / s.sum(), book.max_weight) * g
        return w
    return fn


def run_book(pdata, comp, gate, weight_fn, oos, top_k=20, max_w=0.08, rate=0.35):
    f = gate.join(comp, on=["date", "ticker"], how="left").with_columns(
        comp=pl.when(pl.col("elig")).then(pl.col("comp").cast(pl.Float64))).filter(pl.col("comp").is_not_null())
    wcfg = WalkForwardConfig(oos_start=oos, rebalance_days=21, retrain_days=10**6, min_train_days=5, horizon=21, top_k=top_k,
                             max_weight=max_w, min_dollar_volume=0.0, min_price=0.0, partial_trade_rate=rate,
                             respect_target_gross=True)
    return run_walk_forward(pdata, f, ["comp"], lambda: ColumnAlpha("comp"), wcfg, weight_fn=weight_fn, progress=False).returns()


def long_short(pdata, comp, gate, oos, k=20, cost_bps=20.0, borrow=0.01):
    """Daily returns of a monthly-rebalanced equal-weight top-k / bottom-k dollar-neutral book."""
    tick = {t: i for i, t in enumerate(pdata.tickers)}
    dates = [d for d in pdata.dates if d >= oos]
    di = pdata.date_index()
    sc = (gate.filter(pl.col("elig")).join(comp, on=["date", "ticker"]).drop_nulls("comp")
              .filter(pl.col("ticker").is_in(list(tick))))
    by = {d: g for (d,), g in sc.group_by(["date"])}
    w = np.zeros(len(pdata.tickers)); out = []
    for n, d in enumerate(dates):
        i = di[d]
        r = float((w * pdata.ret[i]).sum())
        cost = 0.0
        if n % 21 == 0 and d in by:
            g = by[d].sort("comp")
            if g.height >= 2 * k:
                new = np.zeros_like(w)
                for t in g.head(k)["ticker"]:
                    new[tick[t]] = -1.0 / k
                for t in g.tail(k)["ticker"]:
                    new[tick[t]] = 1.0 / k
                cost = np.abs(new - w).sum() * cost_bps / 1e4
                w = new
        out.append(r - cost - borrow / 252 * 1.0)   # short leg is 100% of equity
    return np.array(out), dates


def study(pw, pdata, comp, gate, sector_of, regime, oos, label) -> dict:
    bk = replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0)
    no_ovl = replace(bk, regime_scale=1.0, vol_target=None)
    from trading_system.alpha.backtest import _weight_fn
    res = {}
    res["top-20 (production)"] = run_book(pdata, comp, gate, _weight_fn(bk, sector_of, regime), oos)
    res["top-30"] = run_book(pdata, comp, gate, _weight_fn(replace(bk, top_k=30, max_weight=0.06), sector_of, regime), oos, 30, 0.06)
    res["top-50"] = run_book(pdata, comp, gate, _weight_fn(replace(bk, top_k=50, max_weight=0.04, max_per_sector=12), sector_of, regime), oos, 50, 0.04)
    res["top-20 · size tilt √ADV"] = run_book(pdata, comp, gate, sqrt_dv_weights(bk, sector_of, regime), oos)
    res["top-20 · no overlays"] = run_book(pdata, comp, gate, _weight_fn(no_ovl, sector_of, None), oos)
    n = len(res["top-20 (production)"])
    dates = [d for d in pdata.dates if d >= oos][:n]
    tick = list(pdata.tickers); di = pdata.date_index()
    spy = np.array([pdata.ret[di[d], tick.index("SPY")] for d in dates]) if "SPY" in tick else None
    if spy is not None:
        res["50% SPY + 50% top-20"] = 0.5 * spy + 0.5 * res["top-20 (production)"][:len(spy)]
        res["passive: SPY"] = spy
    ls, _ = long_short(pdata, comp, gate, oos)
    res["long-short 20/20 (market-neutral)"] = ls[:n]
    out = {k: LAB.book_stats(v) for k, v in res.items()}
    if spy is not None:
        for k, v in res.items():
            m = min(len(v), len(spy))
            b = np.cov(v[:m], spy[:m])[0, 1] / np.var(spy[:m])
            out[k]["beta"] = float(b)
            out[k]["alpha"] = float((v[:m].mean() - b * spy[:m].mean()) * 252)
    out["_window"] = [str(dates[0]), str(dates[-1])]
    LAB.log(f"{label} construction study done")
    return out


def main():
    results = {}
    # PIT window
    pp = OUT / "pit_panel_batch1.parquet"
    pw = pl.read_parquet(pp if pp.exists() else OUT / "pit_panel.parquet")
    _, allpx, _ = LAB.pit_panel()
    pdata = build_panel(P.sanitize_prices(allpx))
    comp = LAB.zcomp(pl.read_parquet(OUT / "pit_base.parquet"))
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    gate = pw.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07)
                     & (pl.col("log_dv_21") >= np.log1p(20e6)))
    oos = sorted(pw["date"].unique().to_list())[3]
    results["pit"] = study(pw, pdata, comp, gate, sector_of, regime, oos, "PIT")
    (OUT / "construction.json").write_text(json.dumps(results, indent=1, default=str))
    # LONG debiased window (needs the lab's long_base scores)
    lb = OUT / "long_base.parquet"
    if lb.exists():
        pn = P.load_panel(LAB.cfg)
        frame, _ = M.prepare(pn, LAB.BASE_FEATS)
        prices = P.load_prices(LAB.cfg, start="2006-01-01")
        pdl = build_panel(prices)
        sector_l = dict(pn.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
        reg_l = {d: market_regime_on(v) for d, v in pn.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
        gate_l = (frame.select("date", "ticker", "adj_close", "close", "vol_63", "liq_rank")
                       .with_columns(elig=(pl.col("close") >= 5) & (pl.col("liq_rank") <= LAB.EVAL_TOP) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07))
                       .select("date", "ticker", "adj_close", "elig"))
        results["long"] = study(pn, pdl, LAB.zcomp(pl.read_parquet(lb)), gate_l, sector_l, reg_l, date(2009, 1, 1), "LONG")
        (OUT / "construction.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
