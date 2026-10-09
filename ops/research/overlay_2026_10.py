#!/usr/bin/env python3
"""Overlay design study, 2004 → 2026, production model scores, debiased eligibility (top-500 by
liquidity that day). Same signal and book in every row — only the exposure rule changes.

Trend signals are computed on SPY (cap-weighted, no survivorship) — the production overlay uses the
panel's equal-weight index of today's survivors, which is itself biased upward.
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
from trading_system.alpha.backtest import ColumnAlpha, _weight_fn  # noqa: E402
from trading_system.alpha.portfolio import BookConfig, market_regime_on  # noqa: E402
from trading_system.research.wfbacktest import WalkForwardConfig, build_panel, run_walk_forward  # noqa: E402

OOS = date(2004, 1, 2)
EPISODES = [("GFC 2007-09", date(2007, 10, 9), date(2009, 3, 9)), ("Rebound 2009", date(2009, 3, 9), date(2009, 12, 31)),
            ("2011 euro", date(2011, 4, 29), date(2011, 10, 3)), ("2015-16 whipsaw", date(2015, 5, 21), date(2016, 6, 30)),
            ("Q4 2018", date(2018, 9, 20), date(2018, 12, 24)), ("Covid crash", date(2020, 2, 19), date(2020, 3, 23)),
            ("Covid rebound", date(2020, 3, 23), date(2020, 8, 31)), ("2022 bear", date(2022, 1, 3), date(2022, 10, 12)),
            ("2025 tariff + rebound", date(2025, 2, 19), date(2025, 7, 31))]


def spy_signals(dates: list[date]) -> pl.DataFrame:
    s = (pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet", columns=["date", "ticker", "adj_close"])
           .filter(pl.col("ticker") == "SPY").sort("date").with_columns(pl.col("date").cast(pl.Date)))
    s = s.with_columns(sma200=pl.col("adj_close").rolling_mean(200), sma50=pl.col("adj_close").rolling_mean(50),
                       hi252=pl.col("adj_close").rolling_max(252), vol21=(pl.col("adj_close").pct_change().rolling_std(21) * 252 ** 0.5))
    s = s.with_columns(trend=pl.col("adj_close") / pl.col("sma200") - 1, above50=pl.col("adj_close") >= pl.col("sma50"),
                       dd=pl.col("adj_close") / pl.col("hi252") - 1)
    # every signal lagged one session: decided on yesterday's close, known before today's open
    return s.with_columns([pl.col(c).shift(1) for c in ("trend", "above50", "dd", "vol21")]).select("date", "trend", "above50", "dd", "vol21")


def rules(sig: pl.DataFrame) -> dict[str, tuple[dict | None, dict | None]]:
    """name → (regime map date→bool for the binary ½ overlay, gross-multiplier map date→float)."""
    rows = sig.drop_nulls("trend").iter_rows(named=True)
    rows = list(rows)
    trend_on = {r["date"]: r["trend"] >= 0 for r in rows}
    confirm = {r["date"]: not (r["trend"] < 0 and r["dd"] < -0.10) for r in rows}
    # asymmetric: leave when SPY < 200d, come back as soon as SPY > 50d
    asym, state = {}, True
    for r in rows:
        if state and r["trend"] < 0:
            state = False
        elif not state and (r["above50"] or r["trend"] >= 0):
            state = True
        asym[r["date"]] = state
    graded = {r["date"]: float(np.clip(1 + 5 * r["trend"], 0.5, 1.0)) for r in rows}
    volt = {r["date"]: float(np.clip(0.18 / max(r["vol21"] or 0.18, 1e-6), 0.4, 1.0)) for r in rows}
    return {"none": (None, None), "trend (SPY 200d, ½)": (trend_on, None), "trend + confirm (−10% DD)": (confirm, None),
            "trend, fast re-entry (50d)": (asym, None), "graded trend (½ at −10%)": (None, graded),
            "SPY realised-vol target 18%": (None, volt)}


def stats(r: np.ndarray, dates: list[date]) -> dict:
    out = LAB.book_stats(r)
    out["calmar"] = out["cagr"] / abs(out["maxdd"]) if out["maxdd"] < 0 else None
    d = np.array(dates)
    out["episodes"] = {n: float(np.prod(1 + r[(d >= a) & (d <= b)]) - 1) for n, a, b in EPISODES if ((d >= a) & (d <= b)).sum() > 3}
    return out


def main():
    pn = P.load_panel(LAB.cfg)
    frame, _ = M.prepare(pn, LAB.BASE_FEATS)
    sc = pl.read_parquet(LAB.cfg.path("data_bronze").parent / "ledger" / "alpha_backtest_scores.parquet")
    comp = LAB.zcomp(sc.filter(pl.col("horizon").is_in(list(LAB.H))))
    prices = P.load_prices(LAB.cfg, start="2002-01-01")
    pdata = build_panel(prices)
    sector_of = dict(pn.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    gate = (frame.select("date", "ticker", "adj_close", "close", "vol_63", "liq_rank")
                 .with_columns(elig=(pl.col("close") >= 5) & (pl.col("liq_rank") <= LAB.EVAL_TOP) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07))
                 .select("date", "ticker", "adj_close", "elig"))
    f = gate.join(comp, on=["date", "ticker"], how="left").with_columns(
        comp=pl.when(pl.col("elig")).then(pl.col("comp").cast(pl.Float64))).filter(pl.col("comp").is_not_null())
    bk = replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0, vol_target=None, regime_scale=0.5)
    wcfg = WalkForwardConfig(oos_start=OOS, rebalance_days=21, retrain_days=10**6, min_train_days=5, horizon=21, top_k=20,
                             max_weight=0.08, min_dollar_volume=0.0, min_price=0.0, partial_trade_rate=0.35, respect_target_gross=True)
    sig = spy_signals(list(pdata.dates))
    res, dates = {}, None
    panel_trend = {d: market_regime_on(v) for d, v in pn.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    variants = rules(sig)
    variants["PRODUCTION (EW-index trend ½ + 25% vol brake)"] = ("prod", None)
    for name, (reg, mult) in variants.items():
        if reg == "prod":
            wf = _weight_fn(replace(bk, vol_target=0.25), sector_of, panel_trend)
        else:
            wf = _weight_fn(bk, sector_of, reg, mult)
        r = run_walk_forward(pdata, f, ["comp"], lambda: ColumnAlpha("comp"), wcfg, weight_fn=wf, label=name, progress=False)
        dates = r.daily["date"].to_list()
        res[name] = stats(r.returns(), dates)
        LAB.log(f"overlay {name}: CAGR {res[name]['cagr']:+.1%} MaxDD {res[name]['maxdd']:+.1%}")
    tick = list(pdata.tickers); di = pdata.date_index()
    spy = np.array([pdata.ret[di[d], tick.index("SPY")] for d in dates])
    res["passive: SPY"] = stats(spy, dates)
    (LAB.OUT / "overlays.json").write_text(json.dumps({"window": [str(dates[0]), str(dates[-1])], "results": res}, indent=1, default=str))


if __name__ == "__main__":
    main()
