#!/usr/bin/env python3
"""Free fixes for the survivor-trained model, on the honest point-in-time test (2026-10-10).

With complete data the clean test failed (RESEARCH_2026-10 §12): trained only on today's survivors, the
model buys beaten-down names that later fail. Whole-market history would fix the training data but costs
money; these fixes cost nothing. All settings were fixed before the first run:

  monotone   XGBoost monotone constraints with signs from published anomalies, so survivor-only data can't
             flip them: momentum (12-1) +, volatility −, idiosyncratic volatility −, MAX (lottery) −, short
             interest and days-to-cover −, gross profitability / ROA / ROE +, accruals −, asset growth −, SUE +.
  pit-aug    point-in-time training where we have it: from the PIT panel's start (Nov 2024) the training rows
             come from the whole-market point-in-time universe — failures included — instead of survivors;
             before that, survivors (nothing else exists for free). Purged/embargoed exactly like production.
  both       monotone + pit-aug.
  momentum   no model: 12-1 momentum as the score, same book rules.     low-vol: −63-day volatility.

Each with the production book (top-20, gates, overlays, partial trading) and with a point-in-time large-cap
restriction (the 300 most liquid names that day). The panel was inspected before this run (that is how the
failure was found), so a fix that looks good here still has to prove itself live before adoption.
Results → reports/alpha/lab_2026_10/pit_fix.json
"""
from __future__ import annotations

import json
import sys
import time
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
from trading_system.alpha.portfolio import market_regime_on  # noqa: E402
from trading_system.research.wfbacktest import build_panel  # noqa: E402

OUT = LAB.OUT
MONO = (("mom_12_1", 1), ("vol_63", -1), ("idio_vol_63", -1), ("max_ret_21", -1), ("si_ratio", -1),
        ("si_days_to_cover", -1), ("gp_assets", 1), ("roa_ttm", 1), ("roe_ttm", 1), ("accruals", -1),
        ("asset_growth", -1), ("sue", 1))
VARIANTS = {"monotone": (M.TrainSpec(horizons=LAB.H, monotone=MONO), False),
            "pit-aug": (M.TrainSpec(horizons=LAB.H), True),
            "monotone+pit-aug": (M.TrainSpec(horizons=LAB.H, monotone=MONO), True)}


def scores(name: str, spec: M.TrainSpec, pit_aug: bool, pw: pl.DataFrame, hist: pl.DataFrame) -> pl.DataFrame:
    path = OUT / f"pitfix_{name.replace('+', '_')}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    t0 = time.time()
    cal = sorted(pw["date"].unique().to_list())
    cuts = cal[3::63]
    pit_start = cal[0]
    hf, rcols = M.prepare(hist, spec.features)
    wf, _ = M.prepare(pw, spec.features)
    hd = hf.select("date", "didx").unique().sort("didx")
    wd = wf.select("date", "didx").unique().sort("didx")
    parts = []
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else date(2100, 1, 1)
        ci = int(hd.filter(pl.col("date") <= cut)["didx"][-1])
        wi = int(wd.filter(pl.col("date") <= cut)["didx"][-1])
        blk = wf.filter((pl.col("date") >= cut) & (pl.col("date") < nxt))
        cols = {"date": blk["date"], "ticker": blk["ticker"]}
        X = blk.select(rcols).to_numpy().astype(np.float32)
        for h in LAB.H:
            tr = M.training_rows(hf, h, ci, spec)
            if pit_aug:
                tr = tr.filter(pl.col("date") < pit_start)
            Xt, yt = M._xy(tr, rcols, h, spec.label)
            wt = M.sample_weights(tr, ci, spec)
            if pit_aug:
                tw = M.training_rows(wf, h, wi, spec)        # PIT rows whose labels matured by the cut
                if tw.height:
                    X2, y2 = M._xy(tw, rcols, h, spec.label)
                    Xt, yt = np.vstack([Xt, X2]), np.concatenate([yt, y2])
                    w2 = M.sample_weights(tw, wi, spec)
                    wt = None if wt is None else np.concatenate([wt, w2])
            m = M.AlphaGBM(spec, h, rcols).fit(Xt, yt, wt)
            cols[f"s{h}"] = m.predict(X)
        parts.append(pl.DataFrame(cols))
    s = pl.concat(parts).unpivot(index=["date", "ticker"], on=[f"s{h}" for h in LAB.H], variable_name="horizon", value_name="score")
    s = s.with_columns(pl.col("horizon").str.slice(1).cast(pl.Int32))
    s.write_parquet(path)
    LAB.log(f"{name}: {s.height:,} scores in {(time.time() - t0) / 60:.1f} min")
    return s


def main():
    pw, allpx, _ = LAB.pit_panel()
    hist = P.load_panel(LAB.cfg)
    pdata = build_panel(P.sanitize_prices(allpx))
    cal = sorted(pw["date"].unique().to_list())
    oos = cal[3::63][0]
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    base_gate = (pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07)
    liq = pw.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
    gates = {"production universe": liq.select("date", "ticker", "adj_close", elig=base_gate & (pl.col("log_dv_21") >= np.log1p(20e6))),
             "top-300 that day": liq.select("date", "ticker", "adj_close", elig=base_gate & (pl.col("lr") <= 300))}
    labels = pw.select("date", "ticker", "fwd_21", "fwd_63")
    comps = {"production model": LAB.zcomp(pl.read_parquet(OUT / "pit_base.parquet"))}
    ics = {"production model": LAB.ic_summary(LAB.ic_table(pl.read_parquet(OUT / "pit_base.parquet"), labels))}
    for name, (spec, aug) in VARIANTS.items():
        sc = scores(name, spec, aug, pw, hist)
        comps[name], ics[name] = LAB.zcomp(sc), LAB.ic_summary(LAB.ic_table(sc, labels))
    comps["momentum 12-1 (no model)"] = pw.select("date", "ticker", comp=pl.col("mom_12_1").cast(pl.Float64)).drop_nulls("comp")
    comps["low volatility (no model)"] = pw.select("date", "ticker", comp=-pl.col("vol_63").cast(pl.Float64)).drop_nulls("comp")
    res, rets = {"ic": ics, "books": {}}, {}
    for gname, gate in gates.items():
        for cname, comp in comps.items():
            r = LAB.book_run(pdata, comp, gate, sector_of, regime, oos, cname).returns()
            rets[(gname, cname)] = r
            res["books"][f"{cname} · {gname}"] = LAB.book_stats(r)
    base = rets[("production universe", "production model")]
    res["excess_vs_production"] = {f"{c} · {g}": LAB.excess_ci(r, base) for (g, c), r in rets.items() if (g, c) != ("production universe", "production model")}
    tick = list(pdata.tickers); di = pdata.date_index()
    dates = [d for d in pdata.dates if d >= oos][:len(base)]
    for t in ("SPY", "QQQ", "RSP"):
        if t in tick:
            res["books"][f"passive: {t}"] = LAB.book_stats(np.array([pdata.ret[di[d], tick.index(t)] for d in dates]))
    (OUT / "pit_fix.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res["books"].items():
        ex = res["excess_vs_production"].get(k)
        LAB.log(f"{k:<52} CAGR {v['cagr']:+6.1%}  Sharpe {v['sharpe']:5.2f}  MaxDD {v['maxdd']:+6.1%}"
                + (f"  vs production {ex['excess']:+.1%}/yr [{ex['lo']:+.1%}, {ex['hi']:+.1%}]" if ex else ""))
    for k, v in ics.items():
        LAB.log(f"IC {k:<20} 21d {v[21]['ic']:+.4f} (t {v[21]['t']:+.2f})  63d {v[63]['ic']:+.4f} (t {v[63]['t']:+.2f})")


if __name__ == "__main__":
    main()
