#!/usr/bin/env python3
"""Designs from trading-firm competitions, tested on the honest panel (2026-11-13, RESEARCH_2026-11 §10).

Winning entries in Jane Street's, Optiver's, G-Research's and Numerai's contests share a few designs that can
be tried here without new data. Base = P-xgb-debias from panel_2026_11 (point-in-time rows + top-500 training,
refit every 63 sessions). Everything below was fixed before the first run:

  refit21 / refit5   online learning: the largest single gain in Jane Street 2024. Refit every 21 or every 5
                     sessions instead of 63 (production retrains weekly, so refit5 is also the fairer test).
  hl2y               recency weighting: sample-weight half-life 2 years instead of 10.
  neut50 / neut100   Numerai-style neutralisation of the base scores: per day, remove 50% / 100% of the part
                     explained by known risk exposures (beta, 63-day volatility, size, 12-1 momentum, 21-day
                     return, sector), so the book is not a hidden bet on those.
  tranches           Newfound's fix for rebalance-timing luck: hold 4 equal sub-books rebalanced 5 sessions
                     apart. Reported for every variant as the mean of the 4 start-day return series.

Books: (a) index-minus = that day's 300 most liquid names, cap-weighted, minus the worst 30% by score, monthly,
10 bps; (b) the production top-20 book inside the top-300. Each on start days 0/5/10/15 and as 4 tranches.
Adoption rule, set in advance: a variant replaces the base in the challenger only if its 21d and 63d IC is no
lower than the base in EITHER year and its tranched index-minus excess is higher. Otherwise the base stays.
Results → reports/alpha/lab_2026_11/comp.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import panel_2026_11 as PP  # noqa: E402
import tilts_2026_11 as TT  # noqa: E402

LAB, OUT, P, M = PP.LAB, PP.OUT, PP.P, PP.M
H = LAB.H
OFFSETS = (0, 5, 10, 15)
RISK = ("beta_63", "vol_63", "log_mcap", "mom_12_1", "ret_21")
WALKS = {"hl2y": (63, M.TrainSpec(horizons=H, train_top=500, half_life_days=504.0)),
         "refit21": (21, M.TrainSpec(horizons=H, train_top=500)),
         "refit5": (5, M.TrainSpec(horizons=H, train_top=500))}


def walk(name: str, refit: int, spec, pw, hist) -> None:
    path = OUT / f"wf_comp_{name}.parquet"
    if path.exists():
        return
    t0 = time.time()
    cal = sorted(pw["date"].unique().to_list())
    cuts, pit_start = cal[3::refit], cal[0]
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
        for h in H:
            tr = M.training_rows(hf, h, ci, spec).filter(pl.col("date") < pit_start)
            Xt, yt = M._xy(tr, rcols, h, spec.label)
            wt = M.sample_weights(tr, ci, spec)
            tw = M.training_rows(wf, h, wi, spec)
            if tw.height:
                X2, y2 = M._xy(tw, rcols, h, spec.label)
                Xt, yt = np.vstack([Xt, X2]), np.concatenate([yt, y2])
                wt = np.concatenate([wt, M.sample_weights(tw, wi, spec)])
            cols[f"s{h}"] = np.asarray(M.AlphaGBM(spec, h, rcols).fit(Xt, yt, wt).predict(X), dtype=np.float64)
        parts.append(pl.DataFrame(cols))
        if (i + 1) % 8 == 0 or i + 1 == len(cuts):
            LAB.log(f"  {name} fold {i + 1}/{len(cuts)} · {(time.time() - t0) / 60:.1f} min")
    s = pl.concat(parts).unpivot(index=["date", "ticker"], on=[f"s{h}" for h in H], variable_name="horizon", value_name="score")
    s.with_columns(pl.col("horizon").str.slice(1).cast(pl.Int32)).write_parquet(path)


def neutralize(sc: pl.DataFrame, pw: pl.DataFrame, prop: float) -> pl.DataFrame:
    """Per date and horizon: score − prop × (its least-squares projection on the risk exposures)."""
    ex = pw.select("date", "ticker", "sector", *[PP.z(c).alias(c) for c in RISK])
    j = sc.join(ex, on=["date", "ticker"], how="inner")
    out = []
    for (d, h), g in j.group_by(["date", "horizon"]):
        s = g["score"].to_numpy().astype(np.float64)
        s = (s - s.mean()) / (s.std() + 1e-12)
        sec = g["sector"].to_numpy()
        dum = np.column_stack([(sec == u).astype(np.float64) for u in np.unique(sec)])
        X = np.column_stack([g.select(RISK).to_numpy().astype(np.float64), dum])
        s2 = s - prop * (X @ (np.linalg.pinv(X) @ s))
        out.append(pl.DataFrame({"date": g["date"], "ticker": g["ticker"], "horizon": g["horizon"], "score": s2 / (s2.std() + 1e-12)}))
    return pl.concat(out)


def evaluate():
    from trading_system.alpha.portfolio import market_regime_on
    from trading_system.research.wfbacktest import build_panel
    pw, allpx, _ = LAB.pit_panel()
    base = pl.read_parquet(OUT / "wf_P-xgb-debias.parquet")
    raw = {"base": base}
    for n in WALKS:
        if (OUT / f"wf_comp_{n}.parquet").exists():
            raw[n] = pl.read_parquet(OUT / f"wf_comp_{n}.parquet")
    raw["neut50"], raw["neut100"] = neutralize(base, pw, .5), neutralize(base, pw, 1.0)
    labels = pw.select("date", "ticker", "fwd_21", "fwd_63")
    comps = {k: LAB.zcomp(v) for k, v in raw.items()}
    res = {"ic": {k: PP.ic_year(v, labels) for k, v in raw.items()}, "index_minus": {}, "top20": {}}
    # exposure check: rank correlation of each composite with the risk columns
    ex = pw.select("date", "ticker", *RISK)
    res["exposure"] = {k: {c: float(v.join(ex, on=["date", "ticker"]).group_by("date").agg(r=pl.corr("comp", c, method="spearman"))["r"].mean())
                           for c in RISK} for k, v in comps.items()}
    px = allpx.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1)
    R = {(d, t): r for d, t, r in px.select("date", "ticker", "r").iter_rows() if r is not None}
    start = max(c["date"].min() for c in comps.values())
    cal = sorted(pw["date"].unique().to_list())
    days = [d for d in cal if d > start]
    rets_of = {"days": days, "R": R}
    y1 = np.array([d <= PP.Y1_END for d in days])
    cagr = lambda r: float(np.prod(1 + r) ** (252 / len(r)) - 1)
    ctrl = []
    for off in OFFSETS:
        first, prev, by = TT.month_book(pw, comps["base"].filter(pl.col("date") >= start), off)
        ctrl.append(TT.run(rets_of, first, prev, by, "all", "cap")[0])
    ctrl_tr = np.mean(ctrl, axis=0)
    res["index_minus"]["control"] = {"by_day": [cagr(r) for r in ctrl], "tranched": {**LAB.book_stats(ctrl_tr), "y1": float(np.prod(1 + ctrl_tr[y1]) - 1), "y2": float(np.prod(1 + ctrl_tr[~y1]) - 1)}}
    for k, c in comps.items():
        ser, tv = [], []
        for off in OFFSETS:
            first, prev, by = TT.month_book(pw, c.filter(pl.col("date") >= start), off)
            r, t = TT.run(rets_of, first, prev, by, "drop30", "cap")
            ser.append(r); tv.append(t)
        tr = np.mean(ser, axis=0)
        res["index_minus"][k] = {"by_day": [cagr(r) for r in ser], "day_sd": float(np.std([cagr(r) for r in ser])), "turnover": float(np.mean(tv)),
                                 "tranched": {**LAB.book_stats(tr), "y1": float(np.prod(1 + tr[y1]) - 1), "y2": float(np.prod(1 + tr[~y1]) - 1)},
                                 "tranched_vs_control": LAB.excess_ci(tr, ctrl_tr),
                                 "single_vs_control": [LAB.excess_ci(a, b)["excess"] for a, b in zip(ser, ctrl)]}
    pdata = build_panel(P.sanitize_prices(allpx))
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    liq = pw.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
    gate = liq.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07) & (pl.col("lr") <= 300))
    for k, c in comps.items():
        ser = [LAB.book_run(pdata, c, gate, sector_of, regime, cal[3 + off], k).returns() for off in OFFSETS]
        n = min(len(r) for r in ser)
        tr = np.mean([r[-n:] for r in ser], axis=0)
        res["top20"][k] = {"by_day": [cagr(r) for r in ser], "day_sd": float(np.std([cagr(r) for r in ser])), "tranched": LAB.book_stats(tr)}
    (OUT / "comp.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res["ic"].items():
        print(f"IC {k:<8} 21d y1 {v['y1'][21]['ic']:+.3f} y2 {v['y2'][21]['ic']:+.3f} | 63d y1 {v['y1'][63]['ic']:+.3f} y2 {v['y2'][63]['ic']:+.3f} | exposure " + " ".join(f"{c}:{x:+.2f}" for c, x in res["exposure"][k].items()))
    c0 = res["index_minus"]["control"]
    print(f"index control   days {' '.join(f'{x:+.1%}' for x in c0['by_day'])} | tranched {c0['tranched']['cagr']:+.1%} y1 {c0['tranched']['y1']:+.1%} y2 {c0['tranched']['y2']:+.1%} Sh {c0['tranched']['sharpe']:.2f}")
    for k in comps:
        v, t = res["index_minus"][k], res["index_minus"][k]["tranched"]
        e = v["tranched_vs_control"]
        print(f"index-minus {k:<8} days {' '.join(f'{x:+.1%}' for x in v['by_day'])} sd {v['day_sd']:.1%} | tranched {t['cagr']:+.1%} y1 {t['y1']:+.1%} y2 {t['y2']:+.1%} Sh {t['sharpe']:.2f} DD {t['maxdd']:+.1%} | vs control {e['excess']:+.1%} [{e['lo']:+.1%},{e['hi']:+.1%}] turn {v['turnover']:.1f}")
    for k in comps:
        v = res["top20"][k]
        print(f"top-20 {k:<8} days {' '.join(f'{x:+.1%}' for x in v['by_day'])} sd {v['day_sd']:.1%} | tranched {v['tranched']['cagr']:+.1%} Sh {v['tranched']['sharpe']:.2f} DD {v['tranched']['maxdd']:+.1%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        pw, _, _ = LAB.pit_panel()
        hist = P.load_panel(LAB.cfg)
        for n in (sys.argv[1:] or WALKS):
            walk(n, *WALKS[n], pw, hist)
            LAB.log(f"{n}: done")
