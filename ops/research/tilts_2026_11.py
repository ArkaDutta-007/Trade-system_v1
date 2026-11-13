#!/usr/bin/env python3
"""Where does the signal live, and can it be used without betting on 20 names? (November 2026 research pass)

Avramov, Cheng & Metzker (2023) find ML profits concentrated in hard-to-arbitrage and distressed stocks; our
top-20 book buys beaten-down names that later fail. So, on the honest panel, fixed before the first run:

  deciles    mean 21-day return of each score decile relative to the universe, on non-overlapping dates, and
             the share of each decile that has left the point-in-time universe 63 sessions later
  tilts      that day's 300 most liquid names, rebalanced on the first session of each month from the previous
             close, 10 bps on turnover: cap-weight all (the control, close to SPY) · drop the worst 10% / 30% by
             score · keep the best 30% · and the same three with equal weights
  distress   a Campbell-Hilscher-Szilagyi flavoured rank (mean z of −ROA, +leverage, −12-month return, +63-day
             volatility, −size, −price, published signs, no fitting); drop the worst 10% / 20% of the production
             universe before the production book
Models: S-xgb (production), P-xgb (point-in-time training), lit, lit-osap (see panel_2026_11).
Results → reports/alpha/lab_2026_11/tilts.json
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import panel_2026_11 as PP  # noqa: E402

L, LAB, OUT, P = PP.L, PP.LAB, PP.OUT, PP.P
Y1_END = PP.Y1_END


def deciles(pw, comps) -> dict:
    cal = sorted(pw["date"].unique().to_list())
    keep = set(cal[::21])
    pos = {d: i for i, d in enumerate(cal)}
    mem = set(pw.select("date", "ticker").iter_rows())
    later = {d: cal[pos[d] + 63] if pos[d] + 63 < len(cal) else None for d in keep}
    base = pw.filter(pl.col("date").is_in(list(keep))).select("date", "ticker", "fwd_21", "log_dv_21").drop_nulls("fwd_21")
    base = base.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
    gone = [(d, t, (later[d] is not None and (later[d], t) not in mem)) for d, t in base.select("date", "ticker").iter_rows()]
    base = base.join(pl.DataFrame(gone, schema=["date", "ticker", "gone"], orient="row"), on=["date", "ticker"])
    out = {}
    for name, c in comps.items():
        for uname, u in (("all", base), ("top300", base.filter(pl.col("lr") <= 300))):
            j = u.join(c, on=["date", "ticker"]).with_columns(
                dec=((pl.col("comp").rank("ordinal").over("date") - 1) * 10 // pl.col("comp").count().over("date")).cast(pl.Int32),
                rx=pl.col("fwd_21") - pl.col("fwd_21").mean().over("date"))
            for span, f in (("all", pl.lit(True)), ("y1", pl.col("date") <= Y1_END), ("y2", pl.col("date") > Y1_END)):
                g = j.filter(f).group_by("dec").agg(rx=pl.col("rx").mean(), raw=pl.col("fwd_21").mean(), gone=pl.col("gone").mean(), n=pl.len()).sort("dec")
                out[f"{name}|{uname}|{span}"] = {int(r["dec"]): {"rx": r["rx"], "raw": r["raw"], "gone": r["gone"]} for r in g.iter_rows(named=True)}
    return out


def month_book(pw, comp, offset: int = 0):
    """``offset`` sessions after the first of the month: the same rule on a different rebalance day."""
    cal = sorted(pw["date"].unique().to_list())
    start = comp["date"].min()
    days = [d for d in cal if d > start]
    first = [d for i, d in enumerate(days) if i == 0 or d.month != days[i - 1].month]
    first = [days[min(days.index(d) + offset, len(days) - 1)] for d in first]
    prev = {d: p for p, d in zip(cal[:-1], cal[1:])}
    # weights: per-share-class caps (2026-11-13). Before, the filing-based company cap was used: Waters at 1,000×
    # its size from 2026-07 (a units error in its filings) and Alphabet counted twice.
    from trading_system.alpha.challenger import with_log_cap
    cap = pl.when(pl.col("log_cap").is_finite()).then(pl.col("log_cap"))
    uni = (with_log_cap(LAB.cfg, pw.select("date", "ticker", "close", "log_dv_21", "log_mcap"))
             .with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date")).filter(pl.col("lr") <= 300, pl.col("close") >= 5)
             .with_columns(mc=pl.coalesce(cap, pl.col("log_dv_21") + (cap - pl.col("log_dv_21")).median().over("date")))
             .select("date", "ticker", "mc"))
    j = uni.join(comp, on=["date", "ticker"], how="left").with_columns(comp=pl.col("comp").fill_null(pl.col("comp").median().over("date")))
    by = {d: g for (d,), g in j.group_by(["date"])}
    return first, prev, by


def run(rets_of, first, prev, by, rule, weight, cost=.001):
    w, out, turn = {}, [], 0.0
    fs = set(first)
    for d in rets_of["days"]:
        r = sum(x * rets_of["R"].get((d, t), 0.0) for t, x in w.items())
        c = 0.0
        if d in fs and prev[d] in by:
            g = by[prev[d]]
            n = g.height
            if rule == "all":
                g2 = g
            elif rule.startswith("drop"):
                g2 = g.sort("comp").tail(n - int(n * float(rule[4:]) / 100))
            else:
                g2 = g.sort("comp").tail(int(n * float(rule[4:]) / 100))
            v = np.exp(g2["mc"].to_numpy()) if weight == "cap" else np.ones(g2.height)
            v = v / v.sum()
            nw = dict(zip(g2["ticker"].to_list(), v))
            t = sum(abs(nw.get(k, 0) - w.get(k, 0)) for k in set(nw) | set(w))
            c, turn = t * cost, turn + t
            w = nw
        else:                                       # weights drift with returns between rebalances
            tot = sum(x * (1 + rets_of["R"].get((d, t), 0.0)) for t, x in w.items()) or 1.0
            w = {t: x * (1 + rets_of["R"].get((d, t), 0.0)) / tot for t, x in w.items()}
        out.append(r - c)
    return np.array(out), turn


def posthoc():
    """Added AFTER seeing the results, so it is one more look at the same data, not a clean test: the ML models
    were strong in year 1 and weak in year 2, the literature composite the reverse. Does a 50/50 blend keep both?"""
    pw, allpx, _ = LAB.pit_panel()
    ml = LAB.zcomp(pl.read_parquet(OUT / "wf_P-xgb-debias.parquet"))
    lit = PP.lit_scores(pw)["lit"]
    j = ml.join(lit, on=["date", "ticker"], suffix="_l")
    zc = lambda c: (pl.col(c) - pl.col(c).mean().over("date")) / (pl.col(c).std().over("date") + 1e-9)
    bl = j.select("date", "ticker", comp=.5 * zc("comp") + .5 * zc("comp_l"))
    labels = pw.select("date", "ticker", "fwd_21", "fwd_63")
    s = pl.concat([bl.select("date", "ticker", score="comp").with_columns(horizon=pl.lit(h, dtype=pl.Int32)) for h in LAB.H])
    res = {"ic": PP.ic_year(s, labels)}
    px = allpx.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1)
    R = {(d, t): r for d, t, r in px.select("date", "ticker", "r").iter_rows() if r is not None}
    start = bl["date"].min()
    cal = sorted(pw["date"].unique().to_list())
    days = [d for d in cal if d > start]
    rets_of = {"days": days, "R": R}
    y1 = np.array([d <= Y1_END for d in days])
    for rule in ("all", "drop30", "best30"):
        out, ser = [], {}
        for off in (0, 5, 10, 15):
            first, prev, by = month_book(pw, bl, off)
            r, _ = run(rets_of, first, prev, by, rule, "cap")
            ser[off] = r
            out.append(float(np.prod(1 + r) ** (252 / len(days)) - 1))
        res[rule] = {"cagr_by_day": out, "y1": float(np.prod(1 + ser[0][y1]) - 1), "y2": float(np.prod(1 + ser[0][~y1]) - 1), **LAB.book_stats(ser[0])}
        res[f"{rule}_series"] = ser[0]
    res["drop30_vs_ctrl"] = LAB.excess_ci(res.pop("drop30_series"), res["all_series"])
    res["best30_vs_ctrl"] = LAB.excess_ci(res.pop("best30_series"), res.pop("all_series"))
    (OUT / "tilts_blend.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str)[:1500])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "posthoc":
        return posthoc()
    pw, allpx, _ = LAB.pit_panel()
    raw = {n: pl.read_parquet(OUT / f"wf_{n}.parquet") for n in ("S-xgb", "P-xgb", "P-xgb-debias")}
    comps = {k: LAB.zcomp(v) for k, v in raw.items()}
    comps.update(PP.lit_scores(pw))
    res = {"deciles": deciles(pw, comps), "tilts": {}, "tilt_excess": {}, "distress": {}}
    px = allpx.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1)
    R = {(d, t): r for d, t, r in px.select("date", "ticker", "r").iter_rows() if r is not None}
    start = max(c["date"].min() for c in comps.values())
    cal = sorted(pw["date"].unique().to_list())
    days = [d for d in cal if d > start]
    rets_of = {"days": days, "R": R}
    series = {}
    for name, c in comps.items():
        first, prev, by = month_book(pw, c.filter(pl.col("date") >= start))
        for weight in ("cap", "eq"):
            for rule in ("all", "drop10", "drop30", "best30"):
                if rule == "all" and name != "S-xgb":
                    continue
                r, tv = run(rets_of, first, prev, by, rule, weight)
                k = f"{weight}:all" if rule == "all" else f"{name}|{weight}:{rule}"
                series[k] = r
                y1 = np.array([d <= Y1_END for d in days])
                res["tilts"][k] = {**LAB.book_stats(r), "turnover": tv, "y1": float(np.prod(1 + r[y1]) - 1), "y2": float(np.prod(1 + r[~y1]) - 1)}
    for k, r in series.items():
        if "|" in k:
            ctrl = series[f"{k.split('|')[1].split(':')[0]}:all"]
            res["tilt_excess"][k] = LAB.excess_ci(r, ctrl)
    spy = np.array([R.get((d, "SPY"), 0.0) for d in days])
    y1 = np.array([d <= Y1_END for d in days])
    res["tilts"]["passive:SPY"] = {**LAB.book_stats(spy), "y1": float(np.prod(1 + spy[y1]) - 1), "y2": float(np.prod(1 + spy[~y1]) - 1)}
    res["control_vs_spy"] = {"corr": float(np.corrcoef(series["cap:all"], spy)[0, 1]),
                             "tracking_error": float((series["cap:all"] - spy).std() * 252 ** .5)}
    # the same tilts on other rebalance days
    res["tilt_days"] = {}
    for name in ("S-xgb", "P-xgb", "P-xgb-debias"):
        c = comps[name].filter(pl.col("date") >= start)
        for rule in ("all", "drop30", "best30"):
            out = []
            for off in (0, 5, 10, 15):
                first, prev, by = month_book(pw, c, off)
                out.append(float(np.prod(1 + run(rets_of, first, prev, by, rule, "cap")[0]) ** (252 / len(days)) - 1))
            res["tilt_days"][f"{name}|cap:{rule}"] = out
    w = next(iter(by.values()))
    v = np.exp(w["mc"].to_numpy()); v = np.sort(v / v.sum())[::-1]
    res["cap_weight_check_top5"] = [round(float(x), 4) for x in v[:5]]
    # distress filter on the production book
    from trading_system.alpha.portfolio import market_regime_on
    from trading_system.research.wfbacktest import build_panel
    pdata = build_panel(P.sanitize_prices(allpx))
    oos = cal[3::63][0]
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    ds = pw.with_columns(distress=pl.mean_horizontal([-PP.z("roa_ttm"), PP.z("leverage"), -PP.z("ret_252"), PP.z("vol_63"), -PP.z("log_mcap"), -PP.z("log_price")]))
    ds = ds.with_columns(dq=pl.col("distress").rank().over("date") / pl.col("distress").count().over("date"))
    g0 = (pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07) & (pl.col("log_dv_21") >= np.log1p(20e6))
    base = None
    for m in ("S-xgb", "P-xgb"):
        for q in (1.0, .9, .8):
            gate = ds.select("date", "ticker", "adj_close", elig=g0 & (pl.col("dq") <= q))
            r = LAB.book_run(pdata, comps[m], gate, sector_of, regime, oos, m).returns()
            if q == 1.0:
                base = r
            res["distress"][f"{m} drop worst {round((1 - q) * 100)}%"] = {**LAB.book_stats(r), **({"vs_unfiltered": LAB.excess_ci(r, base)} if q < 1 else {})}
    (OUT / "tilts.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res["tilts"].items():
        e = res["tilt_excess"].get(k)
        print(f"{k:<26} CAGR {v['cagr']:+6.1%} Sh {v['sharpe']:5.2f} DD {v['maxdd']:+6.1%} y1 {v['y1']:+6.1%} y2 {v['y2']:+6.1%} turn {v.get('turnover', 0):5.1f}" + (f"  vs ctrl {e['excess']:+6.1%} [{e['lo']:+.1%},{e['hi']:+.1%}]" if e else ""))
    for k, v in res["distress"].items():
        e = v.get("vs_unfiltered")
        print(f"{k:<26} CAGR {v['cagr']:+6.1%} Sh {v['sharpe']:5.2f} DD {v['maxdd']:+6.1%}" + (f"  vs unfiltered {e['excess']:+6.1%} [{e['lo']:+.1%},{e['hi']:+.1%}]" if e else ""))
    print(res["control_vs_spy"], res["cap_weight_check_top5"])
    for k, v in res["tilt_days"].items():
        print(k, " ".join(f"{x:+.1%}" for x in v))
    for k in ("S-xgb|all|all", "P-xgb|all|all", "P-xgb-debias|all|all", "S-xgb|top300|all", "P-xgb|top300|all", "P-xgb-debias|top300|all", "lit|all|all", "lit|top300|all", "lit-osap|top300|all",
              "P-xgb|top300|y1", "P-xgb|top300|y2", "P-xgb-debias|top300|y2", "S-xgb|all|y2"):
        d = res["deciles"][k]
        print(k, " ".join(f"{d[i]['rx'] * 100:+.1f}" for i in sorted(d)), "| gone", " ".join(f"{d[i]['gone'] * 100:.0f}" for i in sorted(d)))


if __name__ == "__main__":
    main()
