#!/usr/bin/env python3
"""The honest test, re-run every week on everything we have (failures included).

Until 2026-11-13 the picks card quoted one frozen number (−1.3%/yr, Nov 24 → Oct 26). The whole-market history
grows by a day every day, so this script repeats the test weekly and the card, the challenger status and the
digest read its result (reports/alpha/honest_latest.json):

  panel     the challenger's point-in-time panel (every stock that traded, 2024-11 → last close)
  models    walk-forward with 63-session refits, purged and embargoed:
            production recipe (today's survivors only) and challenger recipe (point-in-time rows + top-500)
  books     production top-20 book in the production universe · index-minus (top-300 cap-weighted minus the worst
            30% by challenger score) and its control (nothing dropped) · SPY
  luck      every book is run on start days 0/5/10/15 and reported as 4 tranches (their mean), with the range

Run by ops/weekly_retrain.sh (~6 min on the GPU). Nothing here changes a model or a book.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import panel_2026_11 as PP  # noqa: E402
import tilts_2026_11 as TT  # noqa: E402

LAB, P = PP.LAB, PP.P
from trading_system.alpha import challenger as C  # noqa: E402
from trading_system.alpha.portfolio import market_regime_on  # noqa: E402
from trading_system.research.wfbacktest import build_panel  # noqa: E402

OFFSETS = (0, 5, 10, 15)
MODELS = {"production": "S-xgb", "challenger": "P-xgb-debias"}


def stats(r: np.ndarray) -> dict:
    s = LAB.book_stats(r)
    n = min(len(r), 252)
    return {"cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd": s["maxdd"], "vol": s["vol"], "last_year": float(np.prod(1 + r[-n:]) - 1)}


def main() -> int:
    t0 = time.time()
    cfg = LAB.cfg
    work = cfg.path("reports") / "alpha" / "honest_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    PP.OUT = work                                       # PP.walk caches its scores here
    pw = C.load_panel(cfg)
    hist = P.load_panel(cfg)
    px, _ = C.pit_prices(cfg)
    spy = (pl.scan_parquet(cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet").filter(pl.col("ticker") == "SPY")
             .select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect())
    allpx = pl.concat([px, spy]).unique(subset=["date", "ticker"])
    cal = sorted(pw["date"].unique().to_list())
    labels = pw.select("date", "ticker", "fwd_21", "fwd_63")
    comps, ics = {}, {}
    for key, name in MODELS.items():
        PP.walk(name, pw, hist)
        sc = pl.read_parquet(work / f"wf_{name}.parquet")
        comps[key] = LAB.zcomp(sc)
        t = LAB.ic_table(sc, labels)
        last = t.filter(pl.col("date") >= cal[max(0, len(cal) - 252)])
        ics[key] = {"all": {h: v["ic"] for h, v in LAB.ic_summary(t).items()}, "last_year": {h: v["ic"] for h, v in LAB.ic_summary(last).items()}}
    # index-minus and its control, 4 start days
    r1 = allpx.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1)
    R = {(d, t): r for d, t, r in r1.select("date", "ticker", "r").iter_rows() if r is not None}
    start = max(c["date"].min() for c in comps.values())
    days = [d for d in cal if d > start]
    rets_of = {"days": days, "R": R}
    ser = {"index_minus": [], "control": []}
    for off in OFFSETS:
        first, prev, by = TT.month_book(pw, comps["challenger"].filter(pl.col("date") >= start), off)
        ser["index_minus"].append(TT.run(rets_of, first, prev, by, "drop30", "cap")[0])
        ser["control"].append(TT.run(rets_of, first, prev, by, "all", "cap")[0])
    cagr = lambda r: float(np.prod(1 + r) ** (252 / len(r)) - 1)
    out = {"run": str(date.today()), "window": [str(days[0]), str(days[-1])], "sessions": len(days), "ic": ics}
    tr = {k: np.mean(v, axis=0) for k, v in ser.items()}
    for k in ser:
        out[k] = {**stats(tr[k]), "by_start_day": [cagr(r) for r in ser[k]]}
    out["index_minus"]["vs_control"] = LAB.excess_ci(tr["index_minus"], tr["control"])
    n = min(len(days), 252)
    out["index_minus"]["vs_control_last_year"] = LAB.excess_ci(tr["index_minus"][-n:], tr["control"][-n:])
    s = np.array([R.get((d, "SPY"), 0.0) for d in days])
    out["spy"] = stats(s)
    # the production top-20 book (and the challenger's scores through the same book)
    pdata = build_panel(P.sanitize_prices(allpx))
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    gate = pw.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07)
                     & (pl.col("log_dv_21") >= np.log1p(20e6)))
    for key, label in (("production", "production_top20"), ("challenger", "challenger_top20")):
        rs = [LAB.book_run(pdata, comps[key], gate, sector_of, regime, cal[3 + off], key).returns() for off in OFFSETS]
        m = min(len(r) for r in rs)
        out[label] = {**stats(np.mean([r[-m:] for r in rs], axis=0)), "by_start_day": [cagr(r) for r in rs], "median": float(np.median([cagr(r) for r in rs]))}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    C.honest_path(cfg).write_text(json.dumps(out, indent=1, default=str))
    shutil.rmtree(work, ignore_errors=True)
    p, c, i, k = out["production_top20"], out["challenger_top20"], out["index_minus"], out["control"]
    print(f"honest test {out['window'][0]} → {out['window'][1]} ({out['sessions']} sessions, {out['minutes']} min)")
    print(f"  production top-20   {p['cagr']:+.1%}/yr (start days {min(p['by_start_day']):+.1%} … {max(p['by_start_day']):+.1%}) · last 12 months {p['last_year']:+.1%}")
    print(f"  challenger top-20   {c['cagr']:+.1%}/yr (start days {min(c['by_start_day']):+.1%} … {max(c['by_start_day']):+.1%}) · last 12 months {c['last_year']:+.1%}")
    print(f"  index-minus         {i['cagr']:+.1%}/yr · last 12 months {i['last_year']:+.1%} · vs control {i['vs_control']['excess']:+.1%}/yr "
          f"[{i['vs_control']['lo']:+.1%}, {i['vs_control']['hi']:+.1%}] · last 12 months {i['vs_control_last_year']['excess']:+.1%}")
    print(f"  control (all 300)   {k['cagr']:+.1%}/yr · last 12 months {k['last_year']:+.1%}      SPY {out['spy']['cagr']:+.1%}/yr · last 12 months {out['spy']['last_year']:+.1%}")
    print("  IC " + " · ".join(f"{m} 21d {v['all'][21]:+.3f} (last year {v['last_year'][21]:+.3f}) 63d {v['all'][63]:+.3f} ({v['last_year'][63]:+.3f})" for m, v in ics.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
