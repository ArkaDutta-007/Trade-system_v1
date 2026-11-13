#!/usr/bin/env python3
"""Two cheap checks of language-model inputs we already have (2026-11-13, RESEARCH_2026-11 §13).

  tev1 live   every stock-screen verdict the local 4B model has recorded in the daily digest since 2026-10-09
              (ok / caution / avoid), against what the stock did next. Return = close of the judgment day →
              5 and 10 sessions later (so the day-0 reaction is excluded), minus the same day's mean over all
              screened names. Days are heavily overlapping, so the count that matters is independent periods.
  sentiment   Massive's own per-ticker news sentiment (21-session mean, already a model input): rank IC with
              21- and 63-day forward returns on the complete point-in-time panel, alone and after removing what
              the challenger score already explains. This re-runs the October check that predated the news backfill.
Results → reports/alpha/lab_2026_11/llm_checks.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import panel_2026_11 as PP  # noqa: E402

LAB, OUT = PP.LAB, PP.OUT
from trading_system.alpha import challenger as C  # noqa: E402


def tev1_live() -> dict:
    cfg = LAB.cfg
    t = pl.read_parquet(cfg.project_root / "data" / "ledger" / "tev1_judgments.parquet").filter(pl.col("kind") == "screen")
    t = t.select(date="asof", ticker="subject", verdict=pl.col("verdict").fill_null("no news")).unique(subset=["date", "ticker"])
    px = (pl.scan_parquet(cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet").filter(pl.col("ticker").is_in(t["ticker"].unique().to_list()))
            .select("date", "ticker", "adj_close").collect().sort(["ticker", "date"]))
    out = {"dates": t["date"].n_unique(), "judgments": t.height, "first": str(t["date"].min()), "last": str(t["date"].max())}
    for h in (5, 10):
        f = px.with_columns(fwd=pl.col("adj_close").shift(-h).over("ticker") / pl.col("adj_close") - 1)
        j = t.join(f.select("date", "ticker", "fwd"), on=["date", "ticker"]).drop_nulls("fwd").with_columns(rx=pl.col("fwd") - pl.col("fwd").mean().over("date"))
        g = j.group_by("verdict").agg(n=pl.len(), names=pl.col("ticker").n_unique(), rx=pl.col("rx").mean(), raw=pl.col("fwd").mean(), hit=(pl.col("rx") > 0).mean())
        d = (j.group_by("date").agg(avoid=pl.col("rx").filter(pl.col("verdict").is_in(["avoid", "caution"])).mean(), ok=pl.col("rx").filter(pl.col("verdict") == "ok").mean())
              .drop_nulls().with_columns(gap=pl.col("avoid") - pl.col("ok")).sort("date"))
        x = d["gap"].to_numpy()
        xs = x[::h]
        out[f"{h}d"] = {"by_verdict": {r["verdict"]: {k: r[k] for k in ("n", "names", "rx", "raw", "hit")} for r in g.iter_rows(named=True)},
                        "flagged_minus_ok": float(x.mean()) if len(x) else None, "dates": len(x), "independent": len(xs),
                        "t": float(xs.mean() / (xs.std(ddof=1) + 1e-12) * np.sqrt(len(xs))) if len(xs) > 2 else None}
    return out


def sentiment() -> dict:
    pw = C.load_panel(LAB.cfg).select("date", "ticker", "news_sent_21", "news_n_21", "fwd_21", "fwd_63")
    base = LAB.zcomp(pl.read_parquet(OUT / "wf2_base.parquet"))
    j = pw.join(base, on=["date", "ticker"]).filter(pl.col("news_n_21") > 0).drop_nulls("news_sent_21")
    out = {"coverage": j.height / pw.join(base, on=["date", "ticker"]).height}
    rk = lambda c: pl.col(c).rank().over("date") / pl.col(c).count().over("date")
    j = j.with_columns(s=rk("news_sent_21"), b=rk("comp"))
    # residual of the sentiment rank on the challenger rank, per day
    j = j.with_columns(res=pl.col("s") - (pl.cov("s", "b") / pl.col("b").var()).over("date") * (pl.col("b") - pl.col("b").mean().over("date")))
    for h in (21, 63):
        for name, col in (("alone", "s"), ("after challenger", "res"), ("challenger (same names)", "b")):
            g = j.drop_nulls(f"fwd_{h}").group_by("date").agg(ic=pl.corr(col, f"fwd_{h}", method="spearman"), n=pl.len()).filter(pl.col("n") >= 30).sort("date")
            x = g["ic"].to_numpy(); xs = x[::h]
            out[f"{h}d {name}"] = {"ic": float(x.mean()), "t": float(xs.mean() / (xs.std(ddof=1) + 1e-12) * np.sqrt(len(xs))), "dates": len(x),
                                   "y1": float(g.filter(pl.col("date") <= PP.Y1_END)["ic"].mean()), "y2": float(g.filter(pl.col("date") > PP.Y1_END)["ic"].mean())}
    return out


if __name__ == "__main__":
    res = {"tev1_live": tev1_live()}
    if (OUT / "wf2_base.parquet").exists():
        res["sentiment"] = sentiment()
    (OUT / "llm_checks.json").write_text(json.dumps(res, indent=1, default=str))
    t = res["tev1_live"]
    print(f"tev1 screen, live: {t['judgments']} judgments on {t['dates']} days ({t['first']} → {t['last']})")
    for h in ("5d", "10d"):
        v = t[h]
        print(f"  {h}: " + " · ".join(f"{k} {x['rx']:+.2%} (n {x['n']}, {x['names']} names, hit {x['hit']:.0%})" for k, x in sorted(v["by_verdict"].items()))
              + f" | flagged − ok {v['flagged_minus_ok']:+.2%} over {v['dates']} days = {v['independent']} independent, t {v['t'] if v['t'] is None else round(v['t'], 1)}")
    for k, v in res.get("sentiment", {}).items():
        print(f"  sentiment {k}: " + (f"{v:.0%}" if isinstance(v, float) else f"IC {v['ic']:+.3f} (t {v['t']:+.1f}) y1 {v['y1']:+.3f} y2 {v['y2']:+.3f}"))
