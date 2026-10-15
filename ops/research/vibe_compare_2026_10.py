#!/usr/bin/env python3
"""Vibe-Trading's alpha zoo vs our model on the honest point-in-time panel (2026-10-15).

Run with Vibe-Trading's venv (it imports Vibe's factor registry):
    PYTHONPATH=~/Desktop/Vibe-Trading/agent ~/Desktop/Vibe-Trading/.venv/bin/python ops/research/vibe_compare_2026_10.py

Input: data/silver/vibe/pit_long.parquet (ops/research/vibe_export_2026_10.py, our venv). Every zoo factor that
needs only prices (Qlib-158, Kakushadze-101, GTJA-191, academic) is computed by Vibe's own registry on the same
adjusted bars, masked to the point-in-time universe of each day (failures included), and scored with Vibe's own
IC function against the same 21-day forward returns as our models. Fixed in advance:

  halves     H1 = 2024-11-26 → 2025-10-31 (selection), H2 = 2025-11-03 → end (test)
  selection  Vibe's own categories computed on H1: alive (IC > 0.02, IC>0 on ≥ 55% of days, |t| > 2) used as is,
             reversed (IC < −0.02, |t| > 2) sign-flipped; t-stats on non-overlapping 21-day samples
  composite  equal-weight mean of the selected factors' per-day cross-sectional z-scores (clipped ±3)
Outputs: reports/alpha/lab_2026_10/vibe_compare.json and data/silver/vibe/zoo_composite.parquet (for the book test).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
VIBE = Path.home() / "Desktop" / "Vibe-Trading" / "agent"
sys.path.insert(0, str(VIBE))

from src.factors.factor_analysis_core import compute_ic_series  # noqa: E402
from src.factors.registry import SkipAlpha, get_default_registry  # noqa: E402

DATA = REPO / "data" / "silver" / "vibe"
OUT = REPO / "reports" / "alpha" / "lab_2026_10"
ZOOS = ("qlib158", "alpha101", "gtja191", "academic")
H1 = (pd.Timestamp("2024-11-26"), pd.Timestamp("2025-10-31"))
H2 = (pd.Timestamp("2025-11-03"), pd.Timestamp("2100-01-01"))
STEP = 21

_PANEL: dict | None = None
_MEMBER: pd.DataFrame | None = None
_FWD: pd.DataFrame | None = None


def load():
    df = pd.read_parquet(DATA / "pit_long.parquet")
    df["date"] = pd.to_datetime(df["date"])
    wide = {c: df.pivot(index="date", columns="ticker", values=c).sort_index() for c in
            ("open", "high", "low", "close", "volume", "amount", "vwap")}
    member = df.pivot(index="date", columns="ticker", values="member").reindex_like(wide["close"]).fillna(False).astype(bool)
    fwd = df.pivot(index="date", columns="ticker", values="fwd_21").reindex_like(wide["close"]).where(member)
    ours = {k: df.pivot(index="date", columns="ticker", values=k).reindex_like(wide["close"]).where(member)
            for k in ("prod", "pitaug")}
    return wide, member, fwd, ours


def stats(ic: pd.Series, lo, hi) -> dict:
    s = ic[(ic.index >= lo) & (ic.index <= hi)].dropna()
    if len(s) < 30:
        return {"n": len(s)}
    nov = s.iloc[::STEP]
    t = float(nov.mean() / (nov.std(ddof=1) + 1e-12) * np.sqrt(len(nov))) if len(nov) > 3 else float("nan")
    return {"n": len(s), "ic": float(s.mean()), "icir": float(s.mean() / (s.std() + 1e-12)), "pos": float((s > 0).mean()), "t": t}


def category(st: dict) -> str:
    if st.get("ic") is None:
        return "n/a"
    if st["ic"] > 0.02 and st["pos"] >= 0.55 and abs(st["t"]) > 2:
        return "alive"
    if st["ic"] < -0.02 and abs(st["t"]) > 2:
        return "reversed"
    return "dead"


def _init(panel, member, fwd):
    global _PANEL, _MEMBER, _FWD
    _PANEL, _MEMBER, _FWD = panel, member, fwd


def _one(alpha_id: str) -> dict:
    reg = get_default_registry()
    try:
        f = reg.compute(alpha_id, _PANEL)
    except SkipAlpha as e:
        return {"id": alpha_id, "skip": str(e)[:120]}
    except Exception as e:  # noqa: BLE001
        return {"id": alpha_id, "skip": f"{type(e).__name__}: {str(e)[:100]}"}
    f = f.reindex_like(_MEMBER).where(_MEMBER).replace([np.inf, -np.inf], np.nan)
    ic = compute_ic_series(f, _FWD)
    return {"id": alpha_id, "ic": ic}


def zscore(f: pd.DataFrame) -> pd.DataFrame:
    z = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1) + 1e-12, axis=0)
    return z.clip(-3, 3)


def main():
    t0 = time.time()
    wide, member, fwd, ours = load()
    reg = get_default_registry()
    ids = [a for z in ZOOS for a in reg.list(zoo=z)]
    print(f"panel {wide['close'].shape} · {len(ids)} price-only alphas from {ZOOS}", flush=True)
    res, ics = {}, {}
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2), initializer=_init, initargs=(wide, member, fwd)) as ex:
        futs = [ex.submit(_one, a) for a in ids]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if "ic" in r:
                ics[r["id"]] = r["ic"]
            else:
                res[r["id"]] = {"skip": r["skip"]}
            if i % 50 == 0:
                print(f"  {i}/{len(ids)} alphas · {time.time() - t0:.0f}s", flush=True)
    rows = []
    for a, ic in ics.items():
        s1, s2, sf = stats(ic, *H1), stats(ic, *H2), stats(ic, H1[0], H2[1])
        rows.append({"id": a, "zoo": a.split("_")[0], "h1": s1, "h2": s2, "full": sf, "cat_h1": category(s1)})
    ours_ic = {k: compute_ic_series(v, fwd) for k, v in ours.items()}
    ours_st = {k: {"h1": stats(ic, *H1), "h2": stats(ic, *H2), "full": stats(ic, H1[0], H2[1])} for k, ic in ours_ic.items()}
    sel = [(r["id"], 1.0 if r["cat_h1"] == "alive" else -1.0) for r in rows if r["cat_h1"] in ("alive", "reversed")]
    print(f"selected on H1: {sum(1 for _, s in sel if s > 0)} alive + {sum(1 for _, s in sel if s < 0)} reversed", flush=True)
    comp, cnt = None, None
    for a, sgn in sel:
        f = reg.compute(a, wide).reindex_like(member).where(member).replace([np.inf, -np.inf], np.nan)
        z = zscore(f) * sgn
        comp = z.fillna(0) if comp is None else comp.add(z.fillna(0))
        cnt = z.notna().astype(int) if cnt is None else cnt + z.notna().astype(int)
    comp = comp.div(cnt.replace(0, np.nan)) if comp is not None else None
    comp_st = None
    if comp is not None:
        cic = compute_ic_series(comp.where(member), fwd)
        comp_st = {"h1 (in-sample)": stats(cic, *H1), "h2 (out-of-sample)": stats(cic, *H2)}
        long = comp.where(member).stack().rename("comp").reset_index()
        long.columns = ["date", "ticker", "comp"]
        long.to_parquet(DATA / "zoo_composite.parquet")
    full = sorted([r for r in rows if r["full"].get("ic") is not None], key=lambda r: -r["full"]["ic"])
    h2v = np.array([r["h2"]["ic"] for r in rows if r["h2"].get("ic") is not None])
    summary = {
        "alphas_scored": len(rows), "alphas_skipped": len(res),
        "categories_h1": {c: sum(1 for r in rows if r["cat_h1"] == c) for c in ("alive", "reversed", "dead", "n/a")},
        "zoo_h2_ic_distribution": {"median": float(np.median(h2v)), "p90": float(np.percentile(h2v, 90)), "max": float(h2v.max()), "n": len(h2v)},
        "top10_full_window": [{"id": r["id"], "ic": r["full"]["ic"], "t": r["full"]["t"]} for r in full[:10]],
        "ours": ours_st, "zoo_composite": comp_st, "selected": sel,
    }
    (OUT / "vibe_compare.json").write_text(json.dumps({"summary": summary, "rows": rows, "skipped": res}, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "selected"}, indent=1, default=str)[:4000])
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
