#!/usr/bin/env python3
"""Volatility-managed exposure on a century of market data (2026-11-13, RESEARCH_2026-11 §13).

Moreira & Muir (2017) report that holding less of the market when recent volatility is high raises the Sharpe
ratio; Cederburg et al. (2020) question how much survives out of sample. For a long-only investor the usable
form has no leverage. Fixed before the first run, on Ken French's daily US market 1926 →:

  volatility   annualised standard deviation of the last 21 daily returns, two days old when traded (as in §4)
  exposure     min(cap, 15% ÷ volatility); the rest in T-bills; cap = 1.0 (no leverage) and 1.5
  costs        5 bps per unit of exposure changed, traded only when the target moves by more than 0.10;
               above 1× the borrowed part pays T-bill + 0.5% and a 0.9% fee, as in §4
  compared     1× always, the 200-day trend rule at 1×, and volatility management inside the trend rule
Results → reports/alpha/lab_2026_11/volmanaged.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import longrun_2026_11 as LR  # noqa: E402

DD = 252


def managed(m: pd.Series, rf: pd.Series, cap: float, trend: bool = False, target: float = 0.15, band: float = 0.10) -> pd.Series:
    vol = (m.rolling(21).std() * DD ** 0.5).shift(2)
    want = (target / vol).clip(upper=cap).fillna(0.0)
    if trend:
        px = (1 + m).cumprod()
        want = want * (px > px.rolling(200).mean()).astype(float).shift(2).fillna(0.0)
    x = np.zeros(len(m))
    cur = 0.0
    w = want.to_numpy()
    for i in range(len(m)):                      # trade only when the target has moved by more than the band
        if abs(w[i] - cur) > band or (w[i] == 0.0 and cur != 0.0):
            cur = w[i]
        x[i] = cur
    x = pd.Series(x, index=m.index)
    borrow = (x - 1).clip(lower=0)
    r = x * m + (1 - x).clip(lower=0) * rf - borrow * (rf + 0.005 / DD) - (x > 1) * x * 0.009 / DD - x.diff().abs().fillna(0) * 0.0005
    return r[200:], x[200:]


def main():
    f = LR.french("F-F_Research_Data_Factors_daily.csv", None)
    m, rf = f["Mkt-RF"] + f["RF"], f["RF"]
    runs = {"1× always": (m[200:], pd.Series(1.0, index=m.index)[200:]), "1× + 200d trend": (LR.lever(m, rf, 1, True), None),
            "vol-managed, cap 1.0": managed(m, rf, 1.0), "vol-managed, cap 1.5": managed(m, rf, 1.5),
            "vol-managed inside trend, cap 1.0": managed(m, rf, 1.0, trend=True)}
    res = {}
    for k, (r, x) in runs.items():
        d = LR.stats(r)
        rc = LR.rolling_cagr(r, 10)
        d.update(worst_10y=float(rc.min()), since_2005=LR.stats(r.loc["2005":]), since_2015=LR.stats(r.loc["2015":]),
                 avg_exposure=None if x is None else float(x.mean()), trades_per_year=None if x is None else float((x.diff().abs() > 1e-9).sum() / (len(x) / DD)))
        res[k] = d
        print(f"{k:<36} CAGR {d['cagr']:+.1%} vol {d['vol']:.0%} Sharpe {d['sharpe']:.2f} DD {d['maxdd']:+.0%} worst10y {d['worst_10y']:+.1%} | since 2005 {d['since_2005']['cagr']:+.1%} "
              f"Sh {d['since_2005']['sharpe']:.2f} DD {d['since_2005']['maxdd']:+.0%} | since 2015 {d['since_2015']['cagr']:+.1%} Sh {d['since_2015']['sharpe']:.2f}"
              + (f" | avg exposure {d['avg_exposure']:.2f}, {d['trades_per_year']:.0f} trades/yr" if x is not None else ""))
    (LR.OUT / "volmanaged.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
