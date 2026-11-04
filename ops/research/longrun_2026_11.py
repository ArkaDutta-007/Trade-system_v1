#!/usr/bin/env python3
"""What survivor-free long history says about the 30%/yr goal (November 2026 research pass).

Our own honest panel covers two years. These tests use public data built on the full market with failures
and delisting returns included, so nothing here depends on today's survivors. Fixed before the first run:

  A1 base rates   Open Source Asset Pricing (Chen-Zimmermann, release 2026.10), value-weighted decile sorts:
                  excess return of each published signal's long leg over the market, by era and after publication.
  A2 factor timing  each month hold the 10 signals (long legs) with the best trailing 12-month excess return,
                  only among signals published by that year. Cost: 20 bps on the replaced share + 4 bps upkeep.
  A4 trend+leverage  Ken French daily market 1926 →: L × market above the 200-day average, T-bills below; the
                  signal is one day old when traded. L>1 pays T-bill + 0.5% on the borrowed part and 0.9%/yr
                  expense (leveraged-ETF terms); each change in exposure costs 5 bps. Same on the Nasdaq Composite.
  A5 industries   Ken French 49 (and 10) industries, value-weighted: top 5 (top 3) by 12-1 momentum, monthly,
                  10 bps on turnover. Sector ETFs since 1999: top 3 of the nine SPDRs, same rule.
  A6 cross-asset  Faber's five ETFs with the 10-month average (published 2007) and Antonacci's dual momentum
                  (published 2014), 5 bps on turnover; the out-of-sample years start after publication.
Results → reports/alpha/lab_2026_11/longrun.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data" / "external"
OUT = REPO / "reports" / "alpha" / "lab_2026_11"
OUT.mkdir(parents=True, exist_ok=True)
DD = 252


def french(name: str, section: str = "Average Value Weighted Returns -- Daily") -> pd.DataFrame:
    """One table of a Ken French CSV as simple returns; -99.99 → NaN."""
    lines = (EXT / "french" / name).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if section in l) if section else 0
    while not lines[i].startswith(","):
        i += 1
    cols = [c.strip() for c in lines[i].split(",")[1:]]
    rows, idx = [], []
    for l in lines[i + 1:]:
        p = l.split(",")
        if not p[0].strip().isdigit():
            break
        idx.append(p[0].strip()); rows.append([float(x) for x in p[1:]])
    df = pd.DataFrame(rows, index=pd.to_datetime(idx, format="%Y%m%d"), columns=cols)
    return df.where(df > -99) / 100


def stats(r: pd.Series, per: int = DD) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    y = len(r) / per
    return {"cagr": float(eq.iloc[-1] ** (1 / y) - 1), "vol": float(r.std() * per ** .5),
            "sharpe": float(r.mean() / (r.std() + 1e-12) * per ** .5), "maxdd": float((eq / eq.cummax() - 1).min()),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def rolling_cagr(r: pd.Series, years: int, per: int = DD) -> pd.Series:
    n = years * per
    lg = np.log1p(r).rolling(n).sum()
    return np.expm1(lg / years).dropna()


# ── A1/A2 ─────────────────────────────────────────────────────────────────────
def market_monthly() -> pd.Series:
    f = french("F-F_Research_Data_Factors_daily.csv", None)
    m = f["Mkt-RF"] + f["RF"]
    return (1 + m).resample("ME").prod() - 1


def osap_legs() -> tuple[pd.DataFrame, pd.Series]:
    p = pd.read_parquet(EXT / "osap" / "Alt_DecilesVW.parquet")
    p = p[p.port != "LS"]
    p["n"] = p.port.astype(int)
    top = p.loc[p.groupby(["signalname", "date"]).n.transform("max") == p.n]
    w = top.pivot_table(index="date", columns="signalname", values="ret") / 100
    w.index = pd.to_datetime(w.index) + pd.offsets.MonthEnd(0)
    doc = pd.read_parquet(EXT / "osap" / "SignalDoc.parquet").set_index("Acronym")
    yr = pd.to_numeric(doc["Year"], errors="coerce").reindex(w.columns)
    return w, yr


def short_legs(legs: pd.DataFrame, yr: pd.Series, mkt: pd.Series) -> dict:
    """The other end of the same sorts: is avoiding the worst names worth more than picking the best?"""
    p = pd.read_parquet(EXT / "osap" / "Alt_DecilesVW.parquet")
    p = p[p.port == "01"]
    w = p.pivot_table(index="date", columns="signalname", values="ret") / 100
    w.index = pd.to_datetime(w.index) + pd.offsets.MonthEnd(0)
    ex = w.sub(mkt.reindex(w.index), axis=0)
    lx = legs.sub(mkt.reindex(legs.index), axis=0)
    out = {}
    for k, (lo, hi) in {"1926-2004": ("1926", "2004"), "2015-2025": ("2015", "2025"), "2021-2025": ("2021", "2025")}.items():
        m, l = ex.loc[lo:hi].mean() * 12, lx.loc[lo:hi].mean() * 12
        out[k] = {"short_leg_median_excess": float(m.median()), "long_leg_median_excess": float(l.median()),
                  "share_short_legs_below_market": float((m < 0).mean())}
    post = pd.Series({s: ex.loc[str(int(y) + 1):, s].mean() * 12 for s, y in yr.dropna().items()
                      if s in ex and ex.loc[str(int(y) + 1):, s].count() >= 36})
    out["after publication"] = {"short_leg_median_excess": float(post.median()), "share_short_legs_below_market": float((post < 0).mean())}
    return out


def a1_a2() -> dict:
    legs, yr = osap_legs()
    mkt = market_monthly().reindex(legs.index)
    ex = legs.sub(mkt, axis=0)
    eras = {"1926-2004": ("1926", "2004"), "2005-2014": ("2005", "2014"), "2015-2025": ("2015", "2025"), "2021-2025": ("2021", "2025")}
    a1 = {}
    for k, (lo, hi) in eras.items():
        m = ex.loc[lo:hi].mean() * 12
        m = m[ex.loc[lo:hi].count() >= 24]
        a1[k] = {"signals": int(m.size), "median_excess": float(m.median()), "share_positive": float((m > 0).mean()),
                 "p90": float(m.quantile(.9)), "best": m.idxmax(), "best_excess": float(m.max())}
    post = {s: ex.loc[str(int(y) + 1):, s].mean() * 12 for s, y in yr.dropna().items() if ex.loc[str(int(y) + 1):, s].count() >= 36}
    post = pd.Series(post)
    a1["after publication"] = {"signals": int(post.size), "median_excess": float(post.median()), "share_positive": float((post > 0).mean()),
                               "p90": float(post.quantile(.9))}
    s1 = (legs.loc["2021":"2025"].add(1).prod() ** (1 / 5) - 1).dropna()
    a1["best long leg cagr 2021-2025"] = {"best": s1.idxmax(), "cagr": float(s1.max()), "market": float((1 + mkt.loc["2021":"2025"]).prod() ** (1 / 5) - 1)}
    # A2
    tr = (1 + ex).rolling(12).apply(np.prod, raw=True) - 1
    rets, turn, prev, allpub = [], [], set(), []
    idx = legs.index
    for i in range(12, len(idx) - 1):
        t, nx = idx[i], idx[i + 1]
        known = yr.index[yr <= t.year]
        cand = tr.loc[t, known].dropna()
        cand = cand[legs.loc[nx, cand.index].notna()]
        if len(cand) < 20:
            continue
        pick = set(cand.nlargest(10).index)
        tv = len(pick - prev) / 10
        rets.append((nx, legs.loc[nx, list(pick)].mean() - tv * .002 - .0004))
        allpub.append((nx, legs.loc[nx, cand.index].mean() - .0004))
        turn.append(tv); prev = pick
    r = pd.Series(dict(rets)); a = pd.Series(dict(allpub))
    a2 = {}
    for k, (lo, hi) in {"full": (str(r.index[0].year), "2025"), "2005-2014": ("2005", "2014"), "2015-2025": ("2015", "2025"), "2021-2025": ("2021", "2025")}.items():
        a2[k] = {"timed top-10": stats(r.loc[lo:hi], 12), "all published": stats(a.loc[lo:hi], 12), "market": stats(mkt.reindex(r.index).loc[lo:hi], 12)}
    a2["monthly_turnover"] = float(np.mean(turn))
    return {"base_rates": a1, "legs": short_legs(legs, yr, mkt), "factor_timing": a2}


# ── A4 ────────────────────────────────────────────────────────────────────────
def lever(m: pd.Series, rf: pd.Series, L: float, trend: bool, window: int = 200) -> pd.Series:
    px = (1 + m).cumprod()
    on = (px > px.rolling(window).mean()).astype(float).shift(2).fillna(0.0) if trend else pd.Series(1.0, index=m.index)
    x = on * L
    fin = np.where(L > 1, (L - 1) * (rf + .005 / DD) + L * .009 / DD, 0.0)
    r = x * m + (1 - x).clip(lower=0) * rf - on * fin
    return (r - x.diff().abs().fillna(0) * .0005)[window:]


def a4() -> dict:
    f = french("F-F_Research_Data_Factors_daily.csv", None)
    m, rf = f["Mkt-RF"] + f["RF"], f["RF"]
    res = {}
    for name, ser in [("US market 1926-", m)]:
        for L in (1, 1.5, 2, 3):
            for tr in (False, True):
                k = f"{L}x {'+ 200d trend' if tr else 'always'}"
                r = lever(ser, rf, L, tr)
                d = stats(r)
                rc = rolling_cagr(r, 10)
                d.update(worst_10y=float(rc.min()), median_10y=float(rc.median()), p_10y_ge_20=float((rc >= .2).mean()), p_10y_ge_30=float((rc >= .3).mean()),
                         since_2005=stats(r.loc["2005":])["cagr"], since_2005_dd=stats(r.loc["2005":])["maxdd"])
                res[f"{name} {k}"] = d
    try:
        import yfinance as yf
        px = yf.download(["^IXIC", "SPY", "SSO", "UPRO", "QLD", "TQQQ", "QQQ"], start="1971-01-01", auto_adjust=True, progress=False)["Close"]
        ix = px["^IXIC"].dropna().pct_change().dropna()
        ix.index = pd.to_datetime(ix.index).tz_localize(None)
        ix = ix.add(.01 / DD)                                      # price index → rough 1%/yr dividend
        rf2 = rf.reindex(ix.index).ffill()
        for L in (1, 2, 3):
            for tr in (False, True):
                r = lever(ix.loc[:rf.index[-1]], rf2.loc[:rf.index[-1]], L, tr)
                d = stats(r); rc = rolling_cagr(r, 10)
                d.update(worst_10y=float(rc.min()), median_10y=float(rc.median()), p_10y_ge_20=float((rc >= .2).mean()), p_10y_ge_30=float((rc >= .3).mean()))
                res[f"Nasdaq 1971- {L}x {'+ 200d trend' if tr else 'always'}"] = d
        check = {}                                                  # does the synthetic cost model match the real funds?
        for etf, und, L in (("SSO", m, 2), ("UPRO", m, 3)):
            a = px[etf].dropna().pct_change().dropna(); a.index = pd.to_datetime(a.index).tz_localize(None)
            lo, hi = a.index[0], min(a.index[-1], m.index[-1])
            s = (L * m - (L - 1) * (rf + .005 / DD) - L * .009 / DD).loc[lo:hi]
            ra, rs = a.loc[lo:hi], s
            n = len(ra) / DD
            check[etf] = {"real_cagr": float((1 + ra).prod() ** (1 / n) - 1), "synthetic_cagr": float((1 + rs).prod() ** (1 / n) - 1)}
        res["_etf_check"] = check
    except Exception as e:  # noqa: BLE001
        res["_nasdaq_error"] = str(e)[:200]
    return res


# ── A5 ────────────────────────────────────────────────────────────────────────
def mom_rotate(ind: pd.DataFrame, top: int, cost: float = .001) -> tuple[pd.Series, float]:
    mo = (1 + ind).resample("ME").prod(min_count=15) - 1
    sc = (1 + mo).rolling(11, min_periods=11).apply(np.prod, raw=True).shift(1) - 1      # t-12 .. t-2
    w, out, tv = None, {}, []
    for t, nx in zip(mo.index[12:-1], mo.index[13:]):
        c = sc.loc[t].dropna()
        c = c[mo.loc[nx, c.index].notna()]
        if len(c) < top * 2:
            continue
        n = pd.Series(1 / top, index=c.nlargest(top).index)
        turn = 1.0 if w is None else float(n.sub(w, fill_value=0).abs().sum())
        out[nx] = float((mo.loc[nx, n.index] * n).sum() - turn * cost)
        tv.append(turn); w = n
    return pd.Series(out), float(np.mean(tv) * 12)


def a5() -> dict:
    mkt = market_monthly()
    res = {}
    for name, f, top in (("49 industries top-5", "49_Industry_Portfolios_Daily.csv", 5), ("10 industries top-3", "10_Industry_Portfolios_Daily.csv", 3)):
        ind = french(f)
        r, tv = mom_rotate(ind, top)
        for k, (lo, hi) in {"full": ("1927", "2026"), "2005-2014": ("2005", "2014"), "2015-2026": ("2015", "2026"), "2021-2026": ("2021", "2026")}.items():
            res[f"{name} {k}"] = {"rotation": stats(r.loc[lo:hi], 12), "market": stats(mkt.reindex(r.index).loc[lo:hi], 12)}
        res[f"{name} annual_turnover"] = tv
    import yfinance as yf
    etfs = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY"]
    px = yf.download(etfs, start="1998-12-01", auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    r, tv = mom_rotate(px[etfs[:-1]].pct_change().dropna(), 3)
    spy = (1 + px["SPY"].pct_change()).resample("ME").prod() - 1
    res["sector ETFs top-3"] = {"rotation": stats(r, 12), "spy": stats(spy.reindex(r.index), 12), "annual_turnover": tv,
                                "since_2015": stats(r.loc["2015":], 12), "spy_since_2015": stats(spy.reindex(r.index).loc["2015":], 12)}
    return res


# ── A6 ────────────────────────────────────────────────────────────────────────
def a6() -> dict:
    import yfinance as yf
    t = ["SPY", "EFA", "IEF", "DBC", "VNQ", "AGG", "BIL", "QQQ", "TLT", "GLD"]
    px = yf.download(t, start="2002-06-01", auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    me = px.resample("ME").last()
    r = me.pct_change()
    faber = ["SPY", "EFA", "IEF", "DBC", "VNQ"]
    res = {}
    # GTAA5: each ETF held at 20% when above its 10-month average, else in T-bills (BIL / short rate)
    sma = me[faber].rolling(10).mean()
    on = (me[faber] > sma).astype(float).where(me[faber].notna() & sma.notna())
    cash = (french("F-F_Research_Data_Factors_daily.csv", None)["RF"].add(1).resample("ME").prod() - 1)
    g, prev = {}, None
    for i in range(len(me) - 1):
        d, nx = me.index[i], me.index[i + 1]
        o = on.loc[d]
        if o.notna().sum() < 5:
            continue
        w = o * .2
        turn = 1.0 if prev is None else float((w - prev).abs().sum())
        g[nx] = float((w * r.loc[nx, faber]).sum() + (1 - w.sum()) * cash.get(nx, 0.0) - turn * .0005)
        prev = w
    g = pd.Series(g)
    # GEM: 12m return of SPY vs T-bills decides stocks vs AGG; if stocks, the better of SPY and EFA
    d12 = me.pct_change(12)
    gem, prev = {}, None
    for i in range(12, len(me) - 1):
        d, nx = me.index[i], me.index[i + 1]
        if np.isnan(d12.loc[d, ["SPY", "EFA"]]).any() or np.isnan(r.loc[nx, "AGG"]):
            continue
        cum = (1 + cash.loc[:d].tail(12)).prod() - 1
        a = ("SPY" if d12.loc[d, "SPY"] >= d12.loc[d, "EFA"] else "EFA") if d12.loc[d, "SPY"] > cum else "AGG"
        gem[nx] = float(r.loc[nx, a] - (0.0005 * 2 if a != prev and prev else 0)); prev = a
    gem = pd.Series(gem)
    sixty = .6 * r["SPY"] + .4 * r["AGG"]
    for k, s in {"GTAA5 (Faber)": g, "GEM (Antonacci)": gem}.items():
        res[k] = {"full": stats(s, 12), "spy": stats(r["SPY"].reindex(s.index), 12), "60/40": stats(sixty.reindex(s.index), 12)}
    res["GTAA5 (Faber)"]["after_publication_2008"] = {"s": stats(g.loc["2008":], 12), "spy": stats(r["SPY"].reindex(g.index).loc["2008":], 12)}
    res["GEM (Antonacci)"]["after_publication_2015"] = {"s": stats(gem.loc["2015":], 12), "spy": stats(r["SPY"].reindex(gem.index).loc["2015":], 12)}
    return res


def main():
    res = {}
    for k, fn in (("A1_A2", a1_a2), ("A4", a4), ("A5", a5), ("A6", a6)):
        res[k] = fn()
        (OUT / "longrun.json").write_text(json.dumps(res, indent=1, default=str))
        print(k, "done", flush=True)
    print(json.dumps(res, default=lambda o: float(o), indent=None)[:200])


if __name__ == "__main__":
    main()
