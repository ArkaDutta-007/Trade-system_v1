#!/usr/bin/env python3
"""Within-book weighting study, October 2026 — same picks, different weights.

Production holds the top-20 names by risk-adjusted score with weights = ½ equal + ½ inverse-vol, an 8%
cap, a 25% vol brake and the 200-day trend overlay. This study keeps the signal, the picks, the cap and
the overlays fixed and changes ONLY how capital is split across the names, so every difference is the
weighting rule. Schemes (pre-registered — the list and every constant were fixed before any run):

  prod        ½ equal + ½ inverse-vol                                  (production)
  equal       1/N
  invvol      1/σ
  invvar      1/σ²
  conviction  production weights × (1 + score rank within the book)    — top pick 2× the 20th
  grinold     z/σ: Grinold–Kahn α = IC·σ·z under a diagonal risk model → w ∝ α/σ² = IC·z/σ
  erc         equal risk contribution, Ledoit–Wolf covariance of the last 126 sessions
  hrp         hierarchical risk parity (López de Prado 2016), same covariance
  mv40        mean–variance optimiser over the top-40 candidates: max α'w − (λ/2)w'Σw, Σw = 1,
              0 ≤ w ≤ 8%, sector ≤ 30%; α = 0.04·σ·z, λ = 5 (the scale at which the unconstrained
              diagonal solution is about fully invested), Ledoit–Wolf Σ

Windows (the research protocol of docs/RESEARCH_2026-10.md):
  PIT   Nov 2024 → whole-market point-in-time universe, the lab's PIT base scores
  LONG  2009 → debiased (top-500 by liquidity that day), the lab's LONG base scores
  FULL  2004 → same eligibility, the production causal walk-forward scores (includes 2008)

Adoption rule (fixed in advance): a scheme replaces production only if its Sharpe beats production in
all three windows, the paired-bootstrap Sharpe difference has P(>0) ≥ 0.95 on LONG and FULL, and its
max drawdown is no more than 3pp worse anywhere. PBO across the nine schemes is reported.
Results → reports/alpha/lab_2026_10/weighting.json
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
from construction_2026_10 import run_book  # noqa: E402
from trading_system.alpha import model as M  # noqa: E402
from trading_system.alpha import panel as P  # noqa: E402
from trading_system.alpha import portfolio as PF  # noqa: E402
from trading_system.alpha.portfolio import BookConfig, market_regime_on  # noqa: E402
from trading_system.research.stats import deflated_sharpe, paired_bootstrap_ci, pbo_cscv, sharpe  # noqa: E402
from trading_system.research.wfbacktest import build_panel  # noqa: E402

OUT = LAB.OUT
SCHEMES = ["prod", "equal", "invvol", "invvar", "conviction", "grinold", "erc", "hrp", "mv40"]
COV_DAYS, IC, LAMBDA, SECTOR_MAX = 126, 0.04, 5.0, 0.30


# ── risk model ────────────────────────────────────────────────────────────────

class Risk:
    """Ledoit–Wolf covariance of the trailing COV_DAYS daily returns, using data up to the decision date."""

    def __init__(self, pdata):
        self.p = pdata
        self.col = {t: j for j, t in enumerate(pdata.tickers)}
        self.row = pdata.date_index()

    def cov(self, tickers, d, dvol: np.ndarray) -> np.ndarray:
        from sklearn.covariance import LedoitWolf
        i = self.row.get(d)
        k = len(tickers)
        fallback = np.diag(np.maximum(dvol, 1e-4) ** 2) * 252
        if i is None or i < 63:
            return fallback
        lo = max(0, i - COV_DAYS + 1)
        js = [self.col.get(t) for t in tickers]
        if any(j is None for j in js):
            return fallback
        R = self.p.ret[lo:i + 1][:, js]
        alive = self.p.alive[lo:i + 1][:, js]
        good = alive.sum(0) >= 63
        S = fallback.copy()
        if good.sum() >= 2:
            g = np.flatnonzero(good)
            S[np.ix_(g, g)] = LedoitWolf().fit(R[:, g]).covariance_ * 252
        # names with < 63 sessions of history: own variance from dvol, constant 0.3 correlation
        for b in np.flatnonzero(~good):
            sb = np.sqrt(S[b, b])
            for a in range(k):
                if a != b:
                    S[a, b] = S[b, a] = 0.3 * sb * np.sqrt(S[a, a])
        return S


def erc(S: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize
    n = len(S)
    f = lambda w: 0.5 * w @ S @ w - np.log(w).sum() / n            # noqa: E731
    g = lambda w: S @ w - 1.0 / (n * w)                              # noqa: E731
    x0 = 1.0 / np.sqrt(np.diag(S)); x0 /= x0.sum()
    r = minimize(f, x0, jac=g, method="L-BFGS-B", bounds=[(1e-8, None)] * n)
    w = np.maximum(r.x, 0)
    return w / w.sum()


def hrp(S: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    sd = np.sqrt(np.diag(S))
    C = np.clip(S / np.outer(sd, sd), -1, 1)
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    np.fill_diagonal(D, 0.0)
    order = list(leaves_list(linkage(squareform(D, checks=False), "single")))
    w = np.ones(len(S))
    stack = [order]
    while stack:
        c = stack.pop()
        if len(c) < 2:
            continue
        a, b = c[:len(c) // 2], c[len(c) // 2:]
        def cvar(ix):
            s = S[np.ix_(ix, ix)]
            iv = 1 / np.diag(s); iv /= iv.sum()
            return float(iv @ s @ iv)
        va, vb = cvar(a), cvar(b)
        al = 1 - va / (va + vb)
        w[a] *= al; w[b] *= 1 - al
        stack += [a, b]
    return w / w.sum()


def mean_variance(alpha: np.ndarray, S: np.ndarray, cap: float, sectors: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    from scipy.optimize import minimize
    n = len(alpha)
    f = lambda w: -(alpha @ w) + 0.5 * lam * w @ S @ w               # noqa: E731
    g = lambda w: -alpha + lam * S @ w                                # noqa: E731
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    for sec in np.unique(sectors):
        m = (sectors == sec).astype(float)
        if m.sum() * cap > SECTOR_MAX:
            cons.append({"type": "ineq", "fun": lambda w, m=m: SECTOR_MAX - m @ w, "jac": lambda w, m=m: -m})
    x0 = np.full(n, 1.0 / n)
    r = minimize(f, x0, jac=g, method="SLSQP", bounds=[(0.0, cap)] * n, constraints=cons,
                 options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(r.x if r.success else x0, 0, None)
    w[w < 0.005] = 0.0
    return w / w.sum()


# ── the book: production selection + overlays, pluggable weighting ───────────

def select(scores, cols, book: BookConfig, sectors):
    """Production's pick list (identical to PF.target_weights up to the weighting step)."""
    s = np.asarray(scores, dtype=float)
    dvol = np.asarray(cols["dvol"], dtype=float)
    ok = np.isfinite(s) & np.isfinite(dvol) & (dvol > 0)
    if "adv" in cols:
        ok &= np.asarray(cols["adv"], dtype=float) >= book.min_dollar_volume
    if "price" in cols:
        ok &= np.asarray(cols["price"], dtype=float) >= book.min_price
    ok &= dvol <= book.max_daily_vol
    if ok.sum() < max(3, book.top_k // 4):
        return None
    key = np.full(len(s), -np.inf)
    key[ok] = (s[ok] - s[ok].mean()) / np.maximum(dvol[ok], 1e-4) ** book.vol_power
    picked, per = [], {}
    for j in np.argsort(-key):
        if not np.isfinite(key[j]):
            break
        if sectors is not None:
            if per.get(sectors[j], 0) >= book.max_per_sector:
                continue
            per[sectors[j]] = per.get(sectors[j], 0) + 1
        picked.append(j)
        if len(picked) >= book.top_k:
            break
    if not picked:
        return None
    z = np.zeros(len(s))
    z[ok] = (s[ok] - s[ok].mean()) / (s[ok].std() + 1e-12)
    return np.array(picked), key, z, dvol


def finish(w: np.ndarray, dvol: np.ndarray, book: BookConfig, regime_on: bool) -> np.ndarray:
    """Production's tail: cap → normalise → vol brake → trend overlay."""
    w = PF._cap(w, book.max_weight)
    w = w / w.sum() * book.gross_exposure
    if book.vol_target:
        pv = PF.portfolio_vol(w, dvol, book.avg_corr)
        if pv > book.vol_target:
            w = w * (book.vol_target / pv)
    mult = book.regime_scale if not regime_on else 1.0
    mult = max(min(mult, 1.0), book.min_gross_mult) if mult < 1.0 else 1.0
    return w * mult


def scheme_fn(scheme: str, book: BookConfig, sector_of: dict, regime: dict, risk: Risk):
    def fn(scores, cols, cfg):
        n = len(scores)
        tick = np.asarray(cols["ticker"])
        sectors = np.array([sector_of.get(t, "unknown") for t in tick])
        on = regime.get(cols.get("date"), True)
        mv = scheme.startswith("mv")
        k_mv = int(scheme[2:].split("_")[0]) if mv else 0
        lam = float(scheme.split("_l")[1]) if mv and "_l" in scheme else LAMBDA
        sel = select(scores, cols, replace(book, top_k=k_mv, max_per_sector=max(5, k_mv // 4)) if mv else book, sectors)
        if sel is None:
            return np.zeros(n)
        idx, key, z, dvol = sel
        v = np.maximum(dvol[idx], 1e-4)
        if scheme == "prod":
            iv = 1 / v
            raw = book.inv_vol_blend * iv / iv.sum() + (1 - book.inv_vol_blend) / len(idx)
        elif scheme == "equal":
            raw = np.ones(len(idx))
        elif scheme == "invvol":
            raw = 1 / v
        elif scheme == "invvar":
            raw = 1 / v ** 2
        elif scheme == "conviction":
            iv = 1 / v
            base = book.inv_vol_blend * iv / iv.sum() + (1 - book.inv_vol_blend) / len(idx)
            k = key[idx]
            rank = (k.argsort().argsort()) / max(len(idx) - 1, 1)       # 0 = weakest pick, 1 = strongest
            raw = base * (1 + rank)
        elif scheme == "grinold":
            raw = np.maximum(z[idx], 0.1) / v
        else:
            S = risk.cov(list(tick[idx]), cols.get("date"), dvol[idx])
            if scheme == "erc":
                raw = erc(S)
            elif scheme == "hrp":
                raw = hrp(S)
            else:  # mv<K>[_l<λ>]
                alpha = IC * np.sqrt(np.diag(S)) * z[idx]
                raw = mean_variance(alpha, S, book.max_weight, sectors[idx], lam)
        w = np.zeros(n)
        w[idx] = raw / raw.sum()
        return finish(w, dvol, book, on)
    return fn


# ── windows ───────────────────────────────────────────────────────────────────

def window_pit():
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
    return pdata, comp, gate, sector_of, regime, oos


def _hist_gate():
    pn = P.load_panel(LAB.cfg)
    frame, _ = M.prepare(pn, LAB.BASE_FEATS)
    sector_of = dict(pn.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pn.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    gate = (frame.select("date", "ticker", "adj_close", "close", "vol_63", "liq_rank")
                 .with_columns(elig=(pl.col("close") >= 5) & (pl.col("liq_rank") <= LAB.EVAL_TOP) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07))
                 .select("date", "ticker", "adj_close", "elig"))
    return gate, sector_of, regime


def window_long(gate, sector_of, regime):
    pdata = build_panel(P.load_prices(LAB.cfg, start="2006-01-01"))
    return pdata, LAB.zcomp(pl.read_parquet(OUT / "long_base.parquet")), gate, sector_of, regime, date(2009, 1, 1)


def window_full(gate, sector_of, regime):
    sc = pl.read_parquet(LAB.cfg.path("data_bronze").parent / "ledger" / "alpha_backtest_scores.parquet")
    comp = LAB.zcomp(sc.filter(pl.col("horizon").is_in(list(LAB.H))))
    pdata = build_panel(P.load_prices(LAB.cfg, start="2002-01-01"))
    return pdata, comp, gate, sector_of, regime, date(2004, 1, 2)


def check_prod_matches_production(pdata, sector_of, regime, book):
    """Guard: the harness's 'prod' scheme must reproduce production's target_weights exactly."""
    rng = np.random.default_rng(0)
    fn = scheme_fn("prod", book, sector_of, regime, Risk(pdata))
    for _ in range(25):
        n = 120
        tk = rng.choice(np.asarray(pdata.tickers), n, replace=False)
        cols = {"dvol": rng.uniform(0.005, 0.05, n), "adv": np.full(n, 1e8), "price": np.full(n, 50.0), "ticker": tk,
                "date": pdata.dates[-1]}
        s = rng.normal(size=n)
        sectors = np.array([sector_of.get(t, "unknown") for t in tk])
        ref = PF.target_weights(s, cols, book, sectors, regime_on=regime.get(cols["date"], True))
        assert np.allclose(fn(s, cols, None), ref, atol=1e-12), "harness 'prod' differs from production"


def stats(r: np.ndarray, turnover: float | None = None) -> dict:
    out = LAB.book_stats(r)
    out["calmar"] = out["cagr"] / abs(out["maxdd"]) if out["maxdd"] < 0 else None
    if turnover is not None:
        out["turnover"] = turnover
    return out


def run_window(name, pdata, comp, gate, sector_of, regime, oos) -> tuple[dict, dict]:
    book = replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0)
    check_prod_matches_production(pdata, sector_of, regime, book)
    risk = Risk(pdata)
    rets, res = {}, {}
    for sch in SCHEMES:
        t0 = time.time()
        r = run_book(pdata, comp, gate, scheme_fn(sch, book, sector_of, regime, risk), oos)
        rets[sch] = r
        res[sch] = stats(r)
        LAB.log(f"{name} {sch:<10} CAGR {res[sch]['cagr']:+.1%}  Sharpe {res[sch]['sharpe']:.2f}  "
                f"MaxDD {res[sch]['maxdd']:+.1%}  ({time.time() - t0:.0f}s)")
    for sch in SCHEMES[1:]:
        d = paired_bootstrap_ci(rets[sch], rets["prod"], n_boot=500)
        x = rets[sch] - rets["prod"]
        res[sch]["vs_prod"] = {"sharpe_diff": d, "excess_ann": float(x.mean() * 252)}
    R = np.column_stack([rets[s] for s in SCHEMES])
    trials = np.array([sharpe(R[:, j], annualise=False) for j in range(R.shape[1])])
    best = int(np.argmax(trials))
    res["_pbo"] = pbo_cscv(R, n_blocks=16) if len(R) > 400 else None
    res["_best"] = {"scheme": SCHEMES[best], "dsr": deflated_sharpe(R[:, best], trials)}
    res["_window"] = [str(oos), len(R)]
    return res, rets


ROBUST = ["prod", "mv20", "mv30", "mv40_l2.5", "mv40", "mv40_l10", "mv60"]


def robustness():
    """Diagnostics for the optimiser (NOT a selection step — production is compared only with the
    pre-registered mv40 @ λ=5): same-20-picks optimiser (mv20 → is it weighting or selection?),
    candidate-set size and risk aversion λ."""
    global SCHEMES
    SCHEMES = ROBUST
    p = OUT / "weighting.json"
    out = json.loads(p.read_text())
    out["robust_pit"], _ = run_window("PIT", *window_pit())
    gate, sector_of, regime = _hist_gate()
    out["robust_long"], _ = run_window("LONG", *window_long(gate, sector_of, regime))
    out["robust_full"], _ = run_window("FULL", *window_full(gate, sector_of, regime))
    p.write_text(json.dumps(out, indent=1, default=str))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "robust":
        return robustness()
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pit", "long", "full"]
    p = OUT / "weighting.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    if "pit" in which:
        out["pit"], _ = run_window("PIT", *window_pit())
        p.write_text(json.dumps(out, indent=1, default=str))
    if "long" in which or "full" in which:
        gate, sector_of, regime = _hist_gate()
        if "long" in which:
            out["long"], _ = run_window("LONG", *window_long(gate, sector_of, regime))
            p.write_text(json.dumps(out, indent=1, default=str))
        if "full" in which:
            out["full"], _ = run_window("FULL", *window_full(gate, sector_of, regime))
            p.write_text(json.dumps(out, indent=1, default=str))
    LAB.log(f"weighting study → {p}")


if __name__ == "__main__":
    main()
