#!/usr/bin/env python3
"""Small RL / online-learning pilots, October 2026 — where could a small learned policy help the book?

Our decisions are monthly and the signal is weak, so the realistic "RL" here is small: a policy with a
handful of parameters, trained walk-forward, judged against the simple rule it would replace.

  A  Exposure policy. A tiny policy (linear, or an 8-unit MLP ensemble) sets the book's gross exposure
     g ∈ [0.35, 1] each day from market state known at the prior close (SPY trend/vol/drawdown, VIX,
     credit spread, curve, the book's own recent vol and return). Exposure does not change future
     states except through trading costs, so this is a contextual bandit with a known reward; it is
     trained by deterministic policy gradient through that reward (Sharpe of g·r_book − costs, L2), refit
     every January on all prior data (2004 →), applied out of sample 2009 → 2026. Compared in the same
     daily frame with always-invested and with the 200-day trend rule (½ gross below the average).
  B  Horizon blend as online learning with experts. Six fixed blends of the 5/21/63-day forecasts run
     as separate books (production rules, overlays on); each month a learner picks the mix to hold from
     past months only: Hedge (exponential weights, η = 5), discounted Hedge (0.97/month), follow-the-
     leader on trailing 36-month Sharpe, and a contextual leader (best arm in past months with the same
     trend state). Compared with the fixed production blend (21d 30% / 63d 70%).

All constants were fixed before the first run. Results → reports/alpha/lab_2026_10/rl.json
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
from construction_2026_10 import run_book  # noqa: E402
from trading_system.alpha import panel as P  # noqa: E402
from trading_system.alpha.backtest import _weight_fn  # noqa: E402
from trading_system.alpha.portfolio import BookConfig  # noqa: E402
from trading_system.research.stats import paired_bootstrap_ci, sharpe  # noqa: E402
from trading_system.research.wfbacktest import build_panel  # noqa: E402
from weighting_2026_10 import _hist_gate  # noqa: E402

OUT = LAB.OUT
OOS, TEST_FROM = date(2004, 1, 2), 2009
G_LO, COST_BPS = 0.35, 10.0
ARMS = {"21d": {21: 1.0}, "63d": {63: 1.0}, "prod 21/63 = 30/70": {21: 0.3, 63: 0.7}, "21/63 = 50/50": {21: 0.5, 63: 0.5},
        "5/21/63 = 20/30/50": {5: 0.2, 21: 0.3, 63: 0.5}, "5/21/63 equal": {5: 1 / 3, 21: 1 / 3, 63: 1 / 3}}
PROD_ARM = "prod 21/63 = 30/70"


def zcomp_w(sc: pl.DataFrame, w: dict[int, float]) -> pl.DataFrame:
    z = sc.filter(pl.col("horizon").is_in(list(w))).with_columns(
        z=(pl.col("score") - pl.col("score").mean().over(["date", "horizon"])) / (pl.col("score").std().over(["date", "horizon"]) + 1e-9))
    ww = pl.DataFrame({"horizon": list(w), "w": list(w.values())}).with_columns(pl.col("horizon").cast(z.schema["horizon"]))
    return z.join(ww, on="horizon").group_by(["date", "ticker"]).agg(comp=(pl.col("z") * pl.col("w")).sum())


def stats(r: np.ndarray, per_year: int = 252) -> dict:
    eq = np.cumprod(1 + r)
    y = len(r) / per_year
    return {"cagr": float(eq[-1] ** (1 / y) - 1), "sharpe": float(r.mean() / (r.std() + 1e-12) * np.sqrt(per_year)),
            "vol": float(r.std() * np.sqrt(per_year)), "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min())}


# ── shared data ───────────────────────────────────────────────────────────────

def books():
    """Daily returns 2004 → of the production book with and without overlays, + per-arm books (overlays on)."""
    gate, sector_of, regime = _hist_gate()
    pdata = build_panel(P.load_prices(LAB.cfg, start="2002-01-01"))
    sc = pl.read_parquet(LAB.cfg.path("data_bronze").parent / "ledger" / "alpha_backtest_scores.parquet")
    base = replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0)
    no_ovl = replace(base, regime_scale=1.0, vol_target=None)
    out = {"no_overlay": run_book(pdata, zcomp_w(sc, ARMS[PROD_ARM]), gate, _weight_fn(no_ovl, sector_of, None), OOS)}
    for name, w in ARMS.items():
        out[name] = run_book(pdata, zcomp_w(sc, w), gate, _weight_fn(base, sector_of, regime), OOS)
        LAB.log(f"arm {name}: Sharpe {sharpe(out[name]):.2f}")
    n = min(len(v) for v in out.values())
    dates = [d for d in pdata.dates if d >= OOS][:n]
    return {k: v[:n] for k, v in out.items()}, dates


def state(dates: list[date], r_book: np.ndarray) -> pl.DataFrame:
    """Features known at the close of each date (used for the next day's exposure)."""
    spy = (pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet", columns=["date", "ticker", "adj_close"])
             .filter(pl.col("ticker") == "SPY").sort("date").with_columns(pl.col("date").cast(pl.Date)))
    p, r = pl.col("adj_close"), pl.col("adj_close").pct_change()
    spy = spy.with_columns(trend200=p / p.rolling_mean(200) - 1, trend50=p / p.rolling_mean(50) - 1,
                           vol21=r.rolling_std(21) * 252 ** 0.5, vol63=r.rolling_std(63) * 252 ** 0.5,
                           dd252=p / p.rolling_max(252) - 1, ret21=p / p.shift(21) - 1).drop("ticker", "adj_close")
    df = pl.DataFrame({"date": dates, "r_book": r_book}).join(spy, on="date", how="left")
    R = LAB.cfg.path("data_silver") / "regime"
    for f, c in (("fred_VIXCLS", "vix"), ("fred_BAA10Y", "baa_spread"), ("fred_T10Y3M", "curve_10y3m")):
        s = pl.read_parquet(R / f"{f}.parquet").with_columns(pl.col("date").cast(pl.Date)).sort("date")
        s = s.with_columns(pl.col("date") + pl.duration(days=1))                 # FRED publication lag
        df = df.sort("date").join_asof(s, on="date", strategy="backward")
    df = df.with_columns(baa_chg63=pl.col("baa_spread") - pl.col("baa_spread").shift(63),
                         bvol21=pl.col("r_book").rolling_std(21) * 252 ** 0.5,
                         bret63=(1 + pl.col("r_book")).log().rolling_sum(63))
    return df.fill_null(strategy="forward").fill_null(0.0)


FEATS = ["trend200", "trend50", "vol21", "vol63", "dd252", "ret21", "vix", "baa_spread", "baa_chg63", "curve_10y3m", "bvol21", "bret63"]


# ── A: exposure policy ────────────────────────────────────────────────────────

def train_policy(X: np.ndarray, r_next: np.ndarray, hidden: int, seed: int, steps: int = 400, l2: float = 1e-3):
    import torch
    torch.manual_seed(seed)
    Xt, rt = torch.tensor(X, dtype=torch.float32), torch.tensor(r_next, dtype=torch.float32)
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
    else:
        net = torch.nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        net[-1].bias.fill_(2.0) if hidden else net.bias.fill_(2.0)          # start near fully invested
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    for _ in range(steps):
        g = G_LO + (1 - G_LO) * torch.sigmoid(net(Xt).squeeze(-1))
        rp = g * rt - COST_BPS / 1e4 * torch.abs(torch.diff(g, prepend=g[:1]))
        loss = -(rp.mean() / (rp.std() + 1e-8)) * np.sqrt(252) + l2 * sum((w ** 2).sum() for w in net.parameters())
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def exposure_pilot(rets: dict, dates: list[date]) -> dict:
    import torch
    df = state(dates, rets["no_overlay"])
    X_all = df.select(FEATS).to_numpy().astype(float)
    r = df["r_book"].to_numpy()
    yrs = np.array([d.year for d in df["date"].to_list()])
    warm = 252
    g_lin, g_mlp = np.full(len(r), np.nan), np.full(len(r), np.nan)
    for Y in range(TEST_FROM, max(yrs) + 1):
        tr = np.arange(warm, len(r) - 1)
        tr = tr[yrs[tr + 1] < Y]                                                  # reward day must precede the test year too
        te = np.where(yrs == Y)[0]
        mu, sd = X_all[tr].mean(0), X_all[tr].std(0) + 1e-9                      # training-window scaling only
        Xtr, Xte = (X_all[tr] - mu) / sd, (X_all[te] - mu) / sd
        nets = [train_policy(Xtr, r[tr + 1], 0, 0)] + [train_policy(Xtr, r[tr + 1], 8, s) for s in range(5)]
        with torch.no_grad():
            gs = [(G_LO + (1 - G_LO) * torch.sigmoid(n(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1))).numpy() for n in nets]
        g_lin[te], g_mlp[te] = gs[0], np.mean(gs[1:], axis=0)
        LAB.log(f"exposure {Y}: mean g linear {gs[0].mean():.2f} · mlp {np.mean(gs[1:]):.2f}")
    g_lin, g_mlp = np.nan_to_num(g_lin, nan=1.0), np.nan_to_num(g_mlp, nan=1.0)   # before the first test year: invested
    trend = np.where(df["trend200"].to_numpy() >= 0, 1.0, 0.5)
    test = yrs >= TEST_FROM
    def run(g):
        gl = np.roll(g, 1); gl[0] = g[0]                                         # decided at t-1's close
        cost = COST_BPS / 1e4 * np.abs(np.diff(gl, prepend=gl[0]))
        return (gl * r - cost)[test]
    series = {"always invested": r[test], "200-day trend rule (daily, ½ below)": run(trend),
              "learned linear policy": run(g_lin), "learned MLP policy (5-seed ensemble)": run(g_mlp),
              "production book (monthly overlay + vol brake, simulator)": rets[PROD_ARM][test]}
    res = {k: stats(v) for k, v in series.items()}
    ref = series["200-day trend rule (daily, ½ below)"]
    for k in ("learned linear policy", "learned MLP policy (5-seed ensemble)"):
        res[k]["vs_trend_rule"] = paired_bootstrap_ci(series[k], ref, n_boot=500)
        res[k]["mean_gross"] = float(np.nanmean((g_lin if "linear" in k else g_mlp)[test]))
    res["_window"] = [str(df["date"][int(np.argmax(test))]), str(df["date"][-1])]
    eps = {"2008-09": (date(2008, 1, 1), date(2009, 6, 30)), "2020 covid": (date(2020, 2, 19), date(2020, 6, 30)),
           "2022 bear": (date(2022, 1, 3), date(2022, 12, 30)), "2025 tariff + rebound": (date(2025, 2, 19), date(2025, 7, 31))}
    d = np.array(df["date"].to_list())[test]
    res["_episodes"] = {k: {n: float(np.prod(1 + v[(d >= a) & (d <= b)]) - 1) for n, (a, b) in eps.items() if ((d >= a) & (d <= b)).any()}
                        for k, v in series.items()}
    return res


# ── B: online experts over horizon blends ────────────────────────────────────

def monthly(r: np.ndarray, dates: list[date]) -> tuple[np.ndarray, list]:
    df = pl.DataFrame({"date": dates, "r": r}).with_columns(m=pl.col("date").dt.strftime("%Y-%m"))
    g = df.group_by("m", maintain_order=True).agg(r=(1 + pl.col("r")).product() - 1)
    return g["r"].to_numpy(), g["m"].to_list()


def experts_pilot(rets: dict, dates: list[date]) -> dict:
    names = list(ARMS)
    M = np.column_stack([monthly(rets[k], dates)[0] for k in names])
    months = monthly(rets[names[0]], dates)[1]
    trend_m = (state(dates, rets["no_overlay"]).with_columns(m=pl.col("date").dt.strftime("%Y-%m"))
                 .group_by("m", maintain_order=True).agg(pl.col("trend200").last())["trend200"].to_numpy())
    L = np.log1p(M)
    T, K = M.shape
    start = next(i for i, m in enumerate(months) if int(m[:4]) >= TEST_FROM)
    learners = {"Hedge (η=5)": [], "discounted Hedge (0.97/mo)": [], "follow-the-leader (36m Sharpe)": [],
                "contextual leader (same trend state)": []}
    for t in range(start, T):
        past = L[:t]
        w = np.exp(5 * past.sum(0)); learners["Hedge (η=5)"].append(w / w.sum())
        disc = 0.97 ** np.arange(t - 1, -1, -1)[:, None]
        w = np.exp(5 * (disc * past).sum(0)); learners["discounted Hedge (0.97/mo)"].append(w / w.sum())
        win = M[max(0, t - 36):t]
        s = win.mean(0) / (win.std(0) + 1e-9); w = np.eye(K)[int(np.argmax(s))]; learners["follow-the-leader (36m Sharpe)"].append(w)
        same = (np.sign(trend_m[:t]) == np.sign(trend_m[t - 1])) if t > 0 else np.ones(0, bool)
        sub = M[:t][same] if same.sum() >= 12 else M[:t]
        s = sub.mean(0) / (sub.std(0) + 1e-9); w = np.eye(K)[int(np.argmax(s))]; learners["contextual leader (same trend state)"].append(w)
    res = {"arms": {k: stats(M[start:, i], 12) for i, k in enumerate(names)}}
    prod = M[start:, names.index(PROD_ARM)]
    for k, W in learners.items():
        W = np.array(W)
        rr = (W * M[start:]).sum(1)
        res[k] = stats(rr, 12)
        res[k]["vs_production"] = paired_bootstrap_ci(rr, prod, stat=lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(12), n_boot=1000, mean_block=3)
        res[k]["avg_weights"] = {n: round(float(W[:, i].mean()), 3) for i, n in enumerate(names)}
    res["_window"] = [months[start], months[-1]]
    return res


def main():
    rets, dates = books()
    out = {"exposure": exposure_pilot(rets, dates)}
    (OUT / "rl.json").write_text(json.dumps(out, indent=1, default=str))
    out["experts"] = experts_pilot(rets, dates)
    (OUT / "rl.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in out["exposure"].items():
        if not k.startswith("_"):
            LAB.log(f"A {k:<58} CAGR {v['cagr']:+.1%} Sharpe {v['sharpe']:.2f} MaxDD {v['maxdd']:+.1%}")
    for k, v in out["experts"].items():
        if not k.startswith("_") and k != "arms":
            LAB.log(f"B {k:<40} CAGR {v['cagr']:+.1%} Sharpe {v['sharpe']:.2f} MaxDD {v['maxdd']:+.1%} "
                    f"ΔSharpe {v['vs_production']['point']:+.2f} [{v['vs_production']['lo']:+.2f},{v['vs_production']['hi']:+.2f}]")
    for k, v in out["experts"]["arms"].items():
        LAB.log(f"B arm {k:<24} CAGR {v['cagr']:+.1%} Sharpe {v['sharpe']:.2f} MaxDD {v['maxdd']:+.1%}")


if __name__ == "__main__":
    main()
