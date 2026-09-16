#!/usr/bin/env python3
"""Picks v2 — quality-gated, risk-adjusted, diversified long-horizon picks.

WHY THIS EXISTS
---------------
`ts picks` ranks purely on the raw 12m forecast score. Measured on the live
2026-09-04 cross-section that produces:

    corr(score, 20d volatility)   = +0.22    → prefers volatile names
    corr(score, log dollar volume)= -0.19    → prefers ILLIQUID names
    top-25 mean vol 0.74  vs  bottom-25 0.41
    top-10 included PARA ($1.01, $321k/day), SQNS ($2.79, $92k/day),
                    SUMCF ($61k/day)  — untradeable at any real size

That is the classic "expected return ∝ variance" artifact: a high-vol lottery
ticket has the fattest right tail, so it always wins an unadjusted ranking.
The result is a concentrated basket of low-priced, illiquid, single-theme
(AI/crypto) names — exactly the "not good and not varied enough" complaint.

WHAT THIS DOES (each stage is independently switchable)
  1. QUALITY GATE   drop names below price / dollar-volume floors and above a
                    volatility ceiling — removes untradeable lottery tickets.
  2. SANITY GATE    clip absurd forecasts (a +26%/5d, 100%-confidence print is
                    a data artifact, not a signal) via a robust z-score.
  3. RISK ADJUST    rank by score / volatility^alpha (information-ratio style)
                    instead of raw score — neutralises the variance bias.
  4. DIVERSIFY      cap names per theme bucket (ai_semi, ai_cloud, crypto,
                    megacap, …) and per Yahoo sector, then greedily fill to
                    top_n so the book is not one bet expressed nine ways.

Usage:
    python3 picks_v2.py                      # table for the digest
    python3 picks_v2.py --top 15 --json out.json
    python3 picks_v2.py --compare            # v1 vs v2 side by side
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/ad2688/Desktop/Trade-system_v1")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from sectors import load as load_sectors, theme_of  # noqa: E402
from mathlib import (james_stein_shrink, ledoit_wolf, rmt_denoise,  # noqa: E402
                     hrp_weights, effective_n)

# ── tunables ────────────────────────────────────────────────────────────────
MIN_PRICE = 5.0             # no sub-$5 names (spread/borrow/quality cliff)
MIN_DOLLAR_VOL = 20_000_000  # $20M/day: a $10k order is <0.05% of volume
MAX_VOL_20D = 1.10          # drop >110% annualised vol (pure lottery tickets)
VOL_ALPHA = 0.5             # risk adjustment: score / vol**alpha
MAX_PER_THEME = 2           # correlated-cluster cap
MAX_PER_SECTOR = 3          # Yahoo-sector cap
FORECAST_Z_CLIP = 4.0       # robust-z clip for absurd scores
MAX_WEIGHT = 0.20           # single-name cap
CONVICTION_TILT = 1.0       # exp-tilt strength on the standardised signal


def latest_features() -> tuple[pl.DataFrame, object]:
    f = pl.read_parquet(REPO / "data/gold/features.parquet")
    last = f["date"].max()
    cols = ["ticker", "close", "adj_close", "vol_20d", "avg_dollar_volume_20",
            "rsi_14", "sma_gap_50", "mom_120d", "dist_52w_high", "beta_60"]
    d = f.filter(pl.col("date") == last).select([c for c in cols if c in f.columns])
    return d, last


def raw_picks(top_n: int = 400, horizon: int = 252, universe: str = "liquid") -> pl.DataFrame:
    from trading_system.config import get_config
    from trading_system.decision.longterm import build_longterm_picks
    cfg = get_config("configs/default.yaml").use_universe(universe)
    plan = build_longterm_picks(cfg, horizon=horizon, top_n=top_n)
    return pl.DataFrame(plan["picks"])


def apply_gates(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Quality + sanity gates. Returns (kept, rejection counts)."""
    n0 = df.height
    rej = {}

    kept = df.filter(pl.col("close") >= MIN_PRICE)
    rej["price<$%.0f" % MIN_PRICE] = n0 - kept.height

    n = kept.height
    kept = kept.filter(pl.col("avg_dollar_volume_20") >= MIN_DOLLAR_VOL)
    rej["illiquid<$%dM/d" % (MIN_DOLLAR_VOL // 1_000_000)] = n - kept.height

    n = kept.height
    kept = kept.filter(pl.col("vol_20d") <= MAX_VOL_20D)
    rej["vol>%.0f%%" % (MAX_VOL_20D * 100)] = n - kept.height

    # robust z-score on score; clip rather than drop so a genuinely strong name
    # is kept but a data-glitch outlier can't dominate the ranking
    med = kept["score"].median()
    mad = (kept["score"] - med).abs().median() or 1e-9
    kept = kept.with_columns(
        robust_z=((pl.col("score") - med) / (1.4826 * mad))
    ).with_columns(
        score_clipped=pl.when(pl.col("robust_z").abs() > FORECAST_Z_CLIP)
        .then(med + pl.col("robust_z").sign() * FORECAST_Z_CLIP * 1.4826 * mad)
        .otherwise(pl.col("score"))
    )
    rej["score_clipped"] = int(kept.filter(
        pl.col("robust_z").abs() > FORECAST_Z_CLIP).height)
    return kept, rej


MEASURED_IC = 0.011      # deployed ensemble's honest purged daily rank IC


def risk_adjust(df: pl.DataFrame, ic: float = MEASURED_IC) -> pl.DataFrame:
    """Shrink the forecasts, then rank by score per unit of risk.

    Two separate corrections, in order:
      * James-Stein shrinkage toward the cross-sectional mean, scaled by the
        model's MEASURED skill (IC ~ 0.011). Raw scores are wildly
        over-dispersed relative to that skill, so ranking on them treats noise
        as signal.
      * Divide by vol^alpha, converting an expected-return ranking into an
        information-ratio ranking. This is what removes the +0.22
        score-vs-volatility correlation that put $1 stocks at the top.
    """
    shrunk = james_stein_shrink(df["score_clipped"].to_numpy(), ic=ic)
    # DEMEAN before dividing by risk. The cross-sectional mean is common to
    # every name — it is market beta, not alpha. Leaving it in and dividing by
    # vol makes the constant dominate, which silently turns the ranking into a
    # pure low-volatility screen and throws the model's information away
    # (observed: BCS/NXPI at vol 0.18-0.22 topping the list on mean/vol alone).
    # Ranking the *residual* per unit of risk is the information-ratio quantity
    # we actually want.
    mu = float(np.nanmean(shrunk)) if len(shrunk) else 0.0
    return (df.with_columns(score_shrunk=pl.Series(shrunk))
              .with_columns(score_ra=(pl.col("score_shrunk") - mu)
                            / (pl.col("vol_20d").clip(0.05, None) ** VOL_ALPHA))
              .sort("score_ra", descending=True))


def allocate(picks: pl.DataFrame, lookback: int = 252) -> pl.DataFrame:
    """Attach HRP weights built on an RMT-denoised, Ledoit-Wolf covariance.

    Equal weight ignores correlation: ten AI names is one bet. HRP clusters the
    correlation matrix and splits the budget between clusters, so a tight
    cluster shares one allocation instead of getting one each. The covariance
    is shrunk (Ledoit-Wolf) and denoised (Marchenko-Pastur) first, because a
    252x10 sample covariance has noise-dominated small eigenvalues that any
    allocator would otherwise chase.
    """
    tickers = picks["ticker"].to_list()
    px = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet",
                         columns=["date", "ticker", "adj_close"])
    wide = (px.filter(pl.col("ticker").is_in(tickers))
              .pivot(index="date", on="ticker", values="adj_close")
              .sort("date").tail(lookback + 1))
    cols = [c for c in wide.columns if c != "date"]
    if len(cols) < 2:
        return picks.with_columns(weight=pl.lit(1.0 / max(len(tickers), 1)))
    arr = wide.select(cols).to_numpy().astype(float)
    rets = np.diff(arr, axis=0) / arr[:-1]
    ok = ~np.isnan(rets).any(axis=1)
    rets = rets[ok]
    if len(rets) < 30:
        return picks.with_columns(weight=pl.lit(1.0 / len(tickers)))
    cov = rmt_denoise(ledoit_wolf(rets), n_obs=len(rets))
    w_hrp = hrp_weights(cov)
    wmap = dict(zip(cols, w_hrp))
    base = np.array([wmap.get(t, 0.0) for t in tickers], dtype=float)
    if base.sum() <= 0:
        base = np.ones(len(tickers))
    base = base / base.sum()

    # HRP on its own is pure risk parity: it ignores the forecast and will hand
    # the LOWEST-ranked name the largest weight simply because it is quiet
    # (observed: the #10 pick at 45%, effective N 3.75 — worse diversification
    # than equal weight). Tilt the risk budget by conviction (rank-decay), then
    # cap single names and renormalise until the cap binds. This keeps HRP's
    # correlation-awareness while letting the signal set direction.
    n = len(tickers)
    # Exponential tilt on the STANDARDISED signal, not on ordinal rank: a name
    # only earns extra weight to the extent its score/risk actually separates
    # from the field. If the shrunk scores are all similar (which, at IC 0.011,
    # they usually are) the tilt is automatically mild and the book stays close
    # to risk parity — the correct behaviour when there is little to bet on.
    sr = picks["score_ra"].to_numpy().astype(float)
    sd = float(np.nanstd(sr, ddof=1)) if n > 1 else 0.0
    z = (sr - float(np.nanmean(sr))) / sd if sd > 1e-12 else np.zeros(n)
    z = np.clip(np.nan_to_num(z), -3, 3)
    w = base * np.exp(CONVICTION_TILT * z)
    w = w / w.sum()
    for _ in range(50):                          # iterative cap + renormalise
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            w = np.full(n, 1.0 / n)
            break
        w[free] += excess * w[free] / w[free].sum()
    return picks.with_columns(weight=pl.Series(w / w.sum()))


def diversify(df: pl.DataFrame, top_n: int, meta: dict) -> pl.DataFrame:
    """Greedy fill under theme + sector caps."""
    themes: dict[str, int] = {}
    sectors: dict[str, int] = {}
    picked = []
    for row in df.iter_rows(named=True):
        t = row["ticker"]
        th = theme_of(t, meta)
        sec = (meta.get(t) or {}).get("sector", "Unknown")
        if themes.get(th, 0) >= MAX_PER_THEME:
            continue
        if sectors.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        themes[th] = themes.get(th, 0) + 1
        sectors[sec] = sectors.get(sec, 0) + 1
        picked.append({**row, "theme": th, "sector": sec})
        if len(picked) >= top_n:
            break
    return pl.DataFrame(picked)


def build(top_n: int = 10, horizon: int = 252) -> dict:
    feats, asof = latest_features()
    meta = load_sectors()
    raw = raw_picks(horizon=horizon)
    df = raw.join(feats, on="ticker", how="left").drop_nulls(["vol_20d", "close"])
    gated, rej = apply_gates(df)
    ranked = risk_adjust(gated)
    final = diversify(ranked, top_n, meta)
    if final.height:
        final = allocate(final)
    return {
        "as_of": str(asof), "horizon": horizon,
        "n_raw": raw.height, "n_after_gates": gated.height,
        "rejections": rej, "picks": final.to_dicts(),
        "effective_n": (round(effective_n(final["weight"].to_numpy()), 2)
                        if "weight" in final.columns and final.height else None),
        "measured_ic": MEASURED_IC,
        "params": {"min_price": MIN_PRICE, "min_dollar_vol": MIN_DOLLAR_VOL,
                   "max_vol": MAX_VOL_20D, "vol_alpha": VOL_ALPHA,
                   "max_per_theme": MAX_PER_THEME, "max_per_sector": MAX_PER_SECTOR},
    }


def render(plan: dict) -> str:
    L = [f"Picks v2 · {plan['horizon']}d · as of {plan['as_of']} · "
         f"quality-gated + risk-adjusted + diversified",
         f"universe {plan['n_raw']} → {plan['n_after_gates']} after gates "
         f"({', '.join(f'{k}: {v}' for k, v in plan['rejections'].items() if v)})",
         ""]
    L.append(f"{'#':>2}  {'Ticker':<7}{'Entry':>10}{'Score':>9}{'Score/Risk':>11}"
             f"{'Wt%':>7}{'Vol':>7}{'$Vol/d':>9}  {'Theme':<16} {'Timing'}")
    for i, p in enumerate(plan["picks"], 1):
        dv = p["avg_dollar_volume_20"] / 1e6
        rsi = p.get("rsi_14") or 50
        gap = p.get("sma_gap_50") or 0
        timing = ("extended — wait" if rsi > 65 or gap > 0.10
                  else "pullback — accumulate" if rsi < 40 or gap < -0.05
                  else "neutral — scale in")
        theme = p["theme"][:16]
        wt = (p.get("weight") or 0.0) * 100
        L.append(f"{i:>2}  {p['ticker']:<7}{p['close']:>10.2f}{p['score']:>9.3f}"
                 f"{p['score_ra']:>11.3f}{wt:>7.1f}{p['vol_20d']:>7.2f}{dv:>8.0f}M  "
                 f"{theme:<16} {timing}")
    L.append("")
    if plan.get("effective_n"):
        L.append(f"Weights: HRP on RMT-denoised Ledoit-Wolf covariance · "
                 f"effective N = {plan['effective_n']} of {len(plan['picks'])} "
                 f"(equal-weight would be {len(plan['picks'])}) · "
                 f"scores James-Stein shrunk at IC={plan.get('measured_ic')}")
    L.append("Gates: price ≥ $%.0f · $vol ≥ $%dM/d · vol ≤ %.0f%% · "
             "≤%d per theme · ≤%d per sector · rank = score/vol^%.1f"
             % (MIN_PRICE, MIN_DOLLAR_VOL // 1_000_000, MAX_VOL_20D * 100,
                MAX_PER_THEME, MAX_PER_SECTOR, VOL_ALPHA))
    return "\n".join(L)


def compare(top_n: int = 10) -> str:
    feats, _ = latest_features()
    meta = load_sectors()
    raw = raw_picks().join(feats, on="ticker", how="left").drop_nulls(["vol_20d"])
    v1 = raw.sort("score", descending=True).head(top_n)
    v2 = pl.DataFrame(build(top_n)["picks"])
    def stats(d, label):
        th = {theme_of(t, meta) for t in d["ticker"]}
        return (f"{label:<10} median_price ${d['close'].median():>8.2f}  "
                f"median_vol {d['vol_20d'].median():.2f}  "
                f"median_$vol ${d['avg_dollar_volume_20'].median()/1e6:>7.0f}M  "
                f"themes {len(th)}")
    return ("v1 (current `ts picks`): " + ", ".join(v1["ticker"].to_list()) + "\n"
            + "v2 (gated+diversified): " + ", ".join(v2["ticker"].to_list()) + "\n\n"
            + stats(v1, "v1") + "\n" + stats(v2, "v2"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        print(compare(a.top))
        return 0
    plan = build(a.top, a.horizon)
    print(render(plan))
    if a.json:
        a.json.write_text(json.dumps(plan, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
