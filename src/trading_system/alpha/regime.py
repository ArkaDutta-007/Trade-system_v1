"""Macro / fragility regime layer: where are we, when was it like this, and what happened then.

Three jobs:

1. **State** (`regime_frame`): one row per trading day since 1990 describing the
   background — equity vol (VIX, VXN), oil (WTI/Brent level, realised oil vol,
   OVX), credit (Baa–10y spread), the curve (10y–3m), real rates and
   breakevens, the dollar, Nasdaq-vs-market relative strength and drawdown
   (the dot-com fingerprint), and, from the panel, market drawdown, breadth,
   average pairwise correlation, dispersion and an **AI-basket** (semis +
   AI-cloud names) vol / momentum / share-of-volume.  Every column is lagged
   one session (known before the open) and z-scored on an *expanding* window,
   so a value at date *t* uses only history up to *t*.
2. **Analogs** (`similarity`, `episodes`): the kernel-weighted distance from
   today's z-vector to every historical day, the nearest dates, and a
   similarity score against a curated list of stress episodes (Gulf War,
   LTCM, dot-com, 9/11, Iraq, 2008 oil spike, GFC, euro crisis, 2014-16 oil
   crash, China 2015, Q4 2018, Covid, 2022 inflation/Ukraine, SVB, 2025
   tariffs, 2025 Israel–Iran oil).
3. **Recalibration hooks**: per-date **analog weights** the calibrator uses
   as sample weights (so expected returns and bands are fitted on days that
   looked like today, not just on the last three years), a **fragility
   score** the book uses to scale gross, and a handful of date-level
   features the forecaster sees raw (never rank-transformed) so it can learn
   how the cross-section behaves in each background.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

_ANN = math.sqrt(252.0)

FRED_SERIES = {
    "VIXCLS": "vix", "VXNCLS": "vxn", "OVXCLS": "ovx", "DCOILWTICO": "wti", "DCOILBRENTEU": "brent",
    "BAA10Y": "baa_spread", "T10Y3M": "curve_10y3m", "DGS10": "ust10", "DGS2": "ust2", "T10YIE": "breakeven10",
    "DFII10": "real10", "DTWEXBGS": "dollar", "NASDAQCOM": "nasdaq", "DHHNGSP": "natgas",
}

# what the forecaster sees, raw, alongside the cross-sectional ranks
MACRO_FEATURES = ["vix_z", "oil_vol_21", "oil_ret_63", "baa_spread", "curve_10y3m", "mkt_dd_252", "avg_corr_21",
                  "ai_vol_21", "nasdaq_rel_252"]

# the regime vector used for similarity (z-scored, expanding)
REGIME_COLS = ["vix", "vix_ratio", "vxn_vix", "oil_vol_21", "oil_ret_63", "oil_level_z", "ovx", "baa_spread", "baa_chg_63",
               "curve_10y3m", "ust10_chg_63", "real10", "breakeven10", "dollar_ret_63", "nasdaq_rel_252", "nasdaq_dd",
               "mkt_dd_252", "mkt_vol_21", "mkt_trend_200", "breadth_200", "dispersion_21", "avg_corr_21",
               "ai_vol_21", "ai_ret_63", "ai_share", "ai_rel_63"]

# fragility = stress (fast: is the market already under pressure?) + imbalance (slow: how far are oil, rates,
# concentration and tech leadership from their norms — the dot-com / 2008-H1 / 2022 fingerprints)
STRESS_COLS = ["vix", "oil_vol_21", "ovx", "baa_chg_63", "avg_corr_21", "mkt_vol_21"]
IMBALANCE_COLS = ["oil_level_z", "real10", "ust10_chg_63", "ai_share", "nasdaq_rel_252", "ai_vol_21"]

AI_THEME = {
    "NVDA", "AMD", "AVGO", "TSM", "ASML", "LRCX", "AMAT", "KLAC", "MU", "ARM", "MRVL", "SMCI", "GFS", "TER", "SNDK",
    "INTC", "CDNS", "SNPS", "QCOM", "TXN", "CRWV", "IREN", "NBIS", "APLD", "GEV", "VRT", "BE", "STRL", "ORCL", "MSFT",
    "GOOGL", "GOOG", "META", "AMZN", "PLTR", "DELL", "HPE", "ANET", "CIEN", "COHR", "ALAB", "CRDO", "MPWR", "ON", "ADI",
}

EPISODES = [
    ("gulf_war_1990", date(1990, 8, 2), date(1991, 1, 16), "Iraq invades Kuwait; oil doubles, VIX >30, recession"),
    ("ltcm_1998", date(1998, 7, 20), date(1998, 10, 8), "Russia default / LTCM; credit seizure, VIX 45"),
    ("dotcom_bust", date(2000, 3, 10), date(2002, 10, 9), "Nasdaq −78%; tech concentration unwinds over 2.5 years"),
    ("sept11_2001", date(2001, 9, 11), date(2001, 10, 11), "9/11; market shut, oil and vol spike"),
    ("iraq_war_2003", date(2003, 1, 27), date(2003, 3, 31), "Run-up and start of the Iraq war; oil $38, VIX 35"),
    ("oil_spike_2008h1", date(2007, 10, 1), date(2008, 7, 14), "Oil to $147 while credit cracks; the quiet half of the GFC"),
    ("gfc_2008", date(2008, 9, 15), date(2009, 3, 9), "Lehman → March 2009 low; VIX 80, credit 20%"),
    ("euro_debt_2011", date(2011, 7, 22), date(2011, 10, 4), "US downgrade + euro sovereign crisis; −19%"),
    ("oil_crash_2014", date(2014, 10, 1), date(2016, 2, 11), "Oil $100 → $26; energy/credit stress, strong dollar"),
    ("china_2015", date(2015, 8, 11), date(2015, 9, 29), "Yuan devaluation; global growth scare"),
    ("q4_2018", date(2018, 10, 3), date(2018, 12, 24), "Fed tightening + trade war; −20%, oil −40%"),
    ("covid_2020", date(2020, 2, 19), date(2020, 4, 30), "Pandemic crash; VIX 82, oil negative"),
    ("inflation_ukraine_2022", date(2022, 1, 3), date(2022, 10, 12), "War, oil $120, fastest hikes in 40 years; Nasdaq −33%"),
    ("svb_2023", date(2023, 3, 8), date(2023, 3, 31), "Regional-bank run; rates whipsaw"),
    ("tariff_shock_2025", date(2025, 2, 19), date(2025, 4, 21), "Tariff announcements; −19%, VIX 52"),
    ("israel_iran_oil_2025", date(2025, 6, 12), date(2025, 6, 30), "Israel–Iran strikes; oil +20% then round-trip"),
]


# ── FRED (cached, refreshed daily) ────────────────────────────────────────────

def regime_dir(cfg) -> Path:
    return cfg.path("data_silver") / "regime"


def fetch_macro(cfg, max_age_h: float = 20.0, start: str = "1985-01-01") -> pl.DataFrame:
    """Wide daily frame (``date`` + FRED_SERIES values, forward-filled onto a business-day calendar).
    Cached per series under data/silver/regime/; refetched when older than ``max_age_h`` and a key exists;
    falls back to the cache on any failure."""
    d = regime_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("FRED_API_KEY")
    frames = []
    for sid, name in FRED_SERIES.items():
        p = d / f"fred_{sid}.parquet"
        fresh = p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600
        df = None
        if not fresh and key:
            try:
                from fredapi import Fred
                s = Fred(api_key=key).get_series(sid, observation_start=start)
                df = pl.DataFrame({"date": [x.date() for x in s.index.to_pydatetime()], name: s.values.astype(float)})
                df = df.with_columns(pl.col("date").cast(pl.Date), pl.col(name).fill_nan(None)).drop_nulls(name)
                df.write_parquet(p)
            except Exception as e:
                logger.warning(f"regime: FRED {sid} fetch failed ({str(e)[:80]}); using cache")
        if df is None and p.exists():
            df = pl.read_parquet(p)
        if df is not None and df.height:
            frames.append(df)
    if not frames:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="date", how="full", coalesce=True)
    out = out.sort("date")
    # business-day calendar, forward-filled (weekly/monthly series carry until the next print)
    cal = pl.DataFrame({"date": pl.date_range(out["date"].min(), out["date"].max(), interval="1d", eager=True)})
    cal = cal.filter(pl.col("date").dt.weekday() <= 5)
    out = cal.join(out, on="date", how="left").with_columns([pl.col(c).forward_fill() for c in out.columns if c != "date"])
    return out


# ── the state frame ───────────────────────────────────────────────────────────

def _panel_state(panel: pl.DataFrame) -> pl.DataFrame:
    """Date-level state from the panel: market drawdown/vol/trend/breadth/dispersion/correlation + AI basket."""
    T, D = "ticker", "date"
    p = panel.select(D, T, "ret_1", "vol_21", "log_dv_21", "mkt_vol_21", "mkt_trend_200", "breadth_200", "dispersion_21")
    mk = (p.group_by(D).agg(m=pl.col("ret_1").mean(), var_mean=(pl.col("vol_21") / _ANN).pow(2).mean(),
                            mkt_vol_21=pl.col("mkt_vol_21").first(), mkt_trend_200=pl.col("mkt_trend_200").first(),
                            breadth_200=pl.col("breadth_200").first(), dispersion_21=pl.col("dispersion_21").first())
            .sort(D)
            .with_columns(idx=(1 + pl.col("m").fill_null(0.0)).cum_prod())
            .with_columns(mkt_dd_252=(pl.col("idx") / pl.col("idx").rolling_max(252, min_samples=60) - 1),
                          avg_corr_21=((pl.col("mkt_vol_21") / _ANN).pow(2) / (pl.col("var_mean") + 1e-12)).clip(0.0, 1.0)))
    ai = p.filter(pl.col(T).is_in(list(AI_THEME)))
    ai_d = (ai.group_by(D).agg(ai_r=pl.col("ret_1").mean(), ai_dv=(pl.col("log_dv_21").exp() - 1).sum(), n_ai=pl.len()).sort(D)
              .with_columns(ai_vol_21=pl.col("ai_r").rolling_std(21, min_samples=10) * _ANN,
                            ai_idx=(1 + pl.col("ai_r").fill_null(0.0)).cum_prod())
              .with_columns(ai_ret_63=pl.col("ai_idx") / pl.col("ai_idx").shift(63) - 1))
    tot = p.group_by(D).agg(tot_dv=(pl.col("log_dv_21").exp() - 1).sum())
    out = (mk.join(ai_d, on=D, how="left").join(tot, on=D, how="left")
             .with_columns(ai_share=pl.col("ai_dv") / (pl.col("tot_dv") + 1.0),
                           mkt_ret_63=pl.col("idx") / pl.col("idx").shift(63) - 1))
    out = out.with_columns(ai_rel_63=pl.col("ai_ret_63") - pl.col("mkt_ret_63"))
    return out.select(D, "mkt_dd_252", "mkt_vol_21", "mkt_trend_200", "breadth_200", "dispersion_21", "avg_corr_21",
                      "ai_vol_21", "ai_ret_63", "ai_share", "ai_rel_63", "mkt_ret_63", "idx")


def regime_frame(cfg, panel: pl.DataFrame | None = None, macro: pl.DataFrame | None = None,
                 spy: pl.DataFrame | None = None) -> pl.DataFrame:
    """Daily regime state (raw columns, lagged one session so everything is known before the open)."""
    macro = macro if macro is not None else fetch_macro(cfg)
    m = macro.sort("date")
    r = lambda c, n: (pl.col(c) / pl.col(c).shift(1)).log().rolling_std(n, min_samples=n // 2) * _ANN   # realised vol
    m = m.with_columns(
        vix_ratio=pl.col("vix") / pl.col("vix").rolling_mean(63, min_samples=30),
        vxn_vix=pl.col("vxn") / pl.col("vix"),
        oil_vol_21=r("wti", 21), oil_ret_63=pl.col("wti") / pl.col("wti").shift(63) - 1,
        oil_level_z=((pl.col("wti") - pl.col("wti").rolling_mean(252, min_samples=120))
                     / (pl.col("wti").rolling_std(252, min_samples=120) + 1e-9)),
        baa_chg_63=pl.col("baa_spread") - pl.col("baa_spread").shift(63),
        ust10_chg_63=pl.col("ust10") - pl.col("ust10").shift(63),
        dollar_ret_63=pl.col("dollar") / pl.col("dollar").shift(63) - 1,
        nasdaq_dd=pl.col("nasdaq") / pl.col("nasdaq").rolling_max(252, min_samples=120) - 1,
        nasdaq_ret_252=pl.col("nasdaq") / pl.col("nasdaq").shift(252) - 1,
    )
    # market reference for Nasdaq relative strength: SPY (1993→) when given, else the panel EW index
    if spy is not None and spy.height:
        s = spy.select("date", spy_ret_252=pl.col("adj_close") / pl.col("adj_close").shift(252) - 1)
        m = m.join(s, on="date", how="left")
    if panel is not None:
        ps = _panel_state(panel)
        m = m.join(ps, on="date", how="left")
        if "spy_ret_252" not in m.columns:
            m = m.with_columns(spy_ret_252=pl.col("idx") / pl.col("idx").shift(252) - 1)
    if "spy_ret_252" in m.columns:
        m = m.with_columns(nasdaq_rel_252=pl.col("nasdaq_ret_252") - pl.col("spy_ret_252"))
    else:
        m = m.with_columns(nasdaq_rel_252=pl.lit(None, pl.Float64))
    keep = ["date"] + [c for c in REGIME_COLS if c in m.columns]
    out = m.select(keep)
    # lag one session: the row for date t carries the state known at t's open; NaN → null so the
    # expanding statistics skip them instead of being poisoned
    out = out.with_columns([pl.col(c).shift(1).fill_nan(None).alias(c) for c in keep if c != "date"])
    return out


def standardize(rf: pl.DataFrame, min_days: int = 750) -> pl.DataFrame:
    """Expanding z-scores (mean/std of everything up to and including the row) — point-in-time."""
    cols = [c for c in rf.columns if c != "date"]
    out = rf.sort("date")
    exprs = []
    for c in cols:
        mu = pl.col(c).cum_sum() / pl.col(c).cum_count()
        ex2 = pl.col(c).pow(2).cum_sum() / pl.col(c).cum_count()
        sd = (ex2 - mu.pow(2)).clip(0.0).sqrt()
        z = pl.when(pl.col(c).cum_count() >= min_days).then((pl.col(c) - mu) / (sd + 1e-9))
        exprs.append(z.clip(-4.0, 4.0).alias(c + "_z"))
    return out.with_columns(exprs)


# ── similarity ────────────────────────────────────────────────────────────────

@dataclass
class Analogs:
    as_of: date
    weights: pl.DataFrame          # date, dist, w  (kernel weight, normalised)
    nearest: pl.DataFrame          # top-k dates with distance + episode label
    episodes: pl.DataFrame         # episode, similarity (0-1), mean distance, n_days
    state: dict                    # today's raw + z values
    cols: list[str]


def _episode_of(d: date) -> str | None:
    for name, a, b, _ in EPISODES:
        if a <= d <= b:
            return name
    return None


def similarity(zf: pl.DataFrame, as_of: date | None = None, cols: list[str] | None = None, k: int = 15,
               min_gap_days: int = 63, tau: float = 252.0, exclude_recent_days: int = 126) -> Analogs:
    """Similarity of the ``as_of`` regime vector to every past day (NaN-aware: distance is the RMS over the
    dimensions both rows have, with a mild penalty for missing dimensions). Weights are rank-based,
    ``w ∝ exp(−rank/tau)``, so the analog mass always sits on the closest ~``tau`` days regardless of how the
    distances are scaled — a Gaussian kernel with a data-driven bandwidth could not separate 20 genuine
    analog days from the bulk."""
    zcols = [c + "_z" for c in (cols or REGIME_COLS) if c + "_z" in zf.columns]
    zf = zf.sort("date")
    as_of = as_of or zf["date"].max()
    today = zf.filter(pl.col("date") == as_of)
    if today.height == 0:
        today = zf.filter(pl.col("date") <= as_of).tail(1)
        as_of = today["date"][0]
    x = today.select(zcols).to_numpy()[0].astype(float)
    H = zf.filter(pl.col("date") < as_of - timedelta(days=exclude_recent_days))
    M = H.select(zcols).to_numpy().astype(float)
    both = np.isfinite(M) & np.isfinite(x)[None, :]
    n_both = both.sum(axis=1)
    diff = np.where(both, M - x[None, :], 0.0)
    rms = np.sqrt((diff ** 2).sum(axis=1) / np.maximum(n_both, 1))
    coverage = n_both / max(int(np.isfinite(x).sum()), 1)
    min_dims = max(2, min(6, int(0.5 * len(zcols))))
    dist = np.where(n_both >= min_dims, rms / np.sqrt(np.maximum(coverage, 0.25)), np.inf)
    fin = np.isfinite(dist)
    rank = np.full(len(dist), np.inf)
    order_all = np.argsort(np.where(fin, dist, np.inf))
    rank[order_all[: int(fin.sum())]] = np.arange(int(fin.sum()))
    w = np.where(fin, np.exp(-rank / max(tau, 1.0)), 0.0)
    w = w / w.sum() if w.sum() > 0 else w
    h = float(np.quantile(dist[fin], 0.10)) if fin.any() else 1.0     # only used to express episode similarity
    W = pl.DataFrame({"date": H["date"], "dist": dist, "w": w})
    # nearest, with a minimum spacing so one episode does not fill the list
    order = np.argsort(dist)
    picked, last_dates = [], []
    dates = H["date"].to_list()
    for i in order:
        if not np.isfinite(dist[i]):
            break
        if all(abs((dates[i] - d).days) >= min_gap_days for d in last_dates):
            picked.append(i); last_dates.append(dates[i])
        if len(picked) >= k:
            break
    nearest = pl.DataFrame({"date": [dates[i] for i in picked], "dist": [float(dist[i]) for i in picked]})
    nearest = nearest.with_columns(episode=pl.Series([_episode_of(d) or "" for d in nearest["date"]]))
    # episode similarity: kernel weight mass inside each episode vs. its share of days
    rows = []
    for name, a, b, desc in EPISODES:
        sel = W.filter((pl.col("date") >= a) & (pl.col("date") <= b) & pl.col("dist").is_finite())
        if sel.height == 0:
            continue
        rows.append({"episode": name, "from": str(a), "to": str(b), "n_days": sel.height,
                     "mean_dist": float(sel["dist"].mean()),
                     "similarity": float(np.exp(-0.5 * (sel["dist"].mean() / max(h, 1e-6)) ** 2)), "what": desc})
    eps = pl.DataFrame(rows).sort("similarity", descending=True) if rows else pl.DataFrame()
    state = {c: (float(today[c][0]) if today[c][0] is not None else None) for c in zf.columns if c != "date"}
    return Analogs(as_of, W, nearest, eps, state, zcols)


def fragility_score(zf: pl.DataFrame) -> pl.DataFrame:
    """Date → ``stress`` (mean z of the fast stress gauges), ``imbalance`` (mean z of the slow ones),
    ``fragility`` = their average, and a 0-1 ``crisis_like`` squash of the stress score."""
    sc = [c + "_z" for c in STRESS_COLS if c + "_z" in zf.columns]
    ic = [c + "_z" for c in IMBALANCE_COLS if c + "_z" in zf.columns]
    f = zf.select("date", *sc, *ic).with_columns(stress=pl.mean_horizontal(sc), imbalance=pl.mean_horizontal(ic))
    f = f.with_columns(fragility=(pl.col("stress") + pl.col("imbalance")) / 2,
                       crisis_like=(1 / (1 + (-(pl.col("stress") - 1.0) * 2.0).exp())))
    return f.select("date", "stress", "imbalance", "fragility", "crisis_like")


def gross_multiplier(fragility: float | None, lo: float = 1.0, hi: float = 2.0, floor: float = 0.5) -> float:
    """Gross scale from the fragility score: 1 below ``lo``, linearly down to ``floor`` at ``hi``."""
    if fragility is None or not np.isfinite(fragility):
        return 1.0
    if fragility <= lo:
        return 1.0
    return float(max(floor, 1.0 - (1.0 - floor) * (fragility - lo) / (hi - lo)))


# ── conditional performance ───────────────────────────────────────────────────

def conditional_skill(led: pl.DataFrame, weights: pl.DataFrame, horizon: int = 63) -> dict:
    """Analog-weighted vs unconditional realised rank-IC of the backtest ledger."""
    m = led.filter((pl.col("mode") == "backtest") & (pl.col("horizon") == horizon) & pl.col("realized_ret").is_not_null())
    ic = (m.group_by("date").agg(pl.corr("score", "realized_ret", method="spearman").alias("ic"), pl.len().alias("n"))
            .filter(pl.col("n") >= 20).join(weights, on="date", how="inner"))
    if ic.height == 0:
        return {}
    w = ic["w"].to_numpy(); x = ic["ic"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(w)
    w, x = w[ok], x[ok]
    ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
    return {"ic_uncond": float(x.mean()), "ic_analog": float((w * x).sum() / max(w.sum(), 1e-12)),
            "n_dates": int(len(x)), "effective_analog_days": round(ess, 1)}


def conditional_forward(daily: pl.DataFrame, weights: pl.DataFrame, horizon: int = 63) -> dict:
    """What the book (or a benchmark) did in the ``horizon`` sessions after analog days: weighted mean forward
    return, worst-decile forward return, and mean max-drawdown inside the window."""
    d = daily.sort("date")
    eq = d["equity"].to_numpy() if "equity" in d.columns else np.cumprod(1 + d["net_ret"].to_numpy())
    n = len(eq)
    fwd = np.full(n, np.nan); mdd = np.full(n, np.nan)
    for i in range(n - horizon):
        seg = eq[i:i + horizon + 1]
        fwd[i] = seg[-1] / seg[0] - 1
        mdd[i] = float((seg / np.maximum.accumulate(seg) - 1).min())
    f = pl.DataFrame({"date": d["date"], "fwd": fwd, "mdd": mdd}).join(weights, on="date", how="inner").drop_nulls(["fwd"])
    if f.height == 0:
        return {}
    w = f["w"].to_numpy(); x = f["fwd"].to_numpy(); m = f["mdd"].to_numpy()
    wn = w / max(w.sum(), 1e-12)
    order = np.argsort(x); cw = np.cumsum(wn[order])
    q10 = float(x[order][np.searchsorted(cw, 0.10)]); q50 = float(x[order][np.searchsorted(cw, 0.50)])
    return {"fwd_mean_analog": float((wn * x).sum()), "fwd_median_analog": q50, "fwd_q10_analog": q10,
            "mdd_mean_analog": float((wn * m).sum()), "fwd_mean_uncond": float(np.nanmean(x)),
            "mdd_mean_uncond": float(np.nanmean(m)), "n": int(len(x))}


# ── glue ──────────────────────────────────────────────────────────────────────

def load_spy(cfg) -> pl.DataFrame | None:
    p = cfg.path("data_bronze") / "ohlcv_daily.parquet"
    if not p.exists():
        return None
    s = pl.read_parquet(p, columns=["date", "ticker", "adj_close"]).filter(pl.col("ticker") == "SPY").sort("date")
    return s.with_columns(pl.col("date").cast(pl.Date)) if s.height else None


def build_state(cfg, panel: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(raw regime frame, z-scored frame) — the two tables everything else consumes."""
    rf = regime_frame(cfg, panel, spy=load_spy(cfg))
    return rf, standardize(rf)


def macro_features_for_panel(cfg, panel: pl.DataFrame) -> pl.DataFrame:
    """``date`` + MACRO_FEATURES (already lagged), ready to join onto the panel."""
    rf, zf = build_state(cfg, panel)
    f = rf.join(zf.select("date", "vix_z"), on="date", how="left")
    cols = [c for c in MACRO_FEATURES if c in f.columns]
    return f.select("date", *cols)
