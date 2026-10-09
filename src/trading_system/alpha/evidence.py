"""Live evidence: is the alpha engine working in real time? Rules pre-registered 2026-10-09.

What live data CAN and CANNOT decide. The engine's edge is a cross-sectional rank IC of about 0.03 at
21 days, while the IC of a single date swings by ±0.13 (factor regimes move the whole cross-section).
Forecasts made on consecutive days overlap, so 21 daily 21-day forecasts are roughly ONE independent
observation. Telling IC 0.03 from 0 at 2σ therefore takes ~70 independent months — six years. Live
data will not confirm the edge in weeks or months; what it can do quickly is catch a model or data
pipeline that is broken (strongly negative IC) or performing far below what the research measured.
The rules below are written for that job and were fixed before the first 21-day forecast matured:

  RED    live IC t-stat ≤ −2 with ≥ 2 effective (non-overlapping) periods at any horizon
         → something is broken: check data, features and the model before trusting the picks.
  AMBER  live IC ≥ 2 SE below the research expectation with ≥ 3 effective periods
         → skill well below the backtest (decay / selection bias): re-research before relying on it.
  GREEN  otherwise. "Not yet informative" while the standard error exceeds the expected IC.

Statistics: per-date Spearman IC of score vs realised return over every forecast name; overlap-aware
effective sample size (triangular autocorrelation of overlapping h-day windows); per-date IC dispersion
from the causal backtest (2016 →), far more stable than a handful of live dates; posterior P(IC > 0)
with a N(expected, 0.03²) prior. The paper book is reported against SPY and RSP for information only —
at a 3-4%/yr edge and ~10% tracking error a book-level verdict needs decades, not months.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

HORIZONS = (5, 21, 63)
# Research expectations for the LIVE rank IC (today's 1000-name universe, no look-ahead in membership):
#   21d/63d between the debiased 2009→ test (0.024 / 0.036) and the point-in-time test (0.047 / 0.051);
#   5d from the causal backtest 2016→ (0.018), shaded for the same selection bias.
EXPECTED_IC = {5: 0.015, 21: 0.030, 63: 0.040}
IC_SD_DEFAULT = {5: 0.153, 21: 0.127, 63: 0.109}   # backtest 2016→, used when the ledger has no history
PRIOR_SD = 0.03
RED_T, AMBER_Z, RED_MIN_NEFF, AMBER_MIN_NEFF = -2.0, -2.0, 2.0, 3.0
# Review points: the first two fire when the first live forecast of that horizon has been tallied,
# the rest on the calendar date (each alerts once).
CHECKPOINTS = [
    ("21d-first", None, "first 21-day forecasts tallied — pipeline check, no performance verdict possible yet"),
    ("2026-11-16", date(2026, 11, 16), "one month of 21-day results — tripwire check"),
    ("63d-first", None, "first 63-day forecasts tallied — pipeline check"),
    ("2027-03-15", date(2027, 3, 15), "six months live — first consistency check against the research"),
    ("2027-09-15", date(2027, 9, 15), "one year live — consistency check, paper book vs SPY/RSP"),
]


def n_eff(n_dates: int, h: int) -> float:
    """Effective number of independent observations in the mean of ``n_dates`` consecutive daily ICs of
    overlapping h-day windows (autocorrelation 1 − k/h at lag k < h)."""
    n = int(n_dates)
    if n <= 0:
        return 0.0
    s = sum((1 - k / n) * (1 - k / h) for k in range(1, min(n - 1, h - 1) + 1))
    return n / (1 + 2 * s)


def ic_by_date(led: pl.DataFrame, h: int, mode: str = "live", since: date | None = None) -> pl.DataFrame:
    m = led.filter((pl.col("mode") == mode) & (pl.col("horizon") == h) & pl.col("realized_ret").is_not_null())
    if since:
        m = m.filter(pl.col("date") >= since)
    if m.height == 0:
        return pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64, "n": pl.UInt32})
    return (m.group_by("date").agg(ic=pl.corr("score", "realized_ret", method="spearman"), n=pl.len())
             .filter(pl.col("n") >= 30).drop_nans("ic").sort("date"))


def ic_dispersion(led: pl.DataFrame, h: int) -> float:
    bt = ic_by_date(led, h, mode="backtest", since=date(2016, 1, 1))
    return float(bt["ic"].std()) if bt.height >= 250 else IC_SD_DEFAULT[h]


def horizon_evidence(led: pl.DataFrame, h: int, sd: float | None = None) -> dict:
    live = led.filter((pl.col("mode") == "live") & (pl.col("horizon") == h))
    ics = ic_by_date(led, h)
    exp = EXPECTED_IC[h]
    sd = sd if sd is not None else ic_dispersion(led, h)
    out = {"horizon": h, "forecast_dates": live["date"].n_unique(), "first_forecast": live["date"].min(),
           "matured_dates": ics.height, "expected_ic": exp, "ic_sd": sd,
           "months_to_confirm": (2 * sd / exp) ** 2 * h / 21}
    if ics.height == 0:
        return out
    m = float(ics["ic"].mean())
    ne = n_eff(ics.height, h)
    se = sd / math.sqrt(ne)
    prec = 1 / PRIOR_SD ** 2 + 1 / se ** 2
    post = (exp / PRIOR_SD ** 2 + m / se ** 2) / prec
    out.update(ic=m, n_eff=ne, se=se, t=m / se, z_vs_expected=(m - exp) / se,
               p_positive=0.5 * (1 + math.erf(post * math.sqrt(prec) / math.sqrt(2))),
               last_matured=ics["date"].max())
    return out


def verdict(rows: list[dict]) -> tuple[str, str]:
    for r in rows:
        if r.get("t") is not None and r["n_eff"] >= RED_MIN_NEFF and r["t"] <= RED_T:
            return "RED", (f"{r['horizon']}d live IC {r['ic']:+.3f} is {r['t']:.1f} SE below zero over "
                           f"{r['n_eff']:.1f} effective periods — check data, features and model")
    for r in rows:
        if r.get("t") is not None and r["n_eff"] >= AMBER_MIN_NEFF and r["z_vs_expected"] <= AMBER_Z:
            return "AMBER", (f"{r['horizon']}d live IC {r['ic']:+.3f} is {abs(r['z_vs_expected']):.1f} SE below the "
                             f"expected {r['expected_ic']:+.3f} — skill well below the research; re-research")
    scored = [r for r in rows if r.get("t") is not None]
    if not scored:
        return "GREEN", "no live forecast has matured yet"
    if all(r["se"] > r["expected_ic"] for r in scored):
        return "GREEN", "no sign of a problem — not yet informative (the edge is too small to confirm this early)"
    return "GREEN", "live skill consistent with the research"


def book_vs_benchmarks(books_dir: Path | None, prices: pl.DataFrame | None, book: str = "alpha_v2") -> dict | None:
    """The paper book's return against SPY and RSP over exactly the book's marked dates."""
    if books_dir is None or not (books_dir / f"{book}.json").exists():
        return None
    log = json.loads((books_dir / f"{book}.json").read_text()).get("equity_log", [])
    if len(log) < 2:
        return None
    d0, d1 = date.fromisoformat(str(log[0]["date"])), date.fromisoformat(str(log[-1]["date"]))
    out = {"book": book, "since": d0, "through": d1, "sessions": len(log),
           "ret": log[-1]["equity"] / log[0]["equity"] - 1}
    if prices is not None:
        for t in ("SPY", "RSP"):
            p = prices.filter((pl.col("ticker") == t) & pl.col("date").is_in([d0, d1])).sort("date")
            if p.height == 2:
                out[t] = float(p["adj_close"][1] / p["adj_close"][0] - 1)
        if "SPY" in out:
            eq = pl.DataFrame({"date": [date.fromisoformat(str(r["date"])) for r in log],
                               "eq": [float(r["equity"]) for r in log]})
            s = prices.filter(pl.col("ticker") == "SPY").select("date", "adj_close")
            j = eq.join(s, on="date").sort("date")
            if j.height > 5:
                diff = (j["eq"].pct_change() - j["adj_close"].pct_change()).drop_nulls().to_numpy()
                out["tracking_error"] = float(diff.std(ddof=1) * math.sqrt(252))
    return out


def report(led: pl.DataFrame, today: date, books_dir: Path | None = None, prices: pl.DataFrame | None = None) -> dict:
    rows = [horizon_evidence(led, h) for h in HORIZONS]
    v, why = verdict(rows)
    live_dates = led.filter(pl.col("mode") == "live")["date"]
    return {"as_of": today, "live_since": live_dates.min(), "live_dates": live_dates.n_unique(),
            "horizons": rows, "verdict": v, "why": why, "book": book_vs_benchmarks(books_dir, prices)}


def due_checkpoints(rep: dict, today: date, done: set[str]) -> list[tuple[str, str]]:
    by_h = {r["horizon"]: r for r in rep["horizons"]}
    due = []
    for key, when, what in CHECKPOINTS:
        if key in done:
            continue
        if key.endswith("-first"):
            if by_h[int(key.split("d")[0])]["matured_dates"] > 0:
                due.append((key, what))
        elif when is not None and today >= when:
            due.append((key, what))
    return due


def render(rep: dict, today: date, done: set[str] | frozenset = frozenset()) -> str:
    def f(v, fmt):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
    L = [f"Live evidence · as of {today} · live forecasts since {rep['live_since']} ({rep['live_dates']} dates)",
         f"VERDICT: {rep['verdict']} — {rep['why']}", "",
         f"{'horizon':>7}{'matured':>9}{'live IC':>9}{'± SE':>8}{'t':>7}{'expected':>10}{'z vs exp':>10}{'P(IC>0)':>9}"
         f"  to confirm at 2σ"]
    for r in rep["horizons"]:
        conf = f"~{r['months_to_confirm']:.0f} months of live data"
        if r.get("t") is None:
            L.append(f"{str(r['horizon']) + 'd':>7}{r['matured_dates']:>9}{'—':>9}{'':>8}{'':>7}{r['expected_ic']:>+10.3f}"
                     f"{'':>10}{'':>9}  {conf}; none matured yet")
        else:
            L.append(f"{str(r['horizon']) + 'd':>7}{r['matured_dates']:>9}{f(r['ic'], '+.3f'):>9}{f(r['se'], '.3f'):>8}"
                     f"{f(r['t'], '+.1f'):>7}{r['expected_ic']:>+10.3f}{f(r['z_vs_expected'], '+.1f'):>10}"
                     f"{f(r['p_positive'], '.0%'):>9}  {conf}")
    b = rep.get("book")
    if b:
        s = f"Paper book {b['book']} {b['since']} → {b['through']} ({b['sessions']} marks): {b['ret']:+.1%}"
        for t in ("SPY", "RSP"):
            if t in b:
                s += f" · {t} {b[t]:+.1%}"
        if "SPY" in b:
            s += f" → vs SPY {b['ret'] - b['SPY']:+.1%}"
        if "tracking_error" in b:
            s += f" (tracking error {b['tracking_error']:.0%}/yr: information only, a book verdict needs years)"
        L += ["", s]
    nxt = next(((k, w, what) for k, w, what in CHECKPOINTS if k not in done and (w is None or w > today)), None)
    L += ["", "Rules (pre-registered 2026-10-09): RED if a live IC t-stat ≤ −2 over ≥ 2 effective periods "
              "(broken: check data/features/model); AMBER if live IC ≥ 2 SE below the expected over ≥ 3 effective "
              "periods (decay: re-research). Overlapping forecasts count as one period per horizon length."]
    if nxt:
        when = nxt[1]
        if when is None:                          # "<h>d-first": h sessions after the first live forecast, tallied next morning
            h = int(nxt[0].split("d")[0])
            first = next((r["first_forecast"] for r in rep["horizons"] if r["horizon"] == h), None)
            when = f"≈ {np.busday_offset(np.datetime64(first, 'D'), h + 1, roll='forward')}" if first else "—"
        L.append(f"Next checkpoint: {when} — {nxt[2]}")
    return "\n".join(L)


def state_path(cfg) -> Path:
    return cfg.path("reports") / "alpha" / "evidence_state.json"
