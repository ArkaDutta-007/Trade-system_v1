"""Forecast ledger: every forecast is recorded, tallied against what happened, and fed back.

The ledger is one parquet table, ``data/ledger/alpha_forecasts.parquet``, with
one row per (date, ticker, horizon, mode):

    mode          backtest (causal walk-forward) | live (produced by `ts alpha forecast`)
    score         raw model output (gaussian-rank units)
    exp_ret       calibrated expected return over the horizon
    q10 / q90     conformal 80% band for the realised return
    entry_price   adj_close on the forecast date
    realized_ret  filled by ``tally`` once ``horizon`` trading days have passed
    hit / in_band scored outcomes

Continuous learning happens in ``Calibrator``: on every run it is refitted on
the matured rows of the trailing window — an isotonic map score → expected
return per horizon, conformal residual quantiles per score quintile, and a
skill weight per horizon from the trailing rank-IC — so the numbers that reach
the book always reflect the model's *recent realised* accuracy, not what it
looked like at training time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

LEDGER_COLS = {"date": pl.Date, "ticker": pl.Utf8, "horizon": pl.Int32, "mode": pl.Utf8, "model_id": pl.Utf8,
               "score": pl.Float32, "exp_ret": pl.Float32, "q10": pl.Float32, "q90": pl.Float32,
               "entry_price": pl.Float32, "realized_ret": pl.Float32, "realized_date": pl.Date,
               "hit": pl.Boolean, "in_band": pl.Boolean, "recorded_at": pl.Datetime("us")}
KEY = ["date", "ticker", "horizon", "mode"]


def ledger_path(cfg) -> Path:
    return cfg.path("data_bronze").parent / "ledger" / "alpha_forecasts.parquet"


def load(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=LEDGER_COLS)
    return pl.read_parquet(path)


def _conform(df: pl.DataFrame) -> pl.DataFrame:
    for c, t in LEDGER_COLS.items():
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=t).alias(c))
        else:
            df = df.with_columns(pl.col(c).cast(t, strict=False))
    return df.select(list(LEDGER_COLS))


def upsert(path: Path, rows: pl.DataFrame) -> int:
    """Insert/replace by key; realised columns of existing rows are preserved unless the new rows carry them."""
    old = load(path)
    new = _conform(rows.with_columns(recorded_at=pl.lit(datetime.utcnow()).cast(pl.Datetime("us"))))
    if old.height:
        keep_old = old.join(new.select(KEY), on=KEY, how="anti")
        # rows being replaced: carry realised outcomes forward if the replacement has none
        replaced = old.join(new.select(KEY), on=KEY, how="semi").select(KEY + ["realized_ret", "realized_date", "hit", "in_band"])
        if replaced.height:
            new = (new.join(replaced.rename({c: c + "_old" for c in ["realized_ret", "realized_date", "hit", "in_band"]}), on=KEY, how="left")
                      .with_columns([pl.coalesce(pl.col(c), pl.col(c + "_old")).alias(c) for c in ["realized_ret", "realized_date", "hit", "in_band"]])
                      .select(list(LEDGER_COLS)))
        out = pl.concat([keep_old, new], how="vertical_relaxed")
    else:
        out = new
    out = out.sort(["date", "horizon", "ticker"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return new.height


# ── tally ─────────────────────────────────────────────────────────────────────

def tally(path: Path, prices: pl.DataFrame) -> dict:
    """Fill ``realized_*`` for every forecast whose horizon has elapsed on the price calendar.

    ``prices``: long ``date, ticker, adj_close``. Realised return = adj_close[h trading days later] /
    adj_close[date] − 1 (a name that stops printing before maturity is scored at its last print).
    """
    led = load(path)
    if led.height == 0:
        return {"matured": 0, "pending": 0}
    open_rows = led.filter(pl.col("realized_ret").is_null())
    if open_rows.height == 0:
        return {"matured": 0, "pending": 0}
    px = prices.select("date", "ticker", "adj_close").sort(["ticker", "date"])
    cal = px.select("date").unique().sort("date").with_row_index("didx")
    px = px.join(cal, on="date")
    last_didx = int(cal["didx"].max())
    tk_last = px.group_by("ticker").agg(pl.col("didx").max().alias("last_didx"))
    o = (open_rows.drop(["realized_ret", "realized_date", "hit", "in_band"])
                  .join(cal, on="date", how="inner").join(tk_last, on="ticker", how="left"))
    o = o.with_columns(target_didx=(pl.col("didx") + pl.col("horizon")).cast(pl.UInt32))
    o = o.filter(pl.col("target_didx") <= last_didx)          # the calendar has reached maturity
    if o.height == 0:
        return {"matured": 0, "pending": open_rows.height}
    # exit = last print at or before the target index (a delisted name is scored at its final print)
    pxs = px.sort("didx")
    o = (o.sort("target_didx")
          .join_asof(pxs.select("ticker", "didx", exit_price="adj_close", realized_date="date"),
                     left_on="target_didx", right_on="didx", by="ticker", strategy="backward", check_sortedness=False))
    o = (o.sort("didx")
          .join_asof(pxs.select("ticker", "didx", entry_adj="adj_close"), on="didx", by="ticker",
                     strategy="backward", check_sortedness=False))
    o = o.with_columns(realized_ret=(pl.col("exit_price") / pl.col("entry_adj") - 1).cast(pl.Float32))
    # hit: direction of the calibrated expectation when there is one, else of the score (above/below the
    # cross-sectional median forecast); in_band only where a band was issued
    o = o.with_columns(hit=(pl.col("realized_ret") > 0) == (pl.coalesce(pl.col("exp_ret"), pl.col("score")) > 0),
                       in_band=pl.when(pl.col("q10").is_not_null())
                                 .then((pl.col("realized_ret") >= pl.col("q10")) & (pl.col("realized_ret") <= pl.col("q90"))))
    done = o.filter(pl.col("realized_ret").is_not_null()).select(list(LEDGER_COLS))
    n = upsert(path, done) if done.height else 0
    return {"matured": n, "pending": open_rows.height - n}


# ── metrics ───────────────────────────────────────────────────────────────────

def _spearman_by_date(df: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    return (df.group_by(["date", "horizon"]).agg(pl.corr(a, b, method="spearman").alias("ic"), pl.len().alias("n"))
              .filter(pl.col("n") >= 20))


def skill_report(led: pl.DataFrame, since: date | None = None, until: date | None = None,
                 mode: str | None = None) -> pl.DataFrame:
    """Per-horizon realised skill: rank-IC (mean, ICIR, t-stat), hit rate, decile spread, band coverage,
    calibration slope (realised on expected). Uses matured rows only."""
    m = led.filter(pl.col("realized_ret").is_not_null())
    if mode:
        m = m.filter(pl.col("mode") == mode)
    if since:
        m = m.filter(pl.col("date") >= since)
    if until:
        m = m.filter(pl.col("date") <= until)
    if m.height == 0:
        return pl.DataFrame()
    ic = _spearman_by_date(m, "score", "realized_ret").group_by("horizon").agg(
        ic_mean=pl.col("ic").mean(), ic_std=pl.col("ic").std(), n_dates=pl.len())
    ic = ic.with_columns(icir=pl.col("ic_mean") / pl.col("ic_std"),
                         t_stat=pl.col("ic_mean") / pl.col("ic_std") * pl.col("n_dates").sqrt())
    dec = (m.with_columns(dec=((pl.col("score").rank("average").over(["date", "horizon"]) - 1) * 10
                               // pl.col("score").count().over(["date", "horizon"])).cast(pl.Int32))
             .group_by(["date", "horizon", "dec"]).agg(r=pl.col("realized_ret").mean())
             .group_by(["horizon", "dec"]).agg(r=pl.col("r").mean()))
    spread = (dec.filter(pl.col("dec") == 9).select("horizon", top="r")
                 .join(dec.filter(pl.col("dec") == 0).select("horizon", bottom="r"), on="horizon")
                 .with_columns(decile_spread=pl.col("top") - pl.col("bottom")))
    base = m.group_by("horizon").agg(
        n=pl.len(), hit_rate=pl.col("hit").cast(pl.Float64).mean(), coverage_80=pl.col("in_band").cast(pl.Float64).mean(),
        mae=(pl.col("realized_ret") - pl.col("exp_ret")).abs().mean(),
        exp_mean=pl.col("exp_ret").mean(), real_mean=pl.col("realized_ret").mean(),
        first=pl.col("date").min(), last=pl.col("date").max())
    slope = m.group_by("horizon").agg(
        cov=pl.cov("exp_ret", "realized_ret"), var=pl.col("exp_ret").var()).with_columns(
        calib_slope=pl.col("cov") / pl.col("var")).select("horizon", "calib_slope")
    return (base.join(ic, on="horizon", how="left").join(spread.select("horizon", "decile_spread"), on="horizon", how="left")
                .join(slope, on="horizon", how="left").sort("horizon"))


def yearly_ic(led: pl.DataFrame, mode: str | None = None) -> pl.DataFrame:
    m = led.filter(pl.col("realized_ret").is_not_null())
    if mode:
        m = m.filter(pl.col("mode") == mode)
    ic = _spearman_by_date(m, "score", "realized_ret").with_columns(year=pl.col("date").dt.year())
    return (ic.group_by(["year", "horizon"]).agg(ic=pl.col("ic").mean(), icir=pl.col("ic").mean() / pl.col("ic").std(), n=pl.len())
              .sort(["horizon", "year"]))


def rolling_ic(led: pl.DataFrame, horizon: int, window_dates: int = 63, mode: str | None = None) -> pl.DataFrame:
    m = led.filter(pl.col("realized_ret").is_not_null(), pl.col("horizon") == horizon)
    if mode:
        m = m.filter(pl.col("mode") == mode)
    ic = _spearman_by_date(m, "score", "realized_ret").sort("date")
    return ic.with_columns(ic_roll=pl.col("ic").rolling_mean(window_dates, min_samples=max(5, window_dates // 3)))


# ── calibration (the feedback loop) ───────────────────────────────────────────

@dataclass
class HorizonCalibration:
    horizon: int
    x: list[float]                 # isotonic breakpoints (score)
    y: list[float]                 # expected return at breakpoints
    q_lo: list[float]              # residual 10th pct per score quintile (5 values)
    q_hi: list[float]              # residual 90th pct per score quintile
    quintile_edges: list[float]    # 4 interior score edges
    ic: float = 0.0                # trailing mean rank-IC used for skill weighting
    n: int = 0
    fitted_through: str | None = None

    def expected(self, score: np.ndarray) -> np.ndarray:
        return np.interp(score, self.x, self.y, left=self.y[0], right=self.y[-1]).astype(np.float32)

    def band(self, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = self.expected(score)
        q = np.searchsorted(np.asarray(self.quintile_edges), score, side="right")
        lo = e + np.asarray(self.q_lo)[q]
        hi = e + np.asarray(self.q_hi)[q]
        return lo.astype(np.float32), hi.astype(np.float32)


@dataclass
class Calibrator:
    horizons: dict[int, HorizonCalibration] = field(default_factory=dict)
    window_days: int = 365 * 3
    min_rows: int = 5000

    @classmethod
    def fit(cls, led: pl.DataFrame, horizons: Iterable[int], window_days: int = 365 * 3,
            until: date | None = None, mode: str | None = None, min_rows: int = 5000) -> "Calibrator":
        from sklearn.isotonic import IsotonicRegression
        cal = cls(window_days=window_days, min_rows=min_rows)
        m = led.filter(pl.col("realized_ret").is_not_null())
        if mode:
            m = m.filter(pl.col("mode") == mode)
        if until:
            m = m.filter(pl.col("date") <= until)
        for h in horizons:
            mh = m.filter(pl.col("horizon") == h)
            if mh.height == 0:
                continue
            last = mh["date"].max()
            mh = mh.filter(pl.col("date") >= last - timedelta(days=window_days))
            if mh.height < cal.min_rows:
                logger.warning(f"calibrator h={h}: only {mh.height} matured rows in window — skipping")
                continue
            s = mh["score"].to_numpy().astype(np.float64)
            r = np.clip(mh["realized_ret"].to_numpy().astype(np.float64), -0.95, 5.0)
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s, r)
            xs = np.quantile(s, np.linspace(0, 1, 41))
            ys = iso.predict(xs)
            resid = r - iso.predict(s)
            edges = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
            q = np.searchsorted(edges, s, side="right")
            q_lo = [float(np.quantile(resid[q == i], 0.10)) if (q == i).sum() > 50 else float(np.quantile(resid, 0.10)) for i in range(5)]
            q_hi = [float(np.quantile(resid[q == i], 0.90)) if (q == i).sum() > 50 else float(np.quantile(resid, 0.90)) for i in range(5)]
            ic = _spearman_by_date(mh, "score", "realized_ret")["ic"]
            cal.horizons[h] = HorizonCalibration(h, [float(v) for v in xs], [float(v) for v in ys], q_lo, q_hi,
                                                 [float(v) for v in edges], float(ic.mean()) if ic.len() else 0.0,
                                                 int(mh.height), str(last))
        return cal

    def apply(self, scores: pl.DataFrame) -> pl.DataFrame:
        """Add ``exp_ret, q10, q90`` to a ``date, ticker, horizon, score`` frame (nulls where uncalibrated)."""
        parts = []
        for (h,), g in scores.group_by(["horizon"], maintain_order=True):
            c = self.horizons.get(int(h))
            s = g["score"].to_numpy().astype(np.float64)
            if c is None:
                parts.append(g.with_columns(exp_ret=pl.lit(None, pl.Float32), q10=pl.lit(None, pl.Float32), q90=pl.lit(None, pl.Float32)))
                continue
            lo, hi = c.band(s)
            parts.append(g.with_columns(exp_ret=pl.Series(c.expected(s)), q10=pl.Series(lo), q90=pl.Series(hi)))
        return pl.concat(parts) if parts else scores

    def skill_weights(self, horizons: Iterable[int] | None = None, floor: float = 0.0) -> dict[int, float]:
        """Horizon blend weights ∝ trailing IC⁺. A horizon with no realised evidence yet gets no vote
        while others have some; with no evidence at all every horizon votes equally."""
        hs = [int(h) for h in (horizons if horizons is not None else self.horizons)]
        if not hs:
            return {}
        raw = {h: (max(self.horizons[h].ic, 0.0) + floor) if h in self.horizons else 0.0 for h in hs}
        tot = sum(raw.values())
        if tot <= 0:
            return {h: 1.0 / len(hs) for h in hs}
        return {h: v / tot for h, v in raw.items()}

    def to_json(self) -> dict:
        return {"window_days": self.window_days,
                "horizons": {str(h): {"x": c.x, "y": c.y, "q_lo": c.q_lo, "q_hi": c.q_hi, "quintile_edges": c.quintile_edges,
                                      "ic": c.ic, "n": c.n, "fitted_through": c.fitted_through} for h, c in self.horizons.items()}}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        d = json.loads(path.read_text())
        cal = cls(window_days=d.get("window_days", 365 * 3))
        for h, c in d["horizons"].items():
            cal.horizons[int(h)] = HorizonCalibration(int(h), c["x"], c["y"], c["q_lo"], c["q_hi"], c["quintile_edges"],
                                                      c.get("ic", 0.0), c.get("n", 0), c.get("fitted_through"))
        return cal


def composite_score(scores: pl.DataFrame, weights: dict[int, float]) -> pl.DataFrame:
    """Blend per-horizon scores into one ranking signal per (date, ticker) using skill weights.
    Each horizon is z-scored within the date first so the weights act on comparable units."""
    z = scores.with_columns(z=((pl.col("score") - pl.col("score").mean().over(["date", "horizon"]))
                               / (pl.col("score").std().over(["date", "horizon"]) + 1e-9)))
    w = pl.DataFrame({"horizon": [int(h) for h in weights], "w": [float(v) for v in weights.values()]}).with_columns(pl.col("horizon").cast(pl.Int32))
    z = z.join(w, on="horizon", how="inner")
    return z.group_by(["date", "ticker"]).agg(composite=(pl.col("z") * pl.col("w")).sum() / pl.col("w").sum())
