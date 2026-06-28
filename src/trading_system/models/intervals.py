"""Conformalized quantile forecasting of multi-horizon return distributions.

This replaces the old hand-wavy ``forecast_20d = score * 4 * 0.65`` and the
symmetric ±2σ Gaussian fan with a **data-driven, asymmetric, calibrated** band:

  * For each horizon h ∈ {5, 21, 63, 126, 252} trading days we train a LightGBM
    **quantile** regressor per quantile τ ∈ {0.05, 0.25, 0.5, 0.75, 0.95} on the
    realised forward return ``(close[t+h]/close[t] - 1)``.  Quantile regression
    gives skew and fat tails for free — equities don't move symmetrically.
  * The outer band (τ=0.05 / τ=0.95) is wrapped with **Conformalized Quantile
    Regression** (Romano, Patterson & Candès 2019): a holdout calibration set
    yields an additive width correction ``Q`` so the band has *finite-sample*
    ~(1-α) marginal coverage regardless of how well the quantile models fit.

The output per (ticker, horizon) is a return distribution
``{q05, q25, q50, q75, q95, lo, hi}`` which the analyzer turns into price bounds
(min / median / max it can reach over days→months→a year) and the dashboard
renders as a calibrated fan.

Persistence: a single pickled :class:`IntervalBundle` under
``reports/models/intervals/`` plus a ``meta.json`` describing coverage.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZONS = (5, 21, 63, 126, 252)
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntervalBundle:
    """All quantile models + conformal corrections for every horizon."""
    feature_columns: list[str]
    horizons: list[int]
    quantiles: list[float]
    alpha: float
    # models[horizon][quantile] -> fitted LGBMRegressor
    models: dict[int, dict[float, Any]] = field(default_factory=dict)
    # conformal_q[horizon] -> additive width correction for the outer band
    conformal_q: dict[int, float] = field(default_factory=dict)
    # empirical out-of-sample coverage of the calibrated band, per horizon
    coverage: dict[int, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def lo_q(self) -> float:
        return min(self.quantiles)

    @property
    def hi_q(self) -> float:
        return max(self.quantiles)

    def predict_row(self, x: np.ndarray) -> dict[int, dict[str, float]]:
        """Predict the return distribution for a single feature row (1×F)."""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        out: dict[int, dict[str, float]] = {}
        for h in self.horizons:
            qmodels = self.models.get(h, {})
            if not qmodels:
                continue
            qvals = {q: float(m.predict(x)[0]) for q, m in qmodels.items()}
            # enforce monotonicity across quantiles (quantile crossing guard)
            ordered = sorted(qvals)
            running = -np.inf
            for q in ordered:
                running = max(running, qvals[q])
                qvals[q] = running
            corr = self.conformal_q.get(h, 0.0)
            row = {f"q{int(q*100):02d}": v for q, v in qvals.items()}
            row["median"] = qvals.get(0.5, qvals[ordered[len(ordered) // 2]])
            row["lo"] = qvals[self.lo_q] - corr   # conformal-widened lower
            row["hi"] = qvals[self.hi_q] + corr   # conformal-widened upper
            out[h] = row
        return out

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "interval_bundle.pkl", "wb") as f:
            pickle.dump(self, f)
        meta = {
            "horizons": self.horizons,
            "quantiles": self.quantiles,
            "alpha": self.alpha,
            "conformal_q": self.conformal_q,
            "coverage": self.coverage,
            "n_features": len(self.feature_columns),
            "created_at": self.created_at,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return out


def load_interval_bundle(registry: Path | str) -> IntervalBundle | None:
    """Load the interval bundle from reports/models/intervals/, or None."""
    p = Path(registry) / "intervals" / "interval_bundle.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"interval bundle load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _add_forward_returns(features: pl.DataFrame, horizons) -> pl.DataFrame:
    """Compute realised forward returns per horizon from adj_close (targets only)."""
    px = pl.col("adj_close")
    df = features.sort(["ticker", "date"])
    return df.with_columns([
        ((px.shift(-h).over("ticker") / px) - 1).alias(f"fwd_ret_{h}d") for h in horizons
    ])


def train_interval_models(
    features: pl.DataFrame,
    feature_columns: list[str],
    horizons=DEFAULT_HORIZONS,
    quantiles=DEFAULT_QUANTILES,
    alpha: float = 0.10,
    cal_fraction: float = 0.2,
    lgbm_params: dict | None = None,
) -> IntervalBundle:
    """Train per-horizon conformalized quantile models.

    The calibration split is **temporal** (last ``cal_fraction`` of dates) so the
    conformal guarantee isn't contaminated by look-ahead.
    """
    from lightgbm import LGBMRegressor

    horizons = list(horizons)
    quantiles = sorted(quantiles)
    feature_columns = [c for c in feature_columns if c in features.columns]
    if not feature_columns:
        raise ValueError("no usable feature columns for interval training")

    base_params = dict(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1,
    )
    base_params.update(lgbm_params or {})

    df = _add_forward_returns(features, horizons)
    dates = sorted(df["date"].unique().to_list())
    if len(dates) < 50:
        raise ValueError(f"not enough dates ({len(dates)}) to train interval models")
    split_date = dates[int(len(dates) * (1 - cal_fraction))]

    bundle = IntervalBundle(
        feature_columns=feature_columns, horizons=horizons,
        quantiles=quantiles, alpha=alpha,
    )
    lo_q, hi_q = quantiles[0], quantiles[-1]

    for h in horizons:
        tgt = f"fwd_ret_{h}d"
        sub = df.drop_nulls(subset=[tgt] + feature_columns)
        train = sub.filter(pl.col("date") < split_date)
        calib = sub.filter(pl.col("date") >= split_date)
        if train.height < 200 or calib.height < 50:
            logger.warning(f"horizon {h}d: insufficient rows (train={train.height}, cal={calib.height}) — skipping")
            continue

        X_tr = train.select(feature_columns).to_numpy().astype(np.float64)
        y_tr = train[tgt].to_numpy().astype(np.float64)
        X_cal = calib.select(feature_columns).to_numpy().astype(np.float64)
        y_cal = calib[tgt].to_numpy().astype(np.float64)

        qmodels: dict[float, Any] = {}
        for q in quantiles:
            m = LGBMRegressor(objective="quantile", alpha=q, **base_params)
            m.fit(X_tr, y_tr)
            qmodels[q] = m
        bundle.models[h] = qmodels

        # CQR: conformity scores on calibration set for the outer band
        lo_pred = qmodels[lo_q].predict(X_cal)
        hi_pred = qmodels[hi_q].predict(X_cal)
        scores = np.maximum(lo_pred - y_cal, y_cal - hi_pred)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # finite-sample correction
        Q = float(np.quantile(scores, level, method="higher"))
        bundle.conformal_q[h] = Q

        covered = np.mean((y_cal >= lo_pred - Q) & (y_cal <= hi_pred + Q))
        bundle.coverage[h] = float(covered)
        logger.info(
            f"interval {h}d: train={train.height} cal={calib.height} "
            f"Q={Q:.4f} coverage={covered:.1%} (target {1-alpha:.0%})"
        )

    if not bundle.models:
        raise ValueError("no horizons could be trained")
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Prediction → price bounds
# ─────────────────────────────────────────────────────────────────────────────

def bounds_for_ticker(
    bundle: IntervalBundle,
    features: pl.DataFrame,
    ticker: str,
    last_price: float,
    as_of: str | None = None,
) -> dict[str, Any] | None:
    """Return calibrated price bounds (lo/median/hi + quantiles) per horizon.

    Output shape::

        {"ticker","as_of","last_price","horizons":{
            "21d": {"days":21,"return":{...},"price":{"lo":..,"median":..,"hi":..},...}}}
    """
    ticker = ticker.upper()
    last_date = features["date"].max()
    row = features.filter((pl.col("ticker") == ticker) & (pl.col("date") == last_date))
    if row.is_empty():
        return None
    avail = [c for c in bundle.feature_columns if c in row.columns]
    if len(avail) != len(bundle.feature_columns):
        # fill any missing feature columns with 0 to keep the model array shape
        missing = [c for c in bundle.feature_columns if c not in row.columns]
        row = row.with_columns([pl.lit(0.0).alias(c) for c in missing])
    X = row.select(bundle.feature_columns).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float64)

    dist = bundle.predict_row(X[0])
    horizons_out: dict[str, Any] = {}
    label_for = {5: "5d", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
    for h, r in dist.items():
        ret_lo, ret_med, ret_hi = r["lo"], r["median"], r["hi"]
        horizons_out[label_for.get(h, f"{h}d")] = {
            "days": h,
            "return": {k: round(v, 5) for k, v in r.items()},
            "price": {
                "lo": round(last_price * (1 + ret_lo), 4),
                "median": round(last_price * (1 + ret_med), 4),
                "hi": round(last_price * (1 + ret_hi), 4),
                "q25": round(last_price * (1 + r.get("q25", ret_med)), 4),
                "q75": round(last_price * (1 + r.get("q75", ret_med)), 4),
            },
            "coverage": round(bundle.coverage.get(h, float("nan")), 3),
        }
    return {
        "ticker": ticker,
        "as_of": as_of or str(last_date),
        "last_price": round(last_price, 4),
        "method": "conformalized_quantile_lgbm",
        "target_coverage": round(1 - bundle.alpha, 3),
        "horizons": horizons_out,
    }
