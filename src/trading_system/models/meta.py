"""Meta-labeling — P(the forecast is right), the sizing multiplier.

López de Prado's second act: the primary forecaster says *which way*; a
second-stage classifier says *how much to trust that call right now*, and
position size scales by that probability. Same forecasts, better-placed bets.

Training data is the **pseudo-ledger**: the winning family's out-of-sample
predictions across every purged/CPCV fold (persisted by ``ts train-forecast``
as ``models_store/forecast/<h>d/oos_predictions.parquet``) — thousands of
honest (prediction, outcome) pairs available *today*, no waiting for the live
ledger to mature.

The meta features are deliberately **state, not alpha**: the nonlinear
fingerprint (Hurst, permutation/sample entropy, RQA determinism, early-warning
score, tail index, LPPLS), RMT systematic fraction, vol/regime/macro context,
plus the signal's own within-date shape (rank percentile, z, sign) and
cross-sectional dispersion/breadth. The question is never "will the stock go
up" (the primary's job) but "is this name currently in a *forecastable* state"
— which is exactly what those estimators measure.

Honesty: the meta label spans the same [d, d+h] window as the primary label,
so meta CV uses the same purged walk-forward splitter; probabilities are
isotonic-calibrated on pooled OOS folds (never in-sample); and the reported
AUC/Brier/lift come from the purged OOS folds only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..utils import get_logger, get_compute_profile
from .validation import purged_walkforward_splits

logger = get_logger(__name__)

_CAL_PER_TD = 365 / 252

# state-feature patterns (prefix match against the gold panel) — the
# forecastability fingerprint, not the alpha set
META_STATE_PATTERNS: tuple[str, ...] = (
    "hurst_", "permutation_entropy", "sample_entropy", "spectral_entropy",
    "rqa_", "recurrence_", "early_warning", "ar1", "hill_", "lppls_",
    "chaos01", "lyapunov", "wavelet_hf", "higuchi", "rough_vol",
    "dominant_period",
    "rmt_systematic_frac", "rmt_market_beta",
    "vol_20d", "vol_60d", "vol_of_vol_60", "ret_skew_60", "ret_kurt_60",
    "beta_60", "corr_bench_60", "amihud_illiq_20",
    "bull_regime", "high_vol_regime",
    "macro_vix", "macro_hy_oas", "macro_yield_curve",
)

_SIGNAL_COLS = ("pred_rank_pct", "pred_z", "pred_sign",
                "xsec_mom_dispersion", "xsec_breadth")


def resolve_meta_features(df: pl.DataFrame, min_non_null_frac: float = 0.5) -> list[str]:
    """State columns present in the panel with adequate coverage."""
    out = []
    n = df.height
    for c in df.columns:
        if not any(c.startswith(p) for p in META_STATE_PATTERNS):
            continue
        if n and df[c].null_count() / n <= (1 - min_non_null_frac):
            out.append(c)
    return sorted(out)


def _engineer_signal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Within-date shape of the primary signal + cross-sectional context.

    Everything here is a *within-date* transform of ``y_pred`` (or of panel
    columns), so it is computed identically on the training pseudo-ledger and
    on the live cross-section — no train/live skew.
    """
    mom = "mom_20d" if "mom_20d" in df.columns else None
    exprs = [
        (pl.col("y_pred").rank("average").over("date") /
         pl.len().over("date")).alias("pred_rank_pct"),
        ((pl.col("y_pred") - pl.col("y_pred").mean().over("date")) /
         (pl.col("y_pred").std().over("date") + 1e-12)).alias("pred_z"),
        pl.col("y_pred").sign().alias("pred_sign"),
        (pl.col(mom).std().over("date") if mom else pl.lit(0.0)
         ).alias("xsec_mom_dispersion"),
        pl.len().over("date").cast(pl.Float64).alias("xsec_breadth"),
    ]
    return df.with_columns(exprs)


def build_meta_dataset(
    oos: pl.DataFrame,
    features: pl.DataFrame,
    state_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Join the pseudo-ledger with state features; label hit = sign agreement.

    ``hit`` uses the same target the primary was trained on (neutralized →
    "did it beat the market", raw → "did it go up"), which is exactly the call
    the invest planner acts on.
    """
    state_cols = state_cols if state_cols is not None else resolve_meta_features(features)
    keep = ["ticker", "date"] + state_cols + (["mom_20d"] if "mom_20d" in features.columns else [])
    ds = oos.join(features.select(keep), on=["ticker", "date"], how="inner")
    ds = _engineer_signal_features(ds)
    ds = ds.with_columns(
        (pl.col("y_pred").sign() == pl.col("y_true").sign())
        .cast(pl.Int8).alias("hit")
    ).filter(pl.col("y_true").abs() > 1e-9)   # flat outcomes carry no verdict
    cols = state_cols + list(_SIGNAL_COLS)
    return ds, cols


def _meta_estimator(prof):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=31,
        learning_rate=0.05, min_child_samples=60, subsample=0.8,
        colsample_bytree=0.8, verbose=-1, **prof.lgbm_params(),
    )


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def train_meta_for_horizon(
    oos: pl.DataFrame,
    features: pl.DataFrame,
    horizon: int,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> dict[str, Any]:
    """Purged-CV train + isotonic-calibrate the meta classifier for one horizon.

    Returns {model, calibrator, meta_columns, metrics} — metrics all from the
    purged OOS folds (AUC, Brier vs base-rate Brier, decile lift table).
    """
    prof = get_compute_profile()
    ds, cols = build_meta_dataset(oos, features)
    ds = ds.drop_nulls(subset=cols + ["hit"]).sort("date")
    if ds.height < 1000:
        raise ValueError(f"{horizon}d: pseudo-ledger too small after join ({ds.height} rows)")

    X = ds.select(cols).to_numpy().astype(np.float64)
    y = ds["hit"].to_numpy().astype(np.int8)
    dates = ds["date"].to_list()

    splits = purged_walkforward_splits(
        dates, horizon_days=int(round(horizon * _CAL_PER_TD)),
        n_splits=n_splits, embargo_days=embargo_days,
    )
    oof_p, oof_y = [], []
    aucs = []
    for s in splits:
        if len(s.train_idx) < 500 or len(s.test_idx) < 100:
            continue
        m = _meta_estimator(prof)
        m.fit(X[s.train_idx], y[s.train_idx])
        p = m.predict_proba(X[s.test_idx])[:, 1]
        oof_p.append(p)
        oof_y.append(y[s.test_idx])
        a = _auc(y[s.test_idx], p)
        if a is not None:
            aucs.append(a)
    if not oof_p:
        raise ValueError(f"{horizon}d: no viable meta CV folds")
    oof_p = np.concatenate(oof_p)
    oof_y = np.concatenate(oof_y)

    # isotonic calibration on pooled OOS pairs — sizing multiplies by these
    # probabilities, so they must MEAN what they say
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
    calibrator.fit(oof_p, oof_y)
    cal_p = calibrator.predict(oof_p)

    base = float(oof_y.mean())
    brier = float(np.mean((cal_p - oof_y) ** 2))
    brier_base = float(np.mean((base - oof_y) ** 2))
    # decile lift: realized hit-rate by calibrated-probability decile
    order = np.argsort(cal_p)
    deciles = []
    for chunk in np.array_split(order, 10):
        if len(chunk):
            deciles.append({"p_mean": round(float(cal_p[chunk].mean()), 4),
                            "hit_rate": round(float(oof_y[chunk].mean()), 4),
                            "n": int(len(chunk))})
    top, bottom = deciles[-1]["hit_rate"], deciles[0]["hit_rate"]

    model = _meta_estimator(prof)
    model.fit(X, y)

    metrics = {
        "horizon": horizon,
        "n_rows": int(ds.height),
        "n_folds": len(aucs),
        "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "auc_std": round(float(np.std(aucs)), 4) if len(aucs) > 1 else 0.0,
        "base_hit_rate": round(base, 4),
        "brier": round(brier, 4),
        "brier_base": round(brier_base, 4),
        "brier_skill": round(1 - brier / brier_base, 4) if brier_base > 0 else None,
        "decile_lift": deciles,
        "top_minus_bottom_decile": round(top - bottom, 4),
        "meta_columns": cols,
    }
    logger.info(
        f"meta {horizon}d: AUC={metrics['auc_mean']} base={base:.3f} "
        f"brier_skill={metrics['brier_skill']} "
        f"top-bottom decile={metrics['top_minus_bottom_decile']:+.3f} "
        f"({ds.height} rows)"
    )
    return {"model": model, "calibrator": calibrator,
            "meta_columns": cols, "metrics": metrics}


# ── Store ─────────────────────────────────────────────────────────────────────

def save_meta(bundle: dict, horizon: int, store_dir: Path) -> Path:
    from .store import _dump
    hd = Path(store_dir) / "meta" / f"{horizon}d"
    hd.mkdir(parents=True, exist_ok=True)
    _dump({"model": bundle["model"], "calibrator": bundle["calibrator"],
           "meta_columns": bundle["meta_columns"]}, hd / "meta_model.pkl")
    (hd / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2, default=str))
    return hd


def load_meta(horizon: int, store_dir: Path) -> dict | None:
    from .store import _load
    hd = Path(store_dir) / "meta" / f"{horizon}d"
    p = hd / "meta_model.pkl"
    if not p.exists():
        return None
    try:
        bundle = _load(p)
        bundle["metrics"] = json.loads((hd / "metrics.json").read_text()) \
            if (hd / "metrics.json").exists() else {}
        return bundle
    except Exception as e:
        logger.warning(f"failed to load meta model {horizon}d: {e}")
        return None


def meta_probabilities(
    bundle: dict,
    features_latest: pl.DataFrame,
    scores: dict[str, float],
) -> dict[str, float]:
    """Calibrated P(right) for today's cross-section.

    ``features_latest`` is the latest-date slice of the gold panel;
    ``scores`` maps ticker → primary prediction for the relevant horizon
    (within-date transforms are recomputed here exactly as in training).
    """
    tickers = [t for t in features_latest["ticker"].to_list() if t in scores]
    if not tickers:
        return {}
    sub = features_latest.filter(pl.col("ticker").is_in(tickers)).with_columns(
        pl.col("ticker").replace_strict(
            {t: float(scores[t]) for t in tickers}, default=None,
        ).alias("y_pred")
    )
    sub = _engineer_signal_features(sub)
    cols = bundle["meta_columns"]
    missing = [c for c in cols if c not in sub.columns]
    if missing:
        logger.warning(f"meta features missing from panel: {missing[:5]} — skipping meta")
        return {}
    sub = sub.fill_nan(None)
    X = sub.select(cols).to_numpy().astype(np.float64)
    # LightGBM tolerates NaN natively; rows of all-NaN still score
    raw = bundle["model"].predict_proba(X)[:, 1]
    cal = bundle["calibrator"].predict(raw)
    return {t: float(p) for t, p in zip(sub["ticker"].to_list(), cal)}
