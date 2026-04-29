"""Walk-forward training using a comprehensive 14-model ensemble.

For each test window:
  - train on years strictly before the window
  - validate on a held-out slice inside the training window
  - fit 14 base learners + 3 ensemble variants (blend, stack-ridge, stack-lgbm)
  - compute IC/MAE/R² comparative metrics per model
  - predict the test window using the best ensemble
  - emit OOS predictions + per-fold comparative metrics

Returns (fold_records, oos_predictions, comparative_metrics_df)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from ..utils import get_logger
from .ensemble import EnsembleModel

logger = get_logger(__name__)


@dataclass
class FeatureSpec:
    feature_columns: list[str]
    target: str = "forward_return_5d"
    drop_na: bool = True
    extra_drop: list[str] = field(default_factory=lambda: ["forward_return_20d"])


def _make_xy(df: pl.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    keep = ["date", "ticker", spec.target] + spec.feature_columns
    sub = df.select([c for c in keep if c in df.columns])
    if spec.drop_na:
        sub = sub.drop_nulls(subset=[spec.target] + spec.feature_columns)
    X = sub.select(spec.feature_columns).to_numpy().astype(np.float64)
    y = sub[spec.target].to_numpy().astype(np.float64)
    return X, y, sub


def _windows(dates: list[date], train_years: int, test_years: int, step_years: int):
    if not dates:
        return
    start = dates[0]
    end = dates[-1]
    cur = start
    while True:
        train_end = cur.replace(year=cur.year + train_years)
        test_end = train_end.replace(year=train_end.year + test_years)
        if train_end >= end:
            return
        yield (cur, train_end, min(test_end, end))
        cur = cur.replace(year=cur.year + step_years)


def train_walk_forward(
    features: pl.DataFrame,
    spec: FeatureSpec,
    train_years: int = 4,
    test_years: int = 1,
    step_years: int = 1,
    params: dict | None = None,  # kept for back-compat, unused by ensemble
    val_fraction: float = 0.2,   # fraction of train window used for validation
) -> tuple[list, pl.DataFrame, pl.DataFrame]:
    """Run walk-forward ensemble training.

    Returns
    -------
    fold_records : list of dicts with keys: window, ensemble, blend_weights, val_metrics
    oos          : Polars DataFrame with columns: date, ticker, score (best-ensemble score)
    metrics_df   : Polars DataFrame with per-fold × per-model IC/MAE/R²
    """
    dates = sorted(set(features["date"].to_list()))
    fold_records: list[dict] = []
    preds_frames: list[pl.DataFrame] = []
    all_metrics: list[dict] = []

    for fold_i, (tr_start, tr_end, te_end) in enumerate(
        _windows(dates, train_years, test_years, step_years)
    ):
        train_df = features.filter((pl.col("date") >= tr_start) & (pl.col("date") < tr_end))
        test_df = features.filter((pl.col("date") >= tr_end) & (pl.col("date") < te_end))
        if train_df.is_empty() or test_df.is_empty():
            continue

        X_all, y_all, _ = _make_xy(train_df, spec)
        X_te, y_te, test_meta = _make_xy(test_df, spec)
        if len(X_all) < 20 or len(X_te) == 0:
            continue

        # Split train into train/val
        split = max(10, int(len(X_all) * (1 - val_fraction)))
        X_tr, y_tr = X_all[:split], y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:]

        logger.info(
            f"Fold {fold_i} {tr_start}->{tr_end}->{te_end}: "
            f"train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}"
        )

        ensemble = EnsembleModel()
        ensemble.fit(X_tr, y_tr, X_val, y_val, feature_names=spec.feature_columns)

        best = ensemble.best_ensemble_name()
        all_preds = ensemble.predict(X_te)
        score = all_preds.get(best, all_preds.get("ensemble_blend", np.zeros(len(X_te))))

        fold_records.append({
            "window": (tr_start, tr_end, te_end),
            "ensemble": ensemble,
            "blend_weights": ensemble.blend_weights,
            "val_metrics": ensemble.val_metrics,
            "best_variant": best,
        })
        preds_frames.append(
            test_meta.select(["date", "ticker"]).with_columns(score=pl.Series(score))
        )

        # Accumulate comparative metrics
        for row in ensemble.comparative_table():
            all_metrics.append({"fold": fold_i, **row})

    if not preds_frames:
        empty_oos = pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "score": pl.Float64})
        empty_metrics = pl.DataFrame(schema={"fold": pl.Int32, "model": pl.Utf8,
                                             "ic": pl.Float64, "mae": pl.Float64,
                                             "r2": pl.Float64, "weight": pl.Float64})
        return fold_records, empty_oos, empty_metrics

    oos = pl.concat(preds_frames).sort(["date", "ticker"])
    metrics_df = pl.DataFrame(all_metrics)
    return fold_records, oos, metrics_df
