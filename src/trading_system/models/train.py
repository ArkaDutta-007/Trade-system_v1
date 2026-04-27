"""Walk-forward training of a tabular regression/ranking model.

For each test window:
  - train on years strictly before the window
  - predict the test window
  - emit OOS predictions

Returns a tuple of (model_per_fold, oos_predictions). OOS predictions can be fed
to MLRankerStrategy or analyzed for SHAP and ablations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import polars as pl

from ..utils import get_logger

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
    params: dict | None = None,
) -> tuple[list, pl.DataFrame]:
    """Run walk-forward LightGBM training. Returns (models, oos_predictions)."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        **(params or {}),
    }

    dates = sorted(set(features["date"].to_list()))
    models = []
    preds_frames: list[pl.DataFrame] = []

    for tr_start, tr_end, te_end in _windows(dates, train_years, test_years, step_years):
        train_df = features.filter((pl.col("date") >= tr_start) & (pl.col("date") < tr_end))
        test_df = features.filter((pl.col("date") >= tr_end) & (pl.col("date") < te_end))
        if train_df.is_empty() or test_df.is_empty():
            continue

        X_tr, y_tr, _ = _make_xy(train_df, spec)
        X_te, y_te, test_meta = _make_xy(test_df, spec)
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        score = model.predict(X_te)
        models.append({"window": (tr_start, tr_end, te_end), "model": model})
        preds_frames.append(
            test_meta.select(["date", "ticker"]).with_columns(score=pl.Series(score))
        )
        logger.info(f"Fold {tr_start}->{tr_end}->{te_end}: train={len(X_tr)}, test={len(X_te)}")

    if not preds_frames:
        return models, pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "score": pl.Float64})
    oos = pl.concat(preds_frames).sort(["date", "ticker"])
    return models, oos
