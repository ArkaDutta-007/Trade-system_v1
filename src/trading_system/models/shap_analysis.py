"""SHAP analysis for tree models."""
from __future__ import annotations

import numpy as np
import polars as pl


def compute_shap_summary(
    model,
    features: pl.DataFrame,
    feature_columns: list[str],
    sample_size: int = 5000,
    seed: int = 7,
) -> pl.DataFrame:
    """Return mean absolute SHAP value per feature on a random sample.

    For LightGBM, this uses TreeExplainer when available; otherwise falls back
    to model.feature_importances_.
    """
    sub = features.drop_nulls(subset=feature_columns)
    if len(sub) > sample_size:
        sub = sub.sample(n=sample_size, seed=seed)
    X = sub.select(feature_columns).to_numpy()

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        importance = np.abs(sv).mean(axis=0)
    except Exception:
        importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_columns))))
        if importance.sum() > 0:
            importance = importance / importance.sum()

    return pl.DataFrame(
        {"feature": feature_columns, "mean_abs_shap": importance}
    ).sort("mean_abs_shap", descending=True)
