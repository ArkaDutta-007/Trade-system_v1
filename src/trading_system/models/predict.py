"""Inference helpers."""
from __future__ import annotations

import numpy as np
import polars as pl


def predict_with_model(model, features: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Run a single fitted model on a feature frame. Returns date, ticker, score."""
    sub = features.drop_nulls(subset=feature_columns)
    X = sub.select(feature_columns).to_numpy().astype(np.float64)
    score = model.predict(X)
    return sub.select(["date", "ticker"]).with_columns(score=pl.Series(score))
