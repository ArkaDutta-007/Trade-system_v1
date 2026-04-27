"""Position sizing helpers."""
from __future__ import annotations

import polars as pl


def equal_weight(weights: pl.DataFrame) -> pl.DataFrame:
    """Normalize positive weights to equal-weight per (date)."""
    df = weights.filter(pl.col("weight") > 0).with_columns(
        n=pl.len().over("date"),
    )
    return df.with_columns(weight=(1.0 / pl.col("n")).cast(pl.Float64)).select(
        ["date", "ticker", "weight"]
    )


def vol_target_weights(
    weights: pl.DataFrame,
    vol_features: pl.DataFrame,
    target_annual_vol: float = 0.15,
    vol_col: str = "vol_20d",
) -> pl.DataFrame:
    """Scale each non-zero weight by target_vol/realized_vol."""
    df = weights.join(
        vol_features.select(["date", "ticker", vol_col]),
        on=["date", "ticker"], how="left",
    ).with_columns(
        scale=(target_annual_vol / pl.col(vol_col).clip(lower_bound=1e-4)).clip(0.0, 5.0)
    ).with_columns(
        weight=pl.col("weight") * pl.col("scale").fill_null(1.0)
    )
    return df.select(["date", "ticker", "weight"])
