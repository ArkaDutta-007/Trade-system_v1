"""Risk overlays. Hard caps + drawdown kill switch."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class RiskLimits:
    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.0
    max_drawdown_kill_switch: float = 0.25  # set weights to 0 if equity drawdown exceeds this


def enforce_risk_limits(weights: pl.DataFrame, limits: RiskLimits) -> pl.DataFrame:
    df = weights.with_columns(
        weight=pl.col("weight").clip(-limits.max_position_weight, limits.max_position_weight)
    )
    gross = df.group_by("date").agg(g=pl.col("weight").abs().sum())
    df = df.join(gross, on="date", how="left").with_columns(
        scale=pl.when(pl.col("g") > limits.max_gross_exposure)
        .then(limits.max_gross_exposure / pl.col("g"))
        .otherwise(1.0)
    ).with_columns(
        weight=pl.col("weight") * pl.col("scale")
    ).select(["date", "ticker", "weight"])
    return df


def apply_drawdown_kill_switch(
    weights: pl.DataFrame, equity_curve: pl.DataFrame, threshold: float
) -> pl.DataFrame:
    """Zero out weights once running drawdown breaches threshold."""
    eq = equity_curve.sort("date").with_columns(
        peak=pl.col("equity").cum_max(),
    ).with_columns(dd=(pl.col("equity") / pl.col("peak") - 1.0))
    flagged = eq.filter(pl.col("dd") < -abs(threshold)).select("date").to_series().to_list()
    if not flagged:
        return weights
    blocked_from = min(flagged)
    return weights.with_columns(
        weight=pl.when(pl.col("date") >= blocked_from).then(0.0).otherwise(pl.col("weight"))
    )
