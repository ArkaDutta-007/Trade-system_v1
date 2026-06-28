"""Position sizing helpers."""
from __future__ import annotations

import polars as pl


def distribution_sized_weights(
    bounds_by_ticker: dict[str, dict],
    horizon: str = "1m",
    kelly_fraction: float = 0.5,
    max_weight: float = 0.10,
    downside_floor: float = 0.02,
    min_edge: float = 0.0,
) -> dict[str, float]:
    """Size positions from the *distribution*, not a point forecast.

    For each name we read the calibrated return band at ``horizon`` and score it
    by reward-to-downside ``edge / |5th-pct loss|`` (a Sortino-style ratio that
    rewards asymmetric upside and penalises fat left tails).  Scores are turned
    into weights via fractional Kelly, capped per name and normalised to ≤100%.

    Parameters
    ----------
    bounds_by_ticker:
        ``{ticker: bounds_dict}`` where each bounds_dict is the output of
        ``decision.bounds.compute_bounds`` (has ``horizons[label].return``).
    horizon:
        Which horizon label to size on ("5d","1m","3m","6m","12m").
    kelly_fraction:
        Fraction of the Kelly-style weight to actually take (0.5 = half-Kelly).
    max_weight:
        Per-name cap.
    downside_floor:
        Floor on the downside magnitude so a near-zero tail can't blow up size.
    min_edge:
        Only size names whose median return exceeds this.
    """
    raw: dict[str, float] = {}
    for ticker, b in bounds_by_ticker.items():
        hz = (b or {}).get("horizons", {}).get(horizon)
        if not hz:
            continue
        r = hz.get("return", {})
        edge = float(r.get("median", 0.0))
        if edge <= min_edge:
            continue
        downside = max(downside_floor, -float(r.get("lo", 0.0)))
        score = edge / downside  # reward-to-downside
        raw[ticker] = max(0.0, score)

    if not raw:
        return {}

    total = sum(raw.values())
    weights = {t: kelly_fraction * (s / total) for t, s in raw.items()}
    # cap per name
    weights = {t: min(w, max_weight) for t, w in weights.items()}
    # renormalise if capping freed up budget but keep gross ≤ 1.0
    gross = sum(weights.values())
    if gross > 1.0:
        weights = {t: w / gross for t, w in weights.items()}
    return weights


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
