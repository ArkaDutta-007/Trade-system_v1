"""The feature reserve — the canonical catalog the forecasters draw from.

Grouped so training can select "all" or a named subset, and so we have one place
that documents every leakage-safe feature the system produces.  ``resolve_reserve``
intersects the catalog with the columns actually present in a frame (some need
events/macro to be populated), guaranteeing the trainer never references a column
that wasn't built.
"""
from __future__ import annotations

import polars as pl

# ── Groups ──────────────────────────────────────────────────────────────────
TREND = [
    "mom_5d", "mom_10d", "mom_20d", "mom_60d", "mom_120d", "mom_12m1m",
    "sma_gap_10", "sma_gap_20", "sma_gap_50", "sma_gap_200",
    "mom_accel", "dist_52w_high", "dist_52w_low", "bb_pctb_20",
    "breakout_20", "breakdown_20",
]
VOLATILITY = [
    "vol_10d", "vol_20d", "vol_60d", "atr_14",
    "downside_vol_20d", "downside_vol_60d", "vol_of_vol_60",
    "ret_skew_60", "ret_kurt_60", "max_dd_252", "dd_from_high_60",
]
LIQUIDITY = [
    "rel_vol_20", "avg_dollar_volume_20", "amihud_illiq_20",
    "volume_z_60", "overnight_gap",
]
MEAN_REVERSION = ["rsi_14"]
REGIME = [
    "bull_regime", "high_vol_regime", "mom_20d_rank", "excess_ret_1d",
    "beta_60", "corr_bench_60",
]
MACRO = [
    "macro_ust_10y", "macro_yield_curve", "macro_vix", "macro_hy_oas", "macro_fed_funds",
    "macro_ust_10y_chg_20d", "macro_yield_curve_chg_20d", "macro_vix_chg_20d",
    "macro_hy_oas_chg_20d", "macro_fed_funds_chg_20d",
    "macro_ust_10y_z_252", "macro_yield_curve_z_252", "macro_vix_z_252",
    "macro_hy_oas_z_252", "macro_fed_funds_z_252",
]
EVENTS = [
    "event_count", "event_sentiment_mean", "event_magnitude_mean",
    "event_novelty_max", "risk_flag_count",
    "sent_decay_3d", "sent_decay_7d", "sent_decay_14d", "sent_momentum",
    "apprehension_score",
]
CALENDAR = ["days_to_fomc", "days_to_earnings", "macro_event_imminent"]

GROUPS: dict[str, list[str]] = {
    "trend": TREND,
    "volatility": VOLATILITY,
    "liquidity": LIQUIDITY,
    "mean_reversion": MEAN_REVERSION,
    "regime": REGIME,
    "macro": MACRO,
    "events": EVENTS,
    "calendar": CALENDAR,
}

# Nonlinear-dynamics + RMT groups (cross-domain maths). Derived from the feature
# registry so the catalog can never drift from what the builders actually emit.
from .nonlinear import ALL_FEATURES as _NL_FEATURES  # noqa: E402
from .rmt import RMT_COLUMNS as _RMT_COLUMNS         # noqa: E402

for _f in _NL_FEATURES:
    GROUPS.setdefault(_f.group, []).append(_f.name)
GROUPS["rmt"] = list(_RMT_COLUMNS)

# Full reserve, de-duplicated, order-preserving
FEATURE_RESERVE: list[str] = list(dict.fromkeys(
    c for g in GROUPS.values() for c in g
))


def resolve_reserve(
    df: pl.DataFrame,
    groups: list[str] | None = None,
    min_non_null_frac: float = 0.6,
) -> list[str]:
    """Return reserve columns present in ``df`` with enough non-null coverage.

    Parameters
    ----------
    groups:
        Restrict to these named groups (default: all).
    min_non_null_frac:
        Drop a feature if it's null for more than (1 - this) of rows — keeps the
        trainer from leaning on a feature that's mostly missing (e.g. events when
        no news was ingested).
    """
    wanted = (
        [c for g in groups for c in GROUPS.get(g, [])] if groups else FEATURE_RESERVE
    )
    present = [c for c in wanted if c in df.columns]
    if not present:
        return []
    n = max(1, df.height)
    keep = []
    cov = df.select([pl.col(c).is_not_null().sum().alias(c) for c in present]).row(0)
    for c, nn in zip(present, cov):
        if nn / n >= min_non_null_frac:
            keep.append(c)
    return keep


def reserve_report(df: pl.DataFrame) -> dict[str, dict]:
    """Per-group availability summary for `ts features --reserve` / docs."""
    out: dict[str, dict] = {}
    n = max(1, df.height)
    for g, cols in GROUPS.items():
        present = [c for c in cols if c in df.columns]
        out[g] = {"defined": len(cols), "present": len(present), "columns": present}
    out["_total"] = {"reserve_size": len(FEATURE_RESERVE),
                     "present": len([c for c in FEATURE_RESERVE if c in df.columns]),
                     "rows": n}
    return out
