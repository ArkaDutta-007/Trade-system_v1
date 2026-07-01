"""GDELT news features for the panel — causal tone + attention, point-in-time.

Turns the ``gdelt_history`` silver table (daily avg tone + article volume per
ticker, from ``ts backfill-news``) into leakage-safe trained features:

  news_tone      exponentially-decayed media tone (sentiment memory) — causal
  news_tone_mom  short- minus long-EW tone (sentiment improving / deteriorating)
  news_buzz      abnormal attention: z-score of log(1+article count) vs 60d

Because ``gdelt_tone[D]`` is the tone of coverage *on day D* (known by that day's
close) and the aggregations are trailing/EW over rows sorted by (ticker, date),
every value uses only same-day-or-earlier information — safe to pair with the
D→D+h forward-return target. Rows before a ticker's first coverage stay null and
the reserve's coverage gate drops a column that never fills.
"""
from __future__ import annotations

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

GDELT_COLUMNS = ["news_tone", "news_tone_mom", "news_buzz"]


def compute_gdelt_features(
    features: pl.DataFrame,
    gdelt: pl.DataFrame | None,
    half_life: int = 7,
) -> pl.DataFrame:
    """Join causal GDELT tone/attention features onto the (ticker, date) panel."""
    if gdelt is None or gdelt.is_empty() or "gdelt_tone" not in gdelt.columns:
        return features

    g = gdelt.select(["ticker", "date", "gdelt_tone", "gdelt_vol"]).unique(subset=["ticker", "date"])
    out = features.join(g, on=["ticker", "date"], how="left").sort(["ticker", "date"])

    out = out.with_columns(
        # article count: 0 where no coverage that day (a real "quiet day" signal)
        _vol=pl.col("gdelt_vol").fill_null(0),
    ).with_columns(
        # EW tone memory (ignore_nulls → carries the last coverage day forward, causal)
        news_tone=pl.col("gdelt_tone").ewm_mean(half_life=half_life, ignore_nulls=True).over("ticker"),
        _tone_fast=pl.col("gdelt_tone").ewm_mean(half_life=3, ignore_nulls=True).over("ticker"),
        _logvol=(pl.col("_vol") + 1).log(),
    ).with_columns(
        news_tone_mom=(pl.col("_tone_fast") - pl.col("news_tone")),
        # abnormal attention vs the trailing 60 trading days
        _lv_mean=pl.col("_logvol").rolling_mean(60, min_samples=20).over("ticker"),
        _lv_std=pl.col("_logvol").rolling_std(60, min_samples=20).over("ticker"),
    ).with_columns(
        news_buzz=pl.when(pl.col("_lv_std") > 1e-9)
        .then((pl.col("_logvol") - pl.col("_lv_mean")) / pl.col("_lv_std"))
        .otherwise(None),
    )

    # A ticker with no GDELT coverage at all → news_tone stays null (dropped by
    # the reserve). Null out buzz there too so it isn't a spurious 0.
    has_cov = out.group_by("ticker").agg(pl.col("gdelt_tone").is_not_null().any().alias("_cov"))
    out = out.join(has_cov, on="ticker", how="left").with_columns(
        news_buzz=pl.when(pl.col("_cov")).then(pl.col("news_buzz")).otherwise(None),
        news_tone_mom=pl.when(pl.col("_cov")).then(pl.col("news_tone_mom")).otherwise(None),
    )

    drop = [c for c in ["gdelt_tone", "gdelt_vol", "_vol", "_tone_fast", "_logvol",
                        "_lv_mean", "_lv_std", "_cov"] if c in out.columns]
    n_cov = out["news_tone"].is_not_null().sum()
    logger.info(f"GDELT features: news_tone populated on {n_cov:,}/{out.height:,} rows "
                f"({100*n_cov/max(out.height,1):.0f}% of panel)")
    return out.drop(drop)
