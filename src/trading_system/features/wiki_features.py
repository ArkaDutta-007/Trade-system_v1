"""Wikipedia pageview features — causal retail-attention, point-in-time.

Turns the ``wiki_history`` silver table (daily article pageviews per ticker) into
leakage-safe trained features:

  wiki_attention_z    abnormal attention: z-score of log(1+views) vs a trailing 60d
  wiki_attention_mom  attention momentum: fast-EW minus slow-EW log views (rising buzz)

``views[D]`` is the traffic on day D (known by close), and all aggregations are
trailing/EW over rows sorted by (ticker, date) — same-day-or-earlier only. Rows
before a ticker's first coverage / uncovered tickers stay null and are handled by
:mod:`features.sparse_signals`.
"""
from __future__ import annotations

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

WIKI_COLUMNS = ["wiki_attention_z", "wiki_attention_mom"]


def compute_wiki_features(features: pl.DataFrame, wiki: pl.DataFrame | None) -> pl.DataFrame:
    """Join causal Wikipedia attention features onto the (ticker, date) panel."""
    if wiki is None or wiki.is_empty() or "wiki_views" not in wiki.columns:
        return features

    w = wiki.select(["ticker", "date", "wiki_views"]).unique(subset=["ticker", "date"])
    out = features.join(w, on=["ticker", "date"], how="left").sort(["ticker", "date"])

    out = out.with_columns(
        _logv=(pl.col("wiki_views").cast(pl.Float64) + 1).log(),
    ).with_columns(
        _lv_mean=pl.col("_logv").rolling_mean(60, min_samples=20).over("ticker"),
        _lv_std=pl.col("_logv").rolling_std(60, min_samples=20).over("ticker"),
        _fast=pl.col("_logv").ewm_mean(half_life=3, ignore_nulls=True).over("ticker"),
        _slow=pl.col("_logv").ewm_mean(half_life=20, ignore_nulls=True).over("ticker"),
    ).with_columns(
        wiki_attention_z=pl.when(pl.col("_lv_std") > 1e-9)
        .then((pl.col("_logv") - pl.col("_lv_mean")) / pl.col("_lv_std"))
        .otherwise(None),
        wiki_attention_mom=(pl.col("_fast") - pl.col("_slow")),
    )

    # Uncovered tickers → keep null (sparse_signals adds the presence flag + fill).
    has_cov = out.group_by("ticker").agg(pl.col("wiki_views").is_not_null().any().alias("_cov"))
    out = out.join(has_cov, on="ticker", how="left").with_columns(
        wiki_attention_z=pl.when(pl.col("_cov")).then(pl.col("wiki_attention_z")).otherwise(None),
        wiki_attention_mom=pl.when(pl.col("_cov")).then(pl.col("wiki_attention_mom")).otherwise(None),
    )

    drop = [c for c in ["wiki_views", "_logv", "_lv_mean", "_lv_std", "_fast", "_slow", "_cov"]
            if c in out.columns]
    n_cov = out["wiki_attention_z"].is_not_null().sum()
    logger.info(f"Wiki features: attention populated on {n_cov:,}/{out.height:,} rows "
                f"({100*n_cov/max(out.height,1):.0f}% of panel)")
    return out.drop(drop)
