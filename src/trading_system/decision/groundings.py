"""Pull the supporting evidence used by `analyze_symbol`.

Each ground is a small dict so it can be rendered to markdown / JSON without
loss. Fields are designed to be read by humans and stored as audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl


def _latest_row(df: pl.DataFrame, ticker: str) -> dict | None:
    sub = df.filter(pl.col("ticker") == ticker).sort("date").tail(1)
    if sub.is_empty():
        return None
    return sub.to_dicts()[0]


def technical_grounding(features: pl.DataFrame, ticker: str) -> dict[str, Any]:
    row = _latest_row(features, ticker)
    if row is None:
        return {"available": False, "reason": "ticker not in feature matrix"}
    keys = [
        "date", "adj_close",
        "ret_1d", "mom_5d", "mom_20d", "mom_60d", "mom_120d", "mom_12m1m",
        "vol_20d", "vol_60d", "rsi_14", "atr_14",
        "sma_gap_20", "sma_gap_50", "sma_gap_200",
        "breakout_20", "dd_from_high_60", "rel_vol_20", "avg_dollar_volume_20",
    ]
    out = {k: row.get(k) for k in keys if k in row}
    out["available"] = True
    return out


def regime_grounding(features: pl.DataFrame, ticker: str) -> dict[str, Any]:
    row = _latest_row(features, ticker)
    if row is None:
        return {"available": False}
    return {
        "available": True,
        "bull_regime": row.get("bull_regime"),
        "high_vol_regime": row.get("high_vol_regime"),
        "mom_20d_rank": row.get("mom_20d_rank"),
        "excess_ret_1d": row.get("excess_ret_1d"),
    }


def cross_section_grounding(features: pl.DataFrame, ticker: str, top_n: int = 10) -> dict:
    """Where does this ticker rank vs the universe today?"""
    last = features["date"].max()
    today = features.filter(pl.col("date") == last)
    if today.is_empty() or "mom_20d" not in today.columns:
        return {"available": False}
    ranked = today.sort("mom_20d", descending=True, nulls_last=True)
    ranked = ranked.with_columns(rk=pl.int_range(1, ranked.height + 1))
    me = ranked.filter(pl.col("ticker") == ticker).select("rk", "mom_20d").to_dicts()
    if not me:
        return {"available": False}
    return {
        "available": True,
        "as_of": str(last),
        "rank_in_universe": int(me[0]["rk"]),
        "universe_size": ranked.height,
        "top_by_mom_20d": ranked.head(top_n).select("ticker", "mom_20d").to_dicts(),
        "bottom_by_mom_20d": ranked.tail(top_n).select("ticker", "mom_20d").to_dicts(),
    }


def event_grounding(events: pl.DataFrame | None, ticker: str, days: int = 14) -> dict:
    if events is None or events.is_empty():
        return {"available": False, "rows": []}
    cutoff = events["known_at"].max()  # latest event time
    if cutoff is None:
        return {"available": False, "rows": []}
    horizon = cutoff - timedelta(days=days)
    sub = (
        events
        .filter(pl.col("known_at") >= horizon)
        .filter(pl.col("tickers").list.contains(ticker.upper()))
        .sort("known_at", descending=True)
        .head(20)
    )
    if sub.is_empty():
        return {"available": False, "rows": []}
    return {
        "available": True,
        "window_days": days,
        "rows": sub.select(
            "known_at", "source", "event_type", "sentiment", "magnitude", "summary", "source_url"
        ).to_dicts(),
    }


def model_grounding(
    score: float | None,
    feature_columns: list[str] | None,
    shap_summary: pl.DataFrame | None,
    top_features: int = 10,
) -> dict:
    if score is None:
        return {"available": False, "reason": "no model in registry; using rule-based fallback"}
    out = {
        "available": True,
        "score": float(score),
        "score_meaning": "expected forward 5-day return (regression target)",
        "feature_columns": feature_columns or [],
    }
    if shap_summary is not None and not shap_summary.is_empty():
        out["top_features_by_mean_abs_shap"] = (
            shap_summary.head(top_features).to_dicts()
        )
    return out
