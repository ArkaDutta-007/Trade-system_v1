"""Assemble the macro/calendar context inputs for build_feature_matrix().

Centralises the "what's the backdrop" inputs so the daily pipeline and the
single-symbol analyzer build features identically:

  * macro_features    — FRED levels (cached, resilient)
  * economic_calendar — FOMC + CPI/NFP/PPI/Retail/GDP release dates
  * earnings_calendar — per-ticker earnings dates (best-effort via yfinance)

All three are best-effort: any failure returns None for that input and the
feature build simply omits those columns rather than aborting.
"""
from __future__ import annotations

import polars as pl

from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)


def build_macro_inputs(
    cfg: Config,
    tickers: list[str] | None = None,
    with_earnings: bool = True,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None]:
    """Return (macro_features, economic_calendar, earnings_calendar)."""
    macro_features = None
    economic_calendar = None
    earnings_calendar = None

    macro_cfg = cfg.get("macro", {}) or {}
    if macro_cfg.get("enabled", True):
        try:
            from .macro import build_macro_features, DEFAULT_MACRO_SERIES
            macro_features = build_macro_features(
                series_map=macro_cfg.get("series") or DEFAULT_MACRO_SERIES,
                cache_dir=cfg.path("data_silver") / "macro_cache",
                cache_ttl_hours=float(macro_cfg.get("cache_ttl_hours", 12.0)),
            )
            if macro_features is not None and not macro_features.is_empty():
                logger.info(f"macro features: {macro_features.width - 1} cols, latest {macro_features['date'].max()}")
        except Exception as e:
            logger.warning(f"macro features unavailable (non-fatal): {e}")

    try:
        from ..ingestion.calendar_events import fetch_economic_calendar
        economic_calendar = fetch_economic_calendar(lookahead_days=30, lookback_days=400)
    except Exception as e:
        logger.warning(f"economic calendar unavailable (non-fatal): {e}")

    if with_earnings and tickers:
        try:
            from ..ingestion.calendar_events import build_earnings_calendar
            earnings_calendar = build_earnings_calendar(tickers)
        except Exception as e:
            logger.warning(f"earnings calendar unavailable (non-fatal): {e}")

    return macro_features, economic_calendar, earnings_calendar
