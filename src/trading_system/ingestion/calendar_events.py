"""Calendar / corporate-action events. Earnings dates via yfinance when available."""
from __future__ import annotations

from typing import Iterable

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)


def build_earnings_calendar(tickers: Iterable[str]) -> pl.DataFrame:
    """Best-effort earnings calendar from yfinance. Empty for ETFs."""
    import yfinance as yf

    rows = []
    for t in tickers:
        try:
            yt = yf.Ticker(t)
            df = getattr(yt, "earnings_dates", None)
            if df is None or len(df) == 0:
                continue
            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            for _, r in df.iterrows():
                d = r.get("earnings_date") or r.get("earnings_date_")
                if d is None:
                    continue
                try:
                    d = d.date()
                except Exception:
                    pass
                rows.append({"ticker": t.upper(), "event_type": "earnings", "date": d})
        except Exception as e:
            logger.info(f"No earnings data for {t}: {e}")

    if not rows:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "event_type": pl.Utf8, "date": pl.Date})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])
