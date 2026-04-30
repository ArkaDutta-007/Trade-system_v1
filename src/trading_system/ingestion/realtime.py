"""V2: Quasi-realtime price data for dashboard live overlay.

Uses yfinance's fast_info (no API key needed) for current prices
and intraday bars for short-term context.

This module is intentionally read-only — it never executes trades.
The LivePriceFeed can run in a background thread to push price
updates to a queue that the Streamlit dashboard reads from.
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Sequence

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)


def live_price_snapshot(tickers: Sequence[str]) -> dict[str, float]:
    """Fetch the current last-traded price for each ticker.

    Uses yfinance Ticker.fast_info["last_price"] — sub-second, no key required.
    Returns a dict of ticker → price (USD). Missing tickers are omitted.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols (e.g. ["AAPL", "MSFT"]).
    """
    import yfinance as yf

    prices: dict[str, float] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price is not None:
                prices[t.upper()] = float(price)
        except Exception as e:
            logger.debug(f"live_price_snapshot failed for {t}: {e}")
    return prices


def fetch_intraday_bars(
    tickers: Sequence[str],
    interval: str = "5m",
    lookback_days: int = 2,
) -> pl.DataFrame:
    """Fetch intraday OHLCV bars from yfinance.

    Parameters
    ----------
    tickers:
        Ticker symbols to fetch.
    interval:
        Bar interval: "1m", "5m", "15m", "30m", "60m".
        Note: yfinance limits 1m to 7 days, 5m to 60 days.
    lookback_days:
        How many calendar days back to fetch.

    Returns
    -------
    pl.DataFrame with columns:
        datetime (Datetime[ns, UTC]), ticker (Utf8),
        open (Float64), high (Float64), low (Float64),
        close (Float64), volume (Int64)
    """
    import yfinance as yf
    import pandas as pd

    period_map = {1: "1d", 2: "2d", 5: "5d", 7: "7d", 14: "14d", 30: "1mo", 60: "2mo"}
    period = period_map.get(lookback_days, f"{lookback_days}d")

    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.warning(f"fetch_intraday_bars failed: {e}")
        return _empty_intraday_schema()

    if raw.empty:
        return _empty_intraday_schema()

    rows = []
    # Handle single vs multi-ticker yfinance output layout
    if len(tickers) == 1:
        t = list(tickers)[0].upper()
        for idx, row in raw.iterrows():
            rows.append({
                "datetime": idx,
                "ticker": t,
                "open": float(row.get("Open", float("nan"))),
                "high": float(row.get("High", float("nan"))),
                "low": float(row.get("Low", float("nan"))),
                "close": float(row.get("Close", float("nan"))),
                "volume": int(row.get("Volume", 0) or 0),
            })
    else:
        for t in tickers:
            t_upper = t.upper()
            try:
                sub = raw[t_upper] if t_upper in raw.columns.get_level_values(0) else None
            except Exception:
                sub = None
            if sub is None or sub.empty:
                continue
            for idx, row in sub.iterrows():
                rows.append({
                    "datetime": idx,
                    "ticker": t_upper,
                    "open": float(row.get("Open", float("nan"))),
                    "high": float(row.get("High", float("nan"))),
                    "low": float(row.get("Low", float("nan"))),
                    "close": float(row.get("Close", float("nan"))),
                    "volume": int(row.get("Volume", 0) or 0),
                })

    if not rows:
        return _empty_intraday_schema()

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("datetime").cast(pl.Datetime("ns", "UTC")))
        .sort(["ticker", "datetime"])
    )


def _empty_intraday_schema() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "datetime": pl.Datetime("ns", "UTC"),
        "ticker": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
    })


# ---------------------------------------------------------------------------
# LivePriceFeed — background polling thread for dashboard
# ---------------------------------------------------------------------------

class LivePriceFeed:
    """Polls yfinance for live prices every N seconds and pushes to a queue.

    Designed for the Streamlit dashboard to consume without blocking renders.

    Usage::

        feed = LivePriceFeed(["AAPL", "MSFT", "GOOGL"], interval_seconds=300)
        feed.start()
        ...
        prices = feed.latest_prices  # dict[str, float]
        feed.stop()
    """

    def __init__(
        self,
        tickers: Sequence[str],
        interval_seconds: int = 300,
        max_queue_size: int = 10,
    ):
        self._tickers = list(tickers)
        self._interval = interval_seconds
        self._queue: queue.Queue[dict[str, float]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, float] = {}
        self._last_update: datetime | None = None

    @property
    def latest_prices(self) -> dict[str, float]:
        """Most recent price snapshot. Empty dict if feed not started."""
        # Drain queue to get freshest data
        while not self._queue.empty():
            try:
                self._latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return self._latest

    @property
    def last_update_time(self) -> datetime | None:
        return self._last_update

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="LivePriceFeed",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"LivePriceFeed started (interval={self._interval}s, {len(self._tickers)} tickers)")

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("LivePriceFeed stopped.")

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                prices = live_price_snapshot(self._tickers)
                if prices:
                    # Discard oldest entry if queue is full
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put(prices)
                    self._last_update = datetime.now()
                    logger.debug(f"LivePriceFeed: fetched {len(prices)} prices")
            except Exception as e:
                logger.warning(f"LivePriceFeed poll error: {e}")
            self._stop_event.wait(timeout=self._interval)
