"""Daily OHLCV ingestion via yfinance. Writes raw + bronze parquet."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import polars as pl

from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)


def _to_polars(df_pd, ticker: str) -> pl.DataFrame:
    """Convert yfinance pandas frame to a tidy Polars frame."""
    if df_pd is None or len(df_pd) == 0:
        return pl.DataFrame()
    df = df_pd.reset_index()
    # yfinance may return MultiIndex columns when group_by='ticker'
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    pdf = pl.from_pandas(df)
    rename = {}
    for src, dst in [("adj_close", "adj_close"), ("close", "close"), ("open", "open"),
                     ("high", "high"), ("low", "low"), ("volume", "volume"), ("date", "date")]:
        if src in pdf.columns:
            rename[src] = dst
    pdf = pdf.rename(rename)
    if "date" in pdf.columns:
        pdf = pdf.with_columns(pl.col("date").cast(pl.Date))
    pdf = pdf.with_columns(pl.lit(ticker).alias("ticker"))
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"] if c in pdf.columns]
    return pdf.select(keep)


def fetch_ohlcv(
    tickers: Iterable[str],
    start: str | date,
    end: str | date | None = None,
    auto_adjust: bool = False,
) -> pl.DataFrame:
    """Fetch daily OHLCV for tickers. Returns a long-form Polars frame."""
    import yfinance as yf

    if end is None:
        end = datetime.utcnow().date().isoformat()
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    tickers = list(tickers)
    logger.info(f"Fetching OHLCV: {len(tickers)} tickers from {start} to {end}")

    frames: list[pl.DataFrame] = []
    # Batch to avoid yfinance hangs and keep memory bounded
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            df_pd = t.history(start=start, end=end, auto_adjust=auto_adjust, actions=False)
            df = _to_polars(df_pd, ticker)
            if len(df):
                frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed").sort(["ticker", "date"])
    # Compute adjusted_factor and total_return relative columns
    if "adj_close" not in out.columns and "close" in out.columns:
        out = out.with_columns(pl.col("close").alias("adj_close"))
    return out


def ingest_universe(cfg: Config) -> Path:
    """Ingest configured universe and write to bronze parquet. Returns path."""
    tickers = cfg["universe"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")
    df = fetch_ohlcv(tickers, start=start, end=end)
    if df.is_empty():
        raise RuntimeError("No OHLCV data fetched.")
    bronze = cfg.path("data_bronze")
    out = bronze / "ohlcv_daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    logger.info(f"Wrote {len(df)} rows to {out}")
    return out
