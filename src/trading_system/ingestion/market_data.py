"""Daily OHLCV ingestion via yfinance. Writes raw + bronze parquet."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    workers: int | None = None,
    progress: bool = True,
) -> pl.DataFrame:
    """Fetch daily OHLCV for tickers. Returns a long-form Polars frame.

    Fetches are network-bound, so tickers are pulled **concurrently** with a
    thread pool and a live progress bar (count · rate · ETA · last ticker · fails).
    ``workers`` defaults to ``TS_INGEST_WORKERS`` env or 8 (kept modest so
    yfinance doesn't rate-limit). Set ``progress=False`` for quiet/non-TTY use.
    """
    import yfinance as yf

    if end is None:
        end = datetime.utcnow().date().isoformat()
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    tickers = list(tickers)
    workers = workers or int(os.environ.get("TS_INGEST_WORKERS", 8))
    workers = max(1, min(workers, len(tickers) or 1))
    logger.info(f"Fetching OHLCV: {len(tickers)} tickers {start}→{end} ({workers} workers)")

    def _fetch_one(ticker: str) -> pl.DataFrame:
        t = yf.Ticker(ticker)
        df_pd = t.history(start=start, end=end, auto_adjust=auto_adjust, actions=False)
        return _to_polars(df_pd, ticker)

    frames: list[pl.DataFrame] = []
    fails: list[str] = []

    def _run(update=None):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_one, t): t for t in tickers}
            for fut in as_completed(futs):
                tk = futs[fut]
                try:
                    df = fut.result()
                    if len(df):
                        frames.append(df)
                    else:
                        fails.append(tk)
                except Exception as e:
                    fails.append(tk)
                    logger.debug(f"Failed to fetch {tk}: {e}")
                if update:
                    update(tk, len(fails))

    if progress:
        try:
            from rich.progress import (
                Progress, BarColumn, TextColumn, TimeRemainingColumn,
                MofNCompleteColumn, SpinnerColumn,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]ingest"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("· {task.fields[last]}"),
                TimeRemainingColumn(),
                transient=False,
            ) as prog:
                task = prog.add_task("ingest", total=len(tickers), last="…")
                _run(lambda tk, nf: prog.update(
                    task, advance=1, last=f"{tk}" + (f" [red]{nf} fail[/red]" if nf else "")))
        except Exception:
            _run()  # rich unavailable / non-tty → just run quietly
    else:
        _run()

    if fails:
        logger.warning(f"OHLCV: {len(fails)} tickers returned no data: {', '.join(fails[:15])}"
                       + (" …" if len(fails) > 15 else ""))
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed").sort(["ticker", "date"])
    if "adj_close" not in out.columns and "close" in out.columns:
        out = out.with_columns(pl.col("close").alias("adj_close"))
    return out


def sanitize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Drop incomplete/impossible bars so a few yfinance glitches can't fail QC.

    The most common offender is *today's in-progress bar*: yfinance returns it
    with a null close and a half-formed range (low > open etc.). We drop rows
    with a null/≤0 close and any that violate basic OHLC sanity, logging the count.
    """
    if df.is_empty():
        return df
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return df
    n0 = df.height
    clean = df.filter(
        pl.col("close").is_not_null()
        & (pl.col("close") > 0) & (pl.col("open") > 0)
        & (pl.col("high") > 0) & (pl.col("low") > 0)
        & (pl.col("high") >= pl.col("low"))
        & (pl.col("high") >= pl.col("open")) & (pl.col("high") >= pl.col("close"))
        & (pl.col("low") <= pl.col("open")) & (pl.col("low") <= pl.col("close"))
    )
    dropped = n0 - clean.height
    if dropped:
        logger.warning(f"sanitize_ohlcv: dropped {dropped} incomplete/invalid bars "
                       f"(e.g. today's in-progress bar with null close)")
    return clean


def ingest_universe(cfg: Config, workers: int | None = None) -> Path:
    """Ingest configured universe and write to bronze parquet. Returns path."""
    tickers = cfg["universe"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date")
    df = fetch_ohlcv(tickers, start=start, end=end, workers=workers)
    if df.is_empty():
        raise RuntimeError("No OHLCV data fetched.")
    df = sanitize_ohlcv(df)
    bronze = cfg.path("data_bronze")
    out = bronze / "ohlcv_daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    logger.info(f"Wrote {len(df)} rows to {out}")
    return out
