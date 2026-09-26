"""`ts data status` — one table for every dataset the system depends on.

For each store: rows, tickers, date span, file age, how many sessions its newest date lags the
last completed session, and a verdict against the freshness it is supposed to have. Plus the
Massive crawler's liveness/budget and disk headroom. No network.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl


@dataclass
class Store:
    name: str
    rel: str                      # file, or a directory of parquet files
    date_col: str | None          # observation-date column
    role: str                     # who consumes it
    max_lag_sessions: int | None  # expected freshness (None = reference data)
    ticker_col: str | None = "ticker"
    per_file_ticker: bool = False  # directory with one file per ticker
    series_dir: bool = False       # directory of differently-shaped series (only `date` shared)


STORES = [
    Store("prices · whole market (Massive, 2y)", "data/bronze/massive/ohlcv_all.parquet", "date", "alpha panel tail, books", 1),
    Store("prices · deep 1970→", "data/bronze/massive/ohlcv_deep.parquet", "date", "alpha panel history", 7),
    Store("prices · legacy liquid universe", "data/bronze/ohlcv_daily.parquet", "date", "legacy engine, books", 1),
    Store("bars · 1-minute (universe, 2y)", "data/bronze/massive/bars_minute", "ts", "research", 3, None, per_file_ticker=True),
    Store("fundamentals (SEC, 2009→)", "data/bronze/massive/financials.parquet", "filing_date", "value/quality/earnings features", 30),
    Store("news + LLM sentiment (2016→)", "data/bronze/massive/news.parquet", "published_utc", "news features", 2),
    Store("short interest (2017→)", "data/bronze/massive/short_interest.parquet", "settlement_date", "short features", 25),
    Store("short volume (2024→)", "data/bronze/massive/short_volume.parquet", "date", "short features", 5),
    Store("ticker overview (SIC, mcap)", "data/bronze/massive/details.parquet", None, "sector map", None),
    Store("splits", "data/bronze/massive/splits.parquet", None, "price adjustment", None),
    Store("dividends", "data/bronze/massive/dividends.parquet", None, "price adjustment", None),
    Store("regime · FRED oil/vol/credit/rates", "data/silver/regime", "date", "regime layer", 3, None, series_dir=True),
    Store("alpha panel", "data/gold/alpha_panel.parquet", "date", "alpha forecaster", 1),
    Store("alpha forecast ledger", "data/ledger/alpha_forecasts.parquet", "date", "tally, calibration", 1),
    Store("legacy gold features", "data/gold/features.parquet", "date", "legacy engine", 1),
]


def last_session(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def sessions_between(a: date, b: date) -> int:
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        n += d.weekday() < 5
    return n


def inspect(repo: Path, s: Store, today: date) -> dict:
    p = repo / s.rel
    row = {"store": s.name, "role": s.role, "rows": None, "tickers": None, "first": None, "last": None,
           "age_h": None, "lag": None, "status": "missing"}
    files = (sorted(p.glob("*.parquet")) if p.is_dir() else [p]) if p.exists() else []
    if not files:
        return row
    row["age_h"] = round((time.time() - max(f.stat().st_mtime for f in files)) / 3600, 1)
    try:
        if s.series_dir:
            ends = [pl.read_parquet(f, columns=["date"])["date"].max() for f in files]
            row.update(rows=len(files), last=str(sorted(ends)[len(ends) // 2])[:10])   # median: Brent/dollar lag by design
        else:
            lf = pl.scan_parquet([str(f) for f in files])
            cols = lf.collect_schema().names()
            aggs = [pl.len().alias("rows")]
            if s.ticker_col and s.ticker_col in cols:
                aggs.append(pl.col(s.ticker_col).n_unique().alias("tickers"))
            if s.date_col and s.date_col in cols:
                aggs += [pl.col(s.date_col).min().alias("first"), pl.col(s.date_col).max().alias("last")]
            r = lf.select(aggs).collect().row(0, named=True)
            row["rows"] = r["rows"]
            row["tickers"] = len(files) if s.per_file_ticker else r.get("tickers")
            row["first"] = str(r["first"])[:10] if r.get("first") is not None else None
            row["last"] = str(r["last"])[:10] if r.get("last") is not None else None
        if s.max_lag_sessions is None:
            row["status"] = "reference"
        elif row["last"]:
            lag = sessions_between(date.fromisoformat(row["last"]), last_session(today))
            row["lag"] = lag
            row["status"] = "ok" if lag <= s.max_lag_sessions else ("stale" if lag <= 3 * s.max_lag_sessions else "STALE")
        else:
            row["status"] = "no dates"
    except Exception as e:  # diagnostics only
        row["status"] = f"error: {str(e)[:60]}"
    return row


def crawler_summary(repo: Path) -> dict:
    raw = repo / "data/raw/massive"
    out: dict = {"alive": False}
    try:
        st = json.loads((raw / "crawler_state.json").read_text())
        out = {k: st.get(k) for k in ("ts", "current", "idle", "blocked", "failing", "keys_live", "uptime_h")}
        out["alive"] = (Path("/proc") / str(st.get("pid", 0))).exists()
    except Exception:
        pass
    out["calls_by_day"] = {p.name[-8:]: p.stat().st_size for p in sorted(raw.glob(".calls-*"))[-7:]}
    return out


def collect(repo: Path, today: date | None = None) -> dict:
    today = today or date.today()
    du = shutil.disk_usage(repo)
    data_bytes = sum(f.stat().st_size for f in (repo / "data").rglob("*") if f.is_file())
    return {"as_of": str(today), "last_session": str(last_session(today)),
            "stores": [inspect(repo, s, today) for s in STORES], "crawler": crawler_summary(repo),
            "disk": {"free_gb": round(du.free / 1e9, 1), "used_pct": round(100 * du.used / du.total, 1),
                     "data_gb": round(data_bytes / 1e9, 2)}}
