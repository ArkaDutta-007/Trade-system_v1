#!/usr/bin/env python3
"""Export the honest point-in-time panel for the Vibe-Trading alpha-zoo comparison (2026-10-15).

One long parquet → data/silver/vibe/pit_long.parquet with, for every name that was ever in the
whole-market point-in-time top-1000 (Sep 2024 → today, failures included):
  OHLCV (split/dividend adjusted via adj_close/close), amount = close × volume, vwap (Massive's
  when present, else the typical price), member (in the PIT universe that day), fwd_21 / fwd_63
  (the same forward returns our clean test scores against), and our two models' composite scores:
  prod (production training) and pitaug (point-in-time training where it exists, RESEARCH §13).
Vibe-Trading's own venv reads this file (ops/research/vibe_compare_2026_10.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import lab_2026_10 as LAB  # noqa: E402

OUT = LAB.cfg.path("data_silver") / "vibe"


def main():
    pw, allpx, names = LAB.pit_panel()
    src = pl.read_parquet_schema(LAB.cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet")
    px = allpx.filter(pl.col("ticker").is_in(names)).sort(["ticker", "date"])
    if "vwap" in src:
        vw = (pl.scan_parquet(LAB.cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet")
                .filter(pl.col("ticker").is_in(names)).select("date", "ticker", "vwap").collect())
        px = px.join(vw, on=["date", "ticker"], how="left")
    else:
        px = px.with_columns(vwap=pl.lit(None, dtype=pl.Float64))
    f = (pl.col("adj_close") / pl.col("close"))
    px = px.with_columns(open=pl.col("open") * f, high=pl.col("high") * f, low=pl.col("low") * f,
                         vwap=pl.coalesce(pl.col("vwap"), (pl.col("high") + pl.col("low") + pl.col("close")) / 3) * f,
                         amount=pl.col("close") * pl.col("volume")).with_columns(close=pl.col("adj_close"))
    mem = pw.select("date", "ticker", "fwd_21", "fwd_63").with_columns(member=pl.lit(True))
    prod = LAB.zcomp(pl.read_parquet(LAB.OUT / "pit_base.parquet")).rename({"comp": "prod"})
    aug = LAB.zcomp(pl.read_parquet(LAB.OUT / "pitfix_pit-aug.parquet")).rename({"comp": "pitaug"})
    out = (px.select("date", "ticker", "open", "high", "low", "close", "volume", "amount", "vwap")
             .join(mem, on=["date", "ticker"], how="left").with_columns(pl.col("member").fill_null(False))
             .join(prod, on=["date", "ticker"], how="left").join(aug, on=["date", "ticker"], how="left"))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / "pit_long.parquet")
    print(f"{out.height:,} rows · {out['ticker'].n_unique():,} names · {out['date'].min()} → {out['date'].max()} · "
          f"member rows {out['member'].sum():,} · vwap from Massive: {'vwap' in src}")


if __name__ == "__main__":
    main()
