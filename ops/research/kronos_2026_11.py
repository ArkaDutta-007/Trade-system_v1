#!/usr/bin/env python3
"""Kronos (Shi et al., AAAI 2026), the open foundation model for price bars, on the honest point-in-time panel.

Kronos was pre-trained on bars from 45 exchanges and needs no training here. The code is cloned to
~/Desktop/Kronos (MIT licence; Arka approved running it on 2026-11-13) and runs on the RTX 2070.
Fixed before the first run:
  model      Kronos-small + Kronos-Tokenizer-base (the pair its README names), daily adjusted OHLCV + amount
  context    120 sessions; every point-in-time member with 120 clean sessions of history that day
  forecast   21 sessions ahead, 5 sampled paths, T=1, top_p=.9; score = mean forecast close / last close − 1
  dates      the last close of each month from 2025-03 (the first with 120 whole-market sessions): these are
             exactly the scores a book rebalanced on the first session of the month uses
  verdict    rank IC against the same 21-day forward returns as our models, same dates and names; then the
             index-minus book and a 50/50 blend with the challenger (train_2026_11.py). The published weights
             date from mid-2025, so only decision dates from 2025-09 are out of sample for the model itself.
Scores → reports/alpha/lab_2026_11/kronos_scores.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path[:0] = [str(Path.home() / "Desktop/Kronos/repo"), str(Path.home() / "Desktop/Kronos/deps")]
import lab_2026_11 as L  # noqa: E402
from trading_system.alpha import challenger as C  # noqa: E402

LAB, OUT = L.LAB, L.OUT
CTX, PRED, SAMPLES, BATCH = 120, 21, 5, 96


def main(limit: int = 0):
    from model import Kronos, KronosPredictor, KronosTokenizer
    cfg = LAB.cfg
    pw = C.load_panel(cfg).select("date", "ticker")
    px, _ = C.pit_prices(cfg)
    f = pl.col("adj_close") / pl.col("close")
    px = px.sort(["ticker", "date"]).with_columns(open=pl.col("open") * f, high=pl.col("high") * f, low=pl.col("low") * f,
                                                  amount=pl.col("close") * pl.col("volume"), close=pl.col("adj_close"))
    by = {t: g.select("date", "open", "high", "low", "close", "volume", "amount") for (t,), g in px.group_by(["ticker"])}
    cal = sorted(pw["date"].unique().to_list())
    me = [d for d, nx in zip(cal[:-1], cal[1:]) if d.month != nx.month and pd.Timestamp(d) >= pd.Timestamp("2025-03-01")]
    me = me[::-1]                                  # newest first: the out-of-sample dates finish first
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    net = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(net, tok, device="cuda:0", max_context=512)
    path = OUT / "kronos_scores.parquet"
    parts = [pl.read_parquet(path)] if path.exists() else []
    seen = set(parts[0]["date"].unique().to_list()) if parts else set()
    t0 = time.time()
    for k, d in enumerate([x for x in me if x not in seen][:limit or None]):
        names = pw.filter(pl.col("date") == d)["ticker"].to_list()
        ft = pd.bdate_range(pd.Timestamp(d) + pd.Timedelta(days=1), periods=PRED)
        dfs, xs, ys, ok, last = [], [], [], [], []
        for t in names:
            g = by.get(t)
            if g is None:
                continue
            g = g.filter(pl.col("date") <= d).tail(CTX)
            if g.height < CTX or g["volume"].min() <= 0 or g.select(pl.all().is_null().any()).row(0).count(True):
                continue
            p = g.to_pandas()
            dfs.append(p[["open", "high", "low", "close", "volume", "amount"]].astype(float))
            xs.append(pd.Series(pd.to_datetime(p["date"]))); ys.append(pd.Series(ft)); ok.append(t); last.append(float(p["close"].iloc[-1]))
        rows = []
        for i in range(0, len(ok), BATCH):
            out = pred.predict_batch(dfs[i:i + BATCH], xs[i:i + BATCH], ys[i:i + BATCH], pred_len=PRED, T=1.0, top_p=.9,
                                     sample_count=SAMPLES, verbose=False)
            rows += [(ok[i + j], float(o["close"].iloc[-1]) / last[i + j] - 1) for j, o in enumerate(out)]
        parts.append(pl.DataFrame({"date": [d] * len(rows), "ticker": [r[0] for r in rows], "score": [r[1] for r in rows]},
                                  schema={"date": pl.Date, "ticker": pl.Utf8, "score": pl.Float64}))
        pl.concat(parts).write_parquet(path)
        LAB.log(f"kronos {d}: {len(rows)} of {len(names)} names · {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
