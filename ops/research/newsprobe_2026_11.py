#!/usr/bin/env python3
"""A language model's reading of the news, trained on our own returns (2026-11-13, RESEARCH_2026-11 §13).

Earlier passes tried sentiment labels, news types and a local model's judgments; none added to the stock model.
This is the Chen-Kelly-Xiu design ("Expected Returns and Large Language Models"): keep the language model frozen,
take its embedding of each article, and train a small head to predict returns. Fixed before the first run:

  text       title + first 300 characters of the description; articles tagged with at most 3 tickers
  model      BAAI/bge-small-en-v1.5 (384-d sentence embeddings, open weights), CLS pooling, 96 tokens
  timing     an article counts from the first session whose 16:00 ET close comes after it was published
  features   per stock and day: the mean embedding of its articles over the last 5 sessions + log article count
  head       ridge regression on the gaussian-rank labels y_5 and y_21; penalty chosen on 2023-09 → 2024-08
  train      survivor panel, 2021-01 → 2024-08, every 5th session (news coverage starts in 2021)
  test       every point-in-time stock-day from 2024-11 with at least one article in the window
  score      mean of the z-scored 5-day and 21-day predictions → reports/alpha/lab_2026_11/news_scores.parquet
Verdict in train_2026_11.py (`extra`): IC alone, and base + ½·news in the books.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))
import panel_2026_11 as PP  # noqa: E402

LAB, OUT, P = PP.LAB, PP.OUT, PP.P
from trading_system.alpha import challenger as C  # noqa: E402

WORK = LAB.cfg.path("data_silver") / "newsprobe"
MODEL = "BAAI/bge-small-en-v1.5"
TRAIN = (date(2021, 1, 1), date(2024, 8, 31))
VALID_FROM = date(2023, 9, 1)
WIN, STRIDE = 5, 5


def articles() -> pl.DataFrame:
    n = (pl.scan_parquet(LAB.cfg.path("data_bronze") / "massive" / "news.parquet")
           .filter(pl.col("published_utc") >= pl.datetime(TRAIN[0].year, 1, 1, time_zone="UTC"))
           .select("article_id", "published_utc", "ticker", "title", "description").collect())
    n = n.filter(pl.len().over("article_id") <= 3)
    return n.with_columns(text=pl.col("title") + ". " + pl.col("description").fill_null("").str.slice(0, 300))


def embed(texts: list[str]) -> np.ndarray:
    sys.modules.setdefault("torchaudio", None)                # its shared library does not load here; not needed
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    net = AutoModel.from_pretrained(MODEL).half().cuda().eval()
    order = np.argsort([len(t) for t in texts])               # similar lengths together: less padding
    out = np.zeros((len(texts), net.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), 384):
            idx = order[i:i + 384]
            b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
            e = torch.nn.functional.normalize(net(**b).last_hidden_state[:, 0], dim=-1)
            out[idx] = e.float().cpu().numpy().astype(np.float16)
            if (i // 384) % 200 == 0:
                LAB.log(f"  embedded {i:,}/{len(texts):,} · {(time.time() - t0) / 60:.1f} min")
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    n = articles()
    arts = n.select("article_id", "text").unique(subset=["article_id"]).sort("article_id").with_row_index("ai")
    ep = WORK / "emb.npy"
    if ep.exists() and np.load(ep, mmap_mode="r").shape[0] == arts.height:
        E = np.load(ep)
    else:
        E = embed(arts["text"].to_list())
        np.save(ep, E)
    LAB.log(f"{arts.height:,} articles embedded ({E.shape[1]}-d)")
    hist = P.load_panel(LAB.cfg).select("date", "ticker", "y_5", "y_21", "fwd_21")
    pw = C.load_panel(LAB.cfg).select("date", "ticker", "fwd_5", "fwd_21", "fwd_63")
    cal = sorted(set(hist["date"].unique().to_list()) | set(pw["date"].unique().to_list()))
    cdf = pl.DataFrame({"e": cal}).with_row_index("ei").with_columns(pl.col("ei").cast(pl.Int64))
    et = pl.col("published_utc").dt.convert_time_zone("America/New_York")
    rows = (n.join(arts.select("article_id", "ai"), on="article_id")
             .with_columns(d0=pl.when(et.dt.hour() >= 16).then(et.dt.date() + pl.duration(days=1)).otherwise(et.dt.date()))
             .sort("d0").join_asof(cdf.sort("e"), left_on="d0", right_on="e", strategy="forward").drop_nulls("ei")
             .select("ticker", "ai", "ei"))
    # keys: training stock-days (every STRIDE-th session) and every point-in-time stock-day
    tr = (hist.filter((pl.col("date") >= TRAIN[0]) & (pl.col("date") <= TRAIN[1])).join(cdf, left_on="date", right_on="e")
              .filter(pl.col("ei") % STRIDE == 0).with_columns(set=pl.lit("train")))
    te = pw.join(cdf, left_on="date", right_on="e").with_columns(set=pl.lit("test"))
    keys = pl.concat([tr.select("ticker", "ei", "date", "set"), te.select("ticker", "ei", "date", "set")]).unique(subset=["ticker", "ei", "set"]).with_row_index("ki")
    pairs = pl.concat([rows.with_columns(ei=pl.col("ei") + k) for k in range(WIN)]).join(keys.select("ticker", "ei", "ki"), on=["ticker", "ei"])
    LAB.log(f"{keys.height:,} stock-days · {pairs.height:,} article links")
    S = torch.zeros((keys.height, E.shape[1]), dtype=torch.float32)
    cnt = torch.zeros(keys.height, dtype=torch.float32)
    ki, ai = pairs["ki"].to_numpy().astype(np.int64), pairs["ai"].to_numpy().astype(np.int64)
    for i in range(0, len(ki), 400_000):
        k = torch.from_numpy(ki[i:i + 400_000])
        S.index_add_(0, k, torch.from_numpy(E[ai[i:i + 400_000]].astype(np.float32)))
        cnt.index_add_(0, k, torch.ones(len(k)))
    has = (cnt > 0).numpy()
    X = torch.cat([S / cnt.clamp(min=1).unsqueeze(1), torch.log1p(cnt).unsqueeze(1)], 1).numpy()
    keys = keys.with_columns(has=pl.Series(has), n_art=pl.Series(cnt.numpy()))
    trk = keys.filter((pl.col("set") == "train") & pl.col("has")).join(hist, on=["date", "ticker"])
    tek = keys.filter((pl.col("set") == "test") & pl.col("has"))
    mu, sd = X[trk["ki"].to_numpy()].mean(0), X[trk["ki"].to_numpy()].std(0) + 1e-6
    Z = lambda k: (X[k] - mu) / sd

    def ridge(Xa, y, a):
        A = Xa.T @ Xa + a * np.eye(Xa.shape[1])
        return np.linalg.solve(A, Xa.T @ y)

    def day_ic(df, pred, col):
        g = (df.with_columns(p=pl.Series(pred)).drop_nulls(col).group_by("date").agg(ic=pl.corr("p", col, method="spearman"), n=pl.len()).filter(pl.col("n") >= 30))
        return float(g["ic"].mean()) if g.height else float("nan")

    res = {"articles": arts.height, "train_rows": trk.height, "test_rows": tek.height,
           "coverage_test": tek.height / max(1, keys.filter(pl.col("set") == "test").height), "valid": {}}
    preds = {}
    for h in (5, 21):
        d = trk.drop_nulls(f"y_{h}")
        fit, val = d.filter(pl.col("date") < VALID_FROM), d.filter(pl.col("date") >= VALID_FROM)
        Xf, Xv = Z(fit["ki"].to_numpy()), Z(val["ki"].to_numpy())
        yf = fit[f"y_{h}"].to_numpy().astype(np.float64)
        best = None
        for a in (1e2, 1e3, 1e4, 1e5):
            ic = day_ic(val, Xv @ ridge(Xf, yf, a), "fwd_21" if h == 21 else f"y_{h}")
            res["valid"][f"{h}d a={a:g}"] = ic
            if best is None or ic > best[0]:
                best = (ic, a)
        res[f"alpha_{h}"] = best[1]
        beta = ridge(Z(d["ki"].to_numpy()), d[f"y_{h}"].to_numpy().astype(np.float64), best[1])
        preds[h] = Z(tek["ki"].to_numpy()) @ beta
    out = tek.select("date", "ticker", "n_art").with_columns(p5=pl.Series(preds[5]), p21=pl.Series(preds[21]))
    z = lambda c: (pl.col(c) - pl.col(c).mean().over("date")) / (pl.col(c).std().over("date") + 1e-9)
    out = out.with_columns(score=0.5 * z("p5") + 0.5 * z("p21"))
    out.write_parquet(OUT / "news_scores.parquet")
    j = out.join(pw, on=["date", "ticker"])
    for h, col in ((5, "fwd_5"), (21, "fwd_21"), (63, "fwd_63")):
        res[f"test_ic_{h}d"] = day_ic(j, j["score"].to_numpy(), col)
    (OUT / "newsprobe.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=lambda x: round(float(x), 4))[:900])


if __name__ == "__main__":
    main()
