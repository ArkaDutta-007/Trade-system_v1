#!/usr/bin/env python3
"""News-type experiment, October 2026: does *fundamental* news drift while *story* news reverses?

Replicates the design of Kargarzadeh et al., "Buy the Rumor, Sell the News: When Is News Priced In?"
(arXiv 2608.14014) on our own news store, then asks whether the split adds anything to the alpha model.

  classify   every Massive article since 2024-07 (when Massive's point-in-time sentiment starts) gets an
             event tag + three attributes from qwen3.7-flash (cheapest capable model, thinking off).
             Direction is NOT asked of the LLM: it comes from Massive's sentiment, which was produced at
             publication time — so the only LLM output is the article's *type*, a judgement about the
             text that does not need (and is hardly helped by) knowing what happened next.
  analyze    event study (beta-adjusted abnormal returns in the news direction, by type) and signal
             tests (rank IC of type-split news sentiment vs forward returns, incremental to the model's
             composite score) on the point-in-time universe. Results → reports/alpha/lab_2026_10/newstype.json

Pre-registered groups (from the paper, fixed before looking at any result):
  HARD  EARN GUID CAP ANLY          quantified fundamental news — the paper finds it DRIFTS
  SOFT  PROD MACR MGMT              story-driven news — the paper finds it REVERSES
  CORP  MNA DEAL REGL FINC          other corporate events (reported, not pre-judged)
  NOISE MOVE OPIN PR                recaps, opinion, routine releases

Usage:  python3 ops/research/newstype_2026_10.py classify [--pilot N] [--budget USD] [--workers K]
        python3 ops/research/newstype_2026_10.py analyze
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from trading_system.config import get_config  # noqa: E402

cfg = get_config(str(REPO / "configs/default.yaml"))
NEWS = cfg.path("data_bronze") / "massive" / "news.parquet"
WORK = cfg.path("data_silver") / "newstype"
LABELS_JSONL = WORK / "labels.jsonl"
LABELS = WORK / "labels.parquet"
OUT = cfg.path("reports") / "alpha" / "lab_2026_10"
START = datetime(2024, 7, 1)

MODEL = "qwen3.7-flash"
PRICE_IN, PRICE_OUT = 0.03, 0.13          # USD per 1M tokens (DashScope intl, 0-32K tier)
BATCH = 25

TAGS = ["EARN", "GUID", "CAP", "ANLY", "MNA", "DEAL", "REGL", "PROD", "MGMT", "FINC", "MACR", "MOVE", "OPIN", "PR"]
GROUPS = {"HARD": ["EARN", "GUID", "CAP", "ANLY"], "SOFT": ["PROD", "MACR", "MGMT"],
          "CORP": ["MNA", "DEAL", "REGL", "FINC"], "NOISE": ["MOVE", "OPIN", "PR"]}
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}

SYSTEM = """You label financial news articles for a quantitative study. For each numbered article output exactly one line:
<number> <TAG> <Q><N><R>
and nothing else.
TAG = the single best of:
EARN earnings results or pre-announcement
GUID guidance or outlook change/reaffirmation
CAP dividend, buyback or stock split
ANLY analyst rating or price-target change
MNA merger, acquisition, takeover, divestiture, spin-off
DEAL contract, order, partnership, licensing, customer win
REGL regulatory decision or approval, lawsuit, investigation, government action
PROD product, service or technology launch; drug or pipeline data
MGMT executive or board change
FINC share or debt offering, financing, insider or fund buying/selling, index change
MACR macro, sector or market-wide commentary (the company is incidental)
MOVE price-move recap, "why X is up/down", technical analysis, options activity
OPIN opinion, recommendation, stock-picking list, long-term thesis, comparison
PR other press release (event or webcast notice, award, ESG, routine filing)
Tag what the article mainly IS. A piece arguing whether to buy or sell, or retelling a stock's long-term story, is OPIN even if it mentions an event. A "why the stock moved" piece takes the tag of the company event that caused the move (e.g. EARN, ANLY), or MOVE if no specific company event is given. Industry or market research reports are MACR.
Q=1 if the article states specific new numbers about the company's business (revenue, EPS, guidance, deal value, dividend, price target), else 0.
N=1 if it is the first report of new company-specific information; 0 if recap, preview, opinion or follow-up.
R=1 if it rests on rumor, unnamed sources or speculation about a possible event, else 0.
Example output:
1 EARN 110
2 OPIN 000"""

LINE = re.compile(r"^\s*(\d+)\s*[.):]?\s+([A-Z]{2,4})\s+([01])\s*([01])\s*([01])")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── classification ───────────────────────────────────────────────────────────

def articles() -> pl.DataFrame:
    a = (pl.scan_parquet(NEWS).filter(pl.col("published_utc") >= pl.lit(START).dt.replace_time_zone("UTC"))
           .select("article_id", "published_utc", "title", "description")
           .unique(subset=["article_id"]).collect().sort("published_utc"))
    return a.filter(pl.col("title").str.len_chars() > 0)


def _text(title: str, desc: str | None) -> str:
    t = " ".join((title or "").split())[:200]
    d = " ".join((desc or "").split())[:240]
    return f"{t} | {d}" if d else t


class Meter:
    def __init__(self, budget: float):
        self.budget, self.tin, self.tout, self.calls, self.fail = budget, 0, 0, 0, 0
        self.lock = threading.Lock()

    @property
    def usd(self) -> float:
        return self.tin / 1e6 * PRICE_IN + self.tout / 1e6 * PRICE_OUT

    def add(self, tin: int, tout: int) -> None:
        with self.lock:
            self.tin += tin; self.tout += tout; self.calls += 1


def _call(session, url: str, key: str, batch: list[tuple[str, str]], meter: Meter) -> dict[str, tuple]:
    from trading_system.ingestion.llm_config import llm_extra_params
    body = "\n".join(f"{i + 1}. {txt}" for i, (_, txt) in enumerate(batch))
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 12 * len(batch) + 40,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
               **llm_extra_params()}
    for attempt in range(6):
        try:
            r = session.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** attempt * 3)); continue
            r.raise_for_status()
            j = r.json()
            u = j.get("usage") or {}
            meter.add(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
            out = {}
            for line in (j["choices"][0]["message"]["content"] or "").splitlines():
                m = LINE.match(line)
                if m and 1 <= int(m.group(1)) <= len(batch) and m.group(2) in TAGS:
                    out[batch[int(m.group(1)) - 1][0]] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))
            return out
        except Exception as e:  # noqa: BLE001 — network/JSON hiccups: retry, never crash the run
            if attempt == 5:
                log(f"batch failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(min(60, 2 ** attempt * 3))
    with meter.lock:
        meter.fail += 1
    return {}


def classify(pilot: int = 0, budget: float = 8.0, workers: int = 8) -> None:
    import requests
    from dotenv import load_dotenv
    from trading_system.ingestion.llm_config import llm_api_key, llm_base_url
    load_dotenv(REPO / ".env")
    key, url = llm_api_key(), llm_base_url()
    if not key or "dashscope" not in url:
        raise SystemExit("needs the DashScope (Qwen) endpoint configured in .env")
    if not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    WORK.mkdir(parents=True, exist_ok=True)
    done = set()
    if LABELS_JSONL.exists():
        with open(LABELS_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001 — a torn last line from a killed run
                    pass
    a = articles()
    todo = a.filter(~pl.col("article_id").is_in(list(done)))
    if pilot:
        todo = todo.sample(n=min(pilot, todo.height), seed=7)
    log(f"{a.height:,} articles since {START.date()}, {len(done):,} already labelled, {todo.height:,} to do")
    items = [(r[0], _text(r[1], r[2])) for r in todo.select("article_id", "title", "description").iter_rows()]
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    meter, lock = Meter(budget), threading.Lock()
    session = requests.Session()
    t0, n_done = time.time(), 0
    with open(LABELS_JSONL, "a") as fout, ThreadPoolExecutor(workers) as ex:
        futs = {}
        it = iter(batches)
        def submit():
            b = next(it, None)
            if b is not None and meter.usd < budget:
                futs[ex.submit(_call, session, url, key, b, meter)] = b
        for _ in range(workers * 2):
            submit()
        while futs:
            fut = next(as_completed(list(futs)))
            b = futs.pop(fut)
            res = fut.result()
            with lock:
                for aid, (tag, q, n, r) in res.items():
                    fout.write(json.dumps({"id": aid, "tag": tag, "q": q, "n": n, "r": r}) + "\n")
                fout.flush()
            n_done += len(b)
            if meter.calls % 200 == 0:
                rate = n_done / max(time.time() - t0, 1e-9)
                log(f"{n_done:,}/{len(items):,} articles · {meter.calls:,} calls · ${meter.usd:.3f} · "
                    f"{rate:.0f}/s · eta {(len(items) - n_done) / max(rate, 1e-9) / 60:.0f} min · failed batches {meter.fail}")
            submit()
    if meter.usd >= budget:
        log(f"STOPPED at the ${budget:.2f} budget")
    log(f"done: {meter.calls:,} calls, {meter.tin:,} in / {meter.tout:,} out tokens, ${meter.usd:.3f}, failed batches {meter.fail}")
    compact()


def compact() -> pl.DataFrame:
    rows = [json.loads(x) for x in open(LABELS_JSONL) if x.strip()]
    df = pl.DataFrame(rows).unique(subset=["id"], keep="last").rename({"id": "article_id"})
    df.write_parquet(LABELS)
    log(f"labels: {df.height:,} articles → {LABELS}")
    print(df.group_by("tag").len().sort("len", descending=True))
    return df


# ── analysis ─────────────────────────────────────────────────────────────────

def trading_days() -> pl.DataFrame:
    p = cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet"
    d = pl.scan_parquet(p).filter(pl.col("ticker") == "SPY").select(pl.col("date").cast(pl.Date)).collect()
    return d.unique().sort("date")


def events(lab: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """One row per (ticker, event day, group): net news direction from Massive's point-in-time sentiment.
    Event day = the first session whose 16:00 ET close comes after publication."""
    n = (pl.scan_parquet(NEWS).filter(pl.col("published_utc") >= pl.lit(START).dt.replace_time_zone("UTC"),
                                      pl.col("sentiment").is_not_null())
           .select("article_id", "ticker", "published_utc", "sentiment").collect())
    n = n.join(lab.select("article_id", "tag", "q", "n", "r"), on="article_id", how="inner")
    et = pl.col("published_utc").dt.convert_time_zone("America/New_York")
    n = n.with_columns(cand=pl.when(et.dt.hour() >= 16).then(et.dt.date() + pl.duration(days=1)).otherwise(et.dt.date()))
    n = n.sort("cand").join_asof(days.rename({"date": "day0"}), left_on="cand", right_on="day0", strategy="forward")
    n = n.drop_nulls("day0").with_columns(group=pl.col("tag").replace_strict(GROUP_OF, default="NOISE"))
    ev = (n.group_by(["ticker", "day0", "group"])
           .agg(net=pl.col("sentiment").sum(), n_art=pl.len(), q=pl.col("q").max(), new=pl.col("n").max(),
                rumor=pl.col("r").max(), tags=pl.col("tag").unique())
           .with_columns(dir=pl.col("net").sign()))
    return ev


def _nw_t(x: np.ndarray, lag: int) -> float:
    """Newey–West t-stat of the mean of a (date-ordered) series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


WINDOWS = {"AR0": (0, 0), "CAR1_5": (1, 5), "CAR1_20": (1, 20), "CAR2_21": (2, 21), "CAR0_20": (0, 20)}


def event_study(ev: pl.DataFrame, members: pl.DataFrame) -> dict:
    """Market-adjusted abnormal returns (vs SPY) signed by the news direction, by group."""
    p = cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet"
    tick = ev["ticker"].unique().to_list() + ["SPY"]
    px = (pl.scan_parquet(p).filter(pl.col("ticker").is_in(tick), pl.col("adj_close") > 0)
            .select(pl.col("date").cast(pl.Date), "ticker", "adj_close").collect().sort(["ticker", "date"])
            .with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1))
    W = px.pivot(values="r", index="date", on="ticker").sort("date")
    dates = W["date"].to_list()
    di = {d: i for i, d in enumerate(dates)}
    spy = W["SPY"].to_numpy()
    cols = {c: j for j, c in enumerate(W.columns)}
    R = W.to_numpy()
    e = ev.filter(pl.col("dir") != 0).join(members, left_on=["day0", "ticker"], right_on=["date", "ticker"], how="semi")
    rows = []
    for t, d0, g, dr, q, new in e.select("ticker", "day0", "group", "dir", "q", "new").iter_rows():
        i, j = di.get(d0), cols.get(t)
        if i is None or j is None or i + 21 >= len(dates):
            continue
        ar = (R[i:i + 22, j].astype(float) - spy[i:i + 22])
        if not np.isfinite(ar[0]) or np.isnan(ar).mean() > 0.2:
            continue
        ar = np.nan_to_num(ar) * dr
        rows.append({"day0": d0, "group": g, "dir": int(dr), "q": int(q), "new": int(new),
                     **{k: float(ar[a:b + 1].sum()) for k, (a, b) in WINDOWS.items()}})
    es = pl.DataFrame(rows)
    out = {"n_events": es.height}
    def summarise(f: pl.DataFrame) -> dict:
        r = {"n": f.height}
        byd = f.group_by("day0").agg([pl.col(k).mean() for k in WINDOWS]).sort("day0")
        for k, (a, b) in WINDOWS.items():
            r[k] = float(f[k].mean())
            r[k + "_t"] = _nw_t(byd[k].to_numpy(), max(b, 1))
        r["persistence"] = r["CAR0_20"] / r["AR0"] if r["AR0"] else None
        return r
    for g in ["ALL", *GROUPS]:
        f = es if g == "ALL" else es.filter(pl.col("group") == g)
        out[g] = summarise(f)
        for side, sd in (("pos", 1), ("neg", -1)):
            out[g][side] = summarise(f.filter(pl.col("dir") == sd))
    out["HARD_q1_new1"] = summarise(es.filter((pl.col("group") == "HARD") & (pl.col("q") == 1) & (pl.col("new") == 1)))
    return out


def signal_features(ev: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Per (date, ticker): net news direction by group over the trailing 30 calendar days (≈21 sessions),
    from events whose day is ≤ the panel date (published before that session's close)."""
    out = panel.select("date", "ticker")
    for g in GROUPS:
        e = (ev.filter(pl.col("group") == g).group_by(["ticker", "day0"]).agg(d=pl.col("dir").sum())
               .sort(["ticker", "day0"]).with_columns(cum=pl.col("d").cum_sum().over("ticker")))
        cur = out.sort("date").join_asof(e.select("ticker", "day0", "cum").sort("day0"), left_on="date", right_on="day0",
                                         by="ticker", strategy="backward").select("date", "ticker", c1="cum")
        lag = (out.with_columns(lagd=pl.col("date") - pl.duration(days=30)).sort("lagd")
                  .join_asof(e.select("ticker", "day0", "cum").sort("day0"), left_on="lagd", right_on="day0", by="ticker",
                             strategy="backward").select("date", "ticker", c0="cum"))
        out = (out.join(cur, on=["date", "ticker"], how="left").join(lag, on=["date", "ticker"], how="left")
                  .with_columns(((pl.col("c1").fill_null(0) - pl.col("c0").fill_null(0)).cast(pl.Float64)).alias(f"nt_{g}"))
                  .drop("c1", "c0"))
    return out.with_columns(nt_HARD_minus_SOFT=pl.col("nt_HARD") - pl.col("nt_SOFT"))


def ic_tests(df: pl.DataFrame, feats: list[str], controls: list[str]) -> dict:
    """Per-date Spearman IC vs fwd_5 / fwd_21, raw and after residualising the feature's rank on the
    controls' ranks (the model composite and the existing news sentiment feature)."""
    res = {}
    for f in feats:
        d = df.drop_nulls([f, "fwd_21", "fwd_5", *controls])
        rk = d.with_columns([pl.col(c).rank("average").over("date").alias(f"r_{c}") for c in [f, *controls]])
        parts = []
        for (dt,), g in rk.group_by(["date"]):
            if g.height < 100:
                continue
            X = np.column_stack([np.ones(g.height)] + [g[f"r_{c}"].to_numpy() for c in controls])
            y = g[f"r_{f}"].to_numpy().astype(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            row = {"date": dt, "cover": float((g[f] != 0).mean())}
            for h in (5, 21):
                fr = g[f"fwd_{h}"].to_numpy()
                rr = pl.Series(fr).rank("average").to_numpy()
                row[f"ic{h}"] = float(np.corrcoef(y, rr)[0, 1]) if np.std(y) > 0 else np.nan
                row[f"pic{h}"] = float(np.corrcoef(resid, rr)[0, 1]) if np.std(resid) > 0 else np.nan
            parts.append(row)
        t = pl.DataFrame(parts).sort("date")
        r = {"dates": t.height, "coverage": float(t["cover"].mean())}
        for h in (5, 21):
            for k in (f"ic{h}", f"pic{h}"):
                x = t[k].to_numpy()
                xs = x[::h]
                xs = xs[np.isfinite(xs)]
                r[k] = float(np.nanmean(x))
                r[k + "_t"] = float(xs.mean() / (xs.std(ddof=1) + 1e-12) * np.sqrt(len(xs))) if len(xs) > 3 else None
        res[f] = r
    return res


def analyze() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import lab_2026_10 as LAB
    lab = pl.read_parquet(LABELS) if LABELS.exists() else compact()
    days = trading_days()
    ev = events(lab, days)
    log(f"events: {ev.height:,} (ticker, day, group) from {lab.height:,} labelled articles")
    pw = pl.read_parquet(LAB.OUT / "pit_panel.parquet", columns=["date", "ticker", "fwd_5", "fwd_21", "news_sent_21"])
    members = pw.select("date", "ticker")
    out = {"labels": lab.group_by("tag").len().sort("len", descending=True).to_dicts(),
           "groups": GROUPS, "window": [str(START.date()), str(days["date"].max())]}
    out["event_study"] = event_study(ev, members)
    log("event study done")
    feats = signal_features(ev, pw)
    comp = LAB.zcomp(pl.read_parquet(LAB.OUT / "pit_base.parquet"))
    df = pw.join(feats, on=["date", "ticker"], how="left").join(comp, on=["date", "ticker"], how="left")
    fcols = [f"nt_{g}" for g in GROUPS] + ["nt_HARD_minus_SOFT"]
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in fcols]).with_columns(pl.col("news_sent_21").fill_null(0.0))
    out["ic"] = ic_tests(df, fcols, ["comp", "news_sent_21"])
    out["ic_existing_news_sent_21"] = ic_tests(df, ["news_sent_21"], ["comp"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "newstype.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"→ {OUT / 'newstype.json'}")
    es = out["event_study"]
    for g in ["ALL", *GROUPS, "HARD_q1_new1"]:
        r = es[g]
        log(f"{g:<13} n={r['n']:>6,}  AR0 {r['AR0']:+.2%} (t {r['AR0_t']:+.1f})  CAR1-5 {r['CAR1_5']:+.2%} (t {r['CAR1_5_t']:+.1f})  "
            f"CAR1-20 {r['CAR1_20']:+.2%} (t {r['CAR1_20_t']:+.1f})  CAR2-21 {r['CAR2_21']:+.2%} (t {r['CAR2_21_t']:+.1f})")
    for f, r in out["ic"].items():
        log(f"{f:<20} cover {r['coverage']:.1%}  IC5 {r['ic5']:+.4f} (t {r['ic5_t']:+.1f})  IC21 {r['ic21']:+.4f} (t {r['ic21_t']:+.1f})  "
            f"partial IC21 {r['pic21']:+.4f} (t {r['pic21_t']:+.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["classify", "analyze", "compact"])
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.phase == "classify":
        classify(a.pilot, a.budget, a.workers)
    elif a.phase == "compact":
        compact()
    else:
        analyze()


if __name__ == "__main__":
    main()
