#!/usr/bin/env python3
"""Does a language model's reading of earnings guidance predict the next months? (2026-11-13, RESEARCH §13)

In October the one judgment of the local model that survived hiding the company's name was "guidance cut".
That rested on a few dozen cases. This runs it at scale on the names the index-minus book can hold.
Fixed before the first run:

  names      every stock that was among the 300 most liquid on any month-end of the point-in-time panel
  documents  each 8-K with item 2.02 (results of operations) filed 2024-11-01 → today: exhibit 99.1 from SEC EDGAR
  reading    qwen3.7-flash, temperature 0, on the release's opening + first outlook passage with the ticker and
             company name replaced by "the company" (so the model cannot recall what happened next). One JSON:
             guidance = raised | lowered | reaffirmed | provided (no change stated) | withdrawn | none,
             results = beat | inline | miss | unclear,  tone = −2 … 2
  event      entry at the close of the first session AFTER the filing date; return over the next 21 and 63
             sessions minus that day's point-in-time universe mean
  use        base+guidance: the challenger score −1 for a lowered or withdrawn guidance in the last 63 sessions,
             +0.5 for a raised one; judged like every other variant (train_2026_11.verdicts)
Stages: fetch → read → analyze. Cost is logged; the key comes from the repo's LLM config and is never printed.
Results → reports/alpha/lab_2026_11/guidance.json
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import train_2026_11 as T  # noqa: E402

LAB, OUT = T.LAB, T.OUT
from trading_system.alpha import challenger as C  # noqa: E402

WORK = LAB.cfg.path("data_silver") / "guidance"
REL = WORK / "releases"
START = date(2024, 11, 1)
MODEL = "qwen3.7-flash"
PRICE_IN, PRICE_OUT = 0.03, 0.13            # USD per 1M tokens (DashScope intl, 0-32K tier)
SYSTEM = """You read an excerpt from a company's quarterly earnings press release. The company's name is hidden.
Reply with ONE line of JSON and nothing else:
{"guidance": "...", "results": "...", "tone": N}
guidance — the company's own forward outlook (revenue, earnings, margins) compared with its previous outlook:
  "raised" (increased / above prior range), "lowered" (reduced / below prior range), "reaffirmed" (explicitly
  unchanged), "provided" (an outlook is given but the text does not say how it changed), "withdrawn"
  (suspended or no longer provided), "none" (no outlook in the excerpt).
results — the quarter against expectations or the company's own guidance, only if the text says so:
  "beat", "inline", "miss", "unclear".
tone — integer from -2 (very negative) to 2 (very positive) for management's overall tone."""
CATS = ("raised", "lowered", "reaffirmed", "provided", "withdrawn", "none")
_STOP = {"inc", "corp", "corporation", "company", "holdings", "group", "ltd", "plc", "the", "and", "class", "common", "stock", "technologies", "co"}


# the three text helpers below are the ones tested in tev1_decisions_2026_10.py (copied: importing that module
# probes the local model server)
def strip_html(h: str) -> str:
    import html as _html
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    return " ".join(_html.unescape(h).split())


def excerpt(text: str, head: int = 2600, tail: int = 1400) -> str:
    """The release's opening (headline numbers) + the first outlook/guidance passage."""
    out = text[:head]
    m = re.search(r"(?i)(outlook|guidance|expects|forecast)", text[head:])
    if m:
        i = head + m.start()
        out += " … " + text[max(head, i - 200): i + tail]
    return out


def mask(text: str, ticker: str, name: str | None) -> str:
    """Hide the company's identity (look-ahead check, Glasserman & Lin): ticker and name words → 'the company'."""
    words = [ticker] + [w for w in re.split(r"[\s,.]+", name or "") if len(w) > 2 and w.lower() not in _STOP][:2]
    for w in words:
        text = re.sub(rf"(?i)\b{re.escape(w)}('s)?\b", "the company", text)
    return text


def universe(pw: pl.DataFrame) -> list[str]:
    cal = sorted(pw["date"].unique().to_list())
    me = [d for d, nx in zip(cal[:-1], cal[1:]) if d.month != nx.month]
    u = (pw.filter(pl.col("date").is_in(me)).with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
           .filter(pl.col("lr") <= 300))
    return sorted(u["ticker"].unique().to_list())


def fetch(workers: int = 6) -> None:
    from trading_system.ingestion.edgar_fundamentals import store
    REL.mkdir(parents=True, exist_ok=True)
    st = store(LAB.cfg)
    ciks = st.cik_map()
    names = universe(C.load_panel(LAB.cfg))
    idx_p = WORK / "filings.jsonl"
    done = {json.loads(l)["ticker"] for l in idx_p.open()} if idx_p.exists() else set()

    def filings(t: str) -> list[dict]:
        cik = ciks.get(t) or ciks.get(t.replace(".", "-"))
        if not cik:
            return [{"ticker": t, "error": "no CIK"}]
        sub = st._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]
        recs = [sub["recent"]]
        for f in sub.get("files", []):                      # heavy filers (banks) spill older filings into extra files
            if f.get("filingTo", "") >= str(START):
                recs.append(st._get(f"https://data.sec.gov/submissions/{f['name']}").json())
        out = []
        for rec in recs:
            for form, fdate, acc, items in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["items"]):
                if form.startswith("8-K") and "2.02" in (items or "") and fdate >= str(START):
                    out.append({"ticker": t, "cik": cik, "fdate": fdate, "acc": acc})
        return out or [{"ticker": t, "error": "no 8-K 2.02"}]

    todo = [t for t in names if t not in done]
    LAB.log(f"{len(names)} names · {len(todo)} to index")
    with idx_p.open("a") as fo, ThreadPoolExecutor(workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(filings, t) for t in todo]), 1):
            try:
                for r in fu.result():
                    fo.write(json.dumps(r) + "\n")
            except Exception as e:  # noqa: BLE001
                LAB.log(f"index failed: {type(e).__name__}: {str(e)[:100]}")
            if i % 100 == 0:
                fo.flush(); LAB.log(f"  indexed {i}/{len(todo)}")
    fl = [json.loads(l) for l in idx_p.open()]
    fl = [r for r in fl if "acc" in r and not (REL / f"{r['ticker']}_{r['fdate']}.txt").exists()]
    LAB.log(f"{len(fl)} releases to download")

    def release(r: dict) -> int:
        txt = st._get(f"https://www.sec.gov/Archives/edgar/data/{r['cik']}/{r['acc'].replace('-', '')}/{r['acc']}.txt").text
        m = re.search(r"(?is)<TYPE>EX-99\.1\b.*?<TEXT>(.*?)</TEXT>", txt) or re.search(r"(?is)<TYPE>EX-99.*?<TEXT>(.*?)</TEXT>", txt)
        body = strip_html(m.group(1)) if m else ""
        (REL / f"{r['ticker']}_{r['fdate']}.txt").write_text(body[:60_000])
        return len(body)

    with ThreadPoolExecutor(workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(release, r) for r in fl]), 1):
            try:
                fu.result()
            except Exception as e:  # noqa: BLE001
                LAB.log(f"download failed: {type(e).__name__}: {str(e)[:100]}")
            if i % 250 == 0:
                LAB.log(f"  downloaded {i}/{len(fl)}")
    LAB.log(f"releases on disk: {len(list(REL.glob('*.txt')))}")


def read(workers: int = 8, limit: int = 0, budget: float = 3.0) -> None:
    import requests
    from trading_system.ingestion.llm_config import llm_api_key, llm_base_url, llm_extra_params
    key, url = llm_api_key(), llm_base_url()
    if not key:
        raise SystemExit("no LLM key configured")
    if not url.rstrip("/").endswith("chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    det = pl.read_parquet(LAB.cfg.path("data_bronze") / "massive" / "details.parquet", columns=["ticker", "name"])
    name_of = dict(det.iter_rows())
    out_p = WORK / "readings.jsonl"
    done = {(j["ticker"], j["fdate"]) for j in map(json.loads, out_p.open())} if out_p.exists() else set()
    files = sorted(p for p in REL.glob("*.txt") if p.stat().st_size > 400)
    todo = [(p.stem.rsplit("_", 1)[0], p.stem.rsplit("_", 1)[1], p) for p in files]
    todo = [x for x in todo if (x[0], x[1]) not in done][:limit or None]
    LAB.log(f"{len(files)} releases · {len(todo)} to read")
    tin = tout = fail = 0
    lock = threading.Lock()
    sess = requests.Session()

    def one(t: str, fd: str, p: Path) -> dict | None:
        nonlocal tin, tout, fail
        text = mask(excerpt(p.read_text()), t, name_of.get(t))
        payload = {"model": MODEL, "temperature": 0, "max_tokens": 60,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], **llm_extra_params()}
        for attempt in range(5):
            try:
                r = sess.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(60, 2 ** attempt * 3)); continue
                r.raise_for_status()
                j = r.json()
                u = j.get("usage") or {}
                m = re.search(r"\{.*\}", j["choices"][0]["message"]["content"] or "", re.S)
                a = json.loads(m.group(0)) if m else {}
                with lock:
                    tin += int(u.get("prompt_tokens", 0)); tout += int(u.get("completion_tokens", 0))
                g = str(a.get("guidance", "")).lower()
                return {"ticker": t, "fdate": fd, "guidance": g if g in CATS else "unparsed", "results": str(a.get("results", "")).lower(),
                        "tone": a.get("tone")}
            except Exception:  # noqa: BLE001 — retry network/JSON hiccups
                time.sleep(min(30, 2 ** attempt * 2))
        with lock:
            fail += 1
        return None

    with out_p.open("a") as fo, ThreadPoolExecutor(workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(one, *x) for x in todo]), 1):
            r = fu.result()
            if r:
                fo.write(json.dumps(r) + "\n")
            usd = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
            if i % 250 == 0:
                fo.flush(); LAB.log(f"  read {i}/{len(todo)} · ${usd:.3f} · {fail} failed")
            if usd > budget:
                LAB.log("budget reached — stopping"); break
    LAB.log(f"done: {tin:,} tokens in, {tout:,} out, ${tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.3f}, {fail} failed")
    (WORK / "cost.json").write_text(json.dumps({"tokens_in": tin, "tokens_out": tout, "usd": tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, "failed": fail}))


def analyze() -> None:
    pw, allpx = T.load()
    cal = sorted(pw["date"].unique().to_list())
    nxt = {}                                                     # filing date → first session strictly after it
    j = 0
    rd = pl.DataFrame([json.loads(l) for l in (WORK / "readings.jsonl").open()]).unique(subset=["ticker", "fdate"])
    rd = rd.with_columns(fdate=pl.col("fdate").str.to_date())
    for d in sorted(rd["fdate"].unique().to_list()):
        while j < len(cal) and cal[j] <= d:
            j += 1
        nxt[d] = cal[j] if j < len(cal) else None
    m = pl.DataFrame({"fdate": list(nxt), "entry": list(nxt.values())}, schema={"fdate": pl.Date, "entry": pl.Date})
    rd = rd.join(m, on="fdate", how="left").drop_nulls("entry")
    lab = pw.select("date", "ticker", "fwd_21", "fwd_63").with_columns(
        rx21=pl.col("fwd_21") - pl.col("fwd_21").mean().over("date"), rx63=pl.col("fwd_63") - pl.col("fwd_63").mean().over("date"))
    ev = rd.join(lab, left_on=["entry", "ticker"], right_on=["date", "ticker"], how="left")
    res = {"releases": rd.height, "names": rd["ticker"].n_unique(), "matched_to_panel": ev.drop_nulls("fwd_21").height,
           "share": {r["guidance"]: r["len"] / rd.height for r in rd.group_by("guidance").len().iter_rows(named=True)}, "event": {}}

    def tstat(df: pl.DataFrame, col: str) -> tuple[float, float, int]:
        m = df.drop_nulls(col).group_by(pl.col("entry").dt.strftime("%Y-%m")).agg(x=pl.col(col).mean())["x"].to_numpy()
        x = df[col].drop_nulls().to_numpy()
        return (float(x.mean()) if len(x) else float("nan"), float(m.mean() / (m.std(ddof=1) + 1e-12) * np.sqrt(len(m))) if len(m) > 3 else float("nan"), len(x))

    for g in CATS + ("lowered or withdrawn",):
        d = ev.filter(pl.col("guidance").is_in(["lowered", "withdrawn"]) if g == "lowered or withdrawn" else pl.col("guidance") == g)
        if d.height == 0:
            continue
        res["event"][g] = {}
        for col in ("rx21", "rx63"):
            m, t, n = tstat(d, col)
            y1, y2 = d.filter(pl.col("entry") <= T.Y1_END)[col].mean(), d.filter(pl.col("entry") > T.Y1_END)[col].mean()
            res["event"][g][col] = {"mean": m, "t_by_month": t, "n": n, "y1": y1, "y2": y2}
    for name, q in (("results miss", pl.col("results") == "miss"), ("results beat", pl.col("results") == "beat"),
                    ("tone ≤ −1", pl.col("tone").cast(pl.Float64, strict=False) <= -1), ("tone ≥ 1", pl.col("tone").cast(pl.Float64, strict=False) >= 1)):
        d = ev.filter(q)
        res["event"][name] = {col: dict(zip(("mean", "t_by_month", "n"), tstat(d, col))) for col in ("rx21", "rx63")}
    # as an input to the book: the latest reading within 63 sessions, known from its entry day on
    val = pl.when(pl.col("guidance").is_in(["lowered", "withdrawn"])).then(-1.0).when(pl.col("guidance") == "raised").then(0.5).otherwise(0.0)
    sig = rd.select("ticker", "entry", g=val).sort("entry")
    cdf = pl.DataFrame({"date": cal}).with_row_index("i")
    keys = pw.select("date", "ticker").join(cdf, on="date").sort("date")
    sig = sig.join(cdf.rename({"date": "entry", "i": "ei"}), on="entry").sort("entry")
    k = keys.join_asof(sig, left_on="date", right_on="entry", by="ticker", strategy="backward")
    k = k.with_columns(g=pl.when((pl.col("i") - pl.col("ei")) <= 63).then(pl.col("g")).otherwise(0.0).fill_null(0.0))
    base = LAB.zcomp(pl.read_parquet(OUT / "wf2_base.parquet"))
    comps = {"base": base, "base+guidance": base.join(k.select("date", "ticker", "g"), on=["date", "ticker"], how="left")
             .select("date", "ticker", comp=T.zc() + pl.col("g").fill_null(0.0))}
    res["verdicts"] = T.verdicts(comps, pw, allpx)
    res["cost"] = json.loads((WORK / "cost.json").read_text()) if (WORK / "cost.json").exists() else None
    (OUT / "guidance.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{res['releases']} releases · {res['names']} names · matched {res['matched_to_panel']} · shares " + " ".join(f"{g} {v:.0%}" for g, v in sorted(res["share"].items(), key=lambda kv: -kv[1])))
    for g, v in res["event"].items():
        a, b = v["rx21"], v["rx63"]
        print(f"  {g:<22} n {a['n']:>4} · 21d {a['mean']:+.2%} (t {a['t_by_month']:+.1f}) · 63d {b['mean']:+.2%} (t {b['t_by_month']:+.1f})"
              + (f" · 63d y1 {b['y1']:+.2%} y2 {b['y2']:+.2%}" if b.get("y1") is not None and b.get("y2") is not None else ""))
    T.show(res["verdicts"])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if stage == "fetch":
        fetch()
    elif stage == "read":
        read(limit=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        analyze()
