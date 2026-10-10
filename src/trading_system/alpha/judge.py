"""tev1 decision layer — a local 4B decision model as a second opinion next to the algorithms.

Every morning (`ts alpha judge`, daily digest) Together AI's tev1:4b, served by Ollama ≥ 0.35 on this
machine's GPU, answers typed questions about the news, as probabilities:

  switches   the playbook's event switches (configs/tev1_switches.yaml; keys match flag_overrides.yaml)
  monitors   thesis-break conditions over the day's market-wide headlines (keyword pre-filter, then tev1)
  screen     today's picks and current holdings: pending takeover, distress, binary event, guidance cut,
             and "ok / caution / avoid to open a position now"

Informational only: nothing here changes a book, a target or a config. Every answer is logged to
data/ledger/tev1_judgments.parquet so it can be scored against what happened — live, where the model
cannot know the outcome. (Replayed on 2024-26 news the pick screen looked useful, but hiding the
company's name halved the effect: a 2025-26 model partly remembers these companies. Research:
docs/RESEARCH_2026-10.md §11, ops/research/tev1_decisions_2026_10.py.)
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
import yaml

MODEL = "tev1:4b"
CANDIDATES = ("http://127.0.0.1:11434", "http://127.0.0.1:11435")   # system Ollama first, then the user service
FLAGS = ("takeover_pending", "distress", "binary_event", "guidance_cut")
SCREEN = {
    "takeover_pending": {"type": "noul", "instructions": "Has the company agreed to be acquired, or is a takeover offer for it pending?"},
    "distress": {"type": "noul", "instructions": "Do the headlines report fraud allegations, an accounting problem, a regulatory investigation, a going-concern warning or bankruptcy risk for the company?"},
    "binary_event": {"type": "noul", "instructions": "Is a major binary event for the company due within weeks (drug approval decision, trial readout, court ruling, contract award)?"},
    "guidance_cut": {"type": "noul", "instructions": "Did the company recently cut its guidance or warn on results?"},
    "entry": {"type": "choice", "instructions": "Based only on these headlines, is now a reasonable time to open a new long position in the company?",
              "criteria": {"ok": "nothing in the news argues against it", "caution": "a pending event or risk to be aware of",
                           "avoid": "a clear reason not to buy now"}},
}
FLAG_LABEL = {"takeover_pending": "takeover pending", "distress": "distress", "binary_event": "binary event", "guidance_cut": "guidance cut"}
REPLAY_NOTE = ("Replay Nov 2024 → Sep 2026 (company names hidden): picks judged 'avoid' returned +1.2% vs +3.1% for 'ok' "
               "over the next month — suggestive, not significant. Being scored live.")


# ── plumbing ──────────────────────────────────────────────────────────────────

def endpoint(model: str = MODEL, candidates: tuple[str, ...] = CANDIDATES) -> str | None:
    """/v1/systemone of the first Ollama that is ≥ 0.35 and has the model (TEV1_URL overrides)."""
    if os.environ.get("TEV1_URL"):
        return os.environ["TEV1_URL"]
    for base in candidates:
        try:
            v = requests.get(f"{base}/api/version", timeout=3).json()["version"]
            if tuple(int(x) for x in v.split(".")[:2]) < (0, 35):
                continue
            tags = requests.get(f"{base}/api/tags", timeout=5).json().get("models", [])
            if any(model in (m.get("name"), m.get("model")) for m in tags):
                return f"{base}/v1/systemone"
        except Exception:  # noqa: BLE001 — a down or old server just isn't a candidate
            continue
    return None


def ask(url: str, state: dict, questions: dict, timeout: float = 180.0) -> dict:
    """answers{} or {} on any failure (the caller reports 'no answer'). Halves long text once on a 400."""
    for _ in range(2):
        try:
            r = requests.post(url, json={"model": MODEL, "state": state, "questions": questions, "keep_alive": "10m"}, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("answers", {}) or {}
            if r.status_code == 400:
                state = {k: (v[: len(v) // 2] if isinstance(v, str) else v) for k, v in state.items()}
                continue
            return {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def load_news(cfg, since: datetime, tickers: list[str] | None = None) -> pl.DataFrame:
    p = cfg.path("data_bronze") / "massive" / "news.parquet"
    if not p.exists():
        return pl.DataFrame(schema={"article_id": pl.Utf8, "ticker": pl.Utf8, "published_utc": pl.Datetime("us", "UTC"),
                                    "title": pl.Utf8, "description": pl.Utf8})
    lf = pl.scan_parquet(p).filter(pl.col("published_utc") >= since)
    if tickers:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
    return lf.select("article_id", "ticker", "published_utc", "title", "description").collect()


def pack(df: pl.DataFrame, max_chars: int = 3600, desc: int = 140) -> str:
    """Newest first, de-duplicated by title: '• title — start of summary' lines up to the budget."""
    lines, seen, n = [], set(), 0
    for t, d in df.sort("published_utc", descending=True).select("title", "description").iter_rows():
        t = " ".join((t or "").split())
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        line = f"• {t}" + (f" — {' '.join((d or '').split())[:desc]}" if d and desc else "")
        if n + len(line) > max_chars:
            break
        lines.append(line)
        n += len(line) + 1
    return "\n".join(lines)


def keyword_filter(df: pl.DataFrame, pattern: str) -> pl.DataFrame:
    return df.filter((pl.col("title").fill_null("") + " " + pl.col("description").fill_null("")).str.contains("(?i)" + pattern))


def load_conf(cfg) -> dict:
    p = Path(cfg.project_root) / "configs" / "tev1_switches.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {"switches": {}, "monitors": {}}


# ── the three decision families ───────────────────────────────────────────────

def run_switches(cfg, url: str, conf: dict, now: datetime, days: int = 2) -> list[dict]:
    out = []
    for key, sw in (conf.get("switches") or {}).items():
        src = sw.get("source", "macro")
        if src == "macro":
            df = keyword_filter(load_news(cfg, now - timedelta(days=days)).unique(subset=["article_id"]), sw.get("keywords", "."))
        else:
            df = load_news(cfg, now - timedelta(days=days), [src["ticker"]])
        row = {"kind": "switch", "subject": key, "question": sw["question"], "playbook": sw.get("playbook"), "articles": df.height}
        if df.height:
            text = pack(df)
            a = ask(url, {"headlines": text}, {"q": {"type": "noul", "instructions": sw["question"]}})
            row.update(p=(a.get("q") or {}).get("noul"), evidence=text.split("\n")[:2])
        out.append(row)
    return out


def run_monitors(cfg, url: str, conf: dict, now: datetime, days: int = 1) -> list[dict]:
    news = load_news(cfg, now - timedelta(days=days)).unique(subset=["article_id"])
    out = []
    for key, m in (conf.get("monitors") or {}).items():
        df = keyword_filter(news, m.get("keywords", "."))
        row = {"kind": "monitor", "subject": key, "question": m["question"], "playbook": m.get("playbook"), "articles": df.height}
        if df.height:
            text = pack(df, desc=0)
            a = ask(url, {"headlines": text}, {"q": {"type": "noul", "instructions": m["question"]}})
            row.update(p=(a.get("q") or {}).get("noul"), evidence=text.split("\n")[:2])
        out.append(row)
    return out


def run_screen(cfg, url: str, tickers: list[str], now: datetime, days: int = 30, roles: dict | None = None) -> list[dict]:
    news = load_news(cfg, now - timedelta(days=days), tickers)
    out = []
    for t in tickers:
        sub = news.filter(pl.col("ticker") == t)
        row = {"kind": "screen", "subject": t, "role": (roles or {}).get(t, ""), "articles": sub.height}
        if sub.height:
            text = pack(sub, 3600, 140)
            a = ask(url, {"company": t, "headlines": text}, SCREEN)
            if a:
                ent = a.get("entry") or {}
                row.update({k: (a.get(k) or {}).get("noul") for k in FLAGS},
                           verdict=ent.get("choice"), p_avoid=(ent.get("probabilities") or {}).get("avoid"),
                           evidence=text.split("\n")[:1])
        out.append(row)
    return out


# ── ledger + report ───────────────────────────────────────────────────────────

def ledger_path(cfg) -> Path:
    return cfg.path("data_bronze").parent / "ledger" / "tev1_judgments.parquet"


def log_ledger(cfg, rows: list[dict], asof: date) -> int:
    """Upsert on (asof, kind, subject): re-running the same morning replaces that morning's answers."""
    if not rows:
        return 0
    new = pl.DataFrame([{"asof": asof, "kind": r["kind"], "subject": r["subject"], "model": MODEL,
                         "p": r.get("p") if r["kind"] != "screen" else r.get("p_avoid"),
                         "verdict": r.get("verdict"), "details": json.dumps(r, default=str),
                         "recorded_at": datetime.now()} for r in rows],
                       schema={"asof": pl.Date, "kind": pl.Utf8, "subject": pl.Utf8, "model": pl.Utf8, "p": pl.Float64,
                               "verdict": pl.Utf8, "details": pl.Utf8, "recorded_at": pl.Datetime("us")})
    p = ledger_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pl.read_parquet(p)
        old = old.join(new.select("asof", "kind", "subject"), on=["asof", "kind", "subject"], how="anti")
        new = pl.concat([old, new], how="vertical_relaxed")
    new.sort(["asof", "kind", "subject"]).write_parquet(p)
    return len(rows)


def _p(x) -> str:
    return "  —  " if x is None else f"{x:.2f}"


def render(asof: date, url: str, switches: list[dict], monitors: list[dict], screen: list[dict], conf: dict) -> str:
    hi, lo = float(conf.get("alert_p", 0.8)), float(conf.get("watch_p", 0.5))
    mark = lambda p: "🔴 FIRING" if p is not None and p >= hi else ("🟡 watch " if p is not None and p >= lo else "·       ")  # noqa: E731
    L = [f"tev1 judgments · {asof} · {MODEL} via {url.split('/v1')[0]} · informational (scored live, not used by the book)", "",
         "Playbook switches (proposals — set them in configs/flag_overrides.yaml):"]
    for r in switches:
        ev = f"  “{r['evidence'][0][2:110]}”" if r.get("evidence") and (r.get("p") or 0) >= lo else ""
        L.append(f"  {mark(r.get('p'))} {r['subject']:<20} p={_p(r.get('p'))}  ({r['articles']} articles){ev}"
                 + (f"  → playbook: {r['playbook']}" if (r.get("p") or 0) >= hi and r.get("playbook") else ""))
    L += ["", "Thesis-break monitors (today's headlines):"]
    for r in monitors:
        ev = f"  “{r['evidence'][0][2:110]}”" if r.get("evidence") and (r.get("p") or 0) >= lo else ""
        L.append(f"  {mark(r.get('p'))} {r['subject']:<20} p={_p(r.get('p'))}  ({r['articles']} headlines){ev}")
    judged = [r for r in screen if r.get("verdict")]
    flagged = [r for r in judged if r["verdict"] in ("avoid", "caution") or any((r.get(k) or 0) >= hi for k in FLAGS)]
    L += ["", f"Pre-buy / holding screen ({len(judged)} of {len(screen)} names had news in 30 days):"]
    for r in sorted(flagged, key=lambda r: (r["verdict"] != "avoid", -(r.get("p_avoid") or 0))):
        why = ", ".join(f"{FLAG_LABEL[k]} {r[k]:.2f}" for k in FLAGS if (r.get(k) or 0) >= lo)
        ev = f"  “{r['evidence'][0][2:100]}”" if r.get("evidence") else ""
        label = r["verdict"] if r["verdict"] in ("avoid", "caution") else "flag"
        L.append(f"  ⚠ {label:<7} {r['subject']:<6} {('[' + r['role'] + ']') if r.get('role') else '':<16} {why or '—'}{ev}")
    ok = [r["subject"] for r in judged if r not in flagged]
    if ok:
        L.append(f"  ok ({len(ok)}): {', '.join(ok)}")
    L += ["", REPLAY_NOTE]
    return "\n".join(L)
