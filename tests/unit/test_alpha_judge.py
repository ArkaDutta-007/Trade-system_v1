"""tev1 decision layer: endpoint choice, the three decision families on synthetic news, ledger upsert."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from trading_system.alpha import judge as J


class _Cfg:
    def __init__(self, root: Path):
        self.project_root = root

    def path(self, name: str) -> Path:
        return self.project_root / "data" / name.replace("data_", "")


class _Resp:
    def __init__(self, js, status=200):
        self._js, self.status_code = js, status

    def json(self):
        return self._js


def test_endpoint_prefers_a_server_new_enough_and_holding_the_model(monkeypatch):
    def fake_get(url, timeout):
        if "11434/api/version" in url:
            return _Resp({"version": "0.30.11"})                      # system Ollama: too old for /v1/systemone
        if "11435/api/version" in url:
            return _Resp({"version": "0.40.2"})
        if "11435/api/tags" in url:
            return _Resp({"models": [{"name": "tev1:4b"}]})
        raise ConnectionError(url)
    monkeypatch.delenv("TEV1_URL", raising=False)
    monkeypatch.setattr(J.requests, "get", fake_get)
    assert J.endpoint() == "http://127.0.0.1:11435/v1/systemone"
    monkeypatch.setattr(J.requests, "get", lambda url, timeout: (_ for _ in ()).throw(ConnectionError(url)))
    assert J.endpoint() is None                                      # nothing up → the command skips cleanly
    monkeypatch.setenv("TEV1_URL", "http://x/v1/systemone")
    assert J.endpoint() == "http://x/v1/systemone"


@pytest.fixture
def world(tmp_path):
    now = datetime(2026, 9, 18, 9, tzinfo=timezone.utc)
    rows = [
        ("a1", "CRWV", now - timedelta(hours=10), "CoreWeave launches at-the-market offering of 35 million shares", "Equity raise."),
        ("a2", "XOM", now - timedelta(hours=5), "Oil jumps as Strait of Hormuz is closed to shipping", "Tankers halted."),
        ("a3", "AAA", now - timedelta(days=3), "AAA cuts full-year guidance", "Warns on results."),
        ("a4", "BBB", now - timedelta(days=2), "BBB wins a big contract", ""),
        ("a5", "OLD", now - timedelta(days=60), "Ancient news", ""),
    ]
    df = pl.DataFrame(rows, schema=["article_id", "ticker", "published_utc", "title", "description"], orient="row")
    p = tmp_path / "data" / "bronze" / "massive"
    p.mkdir(parents=True)
    df.write_parquet(p / "news.parquet")
    conf = {"alert_p": 0.8, "watch_p": 0.5,
            "switches": {"hormuz_closed": {"source": "macro", "keywords": "hormuz|oil", "question": "Hormuz closed?", "playbook": "O=RED"},
                         "crwv_equity_raise": {"source": {"ticker": "CRWV"}, "question": "Equity raise?", "playbook": "SELL"}},
            "monitors": {"taiwan_escalation": {"keywords": "taiwan", "question": "Taiwan?"}}}
    return _Cfg(tmp_path), now, conf


def fake_ask(url, state, questions, timeout=180.0):
    text = " ".join(str(v) for v in state.values()).lower()
    if "q" in questions:                                             # switches / monitors
        hit = ("hormuz" in text and "Hormuz" in questions["q"]["instructions"]) or ("at-the-market" in text)
        return {"q": {"type": "noul", "noul": 0.95 if hit else 0.02}}
    cut = "cuts" in text
    return {**{k: {"type": "noul", "noul": 0.9 if (k == "guidance_cut" and cut) else 0.05} for k in J.FLAGS},
            "entry": {"type": "choice", "choice": "avoid" if cut else "ok",
                      "probabilities": {"ok": 0.1 if cut else 0.9, "caution": 0.05, "avoid": 0.85 if cut else 0.05}}}


def test_switches_monitors_and_screen_on_synthetic_news(world, monkeypatch):
    cfg, now, conf = world
    monkeypatch.setattr(J, "ask", fake_ask)
    sw = {r["subject"]: r for r in J.run_switches(cfg, "u", conf, now)}
    assert sw["hormuz_closed"]["p"] == 0.95 and sw["hormuz_closed"]["articles"] == 1     # keyword pre-filter kept only XOM's
    assert sw["crwv_equity_raise"]["p"] == 0.95
    mon = J.run_monitors(cfg, "u", conf, now)
    assert mon[0]["articles"] == 0 and mon[0].get("p") is None                           # no Taiwan headline → not asked
    scr = {r["subject"]: r for r in J.run_screen(cfg, "u", ["AAA", "BBB", "OLD"], now, roles={"AAA": "pick"})}
    assert scr["AAA"]["verdict"] == "avoid" and scr["AAA"]["guidance_cut"] == 0.9
    assert scr["BBB"]["verdict"] == "ok" and "verdict" not in scr["OLD"]                # 60-day-old news is outside the window
    text = J.render(date(2026, 9, 18), "u/v1/systemone", list(sw.values()), mon, list(scr.values()), conf)
    assert "🔴 FIRING hormuz_closed" in text and "→ playbook: O=RED" in text
    assert "⚠ avoid   AAA" in text and "ok (1): BBB" in text


def test_ledger_upserts_one_row_per_morning_and_subject(world, monkeypatch):
    cfg, now, conf = world
    monkeypatch.setattr(J, "ask", fake_ask)
    rows = J.run_switches(cfg, "u", conf, now) + J.run_screen(cfg, "u", ["AAA"], now)
    J.log_ledger(cfg, rows, date(2026, 9, 18))
    J.log_ledger(cfg, rows, date(2026, 9, 18))                                          # same morning, re-run
    J.log_ledger(cfg, rows, date(2026, 9, 19))
    led = pl.read_parquet(J.ledger_path(cfg))
    assert led.height == 2 * len(rows)
    assert led.filter((pl.col("kind") == "screen") & (pl.col("subject") == "AAA"))["verdict"].to_list() == ["avoid", "avoid"]
