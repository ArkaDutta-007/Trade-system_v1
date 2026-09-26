"""Massive (ex-Polygon) ingestion — network-free tests.

Covers: the cross-process 5/min limiter, cache/TTL semantics, ``next_url``
pagination, 429 + auth handling, grouped→frame parsing, CRSP-style split and
dividend adjustment against hand-computed values, the yfinance⊕Massive splice
continuity, the daily-update budget cap, and the flatteners.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from trading_system.ingestion import massive as M


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self, t0: float = 1_700_000_000.0):
        self.t = t0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class FakeResp:
    def __init__(self, status: int, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)[:300]

    def json(self):
        return self._payload


class FakeSession:
    """Routes by URL prefix; each route is a list of responses consumed in order (last one sticks)."""

    def __init__(self):
        self.routes: dict[str, list[FakeResp]] = {}
        self.log: list[tuple[str, dict | None, dict]] = []

    def add(self, url_prefix: str, *responses: FakeResp):
        self.routes[url_prefix] = list(responses)

    def get(self, url, params=None, headers=None, timeout=None):
        self.log.append((url, params, headers))
        for prefix, resps in self.routes.items():
            if url.startswith(prefix):
                return resps.pop(0) if len(resps) > 1 else resps[0]
        return FakeResp(404, {"status": "NOT_FOUND"})


def make_client(tmp_path: Path, session: FakeSession | None = None, clock: FakeClock | None = None, rpm=5):
    clock = clock or FakeClock()
    return M.MassiveClient(key="test-key", cache_dir=tmp_path / "raw", rpm=rpm,
                           session=session or FakeSession(), clock=clock, sleep=clock.sleep), clock


def grouped_row(t, c, o=None, h=None, l=None, v=1000, otc=False):
    return {"T": t, "o": o or c, "h": h or c, "l": l or c, "c": c, "v": v, "vw": c, "n": 10, "otc": otc}


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────────────────────────────────────

def test_rate_limiter_allows_rpm_then_blocks(tmp_path):
    clock = FakeClock()
    rl = M.RateLimiter(tmp_path / "rl.json", rpm=5, window_s=62, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        assert rl.acquire() == 0.0
    waited = rl.acquire()                       # 6th must wait until the first stamp ages out
    assert waited == pytest.approx(62.0, abs=0.01)
    assert clock.sleeps and sum(clock.sleeps) == pytest.approx(62.0, abs=0.01)


def test_rate_limiter_is_shared_across_instances(tmp_path):
    """Two 'processes' on the same file share one budget."""
    clock = FakeClock()
    a = M.RateLimiter(tmp_path / "rl.json", rpm=5, window_s=62, clock=clock, sleep=clock.sleep)
    b = M.RateLimiter(tmp_path / "rl.json", rpm=5, window_s=62, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        a.acquire()
    for _ in range(2):
        b.acquire()
    assert b.acquire() > 0                     # b sees a's three stamps


def test_rate_limiter_sliding_window(tmp_path):
    clock = FakeClock()
    rl = M.RateLimiter(tmp_path / "rl.json", rpm=5, window_s=62, clock=clock, sleep=clock.sleep)
    for i in range(5):
        rl.acquire(); clock.t += 10           # stamps at 0,10,20,30,40
    w = rl.acquire()                            # now t=50; oldest (0) expires at 62 → wait 12
    assert w == pytest.approx(12.0, abs=0.01)


def test_rate_limiter_tolerates_corrupt_file(tmp_path):
    p = tmp_path / "rl.json"; p.write_text("not json")
    clock = FakeClock()
    rl = M.RateLimiter(p, rpm=5, clock=clock, sleep=clock.sleep)
    assert rl.acquire() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Client: cache, pagination, errors
# ─────────────────────────────────────────────────────────────────────────────

def test_client_requires_key(tmp_path, monkeypatch):
    for k in M._ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(M.MassiveAuthError):
        M.MassiveClient(cache_dir=tmp_path)


def test_client_bearer_header_and_params(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v2/aggs/grouped", FakeResp(200, {"results": [grouped_row("AAPL", 10)]}))
    c, _ = make_client(tmp_path, s)
    doc = c.grouped_daily(date(2026, 9, 17))
    url, params, headers = s.log[0]
    assert headers["Authorization"] == "Bearer test-key"
    assert url.endswith("/v2/aggs/grouped/locale/us/market/stocks/2026-09-17")
    assert params == {"adjusted": "false", "include_otc": "true"}
    assert doc["results"][0]["T"] == "AAPL" and c.calls == 1


def test_client_cache_immutable_and_ttl(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/x", FakeResp(200, {"results": [1]}), FakeResp(200, {"results": [2]}))
    c, clock = make_client(tmp_path, s)
    assert c.get("/x", cache_key="k", ttl_s=None)["results"] == [1]
    assert c.get("/x", cache_key="k", ttl_s=None)["results"] == [1]      # immutable → cache hit
    assert c.calls == 1 and c.cache_hits == 1
    clock.t += 100
    assert c.get("/x", cache_key="k", ttl_s=50)["results"] == [2]        # stale under TTL → refetch
    assert c.get("/x", cache_key="k", ttl_s=0)["results"] == [2]         # ttl 0 → always refetch
    assert c.calls == 3


def test_client_follows_next_url_and_merges_pages(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v3/reference/tickers",
          FakeResp(200, {"results": [{"ticker": "A"}], "next_url": "https://api.massive.com/v3/reference/tickers?cursor=abc"}))
    s.add("https://api.massive.com/v3/reference/tickers?cursor=abc", FakeResp(200, {"results": [{"ticker": "B"}]}))
    # dict routing is prefix-based; put the cursor route first so it wins
    s.routes = dict(reversed(list(s.routes.items())))
    c, _ = make_client(tmp_path, s)
    doc = c.tickers()
    assert [r["ticker"] for r in doc["results"]] == ["A", "B"] and doc["pages"] == 2
    assert s.log[1][1] is None                                              # cursor carried by next_url, not params


def test_client_429_sleeps_then_retries(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/x", FakeResp(429, {"error": "slow down"}, {"Retry-After": "7"}),
          FakeResp(200, {"results": ["ok"]}))
    c, clock = make_client(tmp_path, s)
    assert c.get("/x")["results"] == ["ok"]
    assert 7.0 in clock.sleeps and c.calls == 2


def test_client_401_retires_key_and_403_is_entitlement(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/x", FakeResp(401, {"error": "bad key"}))
    c, _ = make_client(tmp_path, s)
    with pytest.raises(M.MassiveAuthError):                  # only key → all retired
        c.get("/x")
    assert c.calls == 1 and c.dead == [True]
    s = FakeSession(); s.add("https://api.massive.com/x", FakeResp(403, {"error": "not entitled"}))
    c, _ = make_client(tmp_path, s)
    with pytest.raises(M.MassiveEntitlementError):
        c.get("/x")
    assert c.calls == 1 and c.dead == [False]


def make_client2(tmp_path, session, keys=("k1", "k2")):
    clock = FakeClock()
    return M.MassiveClient(keys=list(keys), cache_dir=tmp_path / "raw", rpm=5, session=session,
                           clock=clock, sleep=clock.sleep), clock


def test_two_keys_double_the_budget_and_alternate(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/x", FakeResp(200, {"results": [1]}))
    c, clock = make_client2(tmp_path, s)
    for _ in range(10):
        c.get("/x")                                            # 10 calls, no sleeping
    assert c.calls == 10 and clock.sleeps == [] and c.key_calls == [5, 5]
    used = [h["Authorization"] for _, _, h in s.log]
    assert used[:4] == ["Bearer k1", "Bearer k2", "Bearer k1", "Bearer k2"]   # round-robin
    c.get("/x")                                                # 11th waits for the earliest slot
    assert len(clock.sleeps) == 1 and clock.sleeps[0] == pytest.approx(M.WINDOW_S, abs=0.01)


def test_429_on_one_key_falls_over_to_the_other_without_sleeping(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/x", FakeResp(429, {"error": "slow"}, {"Retry-After": "30"}), FakeResp(200, {"results": ["ok"]}))
    c, clock = make_client2(tmp_path, s)
    assert c.get("/x")["results"] == ["ok"]
    assert c.calls == 2 and clock.sleeps == []                 # k1 cooled, k2 served immediately
    assert c.cooldown[0] > clock() and c.cooldown[1] == 0.0
    assert [h["Authorization"] for _, _, h in s.log] == ["Bearer k1", "Bearer k2"]


def test_dead_key_is_skipped_and_others_continue(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/x", FakeResp(401, {"error": "bad"}), FakeResp(200, {"results": ["ok"]}))
    c, clock = make_client2(tmp_path, s)
    assert c.get("/x")["results"] == ["ok"]
    assert c.dead == [True, False] and c.live_keys == 1
    for _ in range(4):
        c.get("/x")
    assert all(h["Authorization"] == "Bearer k2" for _, _, h in s.log[1:])


def test_api_keys_discovery(monkeypatch):
    for k in ["MASSIVE_API_KEY", "MASSIVE_API_KEY2", "MASSIVE_API_KEY3", "POLYGON_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", "a"); monkeypatch.setenv("MASSIVE_API_KEY2", "b")
    monkeypatch.setenv("MASSIVE_API_KEY3", "a"); monkeypatch.setenv("POLYGON_API_KEY", " c ")
    assert M.api_keys() == ["a", "b", "c"] and M.api_key() == "a" and M.is_configured()


def test_client_404_is_empty_not_error(tmp_path):
    c, _ = make_client(tmp_path, FakeSession())
    assert c.ticker_details("NOPE")["results"] == []


def test_client_5xx_backs_off_then_gives_up(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/x", FakeResp(503, {}))
    c, clock = make_client(tmp_path, s)
    c.max_retries = 2
    with pytest.raises(M.MassiveError):
        c.get("/x")
    assert c.calls == 3


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def test_grouped_to_frame_normalises_and_skips_bad_rows():
    df = M.grouped_to_frame(date(2026, 9, 17), [grouped_row("BRK.B", 400.0), {"T": "X"}, grouped_row("AAPL", 10, otc=True)])
    assert df["ticker"].to_list() == ["BRK-B", "AAPL"]
    assert df.schema["date"] == pl.Date and df["otc"].to_list() == [False, True]
    assert M.grouped_to_frame(date(2026, 9, 17), []).is_empty()


def _bars(ticker, closes, start=date(2026, 1, 5)):
    ds = M.business_days(start, start + timedelta(days=60))[:len(closes)]
    return pl.DataFrame({"date": ds, "ticker": [ticker] * len(closes),
                         "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
                         "close": closes, "volume": [1000.0] * len(closes)}).with_columns(pl.col("date").cast(pl.Date))


def test_split_adjustment_matches_hand_calc():
    # 4-for-1 split executed on day index 3: raw 100,100,100,25,25 → close 25 everywhere pre-split
    bars = _bars("X", [100.0, 100.0, 100.0, 25.0, 25.0])
    exec_d = bars["date"][3]
    splits = pl.DataFrame({"ticker": ["X"], "execution_date": [exec_d], "split_from": [1.0], "split_to": [4.0], "id": ["s"]}
                          ).with_columns(pl.col("execution_date").cast(pl.Date))
    out = M.apply_corporate_actions(bars, splits, None)
    assert out["close"].to_list() == pytest.approx([25, 25, 25, 25, 25])
    assert out["volume"].to_list() == pytest.approx([4000, 4000, 4000, 1000, 1000])
    assert out["raw_close"].to_list() == pytest.approx([100, 100, 100, 25, 25])
    assert out["adj_close"].to_list() == pytest.approx([25, 25, 25, 25, 25])   # no dividends


def test_dividend_adjustment_matches_hand_calc():
    # $2 dividend with ex-date at index 2; prev close 100 → f = 0.98 applied to bars BEFORE ex-date only
    bars = _bars("X", [100.0, 100.0, 98.0, 98.0])
    ex = bars["date"][2]
    divs = pl.DataFrame({"ticker": ["X"], "ex_dividend_date": [ex], "cash_amount": [2.0]}).with_columns(pl.col("ex_dividend_date").cast(pl.Date))
    out = M.apply_corporate_actions(bars, None, divs)
    assert out["div_factor"].to_list() == pytest.approx([0.98, 0.98, 1.0, 1.0])
    assert out["adj_close"].to_list() == pytest.approx([98.0, 98.0, 98.0, 98.0])
    assert out["close"].to_list() == pytest.approx([100, 100, 98, 98])        # close is NOT dividend-adjusted


def test_two_dividends_compound_and_split_and_div_combine():
    bars = _bars("X", [100.0, 100.0, 99.0, 99.0, 24.5, 24.5])
    d1, split_d, d2 = bars["date"][2], bars["date"][4], bars["date"][5]
    divs = pl.DataFrame({"ticker": ["X", "X"], "ex_dividend_date": [d1, d2], "cash_amount": [1.0, 0.245]}).with_columns(pl.col("ex_dividend_date").cast(pl.Date))
    splits = pl.DataFrame({"ticker": ["X"], "execution_date": [split_d], "split_from": [1.0], "split_to": [4.0], "id": ["s"]}).with_columns(pl.col("execution_date").cast(pl.Date))
    out = M.apply_corporate_actions(bars, splits, divs)
    f1, f2 = 1 - 1.0 / 100.0, 1 - 0.245 / 24.5       # 0.99, 0.99 — raw units at each ex-date
    assert out["div_factor"].to_list() == pytest.approx([f1 * f2, f1 * f2, f2, f2, f2, 1.0])
    assert out["close"].to_list() == pytest.approx([25, 25, 24.75, 24.75, 24.5, 24.5])
    assert out["adj_close"][0] == pytest.approx(25 * f1 * f2)


def test_dividend_before_window_or_bogus_is_ignored():
    bars = _bars("X", [50.0, 50.0, 50.0])
    early = bars["date"][0] - timedelta(days=30)
    divs = pl.DataFrame({"ticker": ["X", "X"], "ex_dividend_date": [early, bars["date"][1]], "cash_amount": [1.0, 500.0]}).with_columns(pl.col("ex_dividend_date").cast(pl.Date))
    out = M.apply_corporate_actions(bars, None, divs)
    assert out["div_factor"].to_list() == pytest.approx([1.0, 1.0, 1.0])     # early → no prev bar; 500>close → clamped


def test_actions_only_touch_their_ticker():
    bars = pl.concat([_bars("X", [100.0, 25.0]), _bars("Y", [10.0, 10.0])])
    splits = pl.DataFrame({"ticker": ["X"], "execution_date": [bars["date"][1]], "split_from": [1.0], "split_to": [4.0], "id": ["s"]}).with_columns(pl.col("execution_date").cast(pl.Date))
    out = M.apply_corporate_actions(bars, splits, None)
    assert out.filter(pl.col("ticker") == "Y")["close"].to_list() == [10.0, 10.0]
    assert out.filter(pl.col("ticker") == "X")["close"].to_list() == [25.0, 25.0]


def test_splice_is_continuous_at_seam_and_keeps_side_only_tickers():
    deep = _bars("X", [10.0, 11.0, 12.0, 13.0]).with_columns((pl.col("close") * 0.9).alias("adj_close"))
    # Massive window starts at deep's 3rd date with prices 2× (e.g. yfinance missed a reverse split)
    recent = deep.slice(2, 2).with_columns([(pl.col(c) * 2).alias(c) for c in ("open", "high", "low", "close")]
                                            + [(pl.col("adj_close") * 2 * 1.05).alias("adj_close"),
                                               (pl.col("volume") / 2).alias("volume")])
    other = _bars("Z", [5.0, 5.0]).with_columns(pl.col("close").alias("adj_close"))
    out = M.splice_history(pl.concat([deep, other]), recent)
    x = out.filter(pl.col("ticker") == "X").sort("date")
    assert x.height == 4
    assert x["close"].to_list() == pytest.approx([20.0, 22.0, 24.0, 26.0])            # deep rescaled ×2
    assert x["adj_close"].to_list() == pytest.approx([c * 0.9 * 2 * 1.05 for c in (10, 11, 12, 13)])
    assert x["volume"].to_list() == pytest.approx([500, 500, 500, 500])
    assert out.filter(pl.col("ticker") == "Z").height == 2                            # deep-only passes through
    assert set(out.columns) == set(M.OHLCV_COLS)


def test_splice_handles_empty_sides():
    d = _bars("X", [1.0, 2.0]).with_columns(pl.col("close").alias("adj_close"))
    assert M.splice_history(pl.DataFrame(), d).height == 2
    assert M.splice_history(d, pl.DataFrame()).height == 2


def test_flatteners():
    fin = M.flatten_financials([{"tickers": ["aapl"], "cik": "1", "fiscal_year": "2026", "fiscal_period": "Q2",
                                 "timeframe": "quarterly", "filing_date": "2026-08-01",
                                 "financials": {"income_statement": {"revenues": {"value": 1e9, "unit": "USD"}},
                                                "balance_sheet": {"assets": {"value": 5e9}}}}])
    assert fin["income_statement__revenues"][0] == 1e9 and fin["ticker"][0] == "AAPL"
    assert fin.schema["filing_date"] == pl.Date
    news = M.flatten_news([{"id": "n1", "published_utc": "2026-09-17T12:00:00Z", "title": "t", "tickers": ["AAPL", "MSFT"],
                            "publisher": {"name": "P"}, "insights": [{"ticker": "AAPL", "sentiment": "positive"}]}])
    assert news.height == 2
    assert news.filter(pl.col("ticker") == "AAPL")["sentiment"][0] == 1.0
    assert news.filter(pl.col("ticker") == "MSFT")["sentiment"][0] is None
    det = M.flatten_details([{"ticker": "BRK.B", "market_cap": "1e12", "type": "CS"}])
    assert det["ticker"][0] == "BRK-B" and det["market_cap"][0] == 1e12
    assert M.month_windows(date(2026, 11, 15), date(2027, 1, 3)) == [
        (date(2026, 11, 1), date(2026, 11, 30)), (date(2026, 12, 1), date(2026, 12, 31)), (date(2027, 1, 1), date(2027, 1, 31))]


# ─────────────────────────────────────────────────────────────────────────────
# Store: crawl bookkeeping + build
# ─────────────────────────────────────────────────────────────────────────────

def _store(tmp_path, session, today=date(2026, 9, 18), rpm=1000):
    client, clock = make_client(tmp_path, session, rpm=rpm)
    st = M.MassiveStore(tmp_path / "raw", tmp_path / "bronze", client=client, today=today)
    return st, client, clock


def test_crawl_grouped_is_resumable_and_budgeted(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v2/aggs/grouped", FakeResp(200, {"results": [grouped_row("AAPL", 10)]}))
    st, client, _ = _store(tmp_path, s)
    start = st.today - timedelta(days=14)               # 11 business days incl. today
    first = st.crawl_grouped(start, max_calls=4)
    assert first.calls == 4 and first.pending_days == len(M.business_days(start, st.today)) - 4
    second = st.crawl_grouped(start)
    assert second.calls == first.pending_days and second.pending_days == 0
    third = st.crawl_grouped(start)                     # everything cached → zero network
    assert third.calls == 0


def test_recent_empty_day_is_retried_but_old_holiday_is_not(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v2/aggs/grouped", FakeResp(200, {"results": []}))
    st, client, clock = _store(tmp_path, s)
    old, recent = st.today - timedelta(days=10), st.today - timedelta(days=1)
    st.fetch_day(old); st.fetch_day(recent)
    assert client.calls == 2
    assert old not in st.missing_days(old, old)         # empty + old → treated as holiday, immutable
    assert recent in st.missing_days(recent, recent)    # empty + recent → not published yet, retry
    clock.t += 7 * 3600
    st.fetch_day(old); st.fetch_day(recent)
    assert client.calls == 3


def test_build_ohlcv_all_applies_actions_from_cache(tmp_path):
    s = FakeSession()
    st, client, _ = _store(tmp_path, s)
    d0, d1 = st.today - timedelta(days=2), st.today - timedelta(days=1)
    # 100 then 4:1 split on d1 → 25; no network: write cache directly
    client.cache_write(f"grouped/{d0.isoformat()}", {"results": [grouped_row("X", 100.0)], "fetched_at": 0})
    client.cache_write(f"grouped/{d1.isoformat()}", {"results": [grouped_row("X", 25.0)], "fetched_at": 0})
    lo, hi = M.month_windows(d1, d1)[0]
    client.cache_write(f"splits/{lo.isoformat()}_{hi.isoformat()}",
                       {"results": [{"ticker": "X", "execution_date": d1.isoformat(), "split_from": 1, "split_to": 4}], "fetched_at": 0})
    client.cache_write(f"dividends/{lo.isoformat()}_{hi.isoformat()}", {"results": [], "fetched_at": 0})
    df = st.build_ohlcv_all()
    assert df.sort("date")["close"].to_list() == pytest.approx([25.0, 25.0])
    assert st.ohlcv_all_path.exists() and st.raw_bars_path.exists()
    # incremental: adding a day only parses that day
    d2 = st.today
    client.cache_write(f"grouped/{d2.isoformat()}", {"results": [grouped_row("X", 26.0)], "fetched_at": 0})
    assert st.build_ohlcv_all().height == 3


def test_update_respects_budget_and_reports(tmp_path, monkeypatch):
    s = FakeSession()
    s.add("https://api.massive.com/v2/aggs/grouped", FakeResp(200, {"results": [grouped_row("AAPL", 10)]}))
    s.add("https://api.massive.com/v3/reference", FakeResp(200, {"results": []}))
    s.add("https://api.massive.com/v2/reference/news", FakeResp(200, {"results": []}))
    st, client, _ = _store(tmp_path, s)

    class Cfg(dict):
        def path(self, k):
            return {"data_raw": tmp_path / "raw", "data_bronze": tmp_path / "bronze"}[k]
    cfg = Cfg(data={"massive": {"update_max_calls": 3}})
    monkeypatch.setattr(M.MassiveStore, "from_config", classmethod(lambda cls, c, client=None: st))
    res = M.update(cfg, ["AAPL"])
    assert res["calls"] == 3 and res["pending_days"] > 0 and res["tables"]["ohlcv_all"] == 3


def test_ingest_spliced_raises_when_window_thin(tmp_path, monkeypatch):
    st, client, _ = _store(tmp_path, FakeSession())
    client.cache_write(f"grouped/{st.today.isoformat()}", {"results": [grouped_row("AAPL", 10)], "fetched_at": 0})

    class Cfg(dict):
        def path(self, k):
            return {"data_raw": tmp_path / "raw", "data_bronze": tmp_path / "bronze"}[k]
    cfg = Cfg(data={"massive": {}}, universe={"tickers": ["AAPL"], "name": "t"})
    monkeypatch.setattr(M.MassiveStore, "from_config", classmethod(lambda cls, c, client=None: st))
    with pytest.raises(M.MassiveNotReady):
        M.ingest_universe_spliced(cfg)


def test_ingest_universe_auto_falls_back_to_yfinance(tmp_path, monkeypatch):
    """data.source=auto with a key but an empty cache must degrade to yfinance, not fail."""
    from trading_system.ingestion import market_data as MD
    monkeypatch.setenv("MASSIVE_API_KEY", "k")
    monkeypatch.setattr(M, "ingest_universe_spliced", lambda cfg: (_ for _ in ()).throw(M.MassiveNotReady("empty")))
    called = {}
    def fake_fetch(tickers, start, end=None, workers=None, progress=True):
        called["yf"] = True
        return pl.DataFrame({"date": [date(2026, 1, 5)], "ticker": ["AAPL"], "open": [1.0], "high": [1.0],
                             "low": [1.0], "close": [1.0], "adj_close": [1.0], "volume": [1]}).with_columns(pl.col("date").cast(pl.Date))
    monkeypatch.setattr(MD, "fetch_ohlcv", fake_fetch)

    class Cfg(dict):
        def path(self, k):
            return tmp_path / "bronze"
    cfg = Cfg(data={"source": "auto", "start_date": "2026-01-01"}, universe={"tickers": ["AAPL"]})
    out = MD.ingest_universe(cfg)
    assert called["yf"] and out.exists()
    with pytest.raises(M.MassiveNotReady):
        MD.ingest_universe(cfg, source="massive")


def test_build_liquid_universe_ranks_by_dollar_volume(tmp_path):
    st, client, _ = _store(tmp_path, FakeSession())
    days = M.business_days(st.today - timedelta(days=120), st.today)
    rows = []
    for d in days:
        rows += [{"date": d, "ticker": "BIG", "close": 100.0, "volume": 1e6, "otc": False},
                 {"date": d, "ticker": "SMALL", "close": 10.0, "volume": 1e4, "otc": False},
                 {"date": d, "ticker": "PENNY", "close": 1.0, "volume": 1e9, "otc": False},
                 {"date": d, "ticker": "PINK", "close": 50.0, "volume": 1e7, "otc": True}]
    pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).write_parquet(st.ohlcv_all_path)
    u = M.build_liquid_universe(st, min_price=5.0, min_dollar_vol=5e5, top=10)
    assert u["ticker"].to_list() == ["BIG"]          # SMALL below $vol floor, PENNY below price, PINK is OTC


# ─────────────────────────────────────────────────────────────────────────────
# Continuous crawler
# ─────────────────────────────────────────────────────────────────────────────

def _crawler(tmp_path, session=None, today=None):
    today = today or (datetime.now(timezone.utc).date() - timedelta(days=3))   # all days publish-ready
    st, client, clock = _store(tmp_path, session or FakeSession(), today=today)
    return M.Crawler(st, ["AAPL", "MSFT"], log=lambda m: None, deep_prices=False), st, client


def test_crawler_priority_order_is_tier0_grouped_first(tmp_path):
    cr, st, client = _crawler(tmp_path)
    tasks = list(cr.due_tasks())
    assert tasks[0].tier == 0 and tasks[0].name.startswith("grouped")
    assert tasks[0].name.endswith(st.today.isoformat())              # newest first
    tiers = [t.tier for t in tasks]
    assert tiers == sorted(tiers)                                       # monotone by tier
    assert any(t.name == "tickers active" for t in tasks) and any(t.name == "details AAPL" for t in tasks)
    assert any(t.name.startswith("aggs-qa") for t in tasks) and any(t.name.startswith("fed ") for t in tasks)
    assert all(t.family for t in tasks)


def test_crawler_skips_cached_and_fresh_items(tmp_path):
    cr, st, client = _crawler(tmp_path)
    for d in M.business_days(st.history_start(), st.today):
        client.cache_write(f"grouped/{d.isoformat()}", {"results": [grouped_row("AAPL", 1)], "fetched_at": 0})
    client.cache_write("details/AAPL", {"results": [{"ticker": "AAPL"}], "fetched_at": 0})
    names = [t.name for t in cr.due_tasks()]
    assert not any(n.startswith("grouped") for n in names)
    assert "details AAPL" not in names and "details MSFT" in names


def test_crawler_respects_holidays_and_publish_lag(tmp_path):
    cr, st, client = _crawler(tmp_path, today=datetime.now(timezone.utc).date())
    today = st.today
    client.cache_write("reference/holidays", {"results": [{"date": (today - timedelta(days=1)).isoformat(),
                                                           "status": "closed", "exchange": "NYSE"}], "fetched_at": 0})
    names = [t.name for t in cr.due_tasks() if t.name.startswith("grouped")]
    assert f"grouped {today.isoformat()}" not in names                 # today not published yet
    assert f"grouped {(today - timedelta(days=1)).isoformat()}" not in names   # holiday


def test_crawler_run_stops_at_budget_and_rebuilds(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v2/aggs/grouped", FakeResp(200, {"results": [grouped_row("AAPL", 10)]}))
    s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path, s)
    n = cr.run(max_calls=7, idle_sleep=0)
    assert n == 7 and client.calls == 7
    assert st.ohlcv_all_path.exists() and cr.state_path.exists()
    state = json.loads(cr.state_path.read_text())
    assert state["calls_session"] == 7 and state["tier_calls"].get("0", 0) + state["tier_calls"].get("1", 0) == 7


def test_crawler_once_drains_then_exits(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path)
    cr.store._client.session = s
    # everything cached & fresh → once-mode exits immediately with zero calls
    for d in M.business_days(st.history_start(), st.today):
        client.cache_write(f"grouped/{d.isoformat()}", {"results": [grouped_row("AAPL", 1)], "fetched_at": 0})
    for key in ["reference/tickers_active", "reference/holidays", "reference/tickers_delisted",
                "details/AAPL", "details/MSFT", "financials/AAPL", "financials/MSFT",
                "news/ticker_deep/AAPL", "news/ticker_deep/MSFT", "events/AAPL", "events/MSFT", "related/AAPL", "related/MSFT",
                "dividends_ticker/AAPL", "dividends_ticker/MSFT", "splits_ticker/AAPL", "splits_ticker/MSFT",
                "short_interest_ticker/AAPL", "short_interest_ticker/MSFT",
                "short_volume_ticker/AAPL", "short_volume_ticker/MSFT",
                "news/ticker_recent/AAPL", "news/ticker_recent/MSFT", "aggs_qa/AAPL", "aggs_qa/MSFT",
                "reference/exchanges", "reference/ticker_types", "reference/ipos",
                "fed/treasury-yields", "fed/inflation", "fed/inflation-expectations",
                f"short_interest/{(st.today - timedelta(days=45)).isoformat()}"]:
        client.cache_write(key, {"results": [], "fetched_at": 0})
    for lo, hi in M.month_windows(st.history_start() - timedelta(days=31), st.today):
        for k in ("splits", "dividends"):
            client.cache_write(f"{k}/{lo.isoformat()}_{hi.isoformat()}", {"results": [], "fetched_at": 0})
    for i in (0, 1, 2):
        client.cache_write(f"news/daily/{(st.today - timedelta(days=i)).isoformat()}", {"results": [], "fetched_at": 0})
    for d in M.business_days(st.today - timedelta(days=10), st.today - timedelta(days=1)):
        client.cache_write(f"short_volume/{d.isoformat()}", {"results": [], "fetched_at": 0})
    for t in ("AAPL", "MSFT"):
        for lo, _ in st.minute_windows():
            client.cache_write(f"aggs_minute/{t}/{lo:%Y-%m}", {"results": [], "fetched_at": 0})
    assert cr.run(once=True, idle_sleep=0) == 0


def test_crawler_parks_family_on_403_and_moves_on(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/vX/reference/financials", FakeResp(403, {"error": "not entitled"}))
    s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path, s)
    for d in M.business_days(st.history_start(), st.today):            # skip the grouped tiers
        client.cache_write(f"grouped/{d.isoformat()}", {"results": [grouped_row("AAPL", 1)], "fetched_at": 0})
    cr.run(once=True, idle_sleep=0)                                     # drains tiers 0-5 then exits
    assert "financials" in cr.blocked_families()
    fin_calls = sum(1 for u, _, _ in s.log if "/vX/reference/financials" in u)
    assert fin_calls == 1                                                # one probe, then parked
    assert not any(t.family == "financials" for t in cr.due_tasks())


def test_extended_tier_takes_top_n_beyond_universe(tmp_path):
    cr, st, client = _crawler(tmp_path)
    cr.extended_top = 4
    pl.DataFrame({"ticker": ["A1", "A2", "A3", "AAPL", "MSFT"], "type": ["CS"] * 5, "active": [True] * 5}).write_parquet(st.tickers_path)
    d = st.today
    pl.DataFrame({"date": [d] * 5, "ticker": ["A1", "A2", "A3", "AAPL", "MSFT"], "close": [1.0] * 5,
                  "volume": [3e6, 2e6, 1e6, 9e6, 9e6], "otc": [False] * 5}).with_columns(pl.col("date").cast(pl.Date)).write_parquet(st.ohlcv_all_path)
    assert cr.extended_tickers() == ["A1", "A2"]                        # 4 − 2 universe names, by $vol
    t5 = [t for t in cr.due_tasks() if t.tier == 5]                     # extended depth
    assert {n.split()[-1] for n in (t.name for t in t5)} == {"A1", "A2"}
    assert any(t.name == "news-deep A1" for t in t5)                     # universe-grade depth
    assert any(t.name == "dividends-deep A1" for t in t5) and any(t.name == "short-interest-deep A1" for t in t5)
    assert any(t.name == "short-volume-deep A1" for t in t5)
    assert not any(t.name.startswith("minute") for t in t5)             # minute bars are universe-only
    t6 = [t for t in cr.due_tasks() if t.tier == 6]                     # rest of market
    assert {n.split()[-1] for n in (t.name for t in t6)} == {"A3"} and not any("deep" in t.name for t in t6)


def test_extra_tables_flatten(tmp_path):
    st, client, _ = _store(tmp_path, FakeSession())
    client.cache_write("events/META", {"results": [{"name": "Meta", "events": [{"type": "ticker_change", "date": "2022-06-09", "ticker_change": {"ticker": "META"}}]}], "fetched_at": 0})
    client.cache_write("related/AAPL", {"results": [{"ticker": "MSFT"}], "fetched_at": 0})
    client.cache_write("fed/treasury-yields", {"results": [{"date": "2026-09-18", "yield_10_year": 4.1, "yield_2_year": 3.7}], "fetched_at": 0})
    client.cache_write("short_interest/2026-08-01", {"results": [{"ticker": "AAPL", "settlement_date": "2026-08-15", "short_interest": 100}], "fetched_at": 0})
    out = st.build_extra_tables()
    assert out == {"events": 1, "related": 1, "fed_treasury_yields": 1, "short_interest": 1}
    ev = pl.read_parquet(st.bronze_dir / "events.parquet")
    assert ev["new_ticker"][0] == "META" and ev["ticker"][0] == "META"
    assert M.flat_rows([{"a": {"b": 1, "c": [1, 2]}, "d": "x"}]) == [{"a__b": 1, "a__c": "[1, 2]", "d": "x"}]


def test_market_tickers_ranks_by_liquidity_with_universe_first(tmp_path):
    cr, st, client = _crawler(tmp_path)
    pl.DataFrame({"ticker": ["ZZZ", "BIG", "ETF", "AAPL"], "type": ["CS", "CS", "ETF", "CS"], "active": [True] * 4}).write_parquet(st.tickers_path)
    d = st.today
    pl.DataFrame({"date": [d] * 3, "ticker": ["ZZZ", "BIG", "AAPL"], "close": [1.0, 100.0, 50.0],
                  "volume": [10.0, 1e6, 1e6], "otc": [False] * 3}).with_columns(pl.col("date").cast(pl.Date)).write_parquet(st.ohlcv_all_path)
    assert cr.market_tickers() == ["AAPL", "MSFT", "BIG", "ZZZ"]         # universe first, then $vol rank, ETF excluded


def test_adjustment_qa_reports_deviation(tmp_path):
    st, client, _ = _store(tmp_path, FakeSession())
    d0, d1 = st.today - timedelta(days=2), st.today - timedelta(days=1)
    pl.DataFrame({"date": [d0, d1], "ticker": ["X", "X"], "close": [25.0, 25.0]}).with_columns(pl.col("date").cast(pl.Date)).write_parquet(st.ohlcv_all_path)
    ms = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    client.cache_write("aggs_qa/X", {"results": [{"t": ms(d0), "c": 25.0}, {"t": ms(d1), "c": 25.5}], "fetched_at": 0})
    qa = M.adjustment_qa(st)
    assert qa.height == 1 and qa["max_abs_dev"][0] == pytest.approx(0.5 / 25.5)


def test_api_ticker_denormalises_for_paths_and_params(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    c, _ = make_client(tmp_path, s)
    c.ticker_details("BRK-B"); c.financials("BRK-B"); c.news(ticker="BRK-B"); c.ticker_events("BRK-B")
    urls = [u for u, _, _ in s.log]; params = [p for _, p, _ in s.log]
    assert urls[0].endswith("/v3/reference/tickers/BRK.B") and urls[3].endswith("/tickers/BRK.B/events")
    assert params[1]["ticker"] == "BRK.B" and params[2]["ticker"] == "BRK.B"
    assert c.cached("details/BRK-B") and c.cached("financials/BRK-B")      # cache keys stay normalised
    assert M.api_ticker("brk-b") == "BRK.B" and M.normalize_ticker("BRK.B") == "BRK-B"


def test_crawler_backs_off_a_failing_task_instead_of_looping(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v3/reference/tickers/AAPL", FakeResp(400, {"error": "Invalid ticker"}))
    s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path, s)
    for d in M.business_days(st.history_start(), st.today):
        client.cache_write(f"grouped/{d.isoformat()}", {"results": [grouped_row("AAPL", 1)], "fetched_at": 0})
    cr.run(once=True, idle_sleep=0)
    bad = sum(1 for u, _, _ in s.log if u.endswith("/v3/reference/tickers/AAPL"))
    assert bad == 1                                                        # one failure → parked 6 h
    assert "details AAPL" in json.loads(cr.state_path.read_text())["failing"]
    assert not any(t.name == "details AAPL" for t in [cr.next_task()] if t)


def test_deep_tasks_only_for_universe_and_extended_and_use_full_history_params(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path, s)
    names = [t.name for t in cr.due_tasks() if t.tier == 3]
    for kind in ("news-deep", "dividends-deep", "splits-deep", "short-interest-deep", "short-volume-deep", "events", "related"):
        assert f"{kind} AAPL" in names
    assert "news-backfill AAPL" not in names
    client.news(ticker="AAPL", gte=datetime(2000, 1, 1, tzinfo=timezone.utc), cache_key="news/ticker_deep/AAPL", max_pages=60)
    client.dividends_ticker("AAPL"); client.short_interest_ticker("AAPL")
    p = [x for _, x, _ in s.log]
    assert p[0]["published_utc.gte"].startswith("2000-01-01") and "published_utc.lt" not in p[0]
    assert p[1] == {"ticker": "AAPL", "limit": 1000, "sort": "ex_dividend_date", "order": "asc"}
    assert p[2]["settlement_date.gte"] == "2000-01-01" and p[2]["limit"] == 50000


def test_corporate_actions_merge_monthly_windows_with_per_ticker_history(tmp_path):
    st, client, _ = _store(tmp_path, FakeSession())
    d1 = st.today - timedelta(days=1)
    lo, hi = M.month_windows(d1, d1)[0]
    client.cache_write(f"splits/{lo.isoformat()}_{hi.isoformat()}",
                       {"results": [{"ticker": "X", "execution_date": d1.isoformat(), "split_from": 1, "split_to": 2}], "fetched_at": 0})
    client.cache_write("splits_ticker/X", {"results": [
        {"ticker": "X", "execution_date": "2005-02-28", "split_from": 1, "split_to": 2},
        {"ticker": "X", "execution_date": d1.isoformat(), "split_from": 1, "split_to": 2}], "fetched_at": 0})    # overlap → deduped
    client.cache_write("dividends_ticker/X", {"results": [{"ticker": "X", "ex_dividend_date": "2012-08-09", "cash_amount": 0.38}], "fetched_at": 0})
    splits, divs = st.build_corporate_actions()
    assert splits.height == 2 and splits["execution_date"].min() == date(2005, 2, 28)
    assert divs.height == 1 and divs["ex_dividend_date"][0] == date(2012, 8, 9)


def test_build_deep_prices_splices_yfinance_onto_massive_window(tmp_path):
    st, client, _ = _store(tmp_path, FakeSession())
    days = M.business_days(st.today - timedelta(days=400), st.today)
    massive_days = days[-100:]
    pl.DataFrame({"date": massive_days, "ticker": ["AAPL"] * 100, "open": [20.0] * 100, "high": [20.0] * 100,
                  "low": [20.0] * 100, "close": [20.0] * 100, "adj_close": [20.0] * 100, "volume": [1.0] * 100,
                  "otc": [False] * 100}).with_columns(pl.col("date").cast(pl.Date)).write_parquet(st.ohlcv_all_path)

    def fake_fetch(tickers, start, end=None, progress=True, workers=None):
        assert start == "1970-01-01" and tickers == ["AAPL", "MSFT"]
        rows = [{"date": d, "ticker": t, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "adj_close": 9.0, "volume": 2}
                for t in tickers for d in days]
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    out = st.build_deep_prices(["AAPL", "MSFT"], start="1970-01-01", fetch=fake_fetch)
    a = out.filter(pl.col("ticker") == "AAPL").sort("date")
    assert a.height == len(days) and a["source"].value_counts().sort("source")["count"].to_list() == [100, len(days) - 100]
    assert a["close"].n_unique() == 1 and a["close"][0] == pytest.approx(20.0)           # rescaled ×2 at the seam
    assert out.filter(pl.col("ticker") == "MSFT")["source"].unique().to_list() == ["yfinance"]
    assert st.deep_path.exists()


def test_maybe_deep_prices_runs_once_in_background(tmp_path, monkeypatch):
    cr, st, client = _crawler(tmp_path)
    cr.deep_prices = True
    calls = []
    monkeypatch.setattr(st, "build_deep_prices", lambda tickers, start: (calls.append((tuple(tickers), start)),
                        st.deep_path.write_text("x"), pl.DataFrame({"ticker": ["AAPL"], "date": [date(2020, 1, 1)]}))[2])
    assert cr.maybe_deep_prices() is True
    cr._deep_thread.join(5)
    assert calls == [(("AAPL", "MSFT"), "1970-01-01")]
    assert cr.maybe_deep_prices() is False                     # fresh file → no second build


# ─────────────────────────────────────────────────────────────────────────────
# Universe intraday (1-minute bars) + per-ticker short volume
# ─────────────────────────────────────────────────────────────────────────────

def _minute_row(ts_ms, px, v=100, n=3):
    return {"t": ts_ms, "o": px, "h": px + 0.5, "l": px - 0.5, "c": px, "vw": px, "v": v, "n": n}


def test_minute_to_frame_tags_sessions_in_new_york_time():
    # 2026-09-18 (EDT, UTC-4): 08:00 ET pre, 09:30 ET regular, 15:59 ET regular, 16:00 ET post
    base = int(datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)     # 08:00 ET
    rows = [_minute_row(base, 10.0), _minute_row(base + 90 * 60_000, 11.0),
            _minute_row(base + (8 * 60 - 1) * 60_000, 12.0), _minute_row(base + 8 * 3600_000, 13.0),
            _minute_row(base, 10.0)]                                                        # duplicate → dropped
    df = M.minute_to_frame("brk.b", rows)
    assert df.columns == M.MINUTE_COLS and df.height == 4
    assert df["ticker"].unique().to_list() == ["BRK-B"]
    assert df["session"].to_list() == ["pre", "regular", "regular", "post"]
    assert df["date"].unique().to_list() == [date(2026, 9, 18)]
    assert df["ts"].dtype.time_zone == "UTC" and df["ts"][0] == datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    assert M.minute_to_frame("X", []).columns == M.MINUTE_COLS


def test_aggs_minute_is_keyed_per_ticker_month_and_unadjusted(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/v2/aggs/ticker/BRK.B/range/1/minute/", FakeResp(200, {"results": [_minute_row(1, 1.0)]}))
    client, _ = make_client(tmp_path, s)
    client.aggs_minute("BRK-B", date(2026, 9, 5), date(2026, 9, 30), ttl_s=None)
    url, params, _ = s.log[0]
    assert url.endswith("/range/1/minute/2026-09-05/2026-09-30") and params["adjusted"] == "false"
    assert client.cached("aggs_minute/BRK-B/2026-09")


def test_minute_windows_clip_to_plan_window(tmp_path):
    st, _, _ = _store(tmp_path, FakeSession())
    w = st.minute_windows()
    assert w[0][0] == st.history_start() and w[-1][1] == st.today
    assert all(lo.month == hi.month for lo, hi in w) and len(w) in (24, 25)


def test_minute_tasks_are_universe_only_newest_first_and_skip_complete_months(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path, s)
    t4 = [t for t in cr.due_tasks() if t.tier == 4]
    assert t4 and all(t.name.startswith("minute ") and t.family == "aggs_minute" for t in t4)
    aapl = [t.name for t in t4 if " AAPL " in t.name]
    assert aapl[0].endswith(f"{st.today:%Y-%m}") and aapl == sorted(aapl, reverse=True)      # newest month first
    assert len(aapl) == len(st.minute_windows()) and not any("MSFT" in n for n in aapl)
    # a closed month fetched after it published is complete → never re-planned
    lo, hi = st.minute_windows()[-2]
    client.cache_write(f"aggs_minute/AAPL/{lo:%Y-%m}", {"results": [], "fetched_at": 0})
    assert f"minute AAPL {lo:%Y-%m}" not in [t.name for t in cr.due_tasks()]
    # …but one fetched BEFORE its last day published is not: backdate the file
    old = (datetime(hi.year, hi.month, hi.day, tzinfo=timezone.utc) - timedelta(days=3)).timestamp()
    f = client._cache_file(f"aggs_minute/AAPL/{lo:%Y-%m}"); os.utime(f, (old, old))
    assert f"minute AAPL {lo:%Y-%m}" in [t.name for t in cr.due_tasks()]


def test_minute_plan_edge_403_backs_off_the_month_instead_of_parking(tmp_path):
    s = FakeSession()
    cr, st, client = _crawler(tmp_path, s)
    lo0, hi0 = st.minute_windows()[0]
    s.add(f"https://api.massive.com/v2/aggs/ticker/AAPL/range/1/minute/{lo0.isoformat()}", FakeResp(403, {"status": "NOT_AUTHORIZED"}))
    s.add("https://api.massive.com/", FakeResp(200, {"results": [_minute_row(1_700_000_000_000, 5.0)]}))
    tasks = {t.name: t for t in cr.due_tasks() if t.tier == 4}
    edge = tasks[f"minute AAPL {lo0:%Y-%m}"]
    with pytest.raises(M.MassiveError) as ei:
        edge.run()
    assert not isinstance(ei.value, M.MassiveEntitlementError) and "plan edge" in str(ei.value)
    cr._note_failure(edge, ei.value)
    assert "aggs_minute" not in cr.blocked_families()
    assert cr.next_task().name != edge.name                                # backed off, others proceed
    newest = tasks[f"minute AAPL {st.today:%Y-%m}"]
    assert newest.run() == 1 and cr._dirty_minute


def test_build_minute_bars_is_per_ticker_and_incremental(tmp_path):
    st, client, _ = _store(tmp_path, FakeSession())
    t0 = int(datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc).timestamp() * 1000)
    client.cache_write("aggs_minute/AAPL/2026-08", {"results": [_minute_row(t0, 1.0), _minute_row(t0 + 60_000, 2.0)], "fetched_at": 0})
    client.cache_write("aggs_minute/AAPL/2026-09", {"results": [_minute_row(t0 + 30 * 86_400_000, 3.0)], "fetched_at": 0})
    client.cache_write("aggs_minute/MSFT/2026-09", {"results": [], "fetched_at": 0})
    out = st.build_minute_bars()
    assert out == {"AAPL": 3} and (st.minute_dir / "AAPL.parquet").exists() and not (st.minute_dir / "MSFT.parquet").exists()
    df = pl.read_parquet(st.minute_dir / "AAPL.parquet")
    assert df["close"].to_list() == [1.0, 2.0, 3.0] and df["session"][0] == "regular"
    assert st.build_minute_bars() == {}                                    # nothing newer → no work
    time.sleep(0.01)
    client.cache_write("aggs_minute/AAPL/2026-09", {"results": [_minute_row(t0 + 30 * 86_400_000, 3.0), _minute_row(t0 + 31 * 86_400_000, 4.0)], "fetched_at": 0})
    f = client._cache_file("aggs_minute/AAPL/2026-09"); now = time.time() + 5; os.utime(f, (now, now))
    assert st.build_minute_bars() == {"AAPL": 4}
    summ = st.minute_summary()
    assert summ["tickers"] == 1 and summ["rows"] == 4 and summ["first"].startswith("2026-08-03 13:30")


def test_short_volume_history_merges_into_short_volume_table(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/stocks/v1/short-volume", FakeResp(200, {"results": [
        {"ticker": "AAPL", "date": "2024-02-06", "short_volume": 5, "total_volume": 10}]}))
    st, client, _ = _store(tmp_path, s)
    client.short_volume_ticker("aapl")
    _, params, _ = s.log[0]
    assert params == {"ticker": "AAPL", "limit": 50000, "sort": "date.asc"} and client.cached("short_volume_ticker/AAPL")
    client.cache_write("short_volume/2026-09-17", {"results": [{"ticker": "AAPL", "date": "2026-09-17", "short_volume": 7, "total_volume": 10},
                                                                 {"ticker": "AAPL", "date": "2024-02-06", "short_volume": 9, "total_volume": 10}], "fetched_at": 0})
    n = st.build_extra_tables()
    df = pl.read_parquet(st.bronze_dir / "short_volume.parquet").sort("date")
    assert n["short_volume"] == 2 and df["date"].to_list() == ["2024-02-06", "2026-09-17"]


def test_financials_keep_acceptance_datetime():
    df = M.flatten_financials([{"tickers": ["AAPL"], "fiscal_year": "2026", "fiscal_period": "Q3", "timeframe": "quarterly",
                                "filing_date": "2026-08-01", "acceptance_datetime": "20260731T203015",
                                "source_filing_file_url": "https://sec/x.htm",
                                "financials": {"income_statement": {"revenues": {"value": 1.0}}}}])
    assert df["acceptance_datetime"][0] == "20260731T203015" and df["source_filing_file_url"][0].endswith("x.htm")


def test_recent_grouped_403_is_not_released_yet_and_never_parks_or_loops(tmp_path):
    s = FakeSession()
    s.add("https://api.massive.com/v2/aggs/grouped", FakeResp(403, {"status": "NOT_AUTHORIZED"}))
    s.add("https://api.massive.com/", FakeResp(200, {"results": []}))
    cr, st, client = _crawler(tmp_path, s)
    t = next(t for t in cr.due_tasks() if t.name.startswith("grouped"))
    with pytest.raises(M.MassiveNotPublished):
        t.run()
    cr._note_failure(t, M.MassiveNotPublished("x"))
    assert "grouped" not in cr.blocked_families()
    assert cr.next_task().name != t.name                                 # backed off ~30 min, others proceed
    assert cr._fail_count.get(t.name, 0) == 0                            # not an escalating failure
    # a parked family is honoured by the grouped tiers (previously it was ignored → retry storm)
    cr.block("grouped", "test")
    assert not any(x.name.startswith("grouped") for x in cr.due_tasks())
    assert M.Crawler.PUBLISH_LAG_H >= 4.0


def test_directory_history_is_lowest_priority_immutable_and_built(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/", FakeResp(200, {"results": [{"ticker": "LEH", "name": "Lehman", "type": "CS"}]}))
    cr, st, client = _crawler(tmp_path, s)
    tasks = list(cr.due_tasks())
    dirs = [t for t in tasks if t.name.startswith("directory ")]
    assert dirs and all(t.tier == 8 for t in dirs) and tasks[-len(dirs):] == dirs          # last in line
    assert dirs[0].name == "directory 2008-01-01" and all(date.fromisoformat(t.name.split()[1]).weekday() < 5 for t in dirs)
    dirs[0].run()
    _, params, _ = s.log[-1]
    assert params["date"] == "2008-01-01" and params["active"] == "true"
    assert "directory 2008-01-01" not in [t.name for t in cr.due_tasks()]                  # immutable once fetched
    n = st.build_extra_tables()
    h = pl.read_parquet(st.bronze_dir / "tickers_history.parquet")
    assert n["tickers_history"] == 1 and h["ticker"][0] == "LEH" and h["snapshot"][0] == date(2008, 1, 1)
