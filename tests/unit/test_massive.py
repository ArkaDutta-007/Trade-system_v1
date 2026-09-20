"""Massive (ex-Polygon) ingestion — network-free tests.

Covers: the cross-process 5/min limiter, cache/TTL semantics, ``next_url``
pagination, 429 + auth handling, grouped→frame parsing, CRSP-style split and
dividend adjustment against hand-computed values, the yfinance⊕Massive splice
continuity, the daily-update budget cap, and the flatteners.
"""
from __future__ import annotations

import json
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


def test_client_auth_error_not_retried(tmp_path):
    s = FakeSession(); s.add("https://api.massive.com/x", FakeResp(403, {"error": "forbidden"}))
    c, _ = make_client(tmp_path, s)
    with pytest.raises(M.MassiveAuthError):
        c.get("/x")
    assert c.calls == 1


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
