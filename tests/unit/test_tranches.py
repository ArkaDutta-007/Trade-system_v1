"""Virtual portfolios (tranches) — book fills, mark to market, SPY alpha.

The book is paper accounting at plan entry prices, so every number here is
hand-computable: fixed entries, fixed synthetic closes, no randomness. That
lets the tests assert *exact* cost/value/pnl and the SPY counterfactual
instead of just "some number came back".
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from trading_system.execution.tranches import (
    _book_path,
    book_fills,
    list_portfolios,
    load_book,
    mark_all,
    mark_to_market,
    save_book,
)


class _StubCfg:
    """Just enough Config: path('data_bronze') → tmp dir (books live next to it)."""

    def __init__(self, root):
        self.root = root

    def path(self, key):
        p = self.root / key.replace("data_", "data/")
        p.mkdir(parents=True, exist_ok=True)
        return p


@pytest.fixture()
def cfg(tmp_path):
    return _StubCfg(tmp_path)


# --- deterministic tape: AAA rises, BBB falls, SPY drifts up ----------------
_DATES = [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)]
_LAST = {"AAA": 12.0, "BBB": 8.0, "SPY": 520.0}


@pytest.fixture()
def ohlcv() -> pl.DataFrame:
    closes = {
        "AAA": [10.0, 11.0, 12.0],
        "BBB": [10.0, 9.0, 8.0],
        "SPY": [500.0, 510.0, 520.0],
    }
    rows = [
        {"date": d, "ticker": tk, "adj_close": px}
        for tk, series in closes.items()
        for d, px in zip(_DATES, series)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _pos(ticker, entry, dollars, stop=None, median_target=None, hold_days=63):
    """Position shaped like an invest-plan position (the booking contract)."""
    return {
        "ticker": ticker, "entry": entry, "dollars": dollars,
        "hold_days": hold_days, "stop": stop, "median_target": median_target,
    }


class TestNameValidation:
    @pytest.mark.parametrize("bad", ["bad name!", "../x"])
    def test_rejects_unsafe_names(self, cfg, bad):
        # path traversal / shell-ish names must never become file paths
        with pytest.raises(ValueError):
            _book_path(cfg, bad)
        with pytest.raises(ValueError):
            load_book(cfg, bad)

    @pytest.mark.parametrize("good", ["p1", "core_2026"])
    def test_accepts_simple_names(self, cfg, good):
        p = _book_path(cfg, good)
        assert p.name == f"{good}.json"
        assert p.parent.name == "portfolios"


class TestBookFills:
    def test_roundtrip_shares_and_spy_counterfactual(self, cfg):
        book_fills(
            cfg, "p1",
            [_pos("AAA", entry=10.0, dollars=1000.0),
             _pos("BBB", entry=10.0, dollars=500.0)],
            as_of="2026-06-01", spy_close=500.0, cash_reserve=250.0,
        )
        book = load_book(cfg, "p1")
        assert book["name"] == "p1"
        assert len(book["fills"]) == 2
        f = {x["ticker"]: x for x in book["fills"]}
        assert f["AAA"]["shares"] == pytest.approx(100.0)       # 1000 / 10
        assert f["AAA"]["spy_shares"] == pytest.approx(2.0)     # 1000 / 500
        assert f["BBB"]["shares"] == pytest.approx(50.0)
        assert f["BBB"]["spy_shares"] == pytest.approx(1.0)
        assert f["AAA"]["as_of"] == "2026-06-01"
        assert f["AAA"]["hold_days"] == 63
        assert book["cash_events"] == [
            {"as_of": "2026-06-01", "amount": 250.0, "plan_id": None}]

    def test_second_booking_appends(self, cfg):
        book_fills(cfg, "p1", [_pos("AAA", 10.0, 1000.0)],
                   as_of="2026-06-01", spy_close=500.0)
        book_fills(cfg, "p1", [_pos("BBB", 10.0, 500.0)],
                   as_of="2026-06-02", spy_close=510.0)
        book = load_book(cfg, "p1")
        assert len(book["fills"]) == 2  # appended, not overwritten
        assert [x["as_of"] for x in book["fills"]] == ["2026-06-01", "2026-06-02"]

    def test_same_plan_id_books_once(self, cfg):
        # a Streamlit re-render / repeated CLI call on a cached plan must not
        # double the book: identical plan_id + ticker + source → skipped
        for _ in range(3):
            book_fills(cfg, "p1", [_pos("AAA", 10.0, 1000.0)],
                       as_of="2026-06-01", spy_close=500.0,
                       cash_reserve=100.0, plan_id="2026-07-02T10:00:00+00:00")
        book = load_book(cfg, "p1")
        assert len(book["fills"]) == 1
        assert len(book["cash_events"]) == 1
        # a NEW plan (different id) with the same ticker still books
        book_fills(cfg, "p1", [_pos("AAA", 11.0, 500.0)],
                   as_of="2026-06-02", spy_close=510.0,
                   plan_id="2026-07-03T10:00:00+00:00")
        assert len(load_book(cfg, "p1")["fills"]) == 2


class TestMarkToMarket:
    def test_exact_pnl_and_spy_alpha(self, cfg, ohlcv):
        book_fills(
            cfg, "p1",
            [_pos("AAA", 10.0, 1000.0, stop=8.0, median_target=11.0),
             _pos("BBB", 10.0, 500.0, stop=9.0, median_target=15.0)],
            as_of="2026-06-01", spy_close=500.0,
        )
        s = mark_to_market(cfg, "p1", ohlcv=ohlcv)
        assert s["n_fills"] == 2
        assert s["as_of"] == "2026-06-03"
        assert s["first_fill"] == "2026-06-01"
        assert s["cost"] == pytest.approx(1500.0)
        # 100 sh × 12 + 50 sh × 8
        assert s["value"] == pytest.approx(1600.0)
        assert s["pnl"] == pytest.approx(100.0)
        assert s["pnl_pct"] == pytest.approx(round(1600 / 1500 - 1, 4))
        # SPY counterfactual: 3.0 spy shares × last SPY close
        assert s["spy_value"] == pytest.approx(3.0 * _LAST["SPY"])
        assert s["spy_pnl_pct"] == pytest.approx(0.04)
        assert s["alpha_vs_spy"] == pytest.approx(s["value"] - s["spy_value"])
        assert s["unpriced"] == []

    def test_stop_and_target_flags(self, cfg, ohlcv):
        book_fills(
            cfg, "p1",
            [_pos("AAA", 10.0, 1000.0, stop=8.0, median_target=11.0),
             _pos("BBB", 10.0, 500.0, stop=9.0, median_target=15.0)],
            as_of="2026-06-01", spy_close=500.0,
        )
        s = mark_to_market(cfg, "p1", ohlcv=ohlcv)
        pos = {p["ticker"]: p for p in s["positions"]}
        # AAA closed at 12 ≥ target 11, well above stop 8
        assert pos["AAA"]["hit_target"] is True
        assert pos["AAA"]["hit_stop"] is False
        # BBB closed at 8 ≤ stop 9, below target 15
        assert pos["BBB"]["hit_stop"] is True
        assert pos["BBB"]["hit_target"] is False
        assert pos["AAA"]["last_price"] == pytest.approx(12.0)
        assert pos["BBB"]["pnl"] == pytest.approx(50 * 8.0 - 500.0)

    def test_tranches_same_ticker_merge(self, cfg, ohlcv):
        book_fills(cfg, "p1", [_pos("AAA", 10.0, 1000.0)],
                   as_of="2026-06-01", spy_close=500.0)
        book_fills(cfg, "p1", [_pos("AAA", 12.0, 600.0)],
                   as_of="2026-06-03", spy_close=520.0)
        s = mark_to_market(cfg, "p1", ohlcv=ohlcv)
        assert s["n_fills"] == 2
        assert len(s["positions"]) == 1  # merged into one line
        p = s["positions"][0]
        assert p["shares"] == pytest.approx(150.0)      # 100 + 50
        assert p["cost"] == pytest.approx(1600.0)
        assert p["avg_cost"] == pytest.approx(round(1600.0 / 150.0, 4))
        assert p["value"] == pytest.approx(150.0 * 12.0)
        assert p["first_fill"] == "2026-06-01"

    def test_unpriced_ticker_held_out_of_pnl(self, cfg, ohlcv):
        book_fills(
            cfg, "p1",
            [_pos("AAA", 10.0, 1000.0), _pos("GHOST", 10.0, 500.0)],
            as_of="2026-06-01", spy_close=500.0,
        )
        s = mark_to_market(cfg, "p1", ohlcv=ohlcv)
        assert s["unpriced"] == ["GHOST"]
        ghost = next(p for p in s["positions"] if p["ticker"] == "GHOST")
        assert ghost["priced"] is False
        assert ghost["last_price"] is None
        # unknown value is NOT a loss — no fake $0 valuation, no fake P&L
        assert ghost["value"] is None
        assert ghost["pnl"] is None
        # summary covers priced names only; GHOST's cost is held out and
        # reported separately so P&L% and alpha stay meaningful
        assert s["value"] == pytest.approx(100.0 * 12.0)
        assert s["cost"] == pytest.approx(1000.0)
        assert s["unpriced_cost"] == pytest.approx(500.0)
        # SPY counterfactual covers only the priced fill's dollars: no scaling
        assert s["spy_pnl_pct"] == pytest.approx(520.0 / 500.0 - 1)

    def test_empty_book(self, cfg, ohlcv):
        assert mark_to_market(cfg, "nothing", ohlcv=ohlcv) == {
            "name": "nothing", "n_fills": 0}


class TestMarkAll:
    def test_lists_every_saved_book(self, cfg, ohlcv):
        book_fills(cfg, "p1", [_pos("AAA", 10.0, 1000.0)],
                   as_of="2026-06-01", spy_close=500.0)
        book_fills(cfg, "core_2026", [_pos("BBB", 10.0, 500.0)],
                   as_of="2026-06-01", spy_close=500.0)
        save_book(cfg, {"name": "empty", "created_at": None, "fills": []})
        assert list_portfolios(cfg) == ["core_2026", "empty", "p1"]
        marks = {m["name"]: m for m in mark_all(cfg, ohlcv=ohlcv)}
        assert set(marks) == {"core_2026", "empty", "p1"}
        assert marks["empty"]["n_fills"] == 0
        assert marks["p1"]["value"] == pytest.approx(1200.0)

    def test_no_bronze_parquet_reports_error_not_empty(self, cfg):
        # without OHLCV the books still exist — the tab must say "no data",
        # not masquerade as "no portfolios yet"
        book_fills(cfg, "p1", [_pos("AAA", 10.0, 1000.0)],
                   as_of="2026-06-01", spy_close=500.0)
        out = mark_all(cfg)
        assert len(out) == 1
        assert out[0]["name"] == "p1" and out[0]["n_fills"] == 1
        assert "error" in out[0]
