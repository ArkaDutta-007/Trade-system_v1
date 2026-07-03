"""Moonshot discovery — asymmetry bands, EDGAR hit parsing, sleeper screen.

Only the pure/deterministic parts are tested (no EDGAR or yfinance calls):
the asymmetry bootstrap is seeded inside the function, the EDGAR parser gets
fixture hits shaped like real full-text-search responses, and the sleeper
screen runs on a hand-built latest-date panel where the ranking is obvious.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from trading_system.decision.discover import (
    _sleepers_from_panel,
    asymmetry_from_returns,
)
from trading_system.ingestion.edgar_discovery import _parse_hit


class TestAsymmetryFromReturns:
    def test_too_short_history_returns_none(self):
        # < 40 obs → can't band; the caller shows "watch" instead
        assert asymmetry_from_returns(np.full(39, 0.01)) is None

    def test_positively_skewed_returns_are_lopsided(self):
        # mostly -1% grind with an occasional +10% pop — the moonshot shape
        rets = np.full(200, -0.01)
        rets[::10] = 0.10
        band = asymmetry_from_returns(rets)
        assert set(band) == {"lo", "median", "hi", "asym"}
        assert band["asym"] > 1.5
        assert band["hi"] > abs(band["lo"])
        assert band["lo"] < 0 < band["hi"]

    def test_symmetric_returns_stay_in_sane_band(self):
        rng = np.random.default_rng(0)
        band = asymmetry_from_returns(rng.normal(0.0, 0.01, 500))
        # log-compounding gives a mild natural upside tilt, but nothing wild
        assert 0.5 < band["asym"] < 2.5

    def test_deterministic_given_same_input(self):
        rets = np.full(100, 0.002)
        rets[::7] = -0.015
        assert asymmetry_from_returns(rets) == asymmetry_from_returns(rets)


class TestParseHit:
    def test_full_display_name(self):
        h = {"_source": {
            "display_names": [
                "First Tracks Biotherapeutics, Inc.  (TRAX)  (CIK 0002091349)"],
            "file_type": "10-12B/A",
            "file_date": "2026-03-27",
        }}
        row = _parse_hit(h)
        assert row["ticker_hint"] == "TRAX"
        assert row["cik"] == "2091349"          # zero-padding stripped
        assert row["company"] == "First Tracks Biotherapeutics, Inc."
        assert row["form"] == "10-12B/A"
        assert row["filed"] == "2026-03-27"

    def test_no_ticker_parenthetical(self):
        # pre-listing filers often have no ticker yet — CIK must still parse
        h = {"_source": {
            "display_names": ["Quiet Spin Co.  (CIK 0000012345)"],
            "file_type": "10-12B",
            "file_date": "2026-05-01",
        }}
        row = _parse_hit(h)
        assert row["ticker_hint"] is None
        assert row["cik"] == "12345"
        assert row["company"] == "Quiet Spin Co."

    def test_empty_display_names(self):
        assert _parse_hit({"_source": {"display_names": []}}) is None
        assert _parse_hit({"_source": {}}) is None
        assert _parse_hit({}) is None


class TestSleepersFromPanel:
    @pytest.fixture()
    def panel(self) -> pl.DataFrame:
        """Latest-date gold-panel slice with an obvious sleeper.

        SLPR: thin ($3M/day) but attention igniting on every channel.
        MEGA: huge dollar volume, fading attention — the opposite of quiet.
        PNNY: $0.50 zombie (fails the $1 floor).
        THIN: $100k/day (fails the $2M exit-liquidity floor).
        BLND: unremarkable mid — just there so z-scores have a middle.
        """
        d = dt.date(2026, 6, 30)
        return pl.DataFrame({
            "ticker": ["SLPR", "MEGA", "PNNY", "THIN", "BLND"],
            "date": [d] * 5,
            "adj_close": [12.0, 300.0, 0.5, 5.0, 50.0],
            "avg_dollar_volume_20": [3e6, 5e9, 2.5e6, 1e5, 5e7],
            "wiki_attention_mom": [2.0, -0.5, 0.1, 0.0, 0.2],
            "news_tone_mom": [1.5, -0.2, 0.0, 0.0, 0.1],
            "sec_form4_90d": [4.0, 0.0, 0.0, 0.0, 1.0],
            "news_buzz": [3.0, 0.5, 0.0, 0.0, 0.5],
        }).with_columns(pl.col("date").cast(pl.Date))

    def test_sleeper_outranks_mega_cap(self, panel):
        out = _sleepers_from_panel(panel, exclude=set())
        order = [r["ticker"] for r in out]
        assert "SLPR" in order and "MEGA" in order
        assert order.index("SLPR") < order.index("MEGA")

    def test_penny_and_illiquid_names_excluded(self, panel):
        tickers = {r["ticker"] for r in _sleepers_from_panel(panel, exclude=set())}
        assert "PNNY" not in tickers   # < $1
        assert "THIN" not in tickers   # < $2M/day dollar volume

    def test_exclude_set_respected(self, panel):
        tickers = {r["ticker"] for r in _sleepers_from_panel(panel, exclude={"SLPR"})}
        assert "SLPR" not in tickers

    def test_output_shape(self, panel):
        out = _sleepers_from_panel(panel, exclude=set())
        assert out, "screen should surface at least the obvious sleeper"
        for r in out:
            assert r["category"] == "sleeper"
            assert {"quiet_pct", "ignition_z", "last_price"} <= set(r)
            assert r["last_price"] >= 1.0

    def test_top_k_caps_output(self, panel):
        assert len(_sleepers_from_panel(panel, exclude=set(), top_k=1)) == 1

    def test_missing_required_columns_returns_empty(self, panel):
        assert _sleepers_from_panel(panel.drop("avg_dollar_volume_20"),
                                    exclude=set()) == []
