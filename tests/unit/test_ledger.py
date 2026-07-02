"""Decision ledger — record → resolve → calibration, on synthetic prices."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from trading_system.monitoring.ledger import (
    calibration_report,
    load_ledger,
    prediction_id,
    record_predictions,
    resolve_ledger,
)


class _StubCfg:
    """Just enough Config for the ledger: path('data_bronze') → tmp dir."""

    def __init__(self, root):
        self.root = root

    def path(self, key):
        p = self.root / key.replace("data_", "data/")
        p.mkdir(parents=True, exist_ok=True)
        return p


def _ohlcv(ticker="AAA", n=80, start="2025-01-01", daily=0.01) -> pl.DataFrame:
    """Deterministic riser: +1%/day → predictions UP always hit."""
    d0 = dt.date.fromisoformat(start)
    dates = [d0 + dt.timedelta(days=i) for i in range(n * 2)]
    dates = [d for d in dates if d.weekday() < 5][:n]
    px = 100 * np.cumprod(np.full(n, 1 + daily))
    return pl.DataFrame({
        "date": dates, "ticker": [ticker] * n,
        "adj_close": px, "close": px,
    }).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture()
def cfg(tmp_path):
    return _StubCfg(tmp_path)


def _pred(ohlcv, horizon=21, up=True, ticker="AAA"):
    as_of = str(ohlcv["date"][10])
    entry = float(ohlcv["adj_close"][10])
    sign = 1.0 if up else -1.0
    return {
        "ticker": ticker, "as_of": as_of, "horizon_days": horizon,
        "entry_price": entry,
        "band_lo": entry * 0.8, "band_median": entry * (1 + sign * 0.1),
        "band_hi": entry * 1.6, "conviction": 0.5 if up else -0.5,
        "weight": 0.25, "dollars": 250.0, "model": "test",
    }


class TestRecord:
    def test_append_and_dedup(self, cfg):
        o = _ohlcv()
        assert record_predictions(cfg, [_pred(o)], source="invest") == 1
        # same (ticker, as_of, horizon, source) → skipped
        assert record_predictions(cfg, [_pred(o)], source="invest") == 0
        # different source → new record
        assert record_predictions(cfg, [_pred(o)], source="picks") == 1
        assert load_ledger(cfg).height == 2

    def test_prediction_id_deterministic(self):
        a = prediction_id("AAA", "2025-01-15", 21, "invest")
        assert a == prediction_id("aaa", "2025-01-15", 21, "invest")
        assert a != prediction_id("AAA", "2025-01-15", 63, "invest")


class TestResolve:
    def test_matured_up_prediction_hits(self, cfg):
        o = _ohlcv(daily=0.01)  # rises 1%/day
        record_predictions(cfg, [_pred(o, horizon=21, up=True)], source="invest")
        counts = resolve_ledger(cfg, ohlcv=o)
        assert counts == {"resolved": 1, "pending": 0, "total": 1}
        df = load_ledger(cfg)
        row = df.to_dicts()[0]
        assert row["hit"] is True
        assert row["realized_return"] > 0.15  # ~21 trading days of +1%
        assert row["in_band"] is True

    def test_down_forecast_on_rising_tape_misses(self, cfg):
        o = _ohlcv(daily=0.01)
        record_predictions(cfg, [_pred(o, horizon=21, up=False)], source="invest")
        resolve_ledger(cfg, ohlcv=o)
        assert load_ledger(cfg).to_dicts()[0]["hit"] is False

    def test_unmatured_stays_pending(self, cfg):
        o = _ohlcv(n=80)
        record_predictions(cfg, [_pred(o, horizon=252)], source="invest")
        counts = resolve_ledger(cfg, ohlcv=o)
        assert counts["resolved"] == 0 and counts["pending"] == 1
        assert load_ledger(cfg).to_dicts()[0]["realized_return"] is None

    def test_resolution_is_idempotent(self, cfg):
        o = _ohlcv()
        record_predictions(cfg, [_pred(o)], source="invest")
        resolve_ledger(cfg, ohlcv=o)
        counts = resolve_ledger(cfg, ohlcv=o)
        assert counts["resolved"] == 0
        assert load_ledger(cfg).height == 1


class TestCalibration:
    def test_report_aggregates(self, cfg):
        o = _ohlcv(daily=0.005)
        preds = [_pred(o, horizon=21, up=True)]
        p2 = _pred(o, horizon=21, up=False, ticker="AAA")
        p2["as_of"] = str(o["date"][12])
        p2["entry_price"] = float(o["adj_close"][12])
        preds.append(p2)
        record_predictions(cfg, preds, source="invest")
        resolve_ledger(cfg, ohlcv=o)
        rep = calibration_report(cfg)
        assert rep["n_predictions"] == 2 and rep["n_resolved"] == 2
        g = rep["groups"][0]
        assert g["horizon_days"] == 21
        assert g["hit_rate"] == 0.5  # one up-hit, one down-miss

    def test_empty_ledger(self, cfg):
        rep = calibration_report(cfg)
        assert rep["n_predictions"] == 0 and rep["groups"] == []
