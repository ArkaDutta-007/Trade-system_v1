"""GDELT news features — causal, null-when-uncovered (tested without network)."""
from datetime import date, timedelta

import polars as pl

from trading_system.features.gdelt_features import compute_gdelt_features, GDELT_COLUMNS


def _panel(n=80):
    rows = []
    d0 = date(2020, 1, 1)
    for i in range(n):
        d = d0 + timedelta(days=i)
        for t in ("AAPL", "NOCOV"):   # NOCOV has no GDELT history
            rows.append({"date": d, "ticker": t, "adj_close": 100.0 + i})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _gdelt():
    d0 = date(2020, 1, 1)
    rows = []
    # AAPL: coverage from day 10 onward, rising tone + variable volume
    for i in range(10, 80):
        rows.append({"ticker": "AAPL", "date": d0 + timedelta(days=i),
                     "gdelt_tone": -2.0 + 0.05 * i, "gdelt_vol": 5 + (i % 7)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_gdelt_columns_added():
    out = compute_gdelt_features(_panel(), _gdelt())
    for c in GDELT_COLUMNS:
        assert c in out.columns


def test_null_before_coverage_and_for_uncovered_ticker():
    out = compute_gdelt_features(_panel(), _gdelt())
    aapl = out.filter(pl.col("ticker") == "AAPL").sort("date")
    # before day 10 → no coverage → null tone
    assert aapl.head(10)["news_tone"].null_count() == 10
    # after coverage begins → populated
    assert aapl.tail(30)["news_tone"].drop_nulls().len() > 0
    # a ticker with zero GDELT rows → all null (dropped by the reserve gate)
    nocov = out.filter(pl.col("ticker") == "NOCOV")
    assert nocov["news_tone"].null_count() == nocov.height
    assert nocov["news_buzz"].null_count() == nocov.height


def test_tone_is_causal_ew_memory():
    out = compute_gdelt_features(_panel(), _gdelt())
    aapl = out.filter(pl.col("ticker") == "AAPL").sort("date")
    ser = aapl["news_tone"].drop_nulls().to_list()
    # rising raw tone → the EW memory is (weakly) increasing over the window
    assert ser[-1] > ser[0]


def test_none_gdelt_is_passthrough():
    p = _panel()
    assert compute_gdelt_features(p, None).equals(p)
