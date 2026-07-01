"""SEC filing features — point-in-time trailing counts + recency (no network)."""
from datetime import date, timedelta

import polars as pl

from trading_system.features.sec_features import compute_sec_features, SEC_COLUMNS


def _panel(n=120):
    d0 = date(2020, 1, 1)
    rows = []
    for i in range(n):
        d = d0 + timedelta(days=i)
        for t in ("AAPL", "ETF"):        # ETF files nothing
            rows.append({"date": d, "ticker": t, "adj_close": 100.0})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _filings():
    d0 = date(2020, 1, 1)
    rows = [
        {"ticker": "AAPL", "date": d0 + timedelta(days=5), "form": "8-K"},
        {"ticker": "AAPL", "date": d0 + timedelta(days=6), "form": "4"},
        {"ticker": "AAPL", "date": d0 + timedelta(days=7), "form": "4/A"},
        {"ticker": "AAPL", "date": d0 + timedelta(days=40), "form": "10-Q"},
        {"ticker": "AAPL", "date": d0 + timedelta(days=41), "form": "424B5"},  # NOT a Form 4
    ]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_columns_added():
    out = compute_sec_features(_panel(), _filings())
    for c in SEC_COLUMNS:
        assert c in out.columns


def test_counts_are_point_in_time_and_form_specific():
    out = compute_sec_features(_panel(), _filings()).sort(["ticker", "date"])
    a = out.filter(pl.col("ticker") == "AAPL").sort("date")
    d0 = date(2020, 1, 1)

    def row(day):
        return a.filter(pl.col("date") == d0 + timedelta(days=day)).row(0, named=True)

    # day 4: nothing filed yet
    assert row(4)["sec_filings_30d"] == 0
    # day 7: three filings in the trailing 30d (8-K, 4, 4/A); two of them are Form 4
    assert row(7)["sec_filings_30d"] == 3
    assert row(7)["sec_8k_30d"] == 1
    assert row(7)["sec_form4_90d"] == 2
    assert row(7)["sec_days_since_filing"] == 0    # a filing landed today
    # day 41: 424B5 must NOT count as a Form 4
    assert row(41)["sec_form4_90d"] == 2           # still just the two early Form-4s
    # 30d window has rolled past the early cluster by day 41
    assert row(41)["sec_filings_30d"] == 2         # 10-Q (d40) + 424B5 (d41)


def test_uncovered_ticker_stays_null():
    out = compute_sec_features(_panel(), _filings())
    etf = out.filter(pl.col("ticker") == "ETF")
    assert etf["sec_filings_30d"].null_count() == etf.height


def test_none_is_passthrough():
    p = _panel()
    assert compute_sec_features(p, None).equals(p)
