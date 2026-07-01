"""Wikipedia attention features — causal z-score/momentum, null-when-uncovered."""
from datetime import date, timedelta

import polars as pl

from trading_system.features.wiki_features import compute_wiki_features, WIKI_COLUMNS


def _panel(n=120):
    d0 = date(2020, 1, 1)
    rows = []
    for i in range(n):
        d = d0 + timedelta(days=i)
        for t in ("AAPL", "NOCOV"):
            rows.append({"date": d, "ticker": t, "adj_close": 100.0})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _wiki():
    d0 = date(2020, 1, 1)
    rows = []
    for i in range(120):
        # noisy baseline ~1000 views (real traffic is never flat), a spike at day 100
        views = 1000 + (i % 7) * 30 + (5000 if i == 100 else 0)
        rows.append({"ticker": "AAPL", "date": d0 + timedelta(days=i), "wiki_views": views})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_columns_added():
    out = compute_wiki_features(_panel(), _wiki())
    for c in WIKI_COLUMNS:
        assert c in out.columns


def test_spike_lifts_attention_z():
    out = compute_wiki_features(_panel(), _wiki()).sort(["ticker", "date"])
    a = out.filter(pl.col("ticker") == "AAPL").sort("date")
    d0 = date(2020, 1, 1)
    spike = a.filter(pl.col("date") == d0 + timedelta(days=100))["wiki_attention_z"][0]
    calm = a.filter(pl.col("date") == d0 + timedelta(days=60))["wiki_attention_z"][0]
    assert spike is not None and calm is not None
    assert spike > 3.0            # a 6x jump is a large positive z
    assert abs(calm) < 1.0        # steady traffic → near zero


def test_uncovered_ticker_stays_null():
    out = compute_wiki_features(_panel(), _wiki())
    nc = out.filter(pl.col("ticker") == "NOCOV")
    assert nc["wiki_attention_z"].null_count() == nc.height


def test_none_is_passthrough():
    p = _panel()
    assert compute_wiki_features(p, None).equals(p)
