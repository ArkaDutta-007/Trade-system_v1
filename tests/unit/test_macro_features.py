"""Macro feature join — leakage-safe forward fill, no back-fill."""
from datetime import date

import polars as pl

from trading_system.features.macro import join_macro_features


def _features():
    rows = []
    for d in [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]:
        for t in ["AAPL", "MSFT"]:
            rows.append({"date": d, "ticker": t, "mom_20d": 0.01})
    return pl.DataFrame(rows)


def test_join_broadcasts_macro_across_tickers_by_date():
    macro = pl.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "macro_vix": [13.0, 14.0, 15.0],
    })
    out = join_macro_features(_features(), macro)
    assert "macro_vix" in out.columns
    jan2 = out.filter(pl.col("date") == date(2024, 1, 2))
    assert set(jan2["macro_vix"].to_list()) == {14.0}  # same value for both tickers


def test_forward_fill_only_no_backfill():
    # macro missing on day 1, present day 2 -> day 1 must stay 0 (no peeking ahead)
    macro = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "macro_vix": [14.0, 15.0],
    })
    out = join_macro_features(_features(), macro).sort(["ticker", "date"])
    aapl = out.filter(pl.col("ticker") == "AAPL").sort("date")
    vals = aapl["macro_vix"].to_list()
    assert vals[0] == 0.0          # day 1 had no macro yet -> zero, NOT back-filled to 14
    assert vals[1] == 14.0
    assert vals[2] == 15.0


def test_none_macro_is_passthrough():
    feats = _features()
    assert join_macro_features(feats, None).equals(feats)
