"""Densify sparse signals — presence flags + neutral fill, gate/row-drop fix."""
from datetime import date

import polars as pl

from trading_system.features.sparse_signals import (
    densify_sparse_signals, PRESENCE_FLAGS,
)
from trading_system.features.reserve import resolve_reserve


def _df():
    # 10 rows; news covered on the last 3 only (like the 2017+ / partial panel)
    return pl.DataFrame({
        "ticker": ["AAPL"] * 10,
        "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), eager=True),
        "news_tone": [None] * 7 + [0.5, 0.6, 0.7],
        "news_tone_mom": [None] * 7 + [0.1, 0.1, 0.1],
        "news_buzz": [None] * 7 + [1.0, 2.0, 0.0],
        "sec_days_since_filing": [None] * 10,          # uncovered SEC → all null
        "sec_filings_30d": [None] * 10,
        "sec_8k_30d": [None] * 10,
        "sec_form4_90d": [None] * 10,
    })


def test_presence_flag_and_neutral_fill():
    out = densify_sparse_signals(_df())
    # flag present and 0/1 valued
    assert "news_present" in out.columns
    assert out["news_present"].to_list() == [0.0] * 7 + [1.0, 1.0, 1.0]
    # tone filled to neutral 0 where absent, real values kept where present
    assert out["news_tone"].null_count() == 0
    assert out["news_tone"].to_list()[:7] == [0.0] * 7
    assert out["news_tone"].to_list()[-1] == 0.7
    # uncovered SEC → flag all 0, recency filled with the "long ago" sentinel
    assert out["sec_present"].to_list() == [0.0] * 10
    assert out["sec_days_since_filing"].to_list() == [999.0] * 10


def test_densified_news_passes_coverage_gate():
    raw = _df()
    # before densify: 30% coverage → dropped by the 60% gate
    assert "news_tone" not in resolve_reserve(raw, groups=["news"])
    # after densify: dense → kept
    dense = densify_sparse_signals(raw)
    kept = resolve_reserve(dense, groups=["news", "presence"])
    assert "news_tone" in kept
    assert "news_present" in kept


def test_absent_source_is_noop():
    df = pl.DataFrame({"ticker": ["A"], "date": [date(2020, 1, 1)], "mom_5d": [0.1]})
    out = densify_sparse_signals(df)
    assert not any(f in out.columns for f in PRESENCE_FLAGS)
    assert out.equals(df)
