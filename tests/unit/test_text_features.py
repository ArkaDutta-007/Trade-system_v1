"""FinBERT text features — aggregation is tested with pre-scored events (no model)."""
from datetime import datetime, timedelta, timezone

import polars as pl

from trading_system.ingestion.news_events import EVENT_SCHEMA
from trading_system.features.text_features import compute_text_features, TEXT_COLUMNS, finbert_available


def _panel():
    rows = []
    d0 = datetime(2024, 1, 1).date()
    for i in range(40):
        d = d0 + timedelta(days=i)
        for t in ("AAPL", "MSFT"):
            rows.append({"date": d, "ticker": t, "adj_close": 100.0 + i})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _scored_events():
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    def ev(t, summ, ka, sent):
        return {
            "event_id": summ[:8], "source": "x", "source_url": "", "published_at": ka,
            "known_at": ka, "tickers": [t], "sectors": [], "event_type": "news",
            "sentiment": 0.0, "confidence": 0.5, "novelty": 0.5, "magnitude": 0.0,
            "time_horizon": "1d", "summary": summ, "content": "", "risk_flags": [],
            "finbert_sentiment": sent,
        }
    rows = [
        ev("AAPL", "great earnings", now, 0.8),
        ev("AAPL", "downgrade risk", now + timedelta(days=2), -0.6),
        ev("MSFT", "cloud strength", now + timedelta(days=1), 0.5),
    ]
    schema = {**EVENT_SCHEMA, "finbert_sentiment": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def test_text_features_from_prescored_events():
    out = compute_text_features(_panel(), events=None, scored_events=_scored_events())
    for c in TEXT_COLUMNS:
        assert c in out.columns
    aapl = out.filter(pl.col("ticker") == "AAPL").sort("date")
    # before the first headline (Jan 10) the sentiment feature is null (causal)
    pre = aapl.filter(pl.col("date") < datetime(2024, 1, 10).date())
    assert pre["finbert_sent"].null_count() == pre.height
    # on/after the headline it is populated and positive (0.8 first)
    post = aapl.filter(pl.col("date") >= datetime(2024, 1, 10).date())
    assert post["finbert_sent"].drop_nulls().len() > 0
    assert float(post.sort("date")["finbert_sent"].drop_nulls()[0]) > 0


def test_no_scored_events_is_passthrough():
    panel = _panel()
    # events with all-null finbert -> feature omitted, panel unchanged
    empty = _scored_events().with_columns(pl.lit(None, dtype=pl.Float64).alias("finbert_sentiment"))
    out = compute_text_features(panel, events=None, scored_events=empty)
    assert out.equals(panel)


def test_finbert_available_is_bool():
    assert isinstance(finbert_available(), bool)
