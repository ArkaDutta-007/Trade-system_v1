"""Point-in-time RAG retrieval over events."""
from datetime import datetime, timedelta, timezone

import polars as pl

from trading_system.ingestion.news_events import EVENT_SCHEMA
from trading_system.ingestion.rag import retrieve_ticker_news, build_context_block


def _event(ticker, summary, content, known_at):
    return {
        "event_id": summary[:8], "source": "newsdata", "source_url": "http://x",
        "published_at": known_at, "known_at": known_at, "tickers": [ticker],
        "sectors": [], "event_type": "news", "sentiment": 0.0, "confidence": 0.5,
        "novelty": 0.5, "magnitude": 0.0, "time_horizon": "1d",
        "summary": summary, "content": content, "risk_flags": [],
    }


def _events():
    now = datetime.now(timezone.utc)
    rows = [
        _event("AAPL", "Apple raises guidance on strong iPhone demand",
               "Apple lifted revenue outlook citing robust iPhone sales", now - timedelta(days=1)),
        _event("AAPL", "Analyst cuts Apple on margin pressure",
               "Downgrade citing gross margin pressure and weak China demand", now - timedelta(days=2)),
        _event("AAPL", "Apple opens retail store in Mumbai",
               "Apple inaugurated a flagship retail store in India", now - timedelta(days=3)),
        _event("AAPL", "FUTURE leak that must be excluded",
               "this is dated in the future", now + timedelta(days=5)),
    ]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)


def test_excludes_future_dated_events():
    now = datetime.now(timezone.utc)
    res = retrieve_ticker_news(_events(), "AAPL", as_of=now, k=10)
    assert all("FUTURE leak" not in r["summary"] for r in res)
    assert len(res) <= 3


def test_relevance_ranking_prefers_query_terms():
    now = datetime.now(timezone.utc)
    res = retrieve_ticker_news(_events(), "AAPL", as_of=now, k=3,
                               query="margin guidance demand earnings")
    # the retail-store article is least relevant to the finance query
    summaries = [r["summary"] for r in res]
    if "Apple opens retail store in Mumbai" in summaries:
        assert summaries.index("Apple opens retail store in Mumbai") == len(summaries) - 1


def test_context_block_respects_char_budget():
    now = datetime.now(timezone.utc)
    res = retrieve_ticker_news(_events(), "AAPL", as_of=now, k=3)
    block = build_context_block(res, max_chars=120)
    assert len(block) <= 200  # block stops adding once budget exceeded


def test_empty_events():
    assert retrieve_ticker_news(pl.DataFrame(schema=EVENT_SCHEMA), "AAPL") == []
