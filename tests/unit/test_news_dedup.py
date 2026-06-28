"""Near-duplicate headline dedup."""
from trading_system.ingestion.dedup import dedup_articles, jaccard, _tokens


def test_jaccard_identical_and_disjoint():
    a = _tokens("Apple beats earnings expectations")
    b = _tokens("Apple beats earnings expectations")
    assert jaccard(a, b) == 1.0
    c = _tokens("Tesla recalls vehicles worldwide")
    assert jaccard(a, c) == 0.0


def test_dedup_collapses_near_duplicates_per_ticker():
    arts = [
        {"ticker": "AAPL", "title": "Apple beats earnings expectations in Q3"},
        {"ticker": "AAPL", "title": "Apple Inc beats earnings expectations for Q3"},  # near-dup
        {"ticker": "AAPL", "title": "Apple unveils new Vision Pro headset"},          # distinct
    ]
    kept = dedup_articles(arts, threshold=0.8)
    assert len(kept) == 2
    # first occurrence in a cluster wins
    assert kept[0]["title"].startswith("Apple beats earnings expectations in Q3")


def test_dedup_keeps_same_headline_across_different_tickers():
    arts = [
        {"ticker": "AAPL", "title": "Tech selloff hits megacaps hard today"},
        {"ticker": "MSFT", "title": "Tech selloff hits megacaps hard today"},
    ]
    kept = dedup_articles(arts, threshold=0.9)
    assert len(kept) == 2


def test_dedup_empty():
    assert dedup_articles([]) == []
