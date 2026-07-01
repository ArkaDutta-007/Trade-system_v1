"""Wiki ingester — name normalisation + failed-resolution retry policy (no network)."""
import json

import polars as pl

from trading_system.ingestion import wiki_pageviews as wp


def test_clean_company_name_strips_legal_suffixes_and_case():
    assert wp._clean_company_name("APPLE INC") == "Apple"
    assert wp._clean_company_name("MICROSOFT CORP") == "Microsoft"
    assert wp._clean_company_name("ALPHABET INC CLASS A") == "Alphabet"
    assert wp._clean_company_name('"NVIDIA CORPORATION"') == "Nvidia"
    # already natural-cased names are left alone
    assert wp._clean_company_name("Berkshire Hathaway") == "Berkshire Hathaway"


def test_failed_resolution_is_not_cached(tmp_path, monkeypatch):
    """A ticker whose title doesn't resolve must NOT be written to the title cache,
    so a later run retries it (the bug that stranded ~all names at 2% coverage)."""
    def fake_resolve(name, sess, retries=4):
        return "Apple Inc." if name == "Apple" else None      # BADX never resolves

    def fake_fetch(ticker, title, start, end, cache_dir=None, session=None, overlap_days=5):
        return [{"date": "2020-01-01", "views": 42}] if title else []

    monkeypatch.setattr(wp, "resolve_wiki_title", fake_resolve)
    monkeypatch.setattr(wp, "fetch_wiki_ticker", fake_fetch)

    df = wp.collect_wiki_history(
        ["AAPL", "BADX"], names={"AAPL": "Apple", "BADX": "Nonexistent Co"},
        cache_dir=tmp_path, workers=1,
    )
    assert df.filter(pl.col("ticker") == "AAPL").height == 1
    assert df.filter(pl.col("ticker") == "BADX").height == 0

    cache = json.loads((tmp_path / "wiki_titles.json").read_text())
    assert cache.get("AAPL") == "Apple Inc."   # success cached
    assert "BADX" not in cache                  # failure NOT cached → retried next run
