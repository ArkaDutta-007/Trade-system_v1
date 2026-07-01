"""Coverage-driven backfill loop — retries the uncovered, stops on plateau,
and resumes without repeating already-done work."""
from trading_system.ingestion.backfill_util import collect_until_covered, BackfillLedger


def test_retries_uncovered_and_stops_on_plateau():
    calls: dict[str, int] = {}

    def fetch_one(t):
        calls[t] = calls.get(t, 0) + 1
        if t == "AAPL":                       # covered on round 1
            return [{"ticker": "AAPL", "v": 1}]
        if t == "MSFT" and calls[t] >= 2:     # flaky: covered only on the 2nd try
            return [{"ticker": "MSFT", "v": 1}]
        return []                             # DEAD: never has data

    rows = collect_until_covered(
        ["AAPL", "MSFT", "DEAD"], fetch_one, source="T",
        workers=1, min_coverage=0.99, max_rounds=8, cooldown_s=0.0,
    )
    covered = {r["ticker"] for r in rows}
    assert covered == {"AAPL", "MSFT"}        # DEAD stays uncovered
    assert calls["MSFT"] >= 2                 # was retried into success
    assert calls["AAPL"] == 1                 # a covered ticker is not re-fetched
    assert calls["DEAD"] < 8                  # plateau stops early, not all max_rounds


def test_stops_immediately_when_target_met():
    calls: dict[str, int] = {}

    def fetch_one(t):
        calls[t] = calls.get(t, 0) + 1
        return [{"ticker": t, "v": 1}]        # everyone covered round 1

    rows = collect_until_covered(
        ["A", "B"], fetch_one, source="T",
        workers=1, min_coverage=0.9, max_rounds=5, cooldown_s=0.0,
    )
    assert {r["ticker"] for r in rows} == {"A", "B"}
    assert calls == {"A": 1, "B": 1}          # single round


def _store_fetch():
    store = {"A": [{"ticker": "A", "v": 1}], "B": [{"ticker": "B", "v": 1}]}
    calls: list[str] = []

    def fetch_one(t):
        calls.append(t)
        return list(store.get(t, []))

    def load_cached(t):
        return list(store.get(t, []))

    return fetch_one, load_cached, calls


def test_ledger_skips_completed_tickers_with_zero_fetch(tmp_path):
    fetch_one, load_cached, calls = _store_fetch()
    led = tmp_path / "_progress.json"
    kw = dict(source="T", workers=1, min_coverage=0.9, max_rounds=3, cooldown_s=0.0,
              load_cached=load_cached)

    # run 1 — both fetched
    r1 = collect_until_covered(["A", "B"], fetch_one,
                               ledger=BackfillLedger(led, "2026-07-01"), **kw)
    assert {r["ticker"] for r in r1} == {"A", "B"}
    assert sorted(calls) == ["A", "B"]

    # run 2, SAME end — served from cache, ZERO fetches (resumability / non-repetition)
    calls.clear()
    r2 = collect_until_covered(["A", "B"], fetch_one,
                               ledger=BackfillLedger(led, "2026-07-01"), **kw)
    assert {r["ticker"] for r in r2} == {"A", "B"}
    assert calls == []

    # run 3, LATER end — everything re-attempted (bring current)
    calls.clear()
    collect_until_covered(["A", "B"], fetch_one,
                          ledger=BackfillLedger(led, "2026-07-02"), **kw)
    assert sorted(calls) == ["A", "B"]


def test_ledger_remembers_empty_so_dead_tickers_are_not_rehammered(tmp_path):
    def fetch_one(t):
        calls.append(t)
        return []                              # DEAD never resolves

    def load_cached(t):
        return []

    calls: list[str] = []
    led = tmp_path / "_progress.json"
    kw = dict(source="T", workers=1, min_coverage=0.99, max_rounds=5, cooldown_s=0.0,
              load_cached=load_cached)

    collect_until_covered(["DEAD"], fetch_one, ledger=BackfillLedger(led, "2026-07-01"), **kw)
    assert calls                                # attempted this run
    # same-day re-run: 'empty' is remembered → not fetched again
    calls.clear()
    collect_until_covered(["DEAD"], fetch_one, ledger=BackfillLedger(led, "2026-07-01"), **kw)
    assert calls == []
