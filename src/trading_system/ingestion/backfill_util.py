"""Coverage-driven, **resumable** backfill loop shared by the GDELT / SEC / Wiki
collectors.

Two problems it solves:

1. **Transient misses.** A single parallel pass strands any ticker that hit a
   rate-limit (429) or a flaky response — how an early run left GDELT at 75% and
   Wiki at 2%. This re-runs **only the still-uncovered tickers** round after round
   until a target fraction has data, then stops (target reached / nothing left /
   two rounds with no new coverage → the remainder genuinely has none).

2. **Repetition on re-run.** A restart (or a same-day re-run to top up) must not
   redo work already done. A :class:`BackfillLedger` records, per ticker, the run
   ``end`` it was brought current through and whether it finished ``covered`` or
   ``empty``; anything already done for that ``end`` is short-circuited with **zero
   network** — covered tickers load from their per-ticker cache, known-empty ones
   are skipped. The ledger is persisted every round, so an interrupted tmux run
   resumes where it stopped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..utils import parallel_map, get_logger

logger = get_logger(__name__)


class BackfillLedger:
    """Per-source progress ledger for resumable, non-repeating backfills.

    ``through`` is the run ``end`` date a ticker was last brought current through
    (ISO string — lexicographic order == chronological), so ``is_done`` is true iff
    the ticker was already handled for the *current* ``end``. A later run (larger
    ``end``) naturally re-checks everything; a same-day re-run skips finished work.
    """

    def __init__(self, path, end):
        self.path = Path(path)
        self.end = end.isoformat() if hasattr(end, "isoformat") else str(end)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}

    def is_done(self, t: str) -> bool:
        rec = self.data.get(t)
        return bool(rec and rec.get("through", "") >= self.end)

    def status(self, t: str) -> str | None:
        rec = self.data.get(t)
        return rec.get("status") if rec else None

    def mark(self, t: str, status: str) -> None:
        self.data[t] = {"through": self.end, "status": status}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))
        except Exception:
            pass


def collect_until_covered(
    tickers,
    fetch_one: Callable[[str], list[dict]],
    *,
    source: str,
    workers: int = 8,
    min_coverage: float = 0.9,
    max_rounds: int = 8,
    cooldown_s: float = 3.0,
    ledger: "BackfillLedger | None" = None,
    load_cached: Callable[[str], list[dict]] | None = None,
) -> list[dict]:
    """Retry uncovered tickers until ``min_coverage`` (fraction) or a plateau.

    ``fetch_one(ticker)`` returns a list of row-dicts each carrying a ``"ticker"``
    key (empty = no data yet). If a ``ledger`` (and ``load_cached``) is given,
    tickers already finished for the ledger's ``end`` are skipped without any fetch.
    Returns the flattened rows for every covered ticker.
    """
    tickers = [t.upper() for t in tickers]
    total = max(len(tickers), 1)
    rows_by_ticker: dict[str, list] = {}

    # ── Resume: pull already-finished tickers from cache, exclude from fetching ──
    if ledger is not None:
        pending, done_cov, done_empty = [], 0, 0
        for t in tickers:
            if not ledger.is_done(t):
                pending.append(t); continue
            if ledger.status(t) == "covered":
                cached = load_cached(t) if load_cached else []
                if cached:
                    rows_by_ticker[t] = cached; done_cov += 1
                else:
                    pending.append(t)          # ledger says covered but cache is gone → refetch
            else:
                done_empty += 1                 # known-empty for this end → skip entirely
        if done_cov or done_empty:
            logger.info(f"{source}: resuming — {done_cov} up-to-date + {done_empty} known-empty "
                        f"skipped (no fetch), {len(pending)} to do")
    else:
        pending = list(tickers)

    # ── Retry the uncovered until target / plateau ──────────────────────────────
    remaining, stall = pending, 0
    for rnd in range(1, max_rounds + 1):
        if not remaining:
            break
        results = parallel_map(fetch_one, remaining, workers=workers,
                               description=f"{source} r{rnd}")
        before = len(rows_by_ticker)
        for sub in results:
            if sub:
                t = sub[0]["ticker"]
                rows_by_ticker[t] = sub
                if ledger is not None:
                    ledger.mark(t, "covered")
        covered = len(rows_by_ticker)
        gained = covered - before
        frac = covered / total
        remaining = [t for t in remaining if t not in rows_by_ticker]
        if ledger is not None:
            ledger.save()                       # persist each round → resumable mid-run
        logger.info(f"{source}: round {rnd} → {covered}/{total} ({frac:.0%}) covered, "
                    f"+{gained} new, {len(remaining)} remaining")

        if frac >= min_coverage or not remaining:
            break
        stall = stall + 1 if gained == 0 else 0
        if stall >= 2:
            logger.warning(
                f"{source}: coverage plateaued at {frac:.0%} after {rnd} rounds — the "
                f"remaining {len(remaining)} tickers likely have no {source} data; stopping."
            )
            break
        time.sleep(cooldown_s)

    # Every ticker still uncovered here WAS attempted at least once this run (each
    # round tries all of `remaining`), so recording it as 'empty for this end' is
    # honest — and stops a same-day re-run from re-hammering it.
    if ledger is not None:
        for t in remaining:
            ledger.mark(t, "empty")
        ledger.save()

    return [r for sub in rows_by_ticker.values() for r in sub]
