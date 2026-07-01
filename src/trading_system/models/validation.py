"""Purged + embargoed walk-forward CV — the leakage-proof core for panel ML.

The old yearly walk-forward leaks: a training row at date *d* carries a label
spanning ``[d, d+h]``.  If ``d+h`` reaches into the test window, the model has
effectively seen the future.  For a 252-day horizon that overlap is a *year* wide
— fatal for honest long-horizon evaluation.

This implements López de Prado's fix for panel (multi-ticker-per-date) data:

  * **Expanding walk-forward**: train on the past, test on a forward block.
  * **Purge**: drop training rows whose label window overlaps the test window
    (``date > test_start - horizon``).
  * **Embargo**: additionally drop a buffer of ``embargo_days`` before the test
    so slow-decaying autocorrelation can't bleed across the seam.

Splits are by *date* (so an entire cross-section moves together), returned as
row-index arrays ready to slice a Polars/NumPy design matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: date
    test_start: date
    test_end: date


def _as_dates(values) -> np.ndarray:
    """Coerce a column of dates to a 1-D numpy array of datetime.date."""
    out = []
    for v in values:
        if isinstance(v, date):
            out.append(v)
        elif hasattr(v, "date"):
            out.append(v.date())
        else:
            out.append(date.fromisoformat(str(v)[:10]))
    return np.array(out, dtype=object)


def purged_walkforward_splits(
    row_dates,
    horizon_days: int,
    n_splits: int = 5,
    embargo_days: int = 5,
    min_train_frac: float = 0.4,
) -> list[Split]:
    """Build purged+embargoed expanding-window splits.

    Parameters
    ----------
    row_dates:
        Per-row dates (list / polars Series / array). Length == n_rows.
    horizon_days:
        Label horizon in **calendar** days (use ~1.4× trading days). Drives purge.
    n_splits:
        Number of forward test blocks.
    embargo_days:
        Extra calendar-day buffer purged before each test block.
    min_train_frac:
        Fraction of the timeline reserved as the initial train before the first
        test block starts.
    """
    dates = _as_dates(row_dates)
    uniq = np.array(sorted(set(dates.tolist())))
    if len(uniq) < n_splits + 2:
        raise ValueError(f"not enough distinct dates ({len(uniq)}) for {n_splits} splits")

    start_i = int(len(uniq) * min_train_frac)
    test_blocks = np.array_split(uniq[start_i:], n_splits)

    # Precompute a date -> row index map for fast masking
    splits: list[Split] = []
    for fold, block in enumerate(test_blocks):
        if len(block) == 0:
            continue
        test_start, test_end = block[0], block[-1]
        purge_cut = test_start - timedelta(days=horizon_days + embargo_days)

        train_idx = np.where(dates < purge_cut)[0]
        test_idx = np.where((dates >= test_start) & (dates <= test_end))[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append(Split(
            fold=fold, train_idx=train_idx, test_idx=test_idx,
            train_end=purge_cut, test_start=test_start, test_end=test_end,
        ))
    if not splits:
        raise ValueError("no valid purged splits produced — check date range vs horizon")
    return splits


def coverage_no_overlap(splits: list[Split], row_dates, horizon_days: int) -> bool:
    """Assert no training label window reaches into its test window (for tests)."""
    dates = _as_dates(row_dates)
    for s in splits:
        if len(s.train_idx) == 0:
            continue
        max_train_label_end = max(dates[i] for i in s.train_idx) + timedelta(days=horizon_days)
        if max_train_label_end >= s.test_start:
            return False
    return True


# ── Combinatorial Purged Cross-Validation (López de Prado) ────────────────────

def combinatorial_purged_splits(
    row_dates,
    horizon_days: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_days: int = 5,
) -> list[Split]:
    """Combinatorial purged CV: many train/test paths instead of one.

    The timeline is cut into ``n_groups`` contiguous date blocks; for every
    combination of ``n_test_groups`` blocks held out as test, the remaining
    blocks train — minus every training row whose label window ``[d, d+h]``
    intersects a test block (purge) and an ``embargo_days`` buffer on each side.
    With ``n_groups=6, n_test_groups=2`` that's C(6,2)=15 paths, so ICIR gets a
    *distribution* rather than a single fragile walk-forward estimate.
    """
    from itertools import combinations

    dates = _as_dates(row_dates)
    uniq = np.array(sorted(set(dates.tolist())))
    if len(uniq) < n_groups * 3:
        raise ValueError(f"not enough distinct dates ({len(uniq)}) for {n_groups} CPCV groups")

    groups = np.array_split(uniq, n_groups)
    pad = timedelta(days=horizon_days + embargo_days)
    splits: list[Split] = []
    for fold, test_combo in enumerate(combinations(range(n_groups), n_test_groups)):
        test_dates = np.concatenate([groups[g] for g in test_combo])
        test_mask = np.isin(dates, test_dates)
        # purge: drop train rows whose label window intersects any test block (± embargo)
        purge = np.zeros(len(dates), dtype=bool)
        for g in test_combo:
            lo, hi = groups[g][0] - pad, groups[g][-1] + pad
            purge |= (dates >= lo) & (dates <= hi)
        train_mask = (~test_mask) & (~purge)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append(Split(
            fold=fold, train_idx=train_idx, test_idx=test_idx,
            train_end=groups[test_combo[0]][0], test_start=test_dates.min(),
            test_end=test_dates.max(),
        ))
    if not splits:
        raise ValueError("no valid CPCV splits produced")
    return splits


# ── Deflated performance (multiple-testing correction) ────────────────────────

def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` iid standard-normal draws (the null).

    Selecting the best of N models inflates the winner's score; this is the score
    a *lucky-but-skilless* best-of-N would post, so a real signal must beat it.
    Uses the standard extreme-value approximation (Bailey & López de Prado).
    """
    from scipy.stats import norm
    n = max(2, int(n_trials))
    gamma = 0.5772156649  # Euler–Mascheroni
    return float((1 - gamma) * norm.ppf(1 - 1.0 / n) + gamma * norm.ppf(1 - 1.0 / (n * np.e)))


def deflated_icir(icir: float, n_folds: int, n_trials: int) -> dict:
    """Haircut ICIR for having trialled ``n_trials`` models over ``n_folds`` folds.

    Under the null (no skill), √n_folds·ICIR ≈ N(0,1), so the best-of-N null ICIR
    is ``expected_max_sharpe(n_trials)/√n_folds``. We report the gap and a pass
    flag: the selected model is credible only if its ICIR clears that bar.
    """
    if n_folds < 2:
        return {"deflated_icir": icir, "null_max_icir": 0.0, "n_trials": n_trials, "pass": icir > 0}
    null_max = expected_max_sharpe(n_trials) / np.sqrt(n_folds)
    return {
        "deflated_icir": round(float(icir - null_max), 4),
        "null_max_icir": round(float(null_max), 4),
        "n_trials": int(n_trials),
        "pass": bool(icir > null_max),
    }
