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
