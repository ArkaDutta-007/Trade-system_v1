"""Purged + embargoed walk-forward CV — the leakage-proofing must hold."""
from datetime import date, timedelta

import pytest

from trading_system.models.validation import (
    purged_walkforward_splits, coverage_no_overlap,
)


def _daily_dates(n=600, start=date(2021, 1, 1)):
    # panel-style: repeat each date a few times (multiple tickers per date)
    dates = []
    for i in range(n):
        d = start + timedelta(days=i)
        dates.extend([d, d, d])
    return dates


def test_no_train_label_overlaps_test():
    dates = _daily_dates()
    for horizon in (21, 63, 252):
        splits = purged_walkforward_splits(dates, horizon_days=horizon, n_splits=4, embargo_days=5)
        assert len(splits) >= 1
        assert coverage_no_overlap(splits, dates, horizon)


def test_train_strictly_before_purge_cut():
    dates = _daily_dates()
    splits = purged_walkforward_splits(dates, horizon_days=63, n_splits=4, embargo_days=5)
    for s in splits:
        # every train row is before the purge cut, which is before test_start
        assert s.train_end < s.test_start
        assert len(s.train_idx) > 0 and len(s.test_idx) > 0


def test_test_blocks_are_disjoint_and_ordered():
    dates = _daily_dates()
    splits = purged_walkforward_splits(dates, horizon_days=21, n_splits=5, embargo_days=2)
    starts = [s.test_start for s in splits]
    assert starts == sorted(starts)
    for a, b in zip(splits, splits[1:]):
        assert a.test_end < b.test_start


def test_raises_when_too_few_dates():
    with pytest.raises(ValueError):
        purged_walkforward_splits([date(2024, 1, 1)] * 3, horizon_days=21, n_splits=5)
