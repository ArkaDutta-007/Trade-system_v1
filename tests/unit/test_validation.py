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


# ── CPCV + deflation ─────────────────────────────────────────────────────────

def test_cpcv_produces_all_paths_for_short_horizon():
    from math import comb
    from trading_system.models.validation import combinatorial_purged_splits
    # short horizon + ample data → every C(6,2)=15 combinatorial path is usable
    splits = combinatorial_purged_splits(_daily_dates(n=1200), horizon_days=21,
                                         n_groups=6, n_test_groups=2, embargo_days=5)
    assert len(splits) == comb(6, 2)


def test_cpcv_no_train_label_overlaps_test():
    from trading_system.models.validation import combinatorial_purged_splits, _as_dates
    from datetime import timedelta
    dates = _daily_dates(n=1500)
    for horizon in (21, 63, 252):
        splits = combinatorial_purged_splits(dates, horizon_days=horizon, n_groups=6,
                                             n_test_groups=2, embargo_days=5)
        assert len(splits) >= 1   # long horizon may drop fully-purged paths (correct)
        d = _as_dates(dates)
        for s in splits:
            # no train row's label window [d, d+h] reaches into any test date
            test_set = set(d[i] for i in s.test_idx)
            for i in s.train_idx:
                window_end = d[i] + timedelta(days=horizon)
                assert not any(d[i] <= t <= window_end for t in test_set)


def test_cpcv_train_and_test_disjoint():
    from trading_system.models.validation import combinatorial_purged_splits
    splits = combinatorial_purged_splits(_daily_dates(600), horizon_days=21,
                                         n_groups=5, n_test_groups=2)
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))
        assert len(s.train_idx) > 0 and len(s.test_idx) > 0


def test_expected_max_sharpe_grows_with_trials():
    from trading_system.models.validation import expected_max_sharpe
    assert expected_max_sharpe(2) < expected_max_sharpe(10) < expected_max_sharpe(100)


def test_deflated_icir_haircut():
    from trading_system.models.validation import deflated_icir
    # more trials -> bigger null bar -> smaller deflated icir
    d1 = deflated_icir(1.0, n_folds=5, n_trials=2)
    d2 = deflated_icir(1.0, n_folds=5, n_trials=20)
    assert d2["null_max_icir"] > d1["null_max_icir"]
    assert d2["deflated_icir"] < d1["deflated_icir"]
    # a strong signal clears the bar; noise doesn't
    assert deflated_icir(2.0, n_folds=5, n_trials=5)["pass"] is True
    assert deflated_icir(0.05, n_folds=5, n_trials=20)["pass"] is False
