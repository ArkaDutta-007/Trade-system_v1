"""Futility pruning in train_horizon — hopeless families stop early,
never win selection, but still count in the deflation trial count."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from trading_system.models.forecast_train import train_horizon


def _panel(n_tickers=30, n_days=420, seed=11) -> tuple[pl.DataFrame, list[str]]:
    """Panel where `sig` genuinely ranks forward returns and `noise` doesn't."""
    rng = np.random.default_rng(seed)
    d0 = dt.date(2020, 1, 1)
    dates = []
    d = d0
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_tickers):
        tk = f"T{i:02d}"
        sig = rng.normal(0, 1, n_days)
        # forward drift follows the signal → cross-sectional predictability
        drift = 0.002 * sig
        rets = drift + rng.normal(0, 0.01, n_days)
        px = 100 * np.cumprod(1 + rets)
        # feature known at t predicts return over the *next* horizon: use the
        # signal that generated the *future* drift, shifted to be causal-ish for
        # the test's purposes (we only care about relative family performance).
        for j, day in enumerate(dates):
            rows.append({"date": day, "ticker": tk, "adj_close": float(px[j]),
                         "sig": float(sig[j]), "noise": float(rng.normal())})
    return (pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)),
            ["sig", "noise"])


class TestFutilityPruning:
    @pytest.fixture(scope="class")
    def panel(self):
        return _panel()

    def test_screen_prunes_and_winner_is_unpruned(self, panel):
        df, cols = panel
        res = train_horizon(df, cols, horizon=5, n_splits=4,
                            models=["lgbm", "xgb", "hist_gbm", "ridge"],
                            screen=True, universe_weight=0.0)
        pruned = {k for k, v in res.per_model.items() if "pruned_after_fold" in v}
        assert res.best_model_name not in pruned
        # pruned families ran fewer folds than the winner
        for k in pruned:
            assert res.per_model[k]["n_folds"] < res.per_model[res.best_model_name]["n_folds"]
        # deflation still counts every trialed family
        assert res.deflation["n_trials"] == 4

    def test_no_screen_runs_all_folds(self, panel):
        df, cols = panel
        res = train_horizon(df, cols, horizon=5, n_splits=4,
                            models=["lgbm", "ridge"],
                            screen=False, universe_weight=0.0)
        assert all("pruned_after_fold" not in v for v in res.per_model.values())
        folds = {v["n_folds"] for v in res.per_model.values()}
        assert len(folds) == 1  # every family saw the same folds

    def test_two_families_never_pruned(self, panel):
        df, cols = panel
        res = train_horizon(df, cols, horizon=5, n_splits=4,
                            models=["lgbm", "ridge"],
                            screen=True, universe_weight=0.0)
        assert all("pruned_after_fold" not in v for v in res.per_model.values())
