"""RMT-cleaned HRP allocation — known-answer tests."""
from __future__ import annotations

import numpy as np
import pytest

from trading_system.portfolio.allocate import (
    blend_weights,
    budget_to_positions,
    clean_correlation,
    hrp_weights,
    marchenko_pastur_lambda_plus,
)


def _one_factor_returns(n_assets=8, n_obs=500, beta=0.8, seed=3):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n_obs)
    idio = rng.normal(0, 0.01, (n_obs, n_assets))
    return beta * market[:, None] + idio


class TestCleanCorrelation:
    def test_pure_noise_flattens_toward_identity(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, (500, 10))
        corr = np.corrcoef(rets, rowvar=False)
        cleaned = clean_correlation(corr, n_obs=500)
        off = cleaned[~np.eye(10, dtype=bool)]
        # noise off-diagonals shrink on average
        assert np.abs(off).mean() <= np.abs(corr[~np.eye(10, dtype=bool)]).mean() + 1e-12
        assert np.allclose(np.diag(cleaned), 1.0)

    def test_one_factor_market_mode_survives(self):
        rets = _one_factor_returns()
        corr = np.corrcoef(rets, rowvar=False)
        cleaned = clean_correlation(corr, n_obs=rets.shape[0])
        top = np.linalg.eigvalsh(cleaned)[-1]
        lam_plus = marchenko_pastur_lambda_plus(8, rets.shape[0])
        assert top > lam_plus  # the real collective mode is preserved

    def test_psd(self):
        rets = _one_factor_returns(n_assets=6)
        corr = np.corrcoef(rets, rowvar=False)
        cleaned = clean_correlation(corr, n_obs=rets.shape[0])
        assert np.linalg.eigvalsh(cleaned).min() > -1e-10


class TestHRP:
    def test_weights_sum_to_one_and_positive(self):
        rets = _one_factor_returns(n_assets=6)
        cov = np.cov(rets, rowvar=False)
        w = hrp_weights(cov)
        assert w.shape == (6,)
        assert np.isclose(w.sum(), 1.0)
        assert (w > 0).all()

    def test_low_vol_uncorrelated_asset_gets_more_weight(self):
        rng = np.random.default_rng(7)
        n = 500
        # two highly correlated risky assets + one quiet independent one
        base = rng.normal(0, 0.02, n)
        rets = np.column_stack([
            base + rng.normal(0, 0.004, n),
            base + rng.normal(0, 0.004, n),
            rng.normal(0, 0.005, n),
        ])
        w = hrp_weights(np.cov(rets, rowvar=False))
        assert w[2] > w[0]
        assert w[2] > w[1]

    def test_single_asset(self):
        assert hrp_weights(np.array([[0.01]]))[0] == 1.0


class TestBlendWeights:
    def test_cap_and_normalisation(self):
        rets = _one_factor_returns(n_assets=4)
        tickers = ["A", "B", "C", "D"]
        kelly = {"A": 10.0, "B": 0.1, "C": 0.1, "D": 0.1}
        w = blend_weights(tickers, kelly, rets, max_weight=0.30)
        assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0
        assert max(w.values()) <= 0.30 + 1e-9

    def test_short_history_falls_back_to_conviction(self):
        tickers = ["A", "B"]
        w = blend_weights(tickers, {"A": 3.0, "B": 1.0}, None, max_weight=0.9)
        assert w["A"] > w["B"]
        assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0

    def test_empty(self):
        assert blend_weights([], {}, None) == {}


class TestBudgetToPositions:
    def test_dollars_and_shares(self):
        pos, leftover = budget_to_positions(
            {"A": 0.6, "B": 0.4}, {"A": 100.0, "B": 50.0},
            deployable=1000.0, min_position=50.0,
        )
        assert pos["A"]["dollars"] == 600.0
        assert pos["A"]["shares"] == 6.0
        assert pos["A"]["whole_shares"] == 6
        assert pos["B"]["dollars"] == 400.0
        assert leftover == 0.0

    def test_min_position_dropped_and_redistributed(self):
        pos, _ = budget_to_positions(
            {"A": 0.9, "B": 0.1}, {"A": 10.0, "B": 10.0},
            deployable=300.0, min_position=50.0,
        )
        assert "B" not in pos            # 0.1 × 300 = $30 < $50 minimum
        assert pos["A"]["dollars"] == 300.0

    def test_zero_budget(self):
        pos, leftover = budget_to_positions({"A": 1.0}, {"A": 10.0}, 0.0)
        assert pos == {} and leftover == 0.0
