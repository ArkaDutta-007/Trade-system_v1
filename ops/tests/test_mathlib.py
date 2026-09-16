"""Unit tests for the estimation/allocation mathematics.

These assert mathematical PROPERTIES (invariants that must hold for any correct
implementation), not remembered outputs — so they catch real regressions
instead of pinning today's numbers.
"""
import numpy as np
import pytest

from mathlib import (james_stein_shrink, ledoit_wolf, rmt_denoise, corr_from_cov,
                     hrp_weights, kelly_fraction, vol_target_scale, effective_n)

rng = np.random.default_rng(7)


# ── James–Stein shrinkage ───────────────────────────────────────────────────
def test_shrink_preserves_mean_and_order():
    s = rng.normal(0, 1, 200)
    out = james_stein_shrink(s, ic=0.02)
    assert np.isclose(np.mean(out), np.mean(s), atol=1e-9)      # mean preserved
    assert np.array_equal(np.argsort(s), np.argsort(out))       # order preserved


def test_shrink_reduces_dispersion_more_when_skill_is_lower():
    s = rng.normal(0, 1, 200)
    lo = np.std(james_stein_shrink(s, ic=0.01))
    hi = np.std(james_stein_shrink(s, ic=0.08))
    assert lo < hi < np.std(s)


def test_shrink_full_trust_at_high_ic_is_identity():
    s = rng.normal(0, 1, 50)
    assert np.allclose(james_stein_shrink(s, ic=0.10), s)


def test_shrink_handles_degenerate_input():
    assert np.allclose(james_stein_shrink(np.array([1.0, 1.0, 1.0])), 1.0)
    assert james_stein_shrink(np.array([1.0])).shape == (1,)     # too few, passthrough


# ── covariance estimators ───────────────────────────────────────────────────
def _returns(T=300, N=20, factor=True):
    if factor:
        f = rng.normal(0, 0.01, (T, 1))
        b = rng.uniform(0.5, 1.5, (1, N))
        return f @ b + rng.normal(0, 0.005, (T, N))
    return rng.normal(0, 0.01, (T, N))


def test_ledoit_wolf_is_symmetric_psd():
    S = ledoit_wolf(_returns())
    assert np.allclose(S, S.T, atol=1e-12)
    assert np.linalg.eigvalsh(S).min() > -1e-10                  # PSD


def test_ledoit_wolf_better_conditioned_than_sample():
    X = _returns(T=30, N=25)                                     # T ~ N: ill-posed
    sample = np.cov(X, rowvar=False)
    shrunk = ledoit_wolf(X)
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)


def test_rmt_denoise_preserves_trace_and_symmetry():
    X = _returns(T=120, N=40)
    C = np.cov(X, rowvar=False)
    D = rmt_denoise(C, n_obs=120)
    assert np.allclose(np.trace(D), np.trace(C), rtol=1e-6)      # variance kept
    assert np.allclose(D, D.T, atol=1e-10)


def test_rmt_denoise_noop_when_more_obs_than_assets_is_false():
    C = np.eye(5)
    assert np.allclose(rmt_denoise(C, n_obs=3), C)               # n<=p => passthrough


def test_corr_from_cov_unit_diagonal_bounded():
    C = ledoit_wolf(_returns())
    R = corr_from_cov(C)
    assert np.allclose(np.diag(R), 1.0)
    assert R.max() <= 1.0 + 1e-12 and R.min() >= -1.0 - 1e-12


# ── HRP allocation ──────────────────────────────────────────────────────────
def test_hrp_weights_sum_to_one_and_nonnegative():
    C = ledoit_wolf(_returns())
    w = hrp_weights(C)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()


def test_hrp_gives_less_weight_to_a_redundant_cluster():
    """Three near-identical assets plus one independent: the lone asset must
    get more weight than any single member of the redundant cluster."""
    T = 500
    base = rng.normal(0, 0.01, T)
    a = base + rng.normal(0, 0.0005, T)
    b = base + rng.normal(0, 0.0005, T)
    c = base + rng.normal(0, 0.0005, T)
    d = rng.normal(0, 0.01, T)
    X = np.column_stack([a, b, c, d])
    w = hrp_weights(ledoit_wolf(X))
    assert w[3] > w[:3].max(), f"independent asset under-weighted: {w}"


def test_hrp_single_and_empty_asset():
    assert np.allclose(hrp_weights(np.array([[0.04]])), [1.0])
    assert hrp_weights(np.zeros((0, 0))).size == 0


# ── sizing ──────────────────────────────────────────────────────────────────
def test_kelly_scales_with_edge_and_is_capped():
    assert kelly_fraction(0.02, 0.04) > kelly_fraction(0.01, 0.04)
    assert kelly_fraction(10.0, 0.001, cap=1.0) == 1.0
    assert kelly_fraction(0.01, 0.0) == 0.0                      # no variance => no bet


def test_kelly_fraction_is_fractional():
    full = kelly_fraction(0.02, 0.04, fraction=1.0)
    quarter = kelly_fraction(0.02, 0.04, fraction=0.25)
    assert np.isclose(quarter, full * 0.25)


def test_vol_target_scale_inverse_and_capped():
    assert vol_target_scale(0.30, target=0.15) == pytest.approx(0.5)
    assert vol_target_scale(0.05, target=0.15, max_leverage=1.5) == 1.5
    assert vol_target_scale(0.0) == 1.0                          # degenerate


def test_effective_n_matches_intuition():
    assert effective_n(np.repeat(0.1, 10)) == pytest.approx(10.0)
    assert effective_n(np.array([1.0])) == pytest.approx(1.0)
    concentrated = np.array([0.9, 0.05, 0.05])
    assert effective_n(concentrated) < 2.0
