"""Known-answer tests for the nonlinear-dynamics estimators.

Each estimator is checked against a signal whose mathematical properties are
known a priori (white noise, AR(1), pure sine, the logistic map at r=4,
Student-t tails, a constructed LPPLS bubble), so a regression in the maths shows
up as a failing invariant — not just "it ran".
"""
from __future__ import annotations

import numpy as np
import pytest

from trading_system.features import nonlinear as nl

N = 500


@pytest.fixture(scope="module")
def sig():
    rng = np.random.default_rng(42)

    def ar1(phi, n=N):
        e = rng.normal(size=n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    t = np.arange(N)
    lg = np.zeros(N)
    lg[0] = 0.4
    for i in range(1, N):
        lg[i] = 4 * lg[i - 1] * (1 - lg[i - 1])
    return {
        "white": rng.normal(size=N),
        "persistent": ar1(0.6),
        "reverting": ar1(-0.6),
        "sine": np.sin(2 * np.pi * t / 20),
        "logistic": lg,
        "student": rng.standard_t(3, size=N),
        "gauss": rng.normal(size=N),
        "ar1": ar1,
        "rng": rng,
    }


def test_hurst_dfa_orders_persistence(sig):
    assert 0.40 < nl.hurst_dfa(sig["white"]) < 0.62
    assert nl.hurst_dfa(sig["persistent"]) > 0.55
    assert nl.hurst_dfa(sig["reverting"]) < 0.45


def test_hurst_rs_persistent_above_random(sig):
    assert 0.40 < nl.hurst_rs(sig["white"]) < 0.66
    assert nl.hurst_rs(sig["persistent"]) > nl.hurst_rs(sig["white"])


def test_higuchi_fd_smooth_vs_noisy(sig):
    assert nl.higuchi_fd(sig["sine"]) < 1.3          # smooth curve → FD≈1
    assert nl.higuchi_fd(sig["white"]) > 1.7         # noisy curve → FD≈2


def test_permutation_entropy_bounds(sig):
    assert nl.permutation_entropy(np.arange(N).astype(float)) < 0.05  # monotone
    assert nl.permutation_entropy(sig["white"]) > 0.95               # disordered
    assert nl.permutation_entropy(sig["sine"]) < nl.permutation_entropy(sig["white"])


def test_sample_entropy_regular_below_random(sig):
    assert nl.sample_entropy(sig["sine"]) < nl.sample_entropy(sig["white"])


def test_spectral_entropy_and_dominant_period(sig):
    assert nl.spectral_entropy(sig["sine"]) < 0.3
    assert nl.spectral_entropy(sig["white"]) > 0.8
    assert abs(nl.dominant_period(sig["sine"]) - 20) < 2


def test_largest_lyapunov_detects_chaos(sig):
    lam_lg = nl.largest_lyapunov(sig["logistic"])
    lam_sine = nl.largest_lyapunov(sig["sine"])
    assert lam_lg > 0                                # positive ⇒ chaotic
    assert lam_sine < lam_lg


def test_rqa_determinism_periodic_high(sig):
    assert nl.recurrence_determinism(sig["sine"]) > 0.9
    assert nl.recurrence_determinism(sig["white"]) < nl.recurrence_determinism(sig["sine"])


def test_chaos01_zero_for_periodic_one_for_chaos(sig):
    assert nl.chaos01(sig["sine"]) < 0.3
    assert nl.chaos01(sig["logistic"]) > 0.6


def test_hill_tail_index_fat_vs_thin(sig):
    a_t = nl.hill_tail_index(sig["student"])
    a_g = nl.hill_tail_index(sig["gauss"])
    assert 1.5 < a_t < 4.5                           # Student-t(3) ≈ 3
    assert a_g > a_t                                 # Gaussian thinner-tailed


def test_early_warning_rises_toward_tipping(sig):
    ar1 = sig["ar1"]
    parts = [ar1(phi, 60) * (1 + 0.1 * k) for k, phi in enumerate(np.linspace(0.0, 0.9, 12))]
    approach = np.concatenate(parts)
    assert nl.early_warning_score(approach) > 0.3
    assert nl.early_warning_score(ar1(0.3, len(approach))) < nl.early_warning_score(approach)


def test_rough_volatility_rough_below_smooth(sig):
    rng = sig["rng"]
    vol_smooth = np.exp(np.cumsum(rng.normal(0, 0.05, N)) * 0.3)
    vol_rough = np.exp(rng.normal(0, 0.3, N))
    assert nl.rough_volatility_hurst(vol_rough) < nl.rough_volatility_hurst(vol_smooth)


def test_lppls_flags_bubble_not_plain_trend(sig):
    rng = sig["rng"]
    t = np.arange(N)
    tc, m, omega = N + 10, 0.5, 9.0
    tau = tc - t
    bubble = (5 - 0.02 * tau ** m
              + 0.004 * tau ** m * np.cos(omega * np.log(tau) - 1.0)
              + rng.normal(0, 0.002, N))
    trend = 5 + 0.001 * t + rng.normal(0, 0.01, N)
    cb = nl.lppls_confidence(bubble, n_samples=120)
    ct = nl.lppls_confidence(trend, n_samples=120)
    assert cb > 0.5                                  # genuine super-exponential bubble
    assert abs(ct) < 0.3                             # ordinary exponential trend is not flagged


def test_wavelet_hf_noisy_above_smooth(sig):
    assert nl.wavelet_hf_ratio(sig["white"]) > nl.wavelet_hf_ratio(sig["sine"])


def _price_panel(n_tickers, n_days, seed=0):
    import polars as pl
    from datetime import date, timedelta
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tickers):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        for i in range(n_days):
            rows.append((date(2022, 1, 3) + timedelta(days=i), f"T{t:02d}", float(px[i])))
    return pl.DataFrame(rows, schema=["date", "ticker", "adj_close"], orient="row")


def test_panel_features_emit_nulls_not_nan_floats():
    """Numpy NaN must become a polars *null*, not a NaN-float (which is_not_null
    would treat as present, defeating coverage gating + leaking NaN into models)."""
    import polars as pl
    from trading_system.features.nonlinear_panel import (
        compute_nonlinear_features, NONLINEAR_FAST_COLUMNS,
    )
    df = _price_panel(6, 300, seed=3)
    out = compute_nonlinear_features(df, deep=False, parallel=False)
    for c in NONLINEAR_FAST_COLUMNS:
        assert c in out.columns
        assert out[c].is_nan().sum() == 0          # no NaN-floats
        assert out[c].is_null().sum() > 0          # pre-window rows are proper nulls

    # causality: the latest value equals a manual recompute on the trailing window
    g = out.filter(pl.col("ticker") == "T00").sort("date")
    lr = np.diff(np.log(g["adj_close"].to_numpy()))
    assert abs(nl.hurst_dfa(lr[-120:]) - g["hurst_dfa_120"].to_numpy()[-1]) < 1e-6


def test_rmt_systematic_fraction_factor_vs_noise():
    import polars as pl
    from datetime import date, timedelta
    from trading_system.features.rmt import compute_rmt_features
    rng = np.random.default_rng(0)
    D, Nt = 180, 30
    days = [date(2022, 1, 3) + timedelta(days=i) for i in range(D)]

    def panel(R, tag):
        rows = []
        for j in range(R.shape[1]):
            px = 100 * np.exp(np.cumsum(R[:, j]))
            for i, d in enumerate(days):
                rows.append((d, f"{tag}{j:02d}", float(px[i])))
        return pl.DataFrame(rows, schema=["date", "ticker", "adj_close"], orient="row")

    f = rng.normal(0, 0.01, D)
    Rf = np.outer(f, rng.uniform(0.5, 1.5, Nt)) + rng.normal(0, 0.004, (D, Nt))   # common factor
    Rn = rng.normal(0, 0.01, (D, Nt))                                              # pure noise

    def last_sys(R, tag):
        o = compute_rmt_features(panel(R, tag), window=120, stride=5, min_tickers=20)
        assert o["rmt_market_beta"].is_nan().sum() == 0                            # nulls, not NaN
        return o.filter(pl.col("date") == o["date"].max())["rmt_systematic_frac"][0]

    assert last_sys(Rf, "F") > 0.3                                                 # strong systematic mode
    assert last_sys(Rn, "N") < 0.15                                                # ~ inside the MP band


def test_estimators_return_nan_on_degenerate_input():
    short = np.arange(3.0)                            # below every estimator's minimum length
    for fn in (nl.hurst_dfa, nl.hurst_rs, nl.permutation_entropy, nl.largest_lyapunov,
               nl.chaos01, nl.early_warning_score, nl.lppls_confidence):
        assert np.isnan(fn(short))
    assert np.isnan(nl.hurst_dfa(np.ones(200)))      # constant series (no fluctuation)
